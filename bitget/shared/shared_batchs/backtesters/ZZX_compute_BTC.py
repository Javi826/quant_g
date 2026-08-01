#shared/shared_batchs/backtesters/ZX_compute_BT.py
#
# Pure Python backup of the Cython backtest engine (ZX_compute_BT.pyx).
# Same logic, same public API, same `prepared_data` structure — much slower,
# intended only as a fallback when the compiled extension is unavailable.

import heapq
import logging
import warnings
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

MIN_PRICE       = 0.00001
INITIAL_BALANCE = 1000
COMISION        = 0.1


# ============================================================
# prepare_static_arrays  (independent of the signal, cacheable)
# ============================================================
def prepare_static_arrays(ohlcv_arrays):
    if not ohlcv_arrays:
        return {
            "symbols": [], "sym_ids": {}, "ts_int_arrays": {}, "close_arrays": {},
            "all_timestamps_int": np.array([], dtype=np.int64),
            "all_timestamps_dt":  np.array([], dtype='datetime64[ns]'),
            "open_2d": np.empty((0, 0)), "close_2d": np.empty((0, 0)),
            "high_2d": np.empty((0, 0)), "low_2d": np.empty((0, 0)),
            "high_time_2d": np.empty((0, 0), dtype=np.int64),
            "low_time_2d":  np.empty((0, 0), dtype=np.int64),
            "ts_int_2d":    np.empty((0, 0), dtype=np.int64),
            "sym_len":      np.array([], dtype=np.int64),
        }

    symbols       = list(ohlcv_arrays.keys())
    n_syms        = len(symbols)
    sym_ids       = {s: i for i, s in enumerate(sorted(symbols))}
    ts_int_arrays = {}
    close_arrays  = {}
    per_sym_ts    = {}
    all_ts_int_lists = []

    for sym in symbols:
        data = ohlcv_arrays[sym]
        ts   = data['ts']
        if ts.dtype.kind != 'M':
            ts = ts.astype('datetime64[ns]')
        ts_int = ts.view('int64')
        per_sym_ts[sym] = (ts_int, len(ts))
        ts_int_arrays[sym] = ts_int
        close_arrays[sym]  = data['close']
        all_ts_int_lists.append(ts_int)

    max_len      = max(per_sym_ts[s][1] for s in symbols)
    open_2d      = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    close_2d     = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    high_2d      = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    low_2d       = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    high_time_2d = np.full((n_syms, max_len), 0,      dtype=np.int64)
    low_time_2d  = np.full((n_syms, max_len), 0,      dtype=np.int64)
    ts_int_2d    = np.full((n_syms, max_len), 0,      dtype=np.int64)
    sym_len      = np.zeros(n_syms, dtype=np.int64)

    for sym in symbols:
        sid  = sym_ids[sym]
        data = ohlcv_arrays[sym]
        ts_int, n = per_sym_ts[sym]
        sym_len[sid]          = n
        open_2d[sid, :n]      = data['open'].astype(np.float64)
        close_2d[sid, :n]     = data['close'].astype(np.float64)
        high_2d[sid, :n]      = data['high'].astype(np.float64)
        low_2d[sid, :n]       = data['low'].astype(np.float64)
        high_time_2d[sid, :n] = data['high_time'].astype(np.int64)
        low_time_2d[sid, :n]  = data['low_time'].astype(np.int64)
        ts_int_2d[sid, :n]    = ts_int.astype(np.int64)

    all_timestamps_int = np.unique(np.concatenate(all_ts_int_lists))
    all_timestamps_dt  = all_timestamps_int.view('datetime64[ns]')

    return {
        "symbols":            symbols,
        "sym_ids":            sym_ids,
        "ts_int_arrays":      ts_int_arrays,
        "close_arrays":       close_arrays,
        "all_timestamps_int": all_timestamps_int,
        "all_timestamps_dt":  all_timestamps_dt,
        "open_2d":            open_2d,
        "close_2d":           close_2d,
        "high_2d":            high_2d,
        "low_2d":             low_2d,
        "high_time_2d":       high_time_2d,
        "low_time_2d":        low_time_2d,
        "ts_int_2d":          ts_int_2d,
        "sym_len":            sym_len,
    }


# ============================================================
# prepare_signal_arrays  (depends on the signal, reuses static_bundle)
# ============================================================
def prepare_signal_arrays(static_bundle, ohlcv_arrays):
    symbols = static_bundle["symbols"]
    sym_ids = static_bundle["sym_ids"]
    sym_len = static_bundle["sym_len"]
    max_len = static_bundle["open_2d"].shape[1]
    n_syms  = len(symbols)

    sym_data = {}
    for sym in symbols:
        data   = ohlcv_arrays[sym]
        sid    = sym_ids[sym]
        n      = int(sym_len[sid])
        ts_int = static_bundle["ts_int_arrays"][sym]
        sym_data[sym] = {
            'ts':        ts_int.view('datetime64[ns]'),
            'ts_int':    ts_int,
            'open':      data['open'],
            'close':     data['close'],
            'high':      data['high'],
            'low':       data['low'],
            'signal':    data['signal'][:n],
            'len':       n,
            'high_time': data['high_time'],
            'low_time':  data['low_time'],
        }

    signal_2d    = np.full((n_syms, max_len), 0, dtype=np.int64)
    event_chunks = []
    for sym in symbols:
        sid        = sym_ids[sym]
        d          = sym_data[sym]
        signal_arr = d['signal']
        n          = d['len']
        signal_2d[sid, :n] = signal_arr.astype(np.int64)

        sig_idxs = np.flatnonzero(signal_arr)
        sig_idxs = sig_idxs[sig_idxs < n]
        if sig_idxs.size:
            ts_ints = d['ts_int'][sig_idxs]
            chunk   = np.empty((sig_idxs.size, 3), dtype=np.int64)
            chunk[:, 0] = ts_ints
            chunk[:, 1] = sid
            chunk[:, 2] = sig_idxs
            event_chunks.append(chunk)

    if event_chunks:
        signal_events = np.concatenate(event_chunks, axis=0)
        # primary key: timestamp asc — secondary key: symbol id asc (same
        # tie-break order used by the Cython lexsort)
        order         = np.lexsort((signal_events[:, 1], signal_events[:, 0]))
        signal_events = signal_events[order]
    else:
        signal_events = np.empty((0, 3), dtype=np.int64)

    ev_col0 = np.ascontiguousarray(signal_events[:, 0], dtype=np.int64) if signal_events.shape[0] > 0 \
        else np.empty((0,), dtype=np.int64)

    arrays = (
        static_bundle["open_2d"], static_bundle["close_2d"],
        static_bundle["high_2d"], static_bundle["low_2d"],
        static_bundle["high_time_2d"], static_bundle["low_time_2d"],
        static_bundle["ts_int_2d"], signal_2d, sym_len,
        signal_events, static_bundle["all_timestamps_int"], ev_col0
    )

    return (sym_data, {}, static_bundle["all_timestamps_int"], static_bundle["all_timestamps_dt"],
            sym_ids, static_bundle["ts_int_arrays"], static_bundle["close_arrays"], arrays)


# ============================================================
# prepare_data / prepare_backtest_data
# ============================================================
def prepare_data(ohlcv_arrays):
    # NOTE: the empty-input case intentionally returns a 7-tuple while the
    # non-empty case returns an 8-tuple — this mirrors the Cython version's
    # behavior exactly (including that inconsistency), for byte-for-byte
    # compatibility with callers written against either backend.
    if not ohlcv_arrays:
        return ({}, {}, np.array([], dtype=np.int64),
                np.array([], dtype='datetime64[ns]'), {}, {}, {})
    static_bundle = prepare_static_arrays(ohlcv_arrays)
    return prepare_signal_arrays(static_bundle, ohlcv_arrays)


def prepare_backtest_data(ohlcv_arrays):
    """Reorganize ohlcv_arrays into internal structures needed by the backtest
    loop. Independent of sell_after/tp_pct/sl_pct — callers evaluating a grid
    of param combinations over the same ohlcv_arrays can call this once and
    reuse the result across combinations via run_backtest_from_prepared."""
    return prepare_data(ohlcv_arrays)


# ============================================================
# _detect_intrabar_exit  (tie-break logic identical to the Cython version)
# ============================================================
def _detect_intrabar_exit(high_row, low_row, high_time_row, low_time_row,
                           buy_idx, sell_idx, tp_price, sl_price, is_short):
    if sell_idx < buy_idx:
        return False, None, 0, None

    high_slice = high_row[buy_idx:sell_idx + 1]
    low_slice  = low_row[buy_idx:sell_idx + 1]

    if is_short:
        tp_hits     = np.where(low_slice  <= tp_price)[0]
        sl_hits     = np.where(high_slice >= sl_price)[0]
        tp_time_arr = low_time_row
        sl_time_arr = high_time_row
    else:
        tp_hits     = np.where(high_slice >= tp_price)[0]
        sl_hits     = np.where(low_slice  <= sl_price)[0]
        tp_time_arr = high_time_row
        sl_time_arr = low_time_row

    tp_first = int(tp_hits[0]) + buy_idx if tp_hits.size > 0 else None
    sl_first = int(sl_hits[0]) + buy_idx if sl_hits.size > 0 else None

    if tp_first is None and sl_first is None:
        return False, None, 0, None

    if tp_first is not None and sl_first is not None:
        if tp_first == sl_first:
            if tp_time_arr[tp_first] <= sl_time_arr[sl_first]:
                return True, tp_first, 1, tp_price
            return True, sl_first, 2, sl_price
        if sl_first < tp_first:
            return True, sl_first, 2, sl_price
        return True, tp_first, 1, tp_price
    if sl_first is not None:
        return True, sl_first, 2, sl_price
    return True, tp_first, 1, tp_price


# ============================================================
# _backtest_core  (pure Python port of the Cython hot loop)
# ============================================================
def _backtest_core(arrays, initial_balance, comi_factor, order_amount,
                    sell_after, tp_pct, sl_pct):

    (open_2d, close_2d, high_2d, low_2d,
     high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
     signal_events, all_timestamps_int, ev_col0) = arrays

    ev_times = signal_events[:, 0] if signal_events.shape[0] > 0 else np.empty((0,), dtype=np.int64)
    ev_sids  = signal_events[:, 1] if signal_events.shape[0] > 0 else np.empty((0,), dtype=np.int64)
    ev_idxs  = signal_events[:, 2] if signal_events.shape[0] > 0 else np.empty((0,), dtype=np.int64)
    n_events = signal_events.shape[0]

    cash_bank    = float(initial_balance)
    blocked_cash = 0.0
    open_heap    = []   # entries: (heap_time_int, counter)
    positions    = {}   # counter -> position dict
    counter      = 0

    tl_sym_id      = []
    tl_buy_time    = []
    tl_buy_price   = []
    tl_sell_time   = []
    tl_sell_price  = []
    tl_qty         = []
    tl_profit      = []
    tl_exit_reason = []
    tl_comm_buy    = []
    tl_comm_sell   = []
    tl_is_short    = []

    for t_int in all_timestamps_int:
        t_int = int(t_int)

        # ── 1. Close expired / intrabar-exit positions ──
        was_empty_before = (len(open_heap) == 0)
        closed_any_tp_sl = False

        while open_heap and open_heap[0][0] <= t_int:
            _, pos_id = heapq.heappop(open_heap)
            pos = positions.pop(pos_id)

            if pos['has_exec'] and pos['exec_time_int'] <= t_int:
                exec_price    = pos['exec_price']
                exec_time_int = pos['exec_time_int']
                reason_code   = pos['exit_reason_code']
            else:
                sid    = pos['sym_id']
                n_sym  = int(sym_len[sid])
                ts_row = ts_int_2d[sid, :n_sym]
                idx    = int(np.searchsorted(ts_row, pos['sell_time_int'], side='right')) - 1
                if idx < 0:
                    idx = 0
                exec_price    = float(close_2d[sid, idx])
                exec_time_int = pos['sell_time_int']
                reason_code   = 0

            qty            = pos['qty']
            buy_price      = pos['buy_price']
            is_short       = pos['is_short']
            comm_buy       = pos['commission_buy']
            blocked_amount = pos['blocked_amount']
            comm_sell      = qty * exec_price * comi_factor

            if is_short:
                cash_bank    -= qty * exec_price + comm_sell
                blocked_cash -= blocked_amount
                profit        = (buy_price - exec_price) * qty - comm_buy - comm_sell
            else:
                cash_bank += qty * exec_price - comm_sell
                profit     = (exec_price - buy_price) * qty - comm_buy - comm_sell

            if -1e-9 < blocked_cash < 0.0:
                blocked_cash = 0.0

            tl_sym_id.append(pos['sym_id'])
            tl_buy_time.append(pos['buy_time_int'])
            tl_buy_price.append(buy_price)
            tl_sell_time.append(exec_time_int)
            tl_sell_price.append(exec_price)
            tl_qty.append(qty)
            tl_profit.append(profit)
            tl_exit_reason.append(reason_code)
            tl_comm_buy.append(comm_buy)
            tl_comm_sell.append(comm_sell)
            tl_is_short.append(1 if is_short else 0)

            if reason_code in (1, 2):
                closed_any_tp_sl = True

        # ── 2. Open new positions if heap empty ──
        if len(open_heap) == 0:
            search_signals = True if was_empty_before else (not closed_any_tp_sl)

            if search_signals and n_events > 0:
                ev_start = int(np.searchsorted(ev_col0, t_int, side='left'))
                ev_scan  = ev_start

                while ev_scan < n_events and ev_times[ev_scan] == t_int:
                    sid     = int(ev_sids[ev_scan])
                    buy_idx = int(ev_idxs[ev_scan])
                    ev_scan += 1

                    n_bars    = int(sym_len[sid])
                    free_cash = cash_bank - blocked_cash

                    if free_cash < order_amount:
                        break

                    sig_val  = signal_2d[sid, buy_idx]
                    is_short = sig_val < 0

                    if is_short and sl_pct == 0.0:
                        continue
                    if is_short and free_cash < order_amount * (sl_pct / 100.0) + order_amount * comi_factor:
                        continue

                    price_t  = float(open_2d[sid, buy_idx])
                    qty      = order_amount / price_t
                    comm_buy = order_amount * comi_factor

                    exit_idx = buy_idx + sell_after
                    if exit_idx >= n_bars:
                        exit_idx = n_bars - 1

                    sell_time_int = int(ts_int_2d[sid, exit_idx])

                    if is_short:
                        tp_price = price_t * (1.0 - tp_pct / 100.0) if tp_pct != 0.0 else -np.inf
                        sl_price = price_t * (1.0 + sl_pct / 100.0) if sl_pct != 0.0 else  np.inf
                        proceeds       = order_amount - comm_buy
                        margin_req     = order_amount * (sl_pct / 100.0) if sl_pct != 0.0 else np.inf
                        blocked_amount = proceeds + margin_req
                        cash_bank     += proceeds
                        blocked_cash  += blocked_amount
                    else:
                        tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else  np.inf
                        sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -np.inf
                        blocked_amount = 0.0
                        cash_bank     -= (order_amount + comm_buy)

                    intra, chosen_idx, reason_code, exec_price = _detect_intrabar_exit(
                        high_2d[sid], low_2d[sid], high_time_2d[sid], low_time_2d[sid],
                        buy_idx, exit_idx, tp_price, sl_price, is_short
                    )

                    pos = {
                        'sym_id':          sid,
                        'qty':             qty,
                        'buy_price':       price_t,
                        'buy_time_int':    int(ts_int_2d[sid, buy_idx]),
                        'sell_time_int':   sell_time_int,
                        'commission_buy':  comm_buy,
                        'is_short':        is_short,
                        'blocked_amount':  blocked_amount,
                    }

                    if intra:
                        exec_time_int = int(ts_int_2d[sid, chosen_idx])
                        pos['has_exec']         = True
                        pos['exec_price']       = float(exec_price)
                        pos['exec_time_int']    = exec_time_int
                        pos['exit_reason_code'] = reason_code
                        heapq.heappush(open_heap, (exec_time_int, counter))
                    else:
                        pos['has_exec'] = False
                        heapq.heappush(open_heap, (sell_time_int, counter))

                    positions[counter] = pos
                    counter += 1

    n_trades = len(tl_sym_id)
    return (
        n_trades,
        np.array(tl_sym_id,      dtype=np.int64),
        np.array(tl_buy_time,    dtype=np.int64),
        np.array(tl_buy_price,   dtype=np.float64),
        np.array(tl_sell_time,   dtype=np.int64),
        np.array(tl_sell_price,  dtype=np.float64),
        np.array(tl_qty,         dtype=np.float64),
        np.array(tl_profit,      dtype=np.float64),
        np.array(tl_exit_reason, dtype=np.int32),
        np.array(tl_comm_buy,    dtype=np.float64),
        np.array(tl_comm_sell,   dtype=np.float64),
        np.array(tl_is_short,    dtype=np.int32),
        cash_bank, blocked_cash
    )


# ============================================================
# _run_core_from_arrays  (full trade_log — matches Cython column set)
# ============================================================
def _run_core_from_arrays(arrays, sym_ids, ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount):
    comi_factor     = float(COMISION) / 100.0
    initial_balance = float(INITIAL_BALANCE)

    (
        n_trades,
        tl_sym_id, tl_buy_time, tl_buy_price,
        tl_sell_time, tl_sell_price, tl_qty,
        tl_profit, tl_exit_reason,
        tl_comm_buy, tl_comm_sell, tl_is_short,
        final_cash_bank, _
    ) = _backtest_core(arrays, initial_balance, comi_factor, order_amount,
                        sell_after, tp_pct, sl_pct)

    symbols = list(ohlcv_arrays.keys())
    n_syms  = len(sym_ids)

    sym_names         = np.empty(n_syms, dtype=object)
    symbol_order_rank = np.empty(n_syms, dtype=np.int64)
    for i, sym in enumerate(symbols):
        sid = sym_ids[sym]
        sym_names[sid]         = sym
        symbol_order_rank[sid] = i

    exit_reason_names   = np.array(['SELL_AFTER', 'TP', 'SL'], dtype=object)
    position_type_names = np.array(['LONG', 'SHORT'], dtype=object)

    trade_log = pd.DataFrame({
        'symbol':          sym_names[tl_sym_id],
        'buy_time':        tl_buy_time.astype('datetime64[ns]'),
        'buy_price':       tl_buy_price,
        'sell_time':       tl_sell_time.astype('datetime64[ns]'),
        'sell_price':      tl_sell_price,
        'qty':             tl_qty,
        'profit':          tl_profit,
        'exit_reason':     exit_reason_names[tl_exit_reason],
        'commission_buy':  tl_comm_buy,
        'commission_sell': tl_comm_sell,
        'position_type':   position_type_names[tl_is_short],
    })

    # Reproduce the original grouping (all profits of symbol 1, then symbol
    # 2, etc., in `symbols` order, preserving chronological order within
    # each symbol) via a single stable argsort.
    rank_per_trade = symbol_order_rank[tl_sym_id]
    order          = np.argsort(rank_per_trade, kind='stable')
    trades_list    = tl_profit[order].tolist()

    return {
        "__PORTFOLIO__": {
            'trades':    trades_list,
            'trade_log': trade_log,
        }
    }


# ============================================================
# run_backtest_from_prepared  (simulate step — reuse prepared data)
# ============================================================
def run_backtest_from_prepared(prepared_data, sell_after, tp_pct, sl_pct, order_amount):
    """Run simulation using data already prepared by prepare_backtest_data.
    Use this when evaluating multiple param combinations over the same
    ohlcv_arrays to avoid repeating the prepare step."""

    (sym_data, _, all_timestamps_int, all_timestamps_dt,
     sym_ids, ts_int_arrays, close_arrays, arrays) = prepared_data

    return _run_core_from_arrays(
        arrays, sym_ids, sym_data,
        sell_after, tp_pct, sl_pct, order_amount
    )


# ============================================================
# _run_core_from_arrays_light  (metrics-only trade_log — no reordering)
# ============================================================
def _run_core_from_arrays_light(arrays, sell_after, tp_pct, sl_pct, order_amount):
    comi_factor     = float(COMISION) / 100.0
    initial_balance = float(INITIAL_BALANCE)

    (
        n_trades,
        tl_sym_id, tl_buy_time, tl_buy_price,
        tl_sell_time, tl_sell_price, tl_qty,
        tl_profit, tl_exit_reason,
        tl_comm_buy, tl_comm_sell, tl_is_short,
        final_cash_bank, _
    ) = _backtest_core(arrays, initial_balance, comi_factor, order_amount,
                        sell_after, tp_pct, sl_pct)

    trade_log = pd.DataFrame({
        'buy_time':  tl_buy_time.astype('datetime64[ns]'),
        'sell_time': tl_sell_time.astype('datetime64[ns]'),
        'profit':    tl_profit,
    })

    return {
        "__PORTFOLIO__": {
            'trade_log': trade_log,
        }
    }


# ============================================================
# run_backtest_from_prepared_light  (simulate step — metrics-only output)
# ============================================================
def run_backtest_from_prepared_light(prepared_data, sell_after, tp_pct, sl_pct, order_amount):
    """Like run_backtest_from_prepared, but the returned trade_log only has
    'buy_time', 'sell_time' and 'profit' — everything compute_metrics() needs
    and nothing else. There is no 'trades' key. Use this for inner param-grid
    search loops; use run_backtest_from_prepared when the full trade detail
    is needed downstream (CSV export, symbol/exit-reason breakdowns, etc)."""

    (sym_data, _, all_timestamps_int, all_timestamps_dt,
     sym_ids, ts_int_arrays, close_arrays, arrays) = prepared_data

    return _run_core_from_arrays_light(
        arrays, sell_after, tp_pct, sl_pct, order_amount
    )


# ============================================================
# run_grid_backtest  --  public API (same signature & output)
# ============================================================
def run_grid_backtest(ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount):
    """Public API — unchanged behavior. Internally delegates to
    prepare_backtest_data + run_backtest_from_prepared."""

    prepared_data = prepare_backtest_data(ohlcv_arrays)

    return run_backtest_from_prepared(
        prepared_data,
        sell_after   = sell_after,
        tp_pct       = tp_pct,
        sl_pct       = sl_pct,
        order_amount = order_amount
    )