# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import heapq
import logging
import warnings
import numpy as np
import pandas as pd

cimport numpy as np
from libc.math cimport HUGE_VAL

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

MIN_PRICE       = 0.00001


# ============================================================
# C-level binary search helpers  (replaces np.searchsorted)
# ============================================================
cdef int _searchsorted_left(long[::1] arr, long val, int n) nogil:
    """Return first index i where arr[i] >= val (left side)."""
    cdef int lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef int _searchsorted_right(long[::1] arr, long val, int n) nogil:
    """Return first index i where arr[i] > val (right side)."""
    cdef int lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo
INITIAL_BALANCE = 1000
COMISION        = 0.1
DEFAULT_CANDLES = 50

# ============================================================
# prepare_data  (pure Python, no Cython types needed)
# ============================================================
def prepare_data(ohlcv_arrays):
    if not ohlcv_arrays:
        return ({}, {}, np.array([], dtype=np.int64),
                np.array([], dtype='datetime64[ns]'), {}, {}, {})

    symbols          = list(ohlcv_arrays.keys())
    sym_data         = {}
    ts_int_arrays    = {}
    close_arrays     = {}
    all_ts_int_lists = []

    for sym in symbols:
        data = ohlcv_arrays[sym]
        ts   = data['ts']
        if ts.dtype.kind != 'M':
            ts = ts.astype('datetime64[ns]')

        ts_int     = ts.view('int64')
        close_view = data['close']
        n          = len(ts)

        sym_data[sym] = {
            'ts':        ts,
            'ts_int':    ts_int,
            'open':      data['open'],
            'close':     close_view,
            'high':      data['high'],
            'low':       data['low'],
            'signal':    data['signal'][:n],
            'len':       n,
            'high_time': data['high_time'],
            'low_time':  data['low_time'],
        }
        ts_int_arrays[sym] = ts_int
        close_arrays[sym]  = close_view
        all_ts_int_lists.append(ts_int)

    n_syms  = len(symbols)
    sym_ids = {s: i for i, s in enumerate(sorted(symbols))}

    event_chunks = []
    for sym in symbols:
        sid        = sym_ids[sym]
        d          = sym_data[sym]
        signal_arr = d['signal']
        sig_idxs   = np.flatnonzero(signal_arr)
        sig_idxs   = sig_idxs[sig_idxs < d['len']]
        if sig_idxs.size:
            ts_ints = d['ts_int'][sig_idxs]
            chunk   = np.empty((sig_idxs.size, 3), dtype=np.int64)
            chunk[:, 0] = ts_ints
            chunk[:, 1] = sid
            chunk[:, 2] = sig_idxs
            event_chunks.append(chunk)

    if event_chunks:
        signal_events = np.concatenate(event_chunks, axis=0)
        order         = np.lexsort((signal_events[:, 1], signal_events[:, 0]))
        signal_events = signal_events[order]
    else:
        signal_events = np.empty((0, 3), dtype=np.int64)

    max_len      = max(sym_data[s]['len'] for s in symbols)
    open_2d      = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    close_2d     = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    high_2d      = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    low_2d       = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    high_time_2d = np.full((n_syms, max_len), 0,      dtype=np.int64)
    low_time_2d  = np.full((n_syms, max_len), 0,      dtype=np.int64)
    ts_int_2d    = np.full((n_syms, max_len), 0,      dtype=np.int64)
    signal_2d    = np.full((n_syms, max_len), 0,      dtype=np.int64)
    sym_len      = np.zeros(n_syms, dtype=np.int64)

    for sym in symbols:
        sid  = sym_ids[sym]
        d    = sym_data[sym]
        n    = d['len']
        sym_len[sid]          = n
        open_2d[sid, :n]      = d['open'].astype(np.float64)
        close_2d[sid, :n]     = d['close'].astype(np.float64)
        high_2d[sid, :n]      = d['high'].astype(np.float64)
        low_2d[sid, :n]       = d['low'].astype(np.float64)
        high_time_2d[sid, :n] = d['high_time'].astype(np.int64)
        low_time_2d[sid, :n]  = d['low_time'].astype(np.int64)
        ts_int_2d[sid, :n]    = d['ts_int'].astype(np.int64)
        signal_2d[sid, :n]    = d['signal'].astype(np.int64)

    all_timestamps_int = np.unique(np.concatenate(all_ts_int_lists))
    all_timestamps_dt  = all_timestamps_int.view('datetime64[ns]')

    arrays = (
        open_2d, close_2d, high_2d, low_2d,
        high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
        signal_events, all_timestamps_int
    )

    return (sym_data, {}, all_timestamps_int, all_timestamps_dt,
            sym_ids, ts_int_arrays, close_arrays, arrays)


# ============================================================
# _detect_intrabar_exit  (typed Cython)
# ============================================================
cdef tuple _detect_intrabar_exit_cy(
    double[::1] high_row,
    double[::1] low_row,
    long[::1]   high_time_row,
    long[::1]   low_time_row,
    int buy_idx, int sell_idx,
    double tp_price, double sl_price,
    bint is_short
):
    cdef int bi, tp_first, sl_first, chosen_idx
    cdef long tp_t, sl_t
    cdef double exec_price
    cdef int reason  # 0=none 1=TP 2=SL

    tp_first = -1
    sl_first = -1

    if is_short:
        for bi in range(buy_idx, sell_idx + 1):
            if tp_first < 0 and low_row[bi] <= tp_price:
                tp_first = bi
            if sl_first < 0 and high_row[bi] >= sl_price:
                sl_first = bi
            if tp_first >= 0 and sl_first >= 0:
                break
    else:
        for bi in range(buy_idx, sell_idx + 1):
            if tp_first < 0 and high_row[bi] >= tp_price:
                tp_first = bi
            if sl_first < 0 and low_row[bi] <= sl_price:
                sl_first = bi
            if tp_first >= 0 and sl_first >= 0:
                break

    if tp_first < 0 and sl_first < 0:
        return False, -1, 0, 0.0

    if tp_first >= 0 and sl_first >= 0:
        if tp_first == sl_first:
            if is_short:
                tp_t = low_time_row[tp_first]
                sl_t = high_time_row[sl_first]
            else:
                tp_t = high_time_row[tp_first]
                sl_t = low_time_row[sl_first]
            if tp_t <= sl_t:
                return True, tp_first, 1, tp_price
            else:
                return True, sl_first, 2, sl_price
        elif sl_first < tp_first:
            return True, sl_first, 2, sl_price
        else:
            return True, tp_first, 1, tp_price
    elif sl_first >= 0:
        return True, sl_first, 2, sl_price
    else:
        return True, tp_first, 1, tp_price


# ============================================================
# _close_position  (typed Cython)
# ============================================================
cdef tuple _close_position_cy(
    dict pos,
    long exec_time_int,
    double exec_price,
    int exit_reason,
    double comi_factor,
    int n_trades,
    long[::1]   tl_sym_id,
    long[::1]   tl_buy_time,
    double[::1] tl_buy_price,
    long[::1]   tl_sell_time,
    double[::1] tl_sell_price,
    double[::1] tl_qty,
    double[::1] tl_profit,
    int[::1]    tl_exit_reason,
    double[::1] tl_comm_buy,
    double[::1] tl_comm_sell,
    int[::1]    tl_is_short,
    double cash_bank,
    double blocked_cash
):
    cdef double qty, buy_price, comm_buy, comm_sell, profit, blocked_amount
    cdef bint is_short

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

    if blocked_cash < 0.0 and blocked_cash > -1e-9:
        blocked_cash = 0.0

    tl_sym_id[n_trades]      = pos['sym_id']
    tl_buy_time[n_trades]    = pos['buy_time_int']
    tl_buy_price[n_trades]   = buy_price
    tl_sell_time[n_trades]   = exec_time_int
    tl_sell_price[n_trades]  = exec_price
    tl_qty[n_trades]         = qty
    tl_profit[n_trades]      = profit
    tl_exit_reason[n_trades] = exit_reason
    tl_comm_buy[n_trades]    = comm_buy
    tl_comm_sell[n_trades]   = comm_sell
    tl_is_short[n_trades]    = 1 if is_short else 0

    return cash_bank, blocked_cash, n_trades + 1


# ============================================================
# _backtest_core  (typed Cython hot loop)
# ============================================================
def _backtest_core(
    np.ndarray[double, ndim=2] open_2d,
    np.ndarray[double, ndim=2] close_2d,
    np.ndarray[double, ndim=2] high_2d,
    np.ndarray[double, ndim=2] low_2d,
    np.ndarray[long,   ndim=2] high_time_2d,
    np.ndarray[long,   ndim=2] low_time_2d,
    np.ndarray[long,   ndim=2] ts_int_2d,
    np.ndarray[long,   ndim=2] signal_2d,
    np.ndarray[long,   ndim=1] sym_len,
    np.ndarray[long,   ndim=2] signal_events,
    np.ndarray[long,   ndim=1] all_timestamps_int,
    double initial_balance,
    double comi_factor,
    double order_amount,
    int sell_after,
    double tp_pct,
    double sl_pct,
    int default_candles
):
    cdef int n_ticks   = len(all_timestamps_int)
    cdef int n_events  = len(signal_events)
    cdef int max_trades = n_events + 1

    # ── Pre-allocated trade log (typed memoryviews) ──
    cdef np.ndarray[long,   ndim=1] tl_sym_id_arr      = np.empty(max_trades, dtype=np.int64)
    cdef np.ndarray[long,   ndim=1] tl_buy_time_arr    = np.empty(max_trades, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] tl_buy_price_arr   = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[long,   ndim=1] tl_sell_time_arr   = np.empty(max_trades, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] tl_sell_price_arr  = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] tl_qty_arr         = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] tl_profit_arr      = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[int,    ndim=1] tl_exit_reason_arr = np.empty(max_trades, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] tl_comm_buy_arr    = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] tl_comm_sell_arr   = np.empty(max_trades, dtype=np.float64)
    cdef np.ndarray[int,    ndim=1] tl_is_short_arr    = np.empty(max_trades, dtype=np.int32)

    cdef long[::1]   tl_sym_id      = tl_sym_id_arr
    cdef long[::1]   tl_buy_time    = tl_buy_time_arr
    cdef double[::1] tl_buy_price   = tl_buy_price_arr
    cdef long[::1]   tl_sell_time   = tl_sell_time_arr
    cdef double[::1] tl_sell_price  = tl_sell_price_arr
    cdef double[::1] tl_qty         = tl_qty_arr
    cdef double[::1] tl_profit      = tl_profit_arr
    cdef int[::1]    tl_exit_reason = tl_exit_reason_arr
    cdef double[::1] tl_comm_buy    = tl_comm_buy_arr
    cdef double[::1] tl_comm_sell   = tl_comm_sell_arr
    cdef int[::1]    tl_is_short    = tl_is_short_arr

    # ── sim_balance ──
    cdef np.ndarray[long,   ndim=1] sb_timestamp_arr = np.empty(n_ticks, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] sb_balance_arr   = np.empty(n_ticks, dtype=np.float64)
    cdef long[::1]   sb_timestamp = sb_timestamp_arr
    cdef double[::1] sb_balance   = sb_balance_arr

    # ── Account state ──
    cdef double cash_bank    = initial_balance
    cdef double blocked_cash = 0.0
    cdef int    n_trades     = 0

    # ── Loop vars ──
    cdef int    tick_i, ev_start, ev_scan
    cdef long   t_int, exp_time
    cdef int    sid, buy_idx, n_bars, exit_idx
    cdef long   sell_time_int, exec_time_int
    cdef double price_t, qty, comm_buy
    cdef double tp_price, sl_price
    cdef double free_cash, proceeds, margin_req, blocked_amount
    cdef int    sig_val
    cdef bint   is_short, intra
    cdef int    chosen_idx, reason_code
    cdef double exec_price_intra
    cdef int    idx
    cdef double total_val, price
    cdef long   n_sym

    # ── Typed memoryviews for 2D arrays ──
    cdef double[:, ::1] open_mv      = open_2d
    cdef double[:, ::1] close_mv     = close_2d
    cdef double[:, ::1] high_mv      = high_2d
    cdef double[:, ::1] low_mv       = low_2d
    cdef long[:, ::1]   high_time_mv = high_time_2d
    cdef long[:, ::1]   low_time_mv  = low_time_2d
    cdef long[:, ::1]   ts_int_mv    = ts_int_2d
    cdef long[:, ::1]   signal_mv    = signal_2d
    cdef long[::1]      sym_len_mv   = sym_len
    cdef long[:, ::1]   ev_mv        = signal_events
    cdef long[::1]      ts_all_mv    = all_timestamps_int
    cdef np.ndarray[long, ndim=1] ev_col0_arr = np.ascontiguousarray(signal_events[:, 0], dtype=np.int64)
    cdef long[::1]      ev_col0      = ev_col0_arr

    open_heap = []
    counter   = 0

    for tick_i in range(n_ticks):
        t_int = ts_all_mv[tick_i]

        # ── 1. Close expired / intrabar-exit positions ──
        while open_heap and open_heap[0][0] <= t_int:
            exp_time, _, pos = heapq.heappop(open_heap)
            if pos.get('closed', False):
                continue

            if 'exec_price' in pos and pos['exec_time_int'] <= t_int:
                exec_price_intra = pos['exec_price']
                exec_time_int    = pos['exec_time_int']
                reason_code      = pos['exit_reason_code']
            else:
                sid    = pos['sym_id']
                n_sym  = sym_len_mv[sid]
                idx    = _searchsorted_right(ts_int_mv[sid], <long>pos['sell_time_int'], <int>n_sym) - 1
                if idx < 0:
                    idx = 0
                exec_price_intra = close_mv[sid, idx]
                exec_time_int    = pos['sell_time_int']
                reason_code      = 0

            cash_bank, blocked_cash, n_trades = _close_position_cy(
                pos, exec_time_int, exec_price_intra, reason_code,
                comi_factor, n_trades,
                tl_sym_id, tl_buy_time, tl_buy_price,
                tl_sell_time, tl_sell_price, tl_qty,
                tl_profit, tl_exit_reason,
                tl_comm_buy, tl_comm_sell, tl_is_short,
                cash_bank, blocked_cash
            )
            pos['closed'] = True

        # ── 2. Open new positions if heap empty ──
        if not open_heap:
            ev_start = _searchsorted_left(ev_col0, t_int, n_events)
            ev_scan  = ev_start

            while ev_scan < n_events and ev_mv[ev_scan, 0] == t_int:
                sid     = <int>ev_mv[ev_scan, 1]
                buy_idx = <int>ev_mv[ev_scan, 2]
                ev_scan += 1

                n_bars = <int>sym_len_mv[sid]

                if sell_after > 0:
                    if buy_idx + sell_after > n_bars:
                        continue
                else:
                    if buy_idx + default_candles >= n_bars:
                        continue

                free_cash = cash_bank - blocked_cash
                if free_cash < order_amount:
                    break

                sig_val  = <int>signal_mv[sid, buy_idx]
                is_short = sig_val < 0

                if is_short and sl_pct == 0.0:
                    continue

                if is_short:
                    if free_cash < order_amount * (sl_pct / 100.0) + order_amount * comi_factor:
                        continue

                price_t  = open_mv[sid, buy_idx]
                qty      = order_amount / price_t
                comm_buy = order_amount * comi_factor

                if sell_after == 0:
                    exit_idx = buy_idx + default_candles
                else:
                    exit_idx = buy_idx + sell_after
                if exit_idx >= n_bars:
                    exit_idx = n_bars - 1

                sell_time_int = ts_int_mv[sid, exit_idx]

                if is_short:
                    tp_price = price_t * (1.0 - tp_pct / 100.0) if tp_pct != 0.0 else -HUGE_VAL
                    sl_price = price_t * (1.0 + sl_pct / 100.0) if sl_pct != 0.0 else  HUGE_VAL
                else:
                    tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else  HUGE_VAL
                    sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -HUGE_VAL

                if is_short:
                    proceeds       = order_amount - comm_buy
                    margin_req     = order_amount * (sl_pct / 100.0) if sl_pct != 0.0 else HUGE_VAL
                    blocked_amount = proceeds + margin_req
                    cash_bank     += proceeds
                    blocked_cash  += blocked_amount
                else:
                    blocked_amount = 0.0
                    cash_bank     -= (order_amount + comm_buy)

                pos = {
                    'sym_id':            sid,
                    'qty':               qty,
                    'buy_price':         price_t,
                    'buy_time_int':      ts_int_mv[sid, buy_idx],
                    'sell_time_int':     sell_time_int,
                    'commission_buy':    comm_buy,
                    'is_short':          is_short,
                    'blocked_amount':    blocked_amount,
                    'closed':            False,
                }

                intra, chosen_idx, reason_code, exec_price_intra = _detect_intrabar_exit_cy(
                    high_mv[sid], low_mv[sid],
                    high_time_mv[sid], low_time_mv[sid],
                    buy_idx, exit_idx, tp_price, sl_price, is_short
                )

                if intra:
                    exec_time_int = ts_int_mv[sid, chosen_idx]
                    pos['exec_price']       = exec_price_intra
                    pos['exec_time_int']    = exec_time_int
                    pos['exit_reason_code'] = reason_code
                    heapq.heappush(open_heap, (exec_time_int, counter, pos))
                else:
                    heapq.heappush(open_heap, (sell_time_int, counter, pos))

                counter += 1

        # ── 3. Snapshot sim_balance ──
        sb_timestamp[tick_i] = t_int
        if open_heap:
            total_val = cash_bank
            seen_sids = {}
            for _, _, p in open_heap:
                if p.get('closed', False):
                    continue
                s = p['sym_id']
                seen_sids[s] = seen_sids.get(s, 0.0) + p['qty']

            for s, qty_sum in seen_sids.items():
                n_sym = sym_len_mv[s]
                idx   = _searchsorted_right(ts_int_mv[s], t_int, <int>sym_len_mv[s]) - 1
                if idx < 0:
                    idx = 0
                price = close_mv[s, idx]
                if any(p['sym_id'] == s and p.get('is_short') and not p.get('closed')
                       for _, _, p in open_heap):
                    total_val -= qty_sum * price
                else:
                    total_val += qty_sum * price
            sb_balance[tick_i] = total_val
        else:
            sb_balance[tick_i] = cash_bank

    return (
        n_trades,
        tl_sym_id_arr[:n_trades],    tl_buy_time_arr[:n_trades],   tl_buy_price_arr[:n_trades],
        tl_sell_time_arr[:n_trades], tl_sell_price_arr[:n_trades], tl_qty_arr[:n_trades],
        tl_profit_arr[:n_trades],    tl_exit_reason_arr[:n_trades],
        tl_comm_buy_arr[:n_trades],  tl_comm_sell_arr[:n_trades],  tl_is_short_arr[:n_trades],
        sb_timestamp_arr, sb_balance_arr,
        cash_bank, blocked_cash
    )


# ============================================================
# compute_annualized_sharpe
# ============================================================
def compute_annualized_sharpe(equity_arr, time_index_int64):
    if equity_arr is None or equity_arr.size < 2:
        return np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        returns = (equity_arr[1:] / equity_arr[:-1]) - 1.0
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return np.nan

    if len(time_index_int64) >= 2:
        deltas_s = np.diff(time_index_int64).astype(np.float64) / 1e9
        positive = deltas_s[deltas_s > 0]
        median_delta_s = float(np.median(positive)) if positive.size > 0 else 24 * 3600
    else:
        median_delta_s = 24 * 3600

    periods_per_year = (365.0 * 24.0 * 3600.0) / median_delta_s if median_delta_s > 0 else 252.0
    mean_p = np.mean(returns)
    std_p  = np.std(returns, ddof=0)
    if not np.isfinite(std_p) or std_p == 0.0:
        return np.nan

    return float((mean_p * periods_per_year) / (std_p * np.sqrt(periods_per_year)))


# ============================================================
# run_grid_backtest  --  public API (same signature & output)
# ============================================================
def run_grid_backtest(ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount):

    cdef_factor     = float(COMISION) / 100.0
    initial_balance = float(INITIAL_BALANCE)

    result = prepare_data(ohlcv_arrays)
    (sym_data, _, all_timestamps_int, all_timestamps_dt,
     sym_ids, ts_int_arrays, close_arrays, arrays) = result

    (open_2d, close_2d, high_2d, low_2d,
     high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
     signal_events, _) = arrays

    # Force C-contiguous arrays before passing to Cython memoryviews
    open_2d      = np.ascontiguousarray(open_2d,      dtype=np.float64)
    close_2d     = np.ascontiguousarray(close_2d,     dtype=np.float64)
    high_2d      = np.ascontiguousarray(high_2d,      dtype=np.float64)
    low_2d       = np.ascontiguousarray(low_2d,       dtype=np.float64)
    high_time_2d = np.ascontiguousarray(high_time_2d, dtype=np.int64)
    low_time_2d  = np.ascontiguousarray(low_time_2d,  dtype=np.int64)
    ts_int_2d    = np.ascontiguousarray(ts_int_2d,    dtype=np.int64)
    signal_2d    = np.ascontiguousarray(signal_2d,    dtype=np.int64)
    sym_len      = np.ascontiguousarray(sym_len,      dtype=np.int64)
    signal_events      = np.ascontiguousarray(signal_events,      dtype=np.int64)
    all_timestamps_int = np.ascontiguousarray(all_timestamps_int, dtype=np.int64)

    symbols   = list(ohlcv_arrays.keys())
    id_to_sym = {v: k for k, v in sym_ids.items()}

    (
        n_trades,
        tl_sym_id, tl_buy_time, tl_buy_price,
        tl_sell_time, tl_sell_price, tl_qty,
        tl_profit, tl_exit_reason,
        tl_comm_buy, tl_comm_sell, tl_is_short,
        sb_timestamp, sb_balance,
        final_cash_bank, _
    ) = _backtest_core(
        open_2d, close_2d, high_2d, low_2d,
        high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
        signal_events, all_timestamps_int,
        initial_balance, cdef_factor, float(order_amount),
        int(sell_after), float(tp_pct), float(sl_pct), int(DEFAULT_CANDLES)
    )

    exit_reason_map = {0: 'SELL_AFTER', 1: 'TP', 2: 'SL'}
    trade_log = pd.DataFrame({
        'symbol':          [id_to_sym[int(s)] for s in tl_sym_id],
        'buy_time':        tl_buy_time.astype('datetime64[ns]'),
        'buy_price':       tl_buy_price,
        'sell_time':       tl_sell_time.astype('datetime64[ns]'),
        'sell_price':      tl_sell_price,
        'qty':             tl_qty,
        'profit':          tl_profit,
        'exit_reason':     [exit_reason_map[int(r)] for r in tl_exit_reason],
        'commission_buy':  tl_comm_buy,
        'commission_sell': tl_comm_sell,
        'position_type':   ['SHORT' if s else 'LONG' for s in tl_is_short],
    })

    trades      = {sym: [] for sym in symbols}
    trade_times = {sym: [] for sym in symbols}
    for i in range(n_trades):
        sym = id_to_sym[int(tl_sym_id[i])]
        trades[sym].append(float(tl_profit[i]))
        trade_times[sym].append(np.datetime64(int(tl_sell_time[i]), 'ns'))

    sim_balance_cols = {
        'timestamp': list(sb_timestamp.astype('datetime64[ns]')),
        'balance':   list(sb_balance),
    }

    sim_values = sb_balance
    sim_ts_int = sb_timestamp

    final_balance    = float(sim_values[-1]) if sim_values.size > 0 else initial_balance
    cummax           = np.maximum.accumulate(sim_values) if sim_values.size > 0 else np.array([initial_balance])
    drawdowns        = (cummax - sim_values) / np.where(cummax == 0, 1, cummax)
    max_dd_portfolio = float(np.max(drawdowns)) if drawdowns.size > 0 else 0.0
    sharpe_portfolio = compute_annualized_sharpe(sim_values, sim_ts_int)

    all_trades = [p for lst in trades.values() for p in lst]
    num_trades = len(all_trades)
    proportion_winners = (
        float(np.sum(np.array(all_trades) > 0.0)) / num_trades if num_trades > 0 else np.nan
    )

    return {
        "__PORTFOLIO__": {
            'trades':              all_trades,
            'final_balance':       final_balance,
            'num_signals':         n_trades,
            'proportion_winners':  proportion_winners,
            'max_dd':              max_dd_portfolio,
            'sim_balance_history': sim_balance_cols,
            'trade_log':           trade_log,
            'sharpe':              sharpe_portfolio,
        }
    }