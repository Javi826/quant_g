# shared/shared_batchs/backtesters/ZX_compute_v2.py
# =============================================================================
# ZX Compute v2 — maximum Python/numpy optimization.
# Identical outputs and signature to ZX_compute.py.
#
# Optimizations vs v1:
# 1. detect_intrabar_exit  — np.flatnonzero replaces np.where; intrabar
#                            timestamps stored as int64 (no datetime conversion)
# 2. close_position        — local alias eliminates repeated dict lookups
# 3. execute_signal        — 'closed' key pre-set; no pos.get() with default
# 4. close_expired_positions — guard on heap[0] avoids function call overhead
# 5. update_sim_balance    — .append bound once per call
# 6. prepare_data          — high_time/low_time stored as int64 views
# 7. run_backtest_loop     — local aliases for all hot-path variables
# =============================================================================
import heapq
import logging
import warnings
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

MIN_PRICE       = 0.0001
INITIAL_BALANCE = 800
COMISION        = 0.1
DEFAULT_CANDLES = 50


# =============================================================================
# PREPARE DATA
# =============================================================================

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

        sym_data[sym] = {
            'ts':        ts,
            'ts_int':    ts_int,
            'open':      data['open'],
            'close':     close_view,
            'high':      data['high'],
            'low':       data['low'],
            'signal':    data['signal'],
            'len':       len(ts),
            'high_time': data['high_time'].view('int64'),
            'low_time':  data['low_time'].view('int64'),
        }

        ts_int_arrays[sym] = ts_int
        close_arrays[sym]  = close_view
        all_ts_int_lists.append(ts_int)

    signals_by_time = {}
    for sym in symbols:
        signal_arr = sym_data[sym]['signal']
        sig_idxs   = np.nonzero(signal_arr)[0]
        if sig_idxs.size > 0:
            ts_int_view = sym_data[sym]['ts_int']
            for t_int, idx in zip(ts_int_view[sig_idxs], sig_idxs):
                key = int(t_int)
                if key not in signals_by_time:
                    signals_by_time[key] = []
                signals_by_time[key].append((sym, int(idx)))

    if all_ts_int_lists:
        all_timestamps_int = np.unique(np.concatenate(all_ts_int_lists))
    else:
        all_timestamps_int = np.array([], dtype=np.int64)

    all_timestamps_dt = all_timestamps_int.view('datetime64[ns]')
    symbol_order      = {s: i for i, s in enumerate(symbols)}

    return (sym_data, signals_by_time, all_timestamps_int, all_timestamps_dt,
            symbol_order, ts_int_arrays, close_arrays)


# =============================================================================
# DETECT INTRABAR EXIT
# =============================================================================

def detect_intrabar_exit(d, buy_idx, sell_idx, tp_price, sl_price, is_short=False):
    if sell_idx < buy_idx:
        return False, None, None, None

    high_slice = d['high'][buy_idx:sell_idx + 1]
    low_slice  = d['low'][buy_idx:sell_idx + 1]

    if is_short:
        tp_hits = np.flatnonzero(low_slice  <= tp_price) if tp_price > -np.inf else np.array([], dtype=np.intp)
        sl_hits = np.flatnonzero(high_slice >= sl_price) if sl_price <  np.inf else np.array([], dtype=np.intp)
    else:
        tp_hits = np.flatnonzero(high_slice >= tp_price) if tp_price <  np.inf else np.array([], dtype=np.intp)
        sl_hits = np.flatnonzero(low_slice  <= sl_price) if sl_price > -np.inf else np.array([], dtype=np.intp)

    tp_first = int(tp_hits[0]) + buy_idx if tp_hits.size > 0 else -1
    sl_first = int(sl_hits[0]) + buy_idx if sl_hits.size > 0 else -1

    if tp_first == -1 and sl_first == -1:
        return False, None, None, None

    if tp_first != -1 and sl_first != -1:
        if tp_first != sl_first:
            if sl_first < tp_first:
                return True, sl_first, 'SL', sl_price
            return True, tp_first, 'TP', tp_price
        # same bar — compare intrabar timestamps stored as int64
        if is_short:
            tp_time = int(d['low_time'][tp_first])
            sl_time = int(d['high_time'][sl_first])
        else:
            tp_time = int(d['high_time'][tp_first])
            sl_time = int(d['low_time'][sl_first])
        if tp_time <= sl_time:
            return True, tp_first, 'TP', tp_price
        return True, sl_first, 'SL', sl_price

    if sl_first != -1:
        return True, sl_first, 'SL', sl_price
    return True, tp_first, 'TP', tp_price


# =============================================================================
# CLOSE POSITION
# =============================================================================

def close_position(pos, exec_time, exec_price, exit_reason, comi_factor,
                   trades, trade_times, trade_log_cols, cash_bank, blocked_cash):
    qty             = pos['qty']
    buy_price       = pos['buy_price']
    is_short        = pos['is_short']
    commission_buy  = pos['commission_buy']
    commission_sell = qty * exec_price * comi_factor

    if is_short:
        cash_bank    -= qty * exec_price + commission_sell
        blocked_cash -= pos['blocked_amount']
        if blocked_cash < 0.0 and abs(blocked_cash) < 1e-12:
            blocked_cash = 0.0
        profit = (buy_price - exec_price) * qty - commission_buy - commission_sell
    else:
        cash_bank += qty * exec_price - commission_sell
        profit     = (exec_price - buy_price) * qty - commission_buy - commission_sell

    sym = pos['symbol']
    trades[sym].append(profit)
    trade_times[sym].append(exec_time)

    tl = trade_log_cols
    tl['symbol'].append(sym)
    tl['buy_time'].append(pos['buy_time'])
    tl['buy_price'].append(buy_price)
    tl['sell_time'].append(exec_time)
    tl['sell_price'].append(exec_price)
    tl['qty'].append(qty)
    tl['profit'].append(profit)
    tl['exit_reason'].append(exit_reason)
    tl['commission_buy'].append(commission_buy)
    tl['commission_sell'].append(commission_sell)
    tl['position_type'].append('SHORT' if is_short else 'LONG')

    return cash_bank, blocked_cash


# =============================================================================
# CLOSE EXPIRED POSITIONS
# =============================================================================

def close_expired_positions(t_int, open_heap, sym_data, ts_int_arrays, close_arrays,
                            comi_factor, trades, trade_times, trade_log_cols,
                            cash_bank, blocked_cash):
    while open_heap and open_heap[0][0] <= t_int:
        _, _, pos = heapq.heappop(open_heap)
        if pos['closed']:
            continue

        if 'exec_price' in pos and pos['exec_time_int'] <= t_int:
            exec_price  = pos['exec_price']
            exec_time   = pos['exec_time']
            exit_reason = pos['exit_reason']
        else:
            sym         = pos['symbol']
            sell_ts_int = pos['sell_time_int']
            ts_arr      = ts_int_arrays[sym]
            close_arr   = close_arrays[sym]
            idx         = np.searchsorted(ts_arr, sell_ts_int, side='right') - 1
            exec_price  = float(close_arr[idx] if idx >= 0 else close_arr[0])
            exec_time   = np.datetime64(int(sell_ts_int), 'ns')
            exit_reason = 'SELL_AFTER'

        cash_bank, blocked_cash = close_position(
            pos, exec_time, exec_price, exit_reason,
            comi_factor, trades, trade_times, trade_log_cols,
            cash_bank, blocked_cash
        )
        pos['closed'] = True

    return cash_bank, blocked_cash


# =============================================================================
# UPDATE SIM BALANCE
# =============================================================================

def update_sim_balance(t_int, open_heap, cash_bank, ts_int_arrays, close_arrays,
                       sim_balance_cols):
    ts_app  = sim_balance_cols['timestamp'].append
    bal_app = sim_balance_cols['balance'].append

    if not open_heap:
        ts_app(np.datetime64(int(t_int), 'ns'))
        bal_app(cash_bank)
        return sim_balance_cols

    symbol_qty_long  = {}
    symbol_qty_short = {}

    for _, _, pos in open_heap:
        if pos['closed']:
            continue
        sym = pos['symbol']
        if pos['is_short']:
            symbol_qty_short[sym] = symbol_qty_short.get(sym, 0.0) + pos['qty']
        else:
            symbol_qty_long[sym]  = symbol_qty_long.get(sym, 0.0) + pos['qty']

    total_value = 0.0
    for sym, qty_sum in symbol_qty_long.items():
        ts_arr    = ts_int_arrays[sym]
        close_arr = close_arrays[sym]
        idx       = np.searchsorted(ts_arr, t_int, side='right') - 1
        total_value += qty_sum * float(close_arr[idx] if idx >= 0 else close_arr[0])

    for sym, qty_sum in symbol_qty_short.items():
        ts_arr    = ts_int_arrays[sym]
        close_arr = close_arrays[sym]
        idx       = np.searchsorted(ts_arr, t_int, side='right') - 1
        total_value -= qty_sum * float(close_arr[idx] if idx >= 0 else close_arr[0])

    ts_app(np.datetime64(int(t_int), 'ns'))
    bal_app(cash_bank + total_value)
    return sim_balance_cols


# =============================================================================
# EXECUTE SIGNAL
# =============================================================================

def execute_signal(sym, buy_idx, cash_bank, blocked_cash, comi_factor, order_amount,
                   sell_after, sym_data, counter, open_heap, tp_pct, sl_pct,
                   is_short=False):
    d       = sym_data[sym]
    price_t = float(d['open'][buy_idx])
    qty     = order_amount / price_t

    commission_buy = float(order_amount * comi_factor)

    if is_short:
        proceeds       = order_amount - commission_buy
        margin         = order_amount * (sl_pct / 100.0) if sl_pct != 0.0 else np.inf
        blocked_amount = proceeds + margin
        cash_bank     += proceeds
    else:
        blocked_amount = 0.0
        cash_bank     -= (order_amount + commission_buy)

    n_velas  = sell_after if sell_after > 0 else DEFAULT_CANDLES
    sell_idx = min(buy_idx + n_velas, d['len'] - 1)

    sell_time_int = int(d['ts_int'][sell_idx])
    sell_time_dt  = d['ts'][sell_idx]

    if is_short:
        tp_price = price_t * (1.0 - tp_pct / 100.0) if tp_pct != 0.0 else -np.inf
        sl_price = price_t * (1.0 + sl_pct / 100.0) if sl_pct != 0.0 else  np.inf
    else:
        tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else  np.inf
        sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -np.inf

    position = {
        'symbol':         sym,
        'qty':            qty,
        'buy_price':      price_t,
        'buy_time':       np.datetime64(int(d['ts_int'][buy_idx]), 'ns'),
        'sell_time':      sell_time_dt,
        'sell_time_int':  sell_time_int,
        'commission_buy': commission_buy,
        'is_short':       is_short,
        'blocked_amount': blocked_amount,
        'closed':         False,
    }

    if is_short and blocked_amount > 0:
        blocked_cash += blocked_amount

    intra, chosen_idx, exit_reason, exec_price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short
    )

    if intra:
        exec_time_int             = int(d['ts_int'][chosen_idx])
        position['exec_price']    = float(exec_price)
        position['exec_time']     = d['ts'][chosen_idx]
        position['exec_time_int'] = exec_time_int
        position['exit_reason']   = exit_reason
        heapq.heappush(open_heap, (exec_time_int, counter, position))
    else:
        heapq.heappush(open_heap, (sell_time_int, counter, position))

    counter += 1
    return cash_bank, blocked_cash, counter


# =============================================================================
# MAIN BACKTEST LOOP
# =============================================================================

def run_backtest_loop(
    all_timestamps_int, sym_data, ts_int_arrays, close_arrays, signals_by_time,
    cash_bank, blocked_cash, order_amount, comi_factor, sell_after, tp_pct, sl_pct,
    trades, trade_times, trade_log_cols, sim_balance_cols
):
    num_signals_executed = 0
    open_heap            = []
    counter              = 0

    oa  = order_amount
    cf  = comi_factor
    sa  = sell_after
    tp  = tp_pct
    sp  = sl_pct
    sd  = sym_data
    ca  = close_arrays
    tia = ts_int_arrays
    sbt = signals_by_time

    for t_int in all_timestamps_int:

        cash_bank, blocked_cash = close_expired_positions(
            t_int, open_heap, sd, tia, ca, cf,
            trades, trade_times, trade_log_cols,
            cash_bank, blocked_cash
        )

        if not open_heap:
            events = sbt.get(int(t_int))
            if events:
                for sym, buy_idx in sorted(events, key=lambda x: x[0]):
                    n_check = sa if sa > 0 else DEFAULT_CANDLES
                    if buy_idx + n_check > len(ca[sym]):
                        continue

                    free_cash = cash_bank - blocked_cash
                    if free_cash < oa:
                        break

                    signal_value = sd[sym]['signal'][buy_idx]
                    if signal_value == 0:
                        continue
                    is_short = signal_value < 0

                    if is_short:
                        if sp == 0.0:
                            continue
                        if free_cash < oa * (sp / 100.0) + oa * cf:
                            continue

                    cash_bank, blocked_cash, counter = execute_signal(
                        sym, buy_idx, cash_bank, blocked_cash, cf, oa, sa,
                        sd, counter, open_heap, tp, sp, is_short
                    )
                    num_signals_executed += 1

        update_sim_balance(t_int, open_heap, cash_bank, tia, ca, sim_balance_cols)

    return cash_bank, blocked_cash, num_signals_executed


# =============================================================================
# METRICS
# =============================================================================

def compute_annualized_sharpe(equity_arr, time_index_int64):
    if equity_arr is None or equity_arr.size < 2:
        return np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = (equity_arr[1:] / equity_arr[:-1]) - 1.0
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return np.nan
    if len(time_index_int64) >= 2:
        deltas_s       = np.diff(time_index_int64).astype(np.float64) / 1e9
        positive       = deltas_s[deltas_s > 0]
        median_delta_s = float(np.median(positive)) if positive.size > 0 else 24 * 3600
    else:
        median_delta_s = 24 * 3600
    periods_per_year = (365.0 * 24.0 * 3600.0) / median_delta_s if median_delta_s > 0 else 252.0
    mean_p = np.mean(returns)
    std_p  = np.std(returns, ddof=0)
    if not np.isfinite(std_p) or std_p == 0.0:
        return np.nan
    return float((mean_p * periods_per_year) / (std_p * np.sqrt(periods_per_year)))


def compute_post_backtest_metrics(symbols, trades, trade_times, all_timestamps_dt,
                                  initial_balance, sim_balance_cols):
    sim_values = np.array(sim_balance_cols['balance'], dtype=np.float64)
    sim_ts_arr = (np.array(sim_balance_cols['timestamp'], dtype='datetime64[ns]')
                  if sim_balance_cols['timestamp']
                  else np.array([], dtype='datetime64[ns]'))
    sim_ts_int = sim_ts_arr.astype('int64') if sim_ts_arr.size > 0 else np.array([], dtype=np.int64)

    final_balance = float(sim_values[-1]) if sim_values.size > 0 else float(initial_balance)
    cummax        = np.maximum.accumulate(sim_values) if sim_values.size > 0 else np.array([initial_balance])
    drawdowns     = (cummax - sim_values) / np.where(cummax == 0, 1, cummax)
    max_dd        = float(np.max(drawdowns)) if drawdowns.size > 0 else 0.0
    all_trades    = [p for lst in trades.values() for p in lst]
    num_trades    = len(all_trades)
    prop_winners  = np.sum(np.array(all_trades) > 0.0) / num_trades if num_trades > 0 else np.nan

    return {
        "final_balance":      final_balance,
        "max_dd_portfolio":   max_dd,
        "sharpe_portfolio":   compute_annualized_sharpe(sim_values, sim_ts_int),
        "proportion_winners": prop_winners,
    }


def build_results_dict(symbols, trades, trade_times,
                       final_balance, num_signals_executed,
                       proportion_winners, max_dd_portfolio,
                       sim_balance_cols, trade_log_cols, sharpe_portfolio):
    return {
        "__PORTFOLIO__": {
            'trades':              [p for lst in trades.values() for p in lst],
            'final_balance':       final_balance,
            'num_signals':         num_signals_executed,
            'proportion_winners':  proportion_winners,
            'max_dd':              max_dd_portfolio,
            'sim_balance_history': sim_balance_cols,
            'trade_log':           pd.DataFrame(trade_log_cols),
            'sharpe':              sharpe_portfolio,
        }
    }


# =============================================================================
# PUBLIC API — identical signature to ZX_compute.py
# =============================================================================

def run_grid_backtest(ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount):
    comi_factor     = float(COMISION) / 100.0
    cash_bank       = float(INITIAL_BALANCE)
    blocked_cash    = 0.0
    initial_balance = INITIAL_BALANCE

    (sym_data, signals_by_time, all_timestamps_int, all_timestamps_dt,
     symbol_order, ts_int_arrays, close_arrays) = prepare_data(ohlcv_arrays)

    symbols        = list(ohlcv_arrays.keys())
    trades         = {sym: [] for sym in symbols}
    trade_times    = {sym: [] for sym in symbols}
    trade_log_cols = {k: [] for k in [
        'symbol', 'buy_time', 'buy_price', 'sell_time', 'sell_price',
        'qty', 'profit', 'exit_reason', 'commission_buy', 'commission_sell',
        'position_type',
    ]}
    sim_balance_cols = {'timestamp': [], 'balance': []}

    cash_bank, blocked_cash, num_signals_executed = run_backtest_loop(
        all_timestamps_int, sym_data, ts_int_arrays, close_arrays, signals_by_time,
        cash_bank, blocked_cash, order_amount, comi_factor, sell_after, tp_pct, sl_pct,
        trades, trade_times, trade_log_cols, sim_balance_cols,
    )

    metrics = compute_post_backtest_metrics(
        symbols, trades, trade_times, all_timestamps_dt, initial_balance, sim_balance_cols
    )

    return build_results_dict(
        symbols, trades, trade_times,
        metrics['final_balance'],
        num_signals_executed,
        metrics['proportion_winners'],
        metrics['max_dd_portfolio'],
        sim_balance_cols,
        trade_log_cols,
        metrics['sharpe_portfolio'],
    )