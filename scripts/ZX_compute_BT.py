import heapq
import logging
import warnings
import numpy as np
import pandas as pd
from numba import njit
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

MIN_PRICE            = 0.0001
INITIAL_BALANCE      = 10000
ORDER_AMOUNT         = 100

# -----------------------------
# CÁLCULO DE GANANCIAS NETAS (igual)
# -----------------------------

@njit
def second_diff(close):
    n = len(close)
    accel_raw = np.zeros(n)
    for i in range(2, n):
        accel_raw[i] = close[i] - 2*close[i-1] + close[i-2]
    return accel_raw


@njit
def delta_numba(close):
    n = len(close)
    delta = np.empty(n)
    delta[0] = 0.0
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    return delta

@njit
def rolling_entropy_numba(delta, window=5, bins=10):
    n = len(delta)
    entropia  = np.zeros(n)
    delta_min = delta.min()
    delta_max = delta.max()
    hist = np.zeros(bins)  # reusar array

    for i in range(n):
        start = max(0, i - window + 1)
        hist[:] = 0.0  # resetear histograma
        for j in range(start, i + 1):
            bin_idx = int((delta[j] - delta_min) / (delta_max - delta_min + 1e-9) * bins)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist[bin_idx] += 1
        s = hist.sum()
        e = 0.0
        for k in range(bins):
            if hist[k] > 0:
                p = hist[k] / s
                e -= p * np.log2(p)
        entropia[i] = e
    return entropia


@njit
def ewm_numba(x, span):
    n = len(x)
    alpha = 2 / (span + 1)
    ewm = np.empty(n)
    ewm[0] = x[0]
    for i in range(1, n):
        ewm[i] = alpha * x[i] + (1 - alpha) * ewm[i - 1]
    return ewm

@njit
def add_indicators_arrays(close, m_accel=5):
    delta = delta_numba(close)
    entropia = rolling_entropy_numba(delta, 5, 10)
    accel_raw = second_diff(close)
    accel = ewm_numba(accel_raw, m_accel)
    return entropia, accel


@njit
def explosive_signal_arrays(entropia, accel, entropia_max=2.0, live=False):
    signal = (entropia < entropia_max) & (accel > 0)
    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = False
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted
    return signal


# ============================
# Helper: get_price_at_int (nivel módulo) para evitar closures no picklables
# ============================

def get_price_at_int(sym, t, sym_data, ts_index_map_by_sym_int):

    # normalizar t a int64 (ns)
    t_int   = int(t) if not isinstance(t, (int, np.integer)) else int(t)
    d       = sym_data[sym]
    idx_map = ts_index_map_by_sym_int[sym]
    idx     = idx_map.get(t_int)
    if idx is not None:
        return float(d['close'][idx])
    # fallback: searchsorted sobre d['ts_int']
    idx = np.searchsorted(d['ts_int'], t_int, side='right') - 1
    if idx >= 0:
        return float(d['close'][idx])
    return None

def run_grid_backtest(
    ohlcv_arrays,
    sell_after,
    initial_balance=10000,
    order_amount=100,
    tp_pct=0.0,
    sl_pct=0.0,
    comi_pct=0.05
):

    # -------------------------
    # Configuración inicial
    # -------------------------
    comi_factor = float(comi_pct) / 100.0
    cash        = float(initial_balance)
    num_signals_executed = 0

    # Preparar sym_data con arrays y ts_int
    symbols = list(ohlcv_arrays.keys())
    sym_data = {}
    for sym in symbols:
        data = ohlcv_arrays[sym]
        ts = data['ts']
        ts_int = ts.astype('int64')
        sym_data[sym] = {
            'ts': ts,
            'ts_int': ts_int,
            'close': data['close'],
            'high': data.get('high', None),
            'low': data.get('low', None),
            'signal': data['signal'],
            'len': len(ts)
        }

    # Señales por timestamp
    signals_by_time = {}
    for sym in symbols:
        d = sym_data[sym]
        sig_idxs = np.nonzero(d['signal'])[0]
        ts_int_arr = d['ts_int']
        for idx in sig_idxs:
            t_int = int(ts_int_arr[idx])
            lst = signals_by_time.get(t_int)
            if lst is None:
                signals_by_time[t_int] = [(sym, int(idx))]
            else:
                lst.append((sym, int(idx)))

    # Array ordenado de todos los timestamps
    all_ts_set = set()
    for d in sym_data.values():
        all_ts_set.update(d['ts_int'].tolist())
    all_timestamps_int = np.array(sorted(all_ts_set), dtype=np.int64)
    all_timestamps_dt = all_timestamps_int.astype('datetime64[ns]')

    # -------------------------
    # Estructuras auxiliares
    # -------------------------
    trades = {sym: [] for sym in symbols}
    trade_times = {sym: [] for sym in symbols}

    # Trade log optimizado con listas
    trade_log_cols = {
        'symbol': [],
        'buy_time': [],
        'buy_price': [],
        'sell_time': [],
        'sell_price': [],
        'qty': [],
        'profit': [],
        'exit_reason': [],
        'commission_buy': [],
        'commission_sell': []
    }

    # Sim balance optimizado con listas separadas
    sim_balance_cols = {
        'timestamp': [],
        'balance': []
    }

    open_positions_heap = []
    counter = 0

    symbol_order = {s: i for i, s in enumerate(symbols)}
    ts_index_map_by_sym_int = {sym: {int(t): idx for idx, t in enumerate(d['ts_int'])} for sym, d in sym_data.items()}

    def _get_price_at_int(sym_local, t_int_local):
        d_local = sym_data[sym_local]
        idx_map_local = ts_index_map_by_sym_int[sym_local]
        idx_local = idx_map_local.get(int(t_int_local))
        if idx_local is not None:
            return float(d_local['close'][idx_local])
        arr_ts = d_local['ts_int']
        idx_local = np.searchsorted(arr_ts, np.int64(t_int_local), side='right') - 1
        if idx_local >= 0:
            return float(d_local['close'][idx_local])
        return None

    def close_position(pos, exec_time, exec_price, exit_reason):
        nonlocal cash
        qty = pos['qty']
        buy_price = pos['buy_price']
        commission_buy = pos.get('commission_buy', 0.0)
        commission_sell = (qty * exec_price) * comi_factor if comi_factor != 0.0 else 0.0
        cash += qty * exec_price - commission_sell
        profit = (exec_price - buy_price) * qty - commission_buy - commission_sell
        trades[pos['symbol']].append(profit)
        trade_times[pos['symbol']].append(exec_time)

        # Llenar listas separadas
        trade_log_cols['symbol'].append(pos['symbol'])
        trade_log_cols['buy_time'].append(pos['buy_time'])
        trade_log_cols['buy_price'].append(buy_price)
        trade_log_cols['sell_time'].append(exec_time)
        trade_log_cols['sell_price'].append(exec_price)
        trade_log_cols['qty'].append(qty)
        trade_log_cols['profit'].append(profit)
        trade_log_cols['exit_reason'].append(exit_reason)
        trade_log_cols['commission_buy'].append(commission_buy)
        trade_log_cols['commission_sell'].append(commission_sell)

    # -------------------------
    # Bucle principal
    # -------------------------
    open_heap = open_positions_heap
    sym_data_local = sym_data
    signals_local = signals_by_time
    symbol_order_local = symbol_order

    for t_int in all_timestamps_int:
        # Cerrar posiciones vencidas
        while open_heap and open_heap[0][0] <= t_int:
            _, _, pos = heapq.heappop(open_heap)
            if pos.get('closed', False):
                continue
            if 'exec_price' in pos and ('exec_time_int' in pos) and pos['exec_time_int'] <= t_int:
                close_position(pos, pos['exec_time'], pos['exec_price'], pos['exit_reason'])
                pos['closed'] = True
            else:
                sym = pos['symbol']
                sell_ts_int = pos.get('sell_time_int', None)
                if sell_ts_int is None:
                    sell_ts_int = int(np.int64(sym_data_local[sym]['ts_int'][-1]))
                exec_price = _get_price_at_int(sym, sell_ts_int)
                if exec_price is not None:
                    exec_time_dt = np.datetime64(int(sell_ts_int), 'ns')
                    close_position(pos, exec_time_dt, exec_price, 'SELL_AFTER')
                else:
                    exec_price = float(sym_data_local[sym]['close'][-1])
                    last_time_dt = sym_data_local[sym]['ts'][-1]
                    close_position(pos, last_time_dt, exec_price, 'FORCED_LAST')
                pos['closed'] = True

        # Actualizar sim balance solo si hay posiciones abiertas
        if open_heap:
            positions_value = sum(pos['qty'] * _get_price_at_int(pos['symbol'], t_int) for _, _, pos in open_heap if not pos.get('closed', False))
            sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
            sim_balance_cols['balance'].append(cash + positions_value)
            continue

        # Abrir nuevas posiciones
        events = signals_local.get(int(t_int), [])
        if events:
            events = sorted(events, key=lambda x: symbol_order_local[x[0]])
            for sym, buy_idx in events:
                if cash < order_amount:
                    break
                d = sym_data_local[sym]
                price_t = float(d['close'][buy_idx])
                qty = order_amount / price_t
                commission_buy = order_amount * comi_factor if comi_factor != 0.0 else 0.0
                cash -= (order_amount + commission_buy)
                num_signals_executed += 1

                sell_idx = min(buy_idx + sell_after, d['len'] - 1)
                sell_time_dt = d['ts'][sell_idx]
                sell_time_int = int(d['ts_int'][sell_idx])
                tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else np.inf
                sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -np.inf

                position = {
                    'symbol': sym,
                    'qty': qty,
                    'buy_price': price_t,
                    'buy_time': np.datetime64(int(t_int), 'ns'),
                    'sell_time': sell_time_dt,
                    'sell_time_int': sell_time_int,
                    'commission_buy': commission_buy
                }

                # Detección intravela
                intravela_detected = False
                if tp_price is not None or sl_price is not None:
                    if d['high'] is not None and d['low'] is not None:
                        start = buy_idx + 1
                        end = sell_idx
                        if end >= start:
                            high_slice = d['high'][start:end+1]
                            low_slice = d['low'][start:end+1]
                            tp_hits = np.where((tp_price is not None) & (high_slice >= tp_price))[0]
                            sl_hits = np.where((sl_price is not None) & (low_slice <= sl_price))[0]
                            tp_first = tp_hits[0] + start if tp_hits.size > 0 else None
                            sl_first = sl_hits[0] + start if sl_hits.size > 0 else None

                            if tp_first is not None and sl_first is not None:
                                if sl_first <= tp_first:
                                    chosen_idx = sl_first
                                    exit_reason = 'SL'
                                    exec_price = sl_price
                                else:
                                    chosen_idx = tp_first
                                    exit_reason = 'TP'
                                    exec_price = tp_price
                                intravela_detected = True
                            elif sl_first is not None:
                                chosen_idx = sl_first
                                exit_reason = 'SL'
                                exec_price = sl_price
                                intravela_detected = True
                            elif tp_first is not None:
                                chosen_idx = tp_first
                                exit_reason = 'TP'
                                exec_price = tp_price
                                intravela_detected = True

                if intravela_detected:
                    exec_time_dt = d['ts'][chosen_idx]
                    exec_time_int = int(d['ts_int'][chosen_idx])
                    position.update({
                        'exec_price': float(exec_price),
                        'exec_time': exec_time_dt,
                        'exec_time_int': exec_time_int,
                        'exit_reason': exit_reason
                    })
                    heapq.heappush(open_heap, (exec_time_int, counter, position))
                    counter += 1
                else:
                    heapq.heappush(open_heap, (sell_time_int, counter, position))
                    counter += 1

        # Registrar balance actual
        positions_value = sum(pos['qty'] * _get_price_at_int(pos['symbol'], t_int) for _, _, pos in open_heap if not pos.get('closed', False))
        sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
        sim_balance_cols['balance'].append(cash + positions_value)

    # -------------------------
    # Cierre final de posiciones
    # -------------------------
    while open_heap:
        _, _, pos = heapq.heappop(open_heap)
        if pos.get('closed', False):
            continue
        sym = pos['symbol']
        d = sym_data_local[sym]
        if 'exec_price' in pos:
            close_position(pos, pos['exec_time'], pos['exec_price'], pos['exit_reason'])
        else:
            sell_ts_int = pos.get('sell_time_int', int(d['ts_int'][-1]))
            exec_price = _get_price_at_int(sym, sell_ts_int)
            if exec_price is not None:
                exec_time_dt = np.datetime64(int(sell_ts_int), 'ns')
                close_position(pos, exec_time_dt, exec_price, 'SELL_AFTER')
            else:
                exec_price = float(d['close'][-1])
                last_time_dt = d['ts'][-1]
                close_position(pos, last_time_dt, exec_price, 'FORCED_LAST')

    # -------------------------
    # Resultados finales
    # -------------------------
    final_balance = cash
    all_trades = []
    for sym in symbols:
        all_trades.extend(trades[sym])
    num_trades = len(all_trades)
    proportion_winners = np.sum(np.array(all_trades) > 0.0) / num_trades if num_trades > 0 else np.nan

    ts_index_map = {np.datetime64(int(t_int), 'ns'): i for i, t_int in enumerate(all_timestamps_int)}
    max_dd_by_symbol = {}
    final_balance_by_symbol = {}
    for sym in symbols:
        profits_series = np.zeros(len(all_timestamps_dt), dtype=np.float64)
        for profit, t_close in zip(trades[sym], trade_times[sym]):
            idx = ts_index_map.get(t_close)
            if idx is None:
                t_close_int = int(t_close.astype('int64'))
                idx = int(np.searchsorted(all_timestamps_int, t_close_int, side='right') - 1)
                if idx < 0:
                    continue
            profits_series[idx] += profit
        equity = initial_balance + np.cumsum(profits_series)
        if equity.size > 0:
            cummax_sym = np.maximum.accumulate(equity)
            drawdowns_sym = (cummax_sym - equity) / np.where(cummax_sym == 0, 1, cummax_sym)
            max_dd_by_symbol[sym] = float(np.nanmax(drawdowns_sym))
            final_balance_by_symbol[sym] = float(equity[-1])
        else:
            max_dd_by_symbol[sym] = 0.0
            final_balance_by_symbol[sym] = float(initial_balance)

    if len(sim_balance_cols['balance']) == 0:
        sim_values = np.array([initial_balance], dtype=np.float64)
    else:
        sim_values = np.array(sim_balance_cols['balance'], dtype=np.float64)

    cummax = np.maximum.accumulate(sim_values)
    drawdowns = (cummax - sim_values) / np.where(cummax == 0, 1, cummax)
    max_dd_portfolio = float(np.nanmax(drawdowns)) if sim_values.size > 0 else 0.0

    results = {}
    for sym in symbols:
        results[sym] = {
            'df': None,
            'trades': trades[sym],
            'final_balance': final_balance_by_symbol[sym],
            'num_signals': len(trades[sym]),
            'proportion_winners': (np.nan if len(trades[sym]) == 0 else np.sum(np.array(trades[sym]) > 0.0) / len(trades[sym])),
            'max_dd': max_dd_by_symbol[sym]
        }

    results["__PORTFOLIO__"] = {
        'df': None,
        'trades': all_trades,
        'final_balance': final_balance,
        'num_signals': num_signals_executed,
        'proportion_winners': proportion_winners,
        'max_dd': max_dd_portfolio,
        'sim_balance_history': sim_balance_cols,  # dict de listas
        'trade_log': pd.DataFrame(trade_log_cols)  # Crear DataFrame una sola vez al final
    }

    return results

