import heapq
import logging
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from numba import jit

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

MIN_PRICE       = 0.0001
INITIAL_BALANCE = 800
COMISION        = 0.1

# ============================
# NUMBA JIT - Funciones críticas
# ============================
@jit(nopython=True, cache=True)
def searchsorted_price(ts_arr, close_arr, t_int):
    """Búsqueda binaria optimizada con JIT para obtener precio"""
    idx = np.searchsorted(ts_arr, t_int, side='right') - 1
    return close_arr[idx] if idx >= 0 else close_arr[0]

@jit(nopython=True, cache=True)
def find_tp_sl_hits_long(high_arr, low_arr, tp_price, sl_price, has_tp, has_sl):
    """Encuentra hits de TP/SL para LONG con JIT"""
    n = len(high_arr)
    tp_idx = -1
    sl_idx = -1
    
    if has_tp:
        for i in range(n):
            if high_arr[i] >= tp_price:
                tp_idx = i
                break
    
    if has_sl:
        for i in range(n):
            if low_arr[i] <= sl_price:
                sl_idx = i
                break
    
    return tp_idx, sl_idx

@jit(nopython=True, cache=True)
def find_tp_sl_hits_short(high_arr, low_arr, tp_price, sl_price, has_tp, has_sl):
    """Encuentra hits de TP/SL para SHORT con JIT"""
    n = len(high_arr)
    tp_idx = -1
    sl_idx = -1
    
    if has_tp:
        for i in range(n):
            if low_arr[i] <= tp_price:
                tp_idx = i
                break
    
    if has_sl:
        for i in range(n):
            if high_arr[i] >= sl_price:
                sl_idx = i
                break
    
    return tp_idx, sl_idx


# ============================
# prepare_data - ULTRA OPTIMIZADO (sin copias redundantes)
# ============================
def prepare_data(ohlcv_arrays):
    if not ohlcv_arrays:
        return ({}, {}, np.array([], dtype=np.int64), 
                np.array([], dtype='datetime64[ns]'), {}, {}, {})
    
    symbols = list(ohlcv_arrays.keys())
    
    # Pre-allocate estructuras
    ts_int_arrays = {}
    close_arrays = {}
    signals_by_time = defaultdict(list)
    
    # OPTIMIZACIÓN: Estimar capacidad total para pre-allocar array de timestamps
    total_len = sum(len(ohlcv_arrays[sym]['ts']) for sym in symbols)
    all_ts_int = np.empty(total_len, dtype=np.int64)
    offset = 0
    
    for sym in symbols:
        data = ohlcv_arrays[sym]
        
        # OPTIMIZACIÓN CRÍTICA: prepare_ohlcv_arrays YA garantiza datetime64[ns]
        # Solo crear view de int64 y guardar metadata adicional IN-PLACE
        ts = data['ts']
        ts_int = ts.view('int64')
        n = len(ts_int)
        
        # OPTIMIZACIÓN: Copiar al array pre-allocado
        all_ts_int[offset:offset+n] = ts_int
        offset += n
        
        # OPTIMIZACIÓN RADICAL: Agregar solo campos nuevos a data (in-place)
        # Evita crear nuevo dict y copiar todas las referencias
        data['ts_int'] = ts_int
        data['len'] = n
        
        # Referencias directas para acceso rápido
        ts_int_arrays[sym] = ts_int
        close_arrays[sym] = data['close']
        
        # OPTIMIZACIÓN: Procesar señales inline con máscara booleana
        signal_arr = data['signal']
        sig_mask = signal_arr != 0
        
        if np.any(sig_mask):
            sig_idxs = np.flatnonzero(sig_mask)  # Más rápido que np.where()[0]
            t_ints = ts_int[sig_idxs]
            
            # OPTIMIZACIÓN: Batch append
            for t_int, idx in zip(t_ints.tolist(), sig_idxs.tolist()):
                signals_by_time[t_int].append((sym, idx))
    
    # OPTIMIZACIÓN: unique sobre array pre-allocado (sin concatenate)
    all_timestamps_int = np.unique(all_ts_int[:offset])
    all_timestamps_dt = all_timestamps_int.view('datetime64[ns]')
    
    symbol_order = {s: i for i, s in enumerate(symbols)}
    
    # Convertir defaultdict a dict
    signals_by_time = dict(signals_by_time)
    
    # OPTIMIZACIÓN: Retornar ohlcv_arrays directamente (ya modificado in-place)
    return ohlcv_arrays, signals_by_time, all_timestamps_int, all_timestamps_dt, symbol_order, ts_int_arrays, close_arrays


# ============================
# detect_intrabar_exit - ULTRA OPTIMIZADO con NUMBA
# ============================
def detect_intrabar_exit(d, buy_idx, sell_idx, tp_price, sl_price, is_short=False):
    """Detección ultra rápida con NUMBA JIT"""
    if tp_price is None and sl_price is None:
        return False, None, None, None

    if sell_idx < buy_idx:
        return False, None, None, None

    # OPTIMIZACIÓN: Slicing directo
    high_slice = d['high'][buy_idx:sell_idx+1]
    low_slice = d['low'][buy_idx:sell_idx+1]
    
    has_tp = tp_price is not None
    has_sl = sl_price is not None

    if is_short:
        tp_idx, sl_idx = find_tp_sl_hits_short(
            high_slice, low_slice, 
            tp_price if has_tp else 0.0, 
            sl_price if has_sl else 0.0,
            has_tp, has_sl
        )
    else:
        tp_idx, sl_idx = find_tp_sl_hits_long(
            high_slice, low_slice,
            tp_price if has_tp else 0.0,
            sl_price if has_sl else 0.0,
            has_tp, has_sl
        )
    
    tp_first = tp_idx + buy_idx if tp_idx >= 0 else None
    sl_first = sl_idx + buy_idx if sl_idx >= 0 else None

    if tp_first is not None and sl_first is not None:
        if is_short:
            tp_time_val = d['low_time'][tp_first]
            sl_time_val = d['high_time'][sl_first]
        else:
            tp_time_val = d['high_time'][tp_first]
            sl_time_val = d['low_time'][sl_first]

        if tp_first == sl_first:
            if tp_time_val <= sl_time_val:
                return True, tp_first, 'TP', tp_price
            else:
                return True, sl_first, 'SL', sl_price
        else:
            if sl_first < tp_first:
                return True, sl_first, 'SL', sl_price
            else:
                return True, tp_first, 'TP', tp_price

    elif sl_first is not None:
        return True, sl_first, 'SL', sl_price

    elif tp_first is not None:
        return True, tp_first, 'TP', tp_price

    return False, None, None, None


# ============================
# close_position - INLINE OPTIMIZADO
# ============================
def close_position(pos, exec_time, exec_price, exit_reason, comi_factor, 
                   trades, trade_times, trade_log_cols, cash_bank, blocked_cash):
    qty = pos['qty']
    buy_price = pos['buy_price']
    is_short = pos.get('is_short', False)
    commission_buy = pos['commission_buy']
    commission_sell = qty * exec_price * comi_factor

    if is_short:
        cash_bank -= qty * exec_price + commission_sell
        blocked_cash -= pos.get('blocked_amount', 0.0)
        profit = (buy_price - exec_price) * qty - commission_buy - commission_sell
    else:
        cash_bank += qty * exec_price - commission_sell
        profit = (exec_price - buy_price) * qty - commission_buy - commission_sell

    sym = pos['symbol']
    trades[sym].append(profit)
    trade_times[sym].append(exec_time)

    # OPTIMIZACIÓN: Batch append para trade_log
    trade_log_cols['symbol'].append(sym)
    trade_log_cols['buy_time'].append(pos['buy_time'])
    trade_log_cols['buy_price'].append(buy_price)
    trade_log_cols['sell_time'].append(exec_time)
    trade_log_cols['sell_price'].append(exec_price)
    trade_log_cols['qty'].append(qty)
    trade_log_cols['profit'].append(profit)
    trade_log_cols['exit_reason'].append(exit_reason)
    trade_log_cols['commission_buy'].append(commission_buy)
    trade_log_cols['commission_sell'].append(commission_sell)
    trade_log_cols['position_type'].append('SHORT' if is_short else 'LONG')

    # Fix floating point precision
    if -1e-12 < blocked_cash < 0:
        blocked_cash = 0.0

    return cash_bank, blocked_cash


# ============================
# close_expired_positions - OPTIMIZADO con NUMBA lookup
# ============================
def close_expired_positions(t_int, open_heap, ohlcv_arrays, ts_int_arrays, close_arrays,
                           comi_factor, trades, trade_times, trade_log_cols, 
                           cash_bank, blocked_cash):
    
    while open_heap and open_heap[0][0] <= t_int:
        _, _, pos = heapq.heappop(open_heap)
        if pos.get('closed', False):
            continue
            
        if 'exec_price' in pos and pos.get('exec_time_int', 0) <= t_int:
            cash_bank, blocked_cash = close_position(
                pos, pos['exec_time'], pos['exec_price'], pos['exit_reason'],
                comi_factor, trades, trade_times, trade_log_cols,
                cash_bank, blocked_cash
            )
            pos['closed'] = True
        else:
            sym = pos['symbol']
            sell_ts_int = pos.get('sell_time_int', int(ohlcv_arrays[sym]['ts_int'][-1]))
            
            # OPTIMIZACIÓN: Usar función NUMBA JIT
            exec_price = float(searchsorted_price(ts_int_arrays[sym], close_arrays[sym], sell_ts_int))
            exec_time_dt = np.datetime64(int(sell_ts_int), 'ns')
            
            cash_bank, blocked_cash = close_position(
                pos, exec_time_dt, exec_price, 'SELL_AFTER',
                comi_factor, trades, trade_times, trade_log_cols,
                cash_bank, blocked_cash
            )
            pos['closed'] = True
    
    return cash_bank, blocked_cash


# ============================
# update_sim_balance - OPTIMIZADO con NUMBA
# ============================
def update_sim_balance(t_int, open_heap, cash_bank, ts_int_arrays, close_arrays, sim_balance_cols):
    
    if not open_heap:
        sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
        sim_balance_cols['balance'].append(cash_bank)
        return sim_balance_cols
    
    # OPTIMIZACIÓN: Acumular en variables locales (más rápido)
    total_value = 0.0
    symbol_qty_long = {}
    symbol_qty_short = {}
    
    # OPTIMIZACIÓN: Un solo bucle para agrupar
    for _, _, pos in open_heap:
        if not pos.get('closed', False):
            sym = pos['symbol']
            qty = pos['qty']
            
            if pos.get('is_short', False):
                symbol_qty_short[sym] = symbol_qty_short.get(sym, 0.0) + qty
            else:
                symbol_qty_long[sym] = symbol_qty_long.get(sym, 0.0) + qty
    
    # OPTIMIZACIÓN: Calcular valor con NUMBA
    for sym, qty_sum in symbol_qty_long.items():
        price = searchsorted_price(ts_int_arrays[sym], close_arrays[sym], t_int)
        total_value += qty_sum * price
    
    for sym, qty_sum in symbol_qty_short.items():
        price = searchsorted_price(ts_int_arrays[sym], close_arrays[sym], t_int)
        total_value -= qty_sum * price
    
    sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
    sim_balance_cols['balance'].append(cash_bank + total_value)
    
    return sim_balance_cols


# ============================
# execute_signal - ULTRA OPTIMIZADO
# ============================
def execute_signal(sym, buy_idx, cash_bank, blocked_cash, comi_factor, order_amount, sell_after,
                   ohlcv_arrays, counter, open_heap, tp_pct, sl_pct, is_short=False):

    d = ohlcv_arrays[sym]
    price_t = float(d['open'][buy_idx])
    
    # OPTIMIZACIÓN: Cálculos inline
    qty = order_amount / price_t
    commission_buy = order_amount * comi_factor
    
    if is_short:
        proceeds = order_amount - commission_buy
        blocked_amount = proceeds + (order_amount * sl_pct / 100.0 if sl_pct != 0.0 else np.inf)
        cash_bank += proceeds
        blocked_cash += blocked_amount
    else:
        blocked_amount = 0.0
        cash_bank -= order_amount + commission_buy

    # OPTIMIZACIÓN: Calcular sell_idx y precios inline
    sell_idx = min(buy_idx + (sell_after if sell_after > 0 else 50), d['len'] - 1)
    sell_time_int = int(d['ts_int'][sell_idx])

    # OPTIMIZACIÓN: Calcular TP/SL con operaciones inline
    if is_short:
        tp_price = price_t * (1.0 - tp_pct * 0.01) if tp_pct != 0.0 else -np.inf
        sl_price = price_t * (1.0 + sl_pct * 0.01) if sl_pct != 0.0 else np.inf
    else:
        tp_price = price_t * (1.0 + tp_pct * 0.01) if tp_pct != 0.0 else np.inf
        sl_price = price_t * (1.0 - sl_pct * 0.01) if sl_pct != 0.0 else -np.inf

    # OPTIMIZACIÓN: Crear dict con valores necesarios solamente
    position = {
        'symbol': sym,
        'qty': qty,
        'buy_price': price_t,
        'buy_time': np.datetime64(int(d['ts_int'][buy_idx]), 'ns'),
        'sell_time': d['ts'][sell_idx],
        'sell_time_int': sell_time_int,
        'commission_buy': commission_buy,
        'is_short': is_short,
        'blocked_amount': blocked_amount
    }

    # Detectar salida intrabar
    intravela_detected, chosen_idx, exit_reason, exec_price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short
    )

    if intravela_detected:
        position['exec_price'] = float(exec_price)
        position['exec_time'] = d['ts'][chosen_idx]
        position['exec_time_int'] = int(d['ts_int'][chosen_idx])
        position['exit_reason'] = exit_reason
        heapq.heappush(open_heap, (position['exec_time_int'], counter, position))
    else:
        heapq.heappush(open_heap, (sell_time_int, counter, position))

    return cash_bank, blocked_cash, counter + 1


# ============================
# run_backtest_loop - MEGA OPTIMIZADO
# ============================
def run_backtest_loop(
    all_timestamps_int, ohlcv_arrays, ts_int_arrays, close_arrays, signals_by_time,
    cash_bank, blocked_cash, order_amount, comi_factor, sell_after, tp_pct, sl_pct,
    trades, trade_times, trade_log_cols, sim_balance_cols
):
    num_signals_executed = 0
    open_heap = []
    counter = 0
    
    # OPTIMIZACIÓN: Pre-computar TODAS las validaciones
    validation_limits = {sym: len(close_arrays[sym]) - (sell_after if sell_after > 0 else 50) 
                        for sym in close_arrays}
    
    # OPTIMIZACIÓN: Set de timestamps con señales para O(1) lookup
    signal_times_set = set(signals_by_time.keys())
    
    # OPTIMIZACIÓN: Variables locales para operaciones frecuentes
    oa = order_amount
    cf = comi_factor
    sa = sell_after
    tp = tp_pct
    sp = sl_pct
    
    # OPTIMIZACIÓN: Pre-calcular valores constantes
    sp_factor = sp * 0.01 if sp != 0.0 else 0.0
    oa_cf = oa * cf
    
    for t_int in all_timestamps_int:
        # Cerrar posiciones expiradas
        cash_bank, blocked_cash = close_expired_positions(
            t_int, open_heap, ohlcv_arrays, ts_int_arrays, close_arrays, 
            cf, trades, trade_times, trade_log_cols, cash_bank, blocked_cash
        )
        
        # OPTIMIZACIÓN: Solo procesar señales si no hay posiciones Y hay señales
        if not open_heap:
            t_int_key = int(t_int)
            if t_int_key in signal_times_set:
                events = signals_by_time[t_int_key]
                
                # OPTIMIZACIÓN: Sort solo si hay múltiples eventos
                if len(events) > 1:
                    events.sort(key=lambda x: x[0])
                
                for sym, buy_idx in events:
                    # OPTIMIZACIÓN: Validación ultra rápida
                    if buy_idx >= validation_limits[sym]:
                        continue
                    
                    # OPTIMIZACIÓN: Early exit si no hay cash
                    free_cash = cash_bank - blocked_cash
                    if free_cash < oa:
                        break
                    
                    signal_value = ohlcv_arrays[sym]['signal'][buy_idx]
                    if signal_value == 0:
                        continue
                    
                    is_short = signal_value < 0

                    # OPTIMIZACIÓN: Validación SHORT con valores pre-calculados
                    if is_short:
                        if sp == 0.0:
                            continue
                        if free_cash < (oa * sp_factor + oa_cf):
                            continue

                    cash_bank, blocked_cash, counter = execute_signal(
                        sym, buy_idx, cash_bank, blocked_cash, cf, oa, sa, 
                        ohlcv_arrays, counter, open_heap, tp, sp, is_short
                    )
                    num_signals_executed += 1
        
        # Actualizar balance
        update_sim_balance(t_int, open_heap, cash_bank, ts_int_arrays, close_arrays, sim_balance_cols)
    
    return cash_bank, blocked_cash, num_signals_executed


# ============================
# Métricas - OPTIMIZADO
# ============================
def compute_annualized_sharpe(equity_arr, time_index_int64):
    if equity_arr is None or equity_arr.size < 2:
        return np.nan

    # OPTIMIZACIÓN: División vectorizada más eficiente
    returns = np.empty(len(equity_arr) - 1, dtype=np.float64)
    returns[:] = equity_arr[1:] / equity_arr[:-1] - 1.0
    
    # Filter finites
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return np.nan

    if len(time_index_int64) >= 2:
        deltas_s = np.diff(time_index_int64).astype(np.float64) * 1e-9
        positive = deltas_s[deltas_s > 0]
        median_delta_s = float(np.median(positive)) if positive.size > 0 else 86400.0
    else:
        median_delta_s = 86400.0

    periods_per_year = 31536000.0 / median_delta_s if median_delta_s > 0 else 252.0

    mean_periodic = np.mean(returns)
    std_periodic = np.std(returns, ddof=0)
    
    if not np.isfinite(std_periodic) or std_periodic == 0.0:
        return np.nan

    return float(mean_periodic * periods_per_year / (std_periodic * np.sqrt(periods_per_year)))


def compute_post_backtest_metrics(symbols, trades, trade_times, all_timestamps_dt, initial_balance, sim_balance_cols):
    sim_values = np.array(sim_balance_cols['balance'], dtype=np.float64)
    
    if sim_values.size == 0:
        return {
            "final_balance": float(initial_balance),
            "max_dd_portfolio": 0.0,
            "sharpe_portfolio": np.nan,
            "proportion_winners": np.nan
        }
    
    sim_ts_arr = np.array(sim_balance_cols['timestamp'], dtype='datetime64[ns]')
    sim_ts_int = sim_ts_arr.astype('int64')

    final_balance = float(sim_values[-1])

    # OPTIMIZACIÓN: Cálculo vectorizado de drawdown
    cummax_portfolio = np.maximum.accumulate(sim_values)
    drawdowns_portfolio = (cummax_portfolio - sim_values) / np.maximum(cummax_portfolio, 1.0)
    max_dd_portfolio = float(np.max(drawdowns_portfolio))

    sharpe_portfolio = compute_annualized_sharpe(sim_values, sim_ts_int)

    # OPTIMIZACIÓN: Flatten trades más eficiente
    all_trades = np.array([p for lst in trades.values() for p in lst])
    num_trades = len(all_trades)
    proportion_winners = float(np.sum(all_trades > 0.0) / num_trades) if num_trades > 0 else np.nan

    return {
        "final_balance": final_balance,
        "max_dd_portfolio": max_dd_portfolio,
        "sharpe_portfolio": sharpe_portfolio,
        "proportion_winners": proportion_winners       
    }


def build_results_dict(symbols, trades, trade_times, 
                       final_balance, num_signals_executed, 
                       proportion_winners, max_dd_portfolio,
                       sim_balance_cols, trade_log_cols, sharpe_portfolio):

    return {
        "__PORTFOLIO__": {
            'trades': [p for lst in trades.values() for p in lst],
            'final_balance': final_balance,
            'num_signals': num_signals_executed,
            'proportion_winners': proportion_winners,
            'max_dd': max_dd_portfolio,
            'sim_balance_history': sim_balance_cols,
            'trade_log': pd.DataFrame(trade_log_cols),
            'sharpe': sharpe_portfolio
        }
    }


# ============================
# FUNCIÓN PRINCIPAL
# ============================
def run_grid_backtest(ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount):
    comi_factor = COMISION * 0.01
    cash_bank = float(INITIAL_BALANCE)
    blocked_cash = 0.0
    
    (ohlcv_arrays, signals_by_time, all_timestamps_int, all_timestamps_dt, 
     symbol_order, ts_int_arrays, close_arrays) = prepare_data(ohlcv_arrays)
    
    symbols = list(ohlcv_arrays.keys())
    
    trades = {sym: [] for sym in symbols}
    trade_times = {sym: [] for sym in symbols}
    trade_log_cols = {k: [] for k in [
        'symbol','buy_time','buy_price','sell_time','sell_price',
        'qty','profit','exit_reason','commission_buy','commission_sell','position_type']}
    sim_balance_cols = {'timestamp': [], 'balance': []}
    
    cash_bank, blocked_cash, num_signals_executed = run_backtest_loop(
        all_timestamps_int, ohlcv_arrays, ts_int_arrays, close_arrays, signals_by_time,
        cash_bank, blocked_cash, order_amount, comi_factor, sell_after, tp_pct, sl_pct,
        trades, trade_times, trade_log_cols, sim_balance_cols
    )
    
    metrics = compute_post_backtest_metrics(
        symbols, trades, trade_times, all_timestamps_dt, INITIAL_BALANCE, sim_balance_cols
    )
    
    results = build_results_dict(
        symbols, trades, trade_times,
        metrics['final_balance'],
        num_signals_executed,
        metrics['proportion_winners'],
        metrics['max_dd_portfolio'],
        sim_balance_cols,
        trade_log_cols,
        metrics['sharpe_portfolio']
    )
    
    return results