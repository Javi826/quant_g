import heapq
import logging
import warnings
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

MIN_PRICE       = 0.0001
INITIAL_BALANCE = 10000
ORDER_AMOUNT    = 100
COMISION = 0.05

# ============================
# Helper: get_price_at_int - OPTIMIZADO
# ============================

def get_price_at_int(sym, t, sym_data, ts_int_arrays, close_arrays):
    """
    Versión optimizada que usa arrays pre-extraídos para evitar
    accesos a diccionarios anidados.
    """
    t_int = np.int64(t)
    ts_arr = ts_int_arrays[sym]
    close_arr = close_arrays[sym]
    
    # Búsqueda binaria directa sin diccionario intermedio
    idx = np.searchsorted(ts_arr, t_int, side='right') - 1
    
    if idx >= 0:
        return close_arr[idx]
    return None

# ============================
# Helper: prepare_data
# ============================

def prepare_data(ohlcv_arrays):
    from collections import defaultdict

    if not ohlcv_arrays:
        return ({}, {}, np.array([], dtype=np.int64), 
                np.array([], dtype='datetime64[ns]'), {}, {}, {})
    
    symbols = list(ohlcv_arrays.keys())
    sym_data, ts_int_arrays, close_arrays = {}, {}, {}
    all_ts_int_lists = []
    
    for sym in symbols:
        data = ohlcv_arrays[sym]
        ts = data['ts']
        if ts.dtype.kind != 'M':
            ts = ts.astype('datetime64[ns]')
        ts_int = ts.view('int64')
        
        close_view = data['close']
        
        sym_data[sym] = {
            'ts': ts,
            'ts_int': ts_int,
            'close': close_view,
            'high': data.get('high'),
            'low': data.get('low'),
            'signal': data['signal'],
            'len': len(ts),
            'high_time': data.get('high_time'),
            'low_time': data.get('low_time')
        }
        
        ts_int_arrays[sym] = ts_int
        close_arrays[sym]  = close_view
        all_ts_int_lists.append(ts_int)
    
    signals_by_time = defaultdict(list)
    for sym in symbols:
        sig_idxs = np.nonzero(sym_data[sym]['signal'])[0]
        if sig_idxs.size > 0:
            t_ints = sym_data[sym]['ts_int'][sig_idxs]
            for t_int, idx in zip(t_ints, sig_idxs):
                signals_by_time[int(t_int)].append((sym, int(idx)))
    
    all_timestamps_int = np.unique(np.concatenate(all_ts_int_lists or [np.array([], dtype=np.int64)]))
    all_timestamps_dt = all_timestamps_int.view('datetime64[ns]')
    symbol_order = {s: i for i, s in enumerate(symbols)}
    
    return sym_data, dict(signals_by_time), all_timestamps_int, all_timestamps_dt, symbol_order, ts_int_arrays, close_arrays


# ============================
# Helper: close_expired_positions - MODIFICADO
# ============================

def close_expired_positions(t_int, open_heap, sym_data_local, ts_int_arrays, close_arrays,
                            comi_factor, trades, trade_times, trade_log_cols, cash):
    while open_heap and open_heap[0][0] <= t_int:
        _, _, pos = heapq.heappop(open_heap)
        if pos.get('closed', False):
            continue
        if 'exec_price' in pos and ('exec_time_int' in pos) and pos['exec_time_int'] <= t_int:
            cash = close_position(pos, pos['exec_time'], pos['exec_price'], pos['exit_reason'],
                                  comi_factor, trades, trade_times, trade_log_cols, cash)
            pos['closed'] = True
        else:
            sym = pos['symbol']
            sell_ts_int = pos.get('sell_time_int', int(sym_data_local[sym]['ts_int'][-1]))
            exec_price = get_price_at_int(sym, sell_ts_int, sym_data_local, ts_int_arrays, close_arrays)
            if exec_price is not None:
                exec_time_dt = np.datetime64(int(sell_ts_int), 'ns')
                cash = close_position(pos, exec_time_dt, exec_price, 'SELL_AFTER',
                                      comi_factor, trades, trade_times, trade_log_cols, cash)
            else:
                exec_price = float(sym_data_local[sym]['close'][-1])
                last_time_dt = sym_data_local[sym]['ts'][-1]
                cash = close_position(pos, last_time_dt, exec_price, 'FORCED_LAST',
                                      comi_factor, trades, trade_times, trade_log_cols, cash)
            pos['closed'] = True
    return cash

# ============================
# Helper: detect_intrabar_exit
# ============================

def detect_intrabar_exit(d, buy_idx, sell_idx, tp_price, sl_price):
    """
    Detecta si TP o SL ocurren dentro de la barra y devuelve la información del cierre.
    """

    intravela_detected = False
    chosen_idx = None
    exit_reason = None
    exec_price = None

    if tp_price is None and sl_price is None:
        return intravela_detected, chosen_idx, exit_reason, exec_price

    # Validar timestamps intrabar
    if (tp_price is not None or sl_price is not None) and (d.get('high_time') is None or d.get('low_time') is None):
        raise ValueError(f"Faltan 'high_time' o 'low_time' para símbolo {d.get('symbol','?')} necesario para TP/SL intrabar")

    start = buy_idx + 1
    end = sell_idx
    if end < start:
        return intravela_detected, chosen_idx, exit_reason, exec_price

    # Slices de precios
    high_slice = d['high'][start:end+1]
    low_slice = d['low'][start:end+1]

    # Detectar hits de TP/SL dentro del slice
    tp_hits = np.where(high_slice >= tp_price)[0] if tp_price is not None else np.array([], dtype=int)
    sl_hits = np.where(low_slice <= sl_price)[0] if sl_price is not None else np.array([], dtype=int)

    tp_first = tp_hits[0] + start if tp_hits.size > 0 else None
    sl_first = sl_hits[0] + start if sl_hits.size > 0 else None

    # Si ambos ocurren
    if tp_first is not None and sl_first is not None:
        tp_time_val = d['high_time'][tp_first]
        sl_time_val = d['low_time'][sl_first]

        if tp_first == sl_first:
            # Ambos en la misma vela → usar timestamps intrabar
            if tp_time_val <= sl_time_val:
                chosen_idx = tp_first
                exit_reason = 'TP'
                exec_price = tp_price
            else:
                chosen_idx = sl_first
                exit_reason = 'SL'
                exec_price = sl_price
        else:
            # Diferentes velas → tomar la primera que ocurre en índice
            if sl_first < tp_first:
                chosen_idx = sl_first
                exit_reason = 'SL'
                exec_price = sl_price
            else:
                chosen_idx = tp_first
                exit_reason = 'TP'
                exec_price = tp_price

        intravela_detected = True

    # Solo SL detectado
    elif sl_first is not None:
        chosen_idx = sl_first
        exit_reason = 'SL'
        exec_price = sl_price
        intravela_detected = True

    # Solo TP detectado
    elif tp_first is not None:
        chosen_idx = tp_first
        exit_reason = 'TP'
        exec_price = tp_price
        intravela_detected = True

    return intravela_detected, chosen_idx, exit_reason, exec_price


# ============================
# Helper: compute_post_backtest_metrics
# ============================

def compute_annualized_sharpe(equity_arr, time_index_int64):
    """
    Calcula el Sharpe ratio anualizado a partir de una serie de equity y sus timestamps.
    Maneja retornos irregulares y casos especiales de NaN o desviación cero.
    """
    if equity_arr is None or equity_arr.size < 2:
        return np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        returns = (equity_arr[1:] / equity_arr[:-1]) - 1.0
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return np.nan

    # Determinar el delta de tiempo entre periodos
    if len(time_index_int64) >= 2:
        deltas_s = np.diff(time_index_int64).astype(np.float64) / 1e9
        positive = deltas_s[deltas_s > 0]
        median_delta_s = float(np.median(positive)) if positive.size > 0 else 24*3600
    else:
        median_delta_s = 24*3600

    periods_per_year = (365.0 * 24.0 * 3600.0) / median_delta_s if median_delta_s > 0 else 252.0

    mean_periodic = np.mean(returns)
    std_periodic = np.std(returns, ddof=0)
    if not np.isfinite(std_periodic) or std_periodic == 0.0:
        return np.nan

    annualized_mean = mean_periodic * periods_per_year
    annualized_std = std_periodic * np.sqrt(periods_per_year)
    return float(annualized_mean / annualized_std)


# ============================
# compute_post_backtest_metrics - solo por portafolio
# ============================
def compute_post_backtest_metrics(symbols, trades, trade_times, all_timestamps_dt, initial_balance, sim_balance_cols):
    """
    Calcula métricas únicamente a nivel de portafolio.
    """
    sim_values = np.array(sim_balance_cols['balance'], dtype=np.float64)
    sim_ts_arr = np.array(sim_balance_cols['timestamp'], dtype='datetime64[ns]') if len(sim_balance_cols['timestamp']) > 0 else np.array([], dtype='datetime64[ns]')
    sim_ts_int = sim_ts_arr.astype('int64') if sim_ts_arr.size > 0 else np.array([], dtype=np.int64)

    final_balance = float(sim_values[-1]) if sim_values.size > 0 else float(initial_balance)

    # Drawdown máximo
    cummax_portfolio = np.maximum.accumulate(sim_values) if sim_values.size > 0 else np.array([initial_balance])
    drawdowns_portfolio = (cummax_portfolio - sim_values) / np.where(cummax_portfolio == 0, 1, cummax_portfolio)
    max_dd_portfolio = float(np.max(drawdowns_portfolio)) if drawdowns_portfolio.size > 0 else 0.0

    # Sharpe anualizado
    sharpe_portfolio = compute_annualized_sharpe(sim_values, sim_ts_int)

    # Trades combinados
    all_trades = [p for lst in trades.values() for p in lst]
    num_trades = len(all_trades)
    proportion_winners = np.sum(np.array(all_trades) > 0.0) / num_trades if num_trades > 0 else np.nan

    return {
        "final_balance": final_balance,
        "max_dd_portfolio": max_dd_portfolio,
        "sharpe_portfolio": sharpe_portfolio,
        "proportion_winners": proportion_winners,
        "final_balance_by_symbol": {},  
        "max_dd_by_symbol": {},         
        "sharpe_by_symbol": {}          
    }


# ============================
# build_results_dict - solo por portafolio
# ============================
def build_results_dict(symbols, trades, trade_times, final_balance_by_symbol, 
                       max_dd_by_symbol, sharpe_by_symbol, 
                       final_balance, num_signals_executed, 
                       proportion_winners, max_dd_portfolio,
                       sim_balance_cols, trade_log_cols, sharpe_portfolio):

    results = {
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
    return results

# ============================
# Helper: update_sim_balance - MODIFICADO
# ============================

def update_sim_balance(t_int, open_heap, cash, sym_data_local, ts_int_arrays, close_arrays, sim_balance_cols):
    active_positions = [pos for _, _, pos in open_heap if not pos.get('closed', False)]
    
    if not active_positions:
        total_value = 0.0
    else:
        # Agrupar por símbolo
        symbol_groups = {}
        for pos in active_positions:
            sym = pos['symbol']
            symbol_groups.setdefault(sym, []).append(pos['qty'])
        
        total_value = 0.0
        for sym, qtys in symbol_groups.items():
            # Obtener precio actual solo una vez por símbolo
            ts_int = ts_int_arrays[sym]
            close_arr = close_arrays[sym]

            # Buscar índice más cercano sin pasar el timestamp actual
            idx = np.searchsorted(ts_int, t_int, side='right') - 1
            if idx < 0:
                price = close_arr[0]
            else:
                price = close_arr[idx]

            # Multiplicar suma de cantidades * precio
            total_value += np.sum(qtys) * price

    sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
    sim_balance_cols['balance'].append(cash + total_value)
    return sim_balance_cols

# ============================
# Helper: execute_signal
# ============================

def execute_signal(sym, buy_idx, cash, comi_factor, order_amount, sell_after,
                   sym_data_local, counter, open_heap, num_signals_executed,
                   tp_pct=0.0, sl_pct=0.0):
    d = sym_data_local[sym]
    price_t = float(d['close'][buy_idx])
    qty = order_amount / price_t

    # Restar del cash el nominal más comisión de compra
    commission_buy = qty * price_t * comi_factor
    cash -= (order_amount + commission_buy)
    num_signals_executed += 1

    # Determinar índice de venta
    sell_idx = min(buy_idx + sell_after, d['len'] - 1)
    sell_time_dt = d['ts'][sell_idx]
    sell_time_int = int(d['ts_int'][sell_idx])

    # Definir precios de TP/SL
    tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else np.inf
    sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -np.inf

    # Crear posición
    position = {
        'symbol': sym,
        'qty': qty,
        'buy_price': price_t,
        'buy_time': np.datetime64(int(d['ts_int'][buy_idx]), 'ns'),
        'sell_time': sell_time_dt,
        'sell_time_int': sell_time_int,
        'commission_buy': commission_buy
    }

    # Detectar si TP/SL ocurre intrabar
    intravela_detected, chosen_idx, exit_reason, exec_price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price
    )

    # Actualizar posición con precio de cierre y empujar al heap
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
    else:
        heapq.heappush(open_heap, (sell_time_int, counter, position))

    counter += 1
    return cash, counter, num_signals_executed, open_heap

def close_position(pos, exec_time, exec_price, exit_reason, comi_factor, trades, trade_times, trade_log_cols, cash):
    qty = pos['qty']
    buy_price = pos['buy_price']

    # Comisiones simétricas con el mismo porcentaje
    commission_buy = qty * buy_price * comi_factor
    commission_sell = qty * exec_price * comi_factor

    cash += qty * exec_price - commission_sell
    profit = (exec_price - buy_price) * qty - commission_buy - commission_sell

    sym = pos['symbol']
    trades[sym].append(profit)
    trade_times[sym].append(exec_time)

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

    return cash


#===================
# Helper: initialize_backtest_structures
# ============================
def initialize_backtest_structures(symbols):
    trades = {sym: [] for sym in symbols}
    trade_times = {sym: [] for sym in symbols}
    trade_log_cols = {k: [] for k in [
        'symbol','buy_time','buy_price','sell_time','sell_price',
        'qty','profit','exit_reason','commission_buy','commission_sell']}
    sim_balance_cols = {'timestamp': [], 'balance': []}
    open_heap = []
    return trades, trade_times, trade_log_cols, sim_balance_cols, open_heap

# ============================
# Helper: process_signals_for_timestamp - MODIFICADO
# ============================

def process_signals_for_timestamp(
    t_int, open_heap, sym_data_local, ts_int_arrays, close_arrays,
    cash, sim_balance_cols, signals_local, symbol_order_local,
    order_amount, comi_factor, sell_after, counter, num_signals_executed,
    tp_pct, sl_pct, trades, trade_times, trade_log_cols,
    close_expired_positions_fn, update_sim_balance_fn, execute_signal_fn
):
    # Evitar búsquedas de nombres dentro del bucle
    close_expired_positions = close_expired_positions_fn
    update_sim_balance = update_sim_balance_fn
    execute_signal = execute_signal_fn

    # Cerrar posiciones expiradas
    cash = close_expired_positions(
        t_int, open_heap, sym_data_local, ts_int_arrays, close_arrays,
        comi_factor, trades, trade_times, trade_log_cols, cash
    )

    # Si hay posiciones abiertas, solo actualizar balance
    if open_heap:
        sim_balance_cols = update_sim_balance(
            t_int, open_heap, cash, sym_data_local, ts_int_arrays, close_arrays, sim_balance_cols
        )
        return cash, counter, num_signals_executed, sim_balance_cols, open_heap

    events = signals_local.get(int(t_int))
    if events:
        from operator import itemgetter
        events = sorted(events, key=itemgetter(0))

        for sym, buy_idx in events:
            total_days = len(close_arrays[sym])

            # Chequeo: no abrir trade si no hay suficientes velas restantes para SELL_AFTER
            if buy_idx + sell_after > total_days:
                continue  # saltar este trade completamente

            if cash < order_amount:
                break

            cash, counter, num_signals_executed, open_heap = execute_signal(
                sym, buy_idx, cash, comi_factor, order_amount, sell_after,
                sym_data_local, counter, open_heap, num_signals_executed,
                tp_pct, sl_pct
            )

    sim_balance_cols = update_sim_balance(
        t_int, open_heap, cash, sym_data_local, ts_int_arrays, close_arrays, sim_balance_cols
    )
    return cash, counter, num_signals_executed, sim_balance_cols, open_heap


def run_backtest_loop(
    all_timestamps_int, sym_data_local, ts_int_arrays, close_arrays, signals_local,
    symbol_order_local, cash, order_amount, comi_factor, sell_after, counter,
    tp_pct, sl_pct, trades, trade_times, trade_log_cols, sim_balance_cols,
    close_expired_positions, update_sim_balance, execute_signal
):
    num_signals_executed = 0
    open_heap = []

    # Binds locales para evitar resolución de nombres costosa
    process_fn = process_signals_for_timestamp
    sdl = sym_data_local
    tsia = ts_int_arrays
    ca = close_arrays
    sl = signals_local
    so = symbol_order_local
    oa = order_amount
    cf = comi_factor
    sa = sell_after
    tp = tp_pct
    sp = sl_pct
    tr = trades
    tt = trade_times
    tlc = trade_log_cols
    cs = close_expired_positions
    usb = update_sim_balance
    es = execute_signal

    for t_int in all_timestamps_int:
        cash, counter, num_signals_executed, sim_balance_cols, open_heap = process_fn(
            t_int, open_heap, sdl, tsia, ca, cash, sim_balance_cols, sl, so,
            oa, cf, sa, counter, num_signals_executed,
            tp, sp, tr, tt, tlc, cs, usb, es
        )

    return cash, counter, num_signals_executed, sim_balance_cols, open_heap


# ============================
# Función principal: run_grid_backtest - MODIFICADO
# ============================

def run_grid_backtest(
    ohlcv_arrays,
    sell_after,
    tp_pct=0.0,
    sl_pct=0.0,
    initial_balance=10000,
    order_amount=100,
    comi_pct=0.05
):
    comi_factor = float(comi_pct) / 100.0
    cash = float(initial_balance)

    # Preparar datos
    (
        sym_data,
        signals_by_time,
        all_timestamps_int,
        all_timestamps_dt,
        symbol_order,
        ts_int_arrays,
        close_arrays
    ) = prepare_data(ohlcv_arrays)

    symbols = list(ohlcv_arrays.keys())

    # Inicializar estructuras de backtest
    trades, trade_times, trade_log_cols, sim_balance_cols, open_heap = \
        initialize_backtest_structures(symbols)
    counter = 0

    # ============================
    # Ejecutar bucle principal
    # ============================
    cash, counter, num_signals_executed, sim_balance_cols, open_heap = run_backtest_loop(
        all_timestamps_int, sym_data, ts_int_arrays, close_arrays, signals_by_time,
        symbol_order, cash, order_amount, comi_factor, sell_after, counter,
        tp_pct, sl_pct, trades, trade_times, trade_log_cols, sim_balance_cols,
        close_expired_positions, update_sim_balance, execute_signal
    )

    # ============================
    # Calcular métricas finales
    # ============================
    metrics = compute_post_backtest_metrics(symbols, trades, trade_times, all_timestamps_dt, initial_balance, sim_balance_cols)

    # ============================
    # Construir diccionario de resultados
    # ============================
    results = build_results_dict(
        symbols, trades, trade_times,
        metrics['final_balance_by_symbol'],
        metrics['max_dd_by_symbol'],
        metrics['sharpe_by_symbol'],
        metrics['final_balance'],
        num_signals_executed,
        metrics['proportion_winners'],
        metrics['max_dd_portfolio'],
        sim_balance_cols,
        trade_log_cols,
        metrics['sharpe_portfolio']
    )

    return results
