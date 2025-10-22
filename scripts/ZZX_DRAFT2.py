import heapq
import logging
import warnings
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

MIN_PRICE       = 0.0001
INITIAL_BALANCE = 10_000
COMISION        = 0.06

# ============================
# prepare_data - 
# ============================
def prepare_data(ohlcv_arrays):

    if not ohlcv_arrays:
        return ({}, {}, np.array([], dtype=np.int64), 
                np.array([], dtype='datetime64[ns]'), {}, {}, {})
    
    symbols = list(ohlcv_arrays.keys())
    
    # Pre-alocar estructuras
    sym_data = {}
    ts_int_arrays = {}
    close_arrays = {}
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
        
        # Referencias directas (evita lookups posteriores)
        ts_int_arrays[sym] = ts_int
        close_arrays[sym] = close_view
        all_ts_int_lists.append(ts_int)
    
    
    signals_by_time = {}
    
    for sym in symbols:
        signal_arr = sym_data[sym]['signal']
        
       
        sig_idxs = np.nonzero(signal_arr)[0]
        
        if sig_idxs.size > 0:
            ts_int_view = sym_data[sym]['ts_int']
            t_ints = ts_int_view[sig_idxs]
            
            
            for t_int, idx in zip(t_ints, sig_idxs):
                t_int_key = int(t_int)
                if t_int_key not in signals_by_time:
                    signals_by_time[t_int_key] = []
                signals_by_time[t_int_key].append((sym, int(idx)))
    
    
    if all_ts_int_lists:
        all_timestamps_int = np.unique(np.concatenate(all_ts_int_lists))
    else:
        all_timestamps_int = np.array([], dtype=np.int64)
       
    all_timestamps_dt = all_timestamps_int.view('datetime64[ns]')
    
    symbol_order = {s: i for i, s in enumerate(symbols)}
    
    return sym_data, signals_by_time, all_timestamps_int, all_timestamps_dt, symbol_order, ts_int_arrays, close_arrays

# ============================
# Helper: detect_intrabar_exit
# ============================
def detect_intrabar_exit(d, buy_idx, sell_idx, tp_price, sl_price):
    intravela_detected = False
    chosen_idx = None
    exit_reason = None
    exec_price = None

    if tp_price is None and sl_price is None:
        return intravela_detected, chosen_idx, exit_reason, exec_price

    start = buy_idx + 1
    end = sell_idx
    if end < start:
        return intravela_detected, chosen_idx, exit_reason, exec_price

    high_slice = d['high'][start:end+1]
    low_slice = d['low'][start:end+1]

    tp_hits = np.where(high_slice >= tp_price)[0] if tp_price is not None else np.array([], dtype=int)
    sl_hits = np.where(low_slice <= sl_price)[0] if sl_price is not None else np.array([], dtype=int)

    tp_first = tp_hits[0] + start if tp_hits.size > 0 else None
    sl_first = sl_hits[0] + start if sl_hits.size > 0 else None

    if tp_first is not None and sl_first is not None:
        tp_time_val = d['high_time'][tp_first]
        sl_time_val = d['low_time'][sl_first]

        if tp_first == sl_first:
            if tp_time_val <= sl_time_val:
                chosen_idx = tp_first
                exit_reason = 'TP'
                exec_price = tp_price
            else:
                chosen_idx = sl_first
                exit_reason = 'SL'
                exec_price = sl_price
        else:
            if sl_first < tp_first:
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

    return intravela_detected, chosen_idx, exit_reason, exec_price


# ============================
# Helper: close_position (sin cambios)
# ============================
def close_position(pos, exec_time, exec_price, exit_reason, comi_factor, 
                   trades, trade_times, trade_log_cols, cash):
    qty = pos['qty']
    buy_price = pos['buy_price']

    commission_buy = pos.get('commission_buy')
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


# ============================
# close_expired_positions - inline y sin lookups
# ============================
def close_expired_positions(t_int, open_heap, sym_data, ts_int_arrays, close_arrays,
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
            sell_ts_int = pos.get('sell_time_int', int(sym_data[sym]['ts_int'][-1]))
            
            # Inline price lookup
            ts_arr = ts_int_arrays[sym]
            close_arr = close_arrays[sym]
            idx = np.searchsorted(ts_arr, sell_ts_int, side='right') - 1
            exec_price = float(close_arr[idx] if idx >= 0 else close_arr[0])
            
            exec_time_dt = np.datetime64(int(sell_ts_int), 'ns')
            
            cash = close_position(pos, exec_time_dt, exec_price, 'SELL_AFTER',
                                  comi_factor, trades, trade_times, trade_log_cols, cash)
            pos['closed'] = True
            
    return cash


# ============================
# update_sim_balance - vectorizado
# ============================
def update_sim_balance(t_int, open_heap, cash, ts_int_arrays, close_arrays, sim_balance_cols):

    if not open_heap:
        sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
        sim_balance_cols['balance'].append(cash)
        return sim_balance_cols
    
    
    symbol_qty = {}
    for _, _, pos in open_heap:
        if pos.get('closed', False):
            continue
        sym = pos['symbol']
        symbol_qty[sym] = symbol_qty.get(sym, 0.0) + pos['qty']
    
   
    total_value = 0.0
    for sym, qty_sum in symbol_qty.items():
        ts_arr = ts_int_arrays[sym]
        close_arr = close_arrays[sym]
        
        
        idx = np.searchsorted(ts_arr, t_int, side='right') - 1
        price = float(close_arr[idx] if idx >= 0 else close_arr[0])
        total_value += qty_sum * price
    
    sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
    sim_balance_cols['balance'].append(cash + total_value)
    return sim_balance_cols


# ============================
# execute_signal - inline y fast
# ============================
def execute_signal(sym, buy_idx, cash, comi_factor, order_amount, sell_after,
                        sym_data, counter, open_heap, tp_pct, sl_pct):

    d = sym_data[sym]
    price_t = float(d['close'][buy_idx])
    qty = order_amount / price_t

    commission_buy = float(order_amount * comi_factor)
    cash -= (order_amount + commission_buy)

    if sell_after == 0:
        sell_idx = d['len'] - 1  # considerar toda la serie de precios hasta el final
    else:
        sell_idx = min(buy_idx + sell_after, d['len'] - 1)
    sell_time_dt = d['ts'][sell_idx]
    sell_time_int = int(d['ts_int'][sell_idx])

    tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else np.inf
    sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -np.inf

    position = {
        'symbol': sym,
        'qty': qty,
        'buy_price': price_t,
        'buy_time': np.datetime64(int(d['ts_int'][buy_idx]), 'ns'),
        'sell_time': sell_time_dt,
        'sell_time_int': sell_time_int,
        'commission_buy': commission_buy
    }

    intravela_detected, chosen_idx, exit_reason, exec_price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price
    )

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
    return cash, counter


# ============================
# Bucle principal 
# ============================
def run_backtest_loop(
    all_timestamps_int, sym_data, ts_int_arrays, close_arrays, signals_by_time,
    cash, order_amount, comi_factor, sell_after, tp_pct, sl_pct,
    trades, trade_times, trade_log_cols, sim_balance_cols
):

    num_signals_executed = 0
    open_heap = []
    counter = 0
    
    
    sd = sym_data
    tia = ts_int_arrays
    ca = close_arrays
    sbt = signals_by_time
    oa = order_amount
    cf = comi_factor
    sa = sell_after
    tp = tp_pct
    sp = sl_pct
    
    for t_int in all_timestamps_int:
        
        cash = close_expired_positions(
            t_int, open_heap, sd, tia, ca, cf, trades, trade_times, trade_log_cols, cash
        )
        
       
        if not open_heap:
            events = sbt.get(int(t_int))
            if events:
               
                events_sorted = sorted(events, key=lambda x: x[0])
                
                for sym, buy_idx in events_sorted:
                    # Validación inline
                    if sa > 0 and buy_idx + sa > len(ca[sym]):
                        continue
                    
                    if cash < oa:
                        break
                    
                    cash, counter = execute_signal(
                        sym, buy_idx, cash, cf, oa, sa, sd, counter, open_heap, tp, sp
                    )
                    num_signals_executed += 1
        
        
        sim_balance_cols = update_sim_balance(
            t_int, open_heap, cash, tia, ca, sim_balance_cols
        )
    
    return cash, num_signals_executed


# ============================
# Métricas (sin cambios
# ============================
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


def compute_post_backtest_metrics(symbols, trades, trade_times, all_timestamps_dt, initial_balance, sim_balance_cols):
    sim_values = np.array(sim_balance_cols['balance'], dtype=np.float64)
    sim_ts_arr = np.array(sim_balance_cols['timestamp'], dtype='datetime64[ns]') if len(sim_balance_cols['timestamp']) > 0 else np.array([], dtype='datetime64[ns]')
    sim_ts_int = sim_ts_arr.astype('int64') if sim_ts_arr.size > 0 else np.array([], dtype=np.int64)

    final_balance = float(sim_values[-1]) if sim_values.size > 0 else float(initial_balance)

    cummax_portfolio = np.maximum.accumulate(sim_values) if sim_values.size > 0 else np.array([initial_balance])
    drawdowns_portfolio = (cummax_portfolio - sim_values) / np.where(cummax_portfolio == 0, 1, cummax_portfolio)
    max_dd_portfolio = float(np.max(drawdowns_portfolio)) if drawdowns_portfolio.size > 0 else 0.0

    sharpe_portfolio = compute_annualized_sharpe(sim_values, sim_ts_int)

    all_trades = [p for lst in trades.values() for p in lst]
    num_trades = len(all_trades)
    proportion_winners = np.sum(np.array(all_trades) > 0.0) / num_trades if num_trades > 0 else np.nan

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
# FUNCIÓN PRINCIPAL 
# ============================
def run_grid_backtest(
    ohlcv_arrays,
    sell_after,
    tp_pct=0.0,
    sl_pct=0.0,
    order_amount=100 
):

    # Constantes
    comi_factor = float(COMISION) / 100.0
    cash = float(INITIAL_BALANCE)
    initial_balance = INITIAL_BALANCE
    

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
    
    # Inicializar estructuras
    trades = {sym: [] for sym in symbols}
    trade_times = {sym: [] for sym in symbols}
    trade_log_cols = {k: [] for k in [
        'symbol','buy_time','buy_price','sell_time','sell_price',
        'qty','profit','exit_reason','commission_buy','commission_sell']}
    sim_balance_cols = {'timestamp': [], 'balance': []}
    
    # Ejecutar backtest ultra-optimizado
    cash, num_signals_executed = run_backtest_loop(
        all_timestamps_int, sym_data, ts_int_arrays, close_arrays, signals_by_time,
        cash, order_amount, comi_factor, sell_after, tp_pct, sl_pct,
        trades, trade_times, trade_log_cols, sim_balance_cols
    )
    
    # Calcular métricas
    metrics = compute_post_backtest_metrics(
        symbols, trades, trade_times, all_timestamps_dt, initial_balance, sim_balance_cols
    )
    
    # Construir resultados
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