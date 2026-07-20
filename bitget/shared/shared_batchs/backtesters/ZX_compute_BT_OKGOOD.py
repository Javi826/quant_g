#shared/shared_batchs/backtesters/ZX_compute_BT.py
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
DEFAULT_CANDLES = 50

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
          
        ts_int     = ts.view('int64')
        close_view = data['close']  
        
        
        sym_data[sym] = {
            'ts': ts,
            'ts_int': ts_int,
            'open': data['open'],
            'close': close_view,
            'high': data['high'],
            'low': data['low'],
            'signal': data['signal'],
            'len': len(ts),
            'high_time': data['high_time'],
            'low_time': data['low_time']
        }
        
        # Referencias directas (evita lookups posteriores)
        ts_int_arrays[sym] = ts_int
        close_arrays[sym]  = close_view
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
# Helper: detect_intrabar_exit - MODIFICADO PARA SHORT
# ============================
def detect_intrabar_exit(d, buy_idx, sell_idx, tp_price, sl_price, is_short=False):
    intravela_detected = False
    chosen_idx  = None
    exit_reason = None
    exec_price  = None

    if tp_price is None and sl_price is None:
        return intravela_detected, chosen_idx, exit_reason, exec_price

    start = buy_idx #VELA
    end = sell_idx
    if end < start:
        return intravela_detected, chosen_idx, exit_reason, exec_price

    high_slice = d['high'][start:end+1]
    low_slice = d['low'][start:end+1]

    if is_short:
        # Para SHORT: TP se alcanza cuando el precio BAJA (low <= tp_price)
        # SL se alcanza cuando el precio SUBE (high >= sl_price)
        tp_hits = np.where(low_slice <= tp_price)[0] if tp_price is not None else np.array([], dtype=int)
        sl_hits = np.where(high_slice >= sl_price)[0] if sl_price is not None else np.array([], dtype=int)
        
        tp_first = tp_hits[0] + start if tp_hits.size > 0 else None
        sl_first = sl_hits[0] + start if sl_hits.size > 0 else None

        if tp_first is not None and sl_first is not None:
            tp_time_val = d['low_time'][tp_first]  # Para SHORT usamos low_time para TP
            sl_time_val = d['high_time'][sl_first]  # Para SHORT usamos high_time para SL

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
    else:
        # LONG (lógica original)
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
# Helper: close_position - MODIFICADO PARA SHORT
# Ahora actualiza BOTH: cash_bank (efectivo total) y blocked_cash (efectivo bloqueado por shorts)
# ============================
def close_position(pos, exec_time, exec_price, exit_reason, comi_factor, 
                   trades, trade_times, trade_log_cols, cash_bank, blocked_cash):
    qty       = pos['qty']
    buy_price = pos['buy_price']
    is_short  = pos.get('is_short', False)

    commission_buy = pos.get('commission_buy')
    commission_sell = qty * exec_price * comi_factor

    if is_short:
        # SHORT: vendimos al inicio (recibimos cash -> fue bloqueado), compramos al final (pagamos)
        # cash_bank se reduce por el coste de recompra + comisión de salida
        cash_bank -= qty * exec_price + commission_sell
        # liberar el efectivo bloqueado asociado a esta posición
        blocked_amount = pos.get('blocked_amount', 0.0)
        blocked_cash -= blocked_amount
        # beneficio/pérdida
        profit = (buy_price - exec_price) * qty - commission_buy - commission_sell
    else:
        # LONG: compramos al inicio, vendemos al final
        cash_bank += qty * exec_price - commission_sell
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
    trade_log_cols['position_type'].append('SHORT' if is_short else 'LONG')

    # evitar pequeños negativos por floating point
    if blocked_cash < 0 and abs(blocked_cash) < 1e-12:
        blocked_cash = 0.0

    return cash_bank, blocked_cash

# ============================
# close_expired_positions - MODIFICADO PARA SHORT
# ============================
def close_expired_positions(t_int, open_heap, sym_data, ts_int_arrays, close_arrays,
                                  comi_factor, trades, trade_times, trade_log_cols, cash_bank, blocked_cash):

    closed_reasons = []

    while open_heap and open_heap[0][0] <= t_int:
        _, _, pos = heapq.heappop(open_heap)
        if pos.get('closed', False):
            continue
            
        if 'exec_price' in pos and ('exec_time_int' in pos) and pos['exec_time_int'] <= t_int:
            cash_bank, blocked_cash = close_position(pos, pos['exec_time'], pos['exec_price'], pos['exit_reason'],
                                                     comi_factor, trades, trade_times, trade_log_cols,
                                                     cash_bank, blocked_cash)
            pos['closed'] = True
            closed_reasons.append(pos['exit_reason'])
        else:
            sym = pos['symbol']
            sell_ts_int = pos.get('sell_time_int', int(sym_data[sym]['ts_int'][-1]))
            
            # Inline price lookup
            ts_arr = ts_int_arrays[sym]
            close_arr = close_arrays[sym]
            idx = np.searchsorted(ts_arr, sell_ts_int, side='right') - 1
            exec_price = float(close_arr[idx] if idx >= 0 else close_arr[0])
            
            exec_time_dt = np.datetime64(int(sell_ts_int), 'ns')
            
            cash_bank, blocked_cash = close_position(pos, exec_time_dt, exec_price, 'SELL_AFTER',
                                                     comi_factor, trades, trade_times, trade_log_cols,
                                                     cash_bank, blocked_cash)
            pos['closed'] = True
            closed_reasons.append('SELL_AFTER')
            
    return cash_bank, blocked_cash, closed_reasons


# ============================
# MODIFICADO PARA SHORT
# Usa cash_bank (efectivo total) para calcular equity = cash_bank + valor_long - valor_short
# ============================



# ============================
# execute_signal - MODIFICADO PARA SHORT
# Ahora actualiza cash_bank y blocked_cash. La lógica de LONGS queda intacta.
# Cambiado: blocked_amount = proceeds + margin_required (antes solo proceeds)
# ============================
def execute_signal(sym, buy_idx, cash_bank, blocked_cash, comi_factor, order_amount, sell_after,
                        sym_data, counter, open_heap, tp_pct, sl_pct, is_short=False):

    d = sym_data[sym]
    price_t = float(d['open'][buy_idx]) #OPEN
    qty = order_amount / price_t

   # print(f"[DEBUG] sym={sym} | signal_candle_ts={d['ts'][buy_idx-1]} signal_candle_close={d['close'][buy_idx-1]} "
      #    f"| exec_candle_ts={d['ts'][buy_idx]} exec_open={price_t}")

    commission_buy = float(order_amount * comi_factor)
    
    if is_short:
        # SHORT: vendemos al inicio -> recibimos efectivo (proceeds), pero bloqueamos proceeds + margen requerido
        proceeds = order_amount - commission_buy

        # margen requerido = pérdida máxima esperada hasta SL
        margin_required = order_amount * (sl_pct / 100.0) if sl_pct != 0.0 else np.inf

        # bloqueamos proceeds + margin_required para que free_cash disminuya al abrir el short
        blocked_amount = proceeds + margin_required

        # añadir proceeds al cash_bank (efectivo total) y bloquear la cantidad
        cash_bank += proceeds
    else:
        # LONG: compramos al inicio, gastamos cash (misma lógica previa)
        blocked_amount = 0.0
        cash_bank -= (order_amount + commission_buy)

    if sell_after == 0:
        n_velas = DEFAULT_CANDLES  
        sell_idx = min(buy_idx + n_velas, d['len'] - 1)
    else:
        sell_idx = min(buy_idx + sell_after, d['len'] - 1)

    sell_time_dt = d['ts'][sell_idx]
    sell_time_int = int(d['ts_int'][sell_idx])

    if is_short:
        # Para SHORT: TP está ABAJO (precio baja), SL está ARRIBA (precio sube)
        tp_price = price_t * (1.0 - tp_pct / 100.0) if tp_pct != 0.0 else -np.inf
        sl_price = price_t * (1.0 + sl_pct / 100.0) if sl_pct != 0.0 else np.inf
    else:
        # Para LONG: TP está ARRIBA, SL está ABAJO
        tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else np.inf
        sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -np.inf

    position = {
        'symbol': sym,
        'qty': qty,
        'buy_price': price_t,
        'buy_time': np.datetime64(int(d['ts_int'][buy_idx]), 'ns'),
        'sell_time': sell_time_dt,
        'sell_time_int': sell_time_int,
        'commission_buy': commission_buy,
        'is_short': is_short,
        'blocked_amount': blocked_amount
    }

    # Si es short y hemos bloqueado algo, añadirlo al total bloqueado
    if is_short and blocked_amount > 0:
        blocked_cash += blocked_amount

    intravela_detected, chosen_idx, exit_reason, exec_price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short
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
    return cash_bank, blocked_cash, counter


# ============================
# Bucle principal - MODIFICADO PARA SHORT
# cash_bank = efectivo total de la cuenta; blocked_cash = suma de los ingresos de shorts reservados
# free_cash = cash_bank - blocked_cash (disponible para abrir nuevas posiciones)
def run_backtest_loop(
    all_timestamps_int, sym_data, ts_int_arrays, close_arrays, signals_by_time,
    cash_bank, blocked_cash, order_amount, comi_factor, sell_after, tp_pct, sl_pct,
    trades, trade_times, trade_log_cols
):

    open_heap = []
    counter = 0

    # Alias locales para velocidad
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

        # Guardar si el heap ya estaba vacío ANTES de cerrar posiciones en esta vela
        was_empty_before = not open_heap

        # Cerrar posiciones expiradas -> devuelve cash_bank, blocked_cash y los motivos de cierre
        cash_bank, blocked_cash, closed_reasons = close_expired_positions(
            t_int, open_heap, sd, tia, ca, cf, trades, trade_times, trade_log_cols, cash_bank, blocked_cash
        )

        if not open_heap:
            if was_empty_before:
                # El heap ya estaba vacío antes de esta vela -> buscar señales normalmente
                search_signals = True
            else:
                # El heap se vació justo en esta vela. Si hubo algún cierre por TP/SL
                # intravela, se retrasa la búsqueda a la vela siguiente (n+1). Si todos los
                # cierres fueron por timeout, se busca en la misma vela (igual que producción).
                had_intrabar_exit = any(r in ('TP', 'SL') for r in closed_reasons)
                search_signals = not had_intrabar_exit

            if search_signals:
                events = sbt.get(int(t_int))
                if events:
                    events_sorted = sorted(events, key=lambda x: x[0])

                    for sym, buy_idx in events_sorted:
                        # Calcular cash libre (no incluir ingresos bloqueados por shorts)
                        free_cash = cash_bank - blocked_cash
                        # Verificar saldo suficiente usando free_cash
                        if free_cash < oa:
                            break

                        # Determinar dirección según la señal: 1 -> long, -1 -> short
                        signal_value = sd[sym]['signal'][buy_idx]
                        if signal_value == 0:
                            continue  # ignorar si no hay señal
                        is_short = signal_value < 0

                        # VALIDACIÓN ADICIONAL PARA SHORTS: si no hay SL (sp == 0) rechazamos porque riesgo ilimitado
                        if is_short:
                            if sp == 0.0:
                                continue

                            # pérdida máxima esperada = order_amount * (sl_pct/100)
                            potential_max_loss = oa * (sp / 100.0)
                            commission_buy = oa * cf
                            # requerimos que free_cash cubra la pérdida máxima + comision de entrada
                            if free_cash < (potential_max_loss + commission_buy):
                                # no hay suficiente capital libre para cubrir la pérdida máxima estimada
                                continue

                        # Ejecutar señal (ahora devuelve cash_bank y blocked_cash)
                        cash_bank, blocked_cash, counter = execute_signal(
                            sym, buy_idx, cash_bank, blocked_cash, cf, oa, sa, sd, counter, open_heap, tp, sp, is_short
                        )

    return cash_bank, blocked_cash

# ============================
# Métricas (sin cambios)
# ============================
def build_results_dict(trades, trade_log_cols):
    """The backtester returns only raw simulation output — trades and the trade
    log. All evaluation metrics (Sharpe, drawdown, win rate, Calmar...) are
    computed exclusively by batch_metrics.compute_metrics, the single source
    of truth for metrics in the pipeline."""
    results = {
        "__PORTFOLIO__": {
            'trades':    [p for lst in trades.values() for p in lst],
            'trade_log': pd.DataFrame(trade_log_cols),
        }
    }
    return results


# ============================
# FUNCIÓN PRINCIPAL - MODIFICADO PARA SOPORTAR LONG/SHORT
# Ahora inicializa cash_bank y blocked_cash y los pasa al loop
# ============================
def run_grid_backtest(
    ohlcv_arrays,
    sell_after,
    tp_pct,
    sl_pct,
    order_amount  # NUEVO PARÁMETRO: monto por orden
):

    # Constantes
    comi_factor     = float(COMISION) / 100.0
    cash_bank       = float(INITIAL_BALANCE)   # efectivo total de la cuenta
    blocked_cash    = 0.0                      # efectivo bloqueado (ingresos de shorts)
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
        'qty','profit','exit_reason','commission_buy','commission_sell','position_type']}
    
    # Ejecutar backtest con el tipo de posición especificado
    cash_bank, blocked_cash = run_backtest_loop(
        all_timestamps_int, sym_data, ts_int_arrays, close_arrays, signals_by_time,
        cash_bank, blocked_cash, order_amount, comi_factor, sell_after, tp_pct, sl_pct,
        trades, trade_times, trade_log_cols
    )
    
    # Construir resultados — solo datos crudos; las métricas se calculan en batch_metrics
    results = build_results_dict(trades, trade_log_cols)
    
    return results