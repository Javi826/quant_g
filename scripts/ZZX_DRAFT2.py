# Z_compute_03.py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from numba import njit
import logging
logging.basicConfig(level=logging.INFO)

# -----------------------------
# CÁLCULO DE GANANCIAS NETAS
# -----------------------------
def calculate_data(results, initial_balance=10000):
    portfolio = results.get("__PORTFOLIO__", None)
    if portfolio is None:
        return 0.0, 0.0
    final_balance  = portfolio['final_balance']
    net_gain_total = final_balance - initial_balance
    net_gain_pct   = (net_gain_total / initial_balance) * 100
    return net_gain_total, net_gain_pct

# -----------------------------
# INDICADORES
# -----------------------------
@njit
def delta_numba(close):
    n = len(close)
    delta = np.empty(n)
    delta[0] = 0.0
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    return delta

@njit
def rolling_entropy_numba(delta, window, bins):
    n = len(delta)
    entropia = np.zeros(n)
    delta_min = delta.min()
    delta_max = delta.max()
    
    for i in range(n):
        start = max(0, i - window + 1)
        hist = np.zeros(bins)
        for j in range(start, i + 1):
            bin_idx = int((delta[j] - delta_min) / (delta_max - delta_min + 1e-9) * bins)
            if bin_idx == bins:
                bin_idx = bins - 1
            hist[bin_idx] += 1
        prob = hist / hist.sum()
        e = 0.0
        for p in prob:
            if p > 0:
                e -= p * np.log2(p)
        entropia[i] = e
        
    return entropia

def add_indicators(df, m_accel=5):
    close = df['close'].values
    delta = delta_numba(close)
    entropia = rolling_entropy_numba(delta, 5, 10)
    df['entropia'] = entropia
    df['accel_raw'] = df['close'].diff().diff().fillna(0)
    df['accel'] = df['accel_raw'].ewm(span=m_accel, adjust=False).mean()
    return df

# -----------------------------
# EJEMPLO DE FUNCIÓN DE SEÑAL (estrategia)
# -----------------------------
def explosive_signal(df, entropia_max=2.0, live=False):

    signal = (df['entropia'] < entropia_max) & (df['accel'] > 0)
    if not live:
        signal = signal.shift(1) 
    df['signal'] = signal.fillna(False)
    return df

# -----------------------------
# FUNCIÓN PRINCIPAL DE BACKTEST GRID (CON TP/SL INTRAVELA)
# -----------------------------
def run_grid_backtest(
    ohlcv_data,
    sell_after,
    n_jobs=1,
    initial_balance=10000,
    order_amount=100,
    keep_df=True,
    tp_pct=0.0,
    sl_pct=0.0,
    comi_pct=0.05,   # <-- nuevo parámetro: comisión en % sobre monto USDT (ej: 0.1 => 0.1%)
):

    symbols = list(ohlcv_data.keys())
    symbol_data = {}
    for sym in symbols:
        df = ohlcv_data[sym].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Comprobación columnaria mínima
        if "signal" not in df.columns:
            raise ValueError(f"El DataFrame de {sym} no contiene columna 'signal'. Debes calcular las señales antes de llamar al backtest.")
        if tp_pct != 0.0 or sl_pct != 0.0:
            # si quieres alta garantía, comprobamos que exista high/low en el df (si no existen, fallback a close)
            if ('high' not in df.columns) or ('low' not in df.columns):
                logging.warning(f"El DataFrame de {sym} no contiene 'high'/'low'. Se hará fallback a 'close' para detectar TP/SL.")
        symbol_data[sym] = df

    # Map timestamp -> index por símbolo
    idx_maps = {sym: {ts: i for i, ts in enumerate(df.index)} for sym, df in symbol_data.items()}
    last_index_per_symbol = {sym: df.index[-1] for sym, df in symbol_data.items()}

    # Timeline global
    all_timestamps = sorted({ts for df in symbol_data.values() for ts in df.index})

    # Posiciones y trades
    positions   = {sym: [] for sym in symbols}
    trades      = {sym: [] for sym in symbols}           
    trade_times = {sym: [] for sym in symbols}       

    # trade_log para auditoría
    trade_log_rows = []

    cash = float(initial_balance)
    sim_balance_history = []
    num_signals_executed = 0

    use_tp_sl = (tp_pct != 0.0) or (sl_pct != 0.0)
    comi_factor = float(comi_pct) / 100.0

    # --- Loop temporal global
    for t in all_timestamps:
        # 0) Precio actual por símbolo (si existe). También obtenemos high/low si la vela existe.
        price_at_t = {}
        high_at_t = {}
        low_at_t = {}
        has_veta_at_t = {}
        for sym, df in symbol_data.items():
            if t in idx_maps[sym]:
                price_at_t[sym] = float(df.loc[t, 'close'])
                # si existen high/low, extraerlos
                if 'high' in df.columns and 'low' in df.columns:
                    high_at_t[sym] = float(df.loc[t, 'high'])
                    low_at_t[sym]  = float(df.loc[t, 'low'])
                    has_veta_at_t[sym] = True
                else:
                    high_at_t[sym] = None
                    low_at_t[sym] = None
                    has_veta_at_t[sym] = False
            else:
                # si el timestamp no existe para ese símbolo, intentar usar el precio anterior más cercano (no hay high/low intravela)
                earlier_idx = df.index.searchsorted(t) - 1
                if earlier_idx >= 0:
                    price_at_t[sym] = float(df.iloc[earlier_idx]['close'])
                else:
                    price_at_t[sym] = None
                high_at_t[sym] = None
                low_at_t[sym] = None
                has_veta_at_t[sym] = False

        # 1) Ejecutar cierres: por TP/SL intravela o por sell_time
        for sym, df in symbol_data.items():
            remaining = []
            price_t = price_at_t.get(sym, None)
            for pos in positions[sym]:
                closed = False
                exec_price = None
                exit_reason = None

                tp_price = pos.get('tp_price', None)
                sl_price = pos.get('sl_price', None)

                # 1.a) Si disponemos de high/low para esta vela -> comprobación intravela
                if use_tp_sl and has_veta_at_t.get(sym, False):
                    high_t = high_at_t.get(sym)
                    low_t  = low_at_t.get(sym)

                    # ambos en la misma vela: priorizamos SL (tal y como pediste)
                    if (tp_price is not None) and (sl_price is not None) and (high_t >= tp_price) and (low_t <= sl_price):
                        exec_price = sl_price
                        exit_reason = 'SL'
                        closed = True
                    else:
                        # SL solo
                        if (sl_price is not None) and (low_t <= sl_price):
                            exec_price = sl_price
                            exit_reason = 'SL'
                            closed = True
                        # TP solo
                        elif (tp_price is not None) and (high_t >= tp_price):
                            exec_price = tp_price
                            exit_reason = 'TP'
                            closed = True

                    # Si cerramos intravela, registramos
                    if closed and exec_price is not None:
                        qty = pos['qty']
                        buy_price = pos['buy_price']
                        # Comisión en la venta (sobre monto recibido)
                        commission_sell = (qty * exec_price) * comi_factor if comi_factor != 0.0 else 0.0
                        cash += qty * exec_price - commission_sell
                        # profit neto (restando comisión de compra almacenada y comisión de venta)
                        commission_buy = pos.get('commission_buy', 0.0)
                        profit = (exec_price - buy_price) * qty - commission_buy - commission_sell
                        trades[sym].append(profit)
                        trade_times[sym].append(t)
                        # log
                        trade_log_rows.append({
                            'symbol': sym,
                            'buy_time': pos['buy_time'],
                            'buy_price': buy_price,
                            'sell_time': t,
                            'sell_price': exec_price,
                            'qty': qty,
                            'profit': profit,
                            'exit_reason': exit_reason,
                            'commission_buy': commission_buy,
                            'commission_sell': commission_sell
                        })
                # 1.b) Si no intravela o no hay high/low, fallback a comparación por close/price_t
                if (not closed) and use_tp_sl and (price_t is not None) and not has_veta_at_t.get(sym, False):
                    # comprobación por price_t (último close conocido)
                    if (tp_price is not None) and (price_t >= tp_price):
                        exec_price = price_t
                        exit_reason = 'TP'
                        closed = True
                    elif (sl_price is not None) and (price_t <= sl_price):
                        exec_price = price_t
                        exit_reason = 'SL'
                        closed = True

                    if closed and exec_price is not None:
                        qty = pos['qty']
                        buy_price = pos['buy_price']
                        commission_sell = (qty * exec_price) * comi_factor if comi_factor != 0.0 else 0.0
                        cash += qty * exec_price - commission_sell
                        commission_buy = pos.get('commission_buy', 0.0)
                        profit = (exec_price - buy_price) * qty - commission_buy - commission_sell
                        trades[sym].append(profit)
                        trade_times[sym].append(t)
                        trade_log_rows.append({
                            'symbol': sym,
                            'buy_time': pos['buy_time'],
                            'buy_price': buy_price,
                            'sell_time': t,
                            'sell_price': exec_price,
                            'qty': qty,
                            'profit': profit,
                            'exit_reason': exit_reason,
                            'commission_buy': commission_buy,
                            'commission_sell': commission_sell
                        })

                # 1.c) Venta programada por sell_time si no se ha cerrado ya
                if (not closed) and (pos['sell_time'] is not None) and (pos['sell_time'] == t):
                    # usar close de la vela si existe, si no usar price_t o último precio del df
                    if (t in idx_maps[sym]) and ('close' in df.columns):
                        price_exec = float(df.loc[t, 'close'])
                    else:
                        price_exec = price_t if price_t is not None else float(df['close'].iloc[-1])
                    qty = pos['qty']
                    buy_price = pos['buy_price']
                    commission_sell = (qty * price_exec) * comi_factor if comi_factor != 0.0 else 0.0
                    cash += qty * price_exec - commission_sell
                    commission_buy = pos.get('commission_buy', 0.0)
                    profit = (price_exec - buy_price) * qty - commission_buy - commission_sell
                    trades[sym].append(profit)
                    trade_times[sym].append(t)
                    trade_log_rows.append({
                        'symbol': sym,
                        'buy_time': pos['buy_time'],
                        'buy_price': buy_price,
                        'sell_time': t,
                        'sell_price': price_exec,
                        'qty': qty,
                        'profit': profit,
                        'exit_reason': 'SELL_AFTER',
                        'commission_buy': commission_buy,
                        'commission_sell': commission_sell
                    })
                    closed = True

                if not closed:
                    remaining.append(pos)
            positions[sym] = remaining

        # 2) Si hay posiciones abiertas -> bloqueo global (no se compra nada)
        total_open_positions = sum(len(v) for v in positions.values())
        if total_open_positions > 0:
            positions_value = 0.0
            for sym, pos_list in positions.items():
                if len(pos_list) == 0:
                    continue
                price_map = price_at_t.get(sym, None)
                if price_map is None:
                    # fallback a último precio conocido
                    df = symbol_data[sym]
                    earlier_idx = df.index.searchsorted(t) - 1
                    if earlier_idx >= 0:
                        price_map = float(df.iloc[earlier_idx]['close'])
                if price_map is None:
                    continue
                for pos in pos_list:
                    positions_value += pos['qty'] * price_map
            sim_balance = cash + positions_value
            sim_balance_history.append((t, sim_balance))
            continue

        # 3) Compras según señales
        for sym, df in symbol_data.items():
            if t not in idx_maps[sym]:
                continue
            if not bool(df.loc[t, 'signal']):
                continue
            if cash < order_amount:
                continue
            price_t = float(df.loc[t, 'close'])
            if price_t <= 0:
                continue

            qty = order_amount / price_t
            # comisión en la compra sobre el monto USDT (order_amount)
            commission_buy = order_amount * comi_factor if comi_factor != 0.0 else 0.0
            # descontar del cash el monto de la orden + comisión de compra
            cash -= (order_amount + commission_buy)
            num_signals_executed += 1

            pos_idx = idx_maps[sym][t]
            sell_idx = pos_idx + sell_after
            if sell_idx < len(df):
                sell_time = df.index[sell_idx]
            else:
                # Si no hay suficientes velas, cerrar en la última vela disponible
                sell_time = last_index_per_symbol[sym]

            # Preparar TP/SL si aplica
            tp_price = None
            sl_price = None
            if tp_pct != 0.0:
                tp_price = price_t * (1.0 + tp_pct / 100.0)
            if sl_pct != 0.0:
                sl_price = price_t * (1.0 - sl_pct / 100.0)

            positions[sym].append({
                'qty': qty,
                'buy_price': price_t,
                'buy_time': t,
                'sell_time': sell_time,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'commission_buy': commission_buy
            })

        # 4) Balance
        positions_value = 0.0
        for sym, pos_list in positions.items():
            if len(pos_list) == 0:
                continue
            df = symbol_data[sym]
            price_map = price_at_t.get(sym, None)
            if price_map is None:
                earlier_idx = df.index.searchsorted(t) - 1
                if earlier_idx >= 0:
                    price_map = float(df.iloc[earlier_idx]['close'])
            if price_map is None:
                continue
            for pos in pos_list:
                positions_value += pos['qty'] * price_map

        sim_balance = cash + positions_value
        sim_balance_history.append((t, sim_balance))

    # --- Cierre forzado de posiciones al final (último precio disponible por símbolo)
    # Procesamos cierres finales con la misma lógica de comisión
    for sym, pos_list in positions.items():
        if len(pos_list) == 0:
            continue
        df = symbol_data[sym]
        last_price = float(df['close'].iloc[-1])
        last_time = df.index[-1]
        for pos in pos_list:
            qty = pos['qty']
            buy_price = pos['buy_price']
            commission_buy = pos.get('commission_buy', 0.0)
            commission_sell = (qty * last_price) * comi_factor if comi_factor != 0.0 else 0.0
            cash += qty * last_price - commission_sell
            profit = (last_price - buy_price) * qty - commission_buy - commission_sell
            trades[sym].append(profit)
            trade_times[sym].append(last_time)
            trade_log_rows.append({
                'symbol': sym,
                'buy_time': pos['buy_time'],
                'buy_price': buy_price,
                'sell_time': last_time,
                'sell_price': last_price,
                'qty': qty,
                'profit': profit,
                'exit_reason': 'FORCED_LAST',
                'commission_buy': commission_buy,
                'commission_sell': commission_sell
            })
        positions[sym] = []

    final_balance = cash

    # --- Métricas globales (portafolio)
    if len(sim_balance_history) == 0:
        timestamps = [pd.Timestamp.now()]
        sim_values = np.array([initial_balance], dtype=np.float64)
    else:
        timestamps, sim_values = zip(*sim_balance_history)
        sim_values = np.array(sim_values, dtype=np.float64)

    cummax    = np.maximum.accumulate(sim_values)
    # proteger división por cero
    drawdowns = (cummax - sim_values) / np.where(cummax == 0, 1, cummax)
    max_dd_portfolio    = float(np.nanmax(drawdowns)) if drawdowns.size > 0 else 0.0

    all_trades = []
    for sym in trades:
        all_trades.extend(trades[sym])
    num_trades = len(all_trades)
    if num_trades > 0:
        num_winners = int(np.sum(np.array(all_trades) > 0.0))
        proportion_winners = num_winners / num_trades
    else:
        proportion_winners = np.nan

    # --- Calcular drawdown por símbolo usando los trades y sus timestamps
    max_dd_by_symbol = {}
    final_balance_by_symbol = {}
    for sym in symbols:
        # Serie de ganancias por timestamp inicializada a 0
        profits_series = pd.Series(0.0, index=pd.Index(all_timestamps))
        # Sumar ganancias en los timestamps donde cerró cada trade
        for profit, t_close in zip(trades[sym], trade_times[sym]):
            if t_close in profits_series.index:
                profits_series.loc[t_close] += profit
            else:
                idx = profits_series.index.searchsorted(t_close)
                if idx > 0:
                    profits_series.iloc[idx-1] += profit
                else:
                    profits_series.iloc[0] += profit
        # equity acumulada (initial + cumsum)
        equity = initial_balance + profits_series.cumsum().values
        if equity.size == 0:
            max_dd_by_symbol[sym] = 0.0
            final_balance_by_symbol[sym] = float(initial_balance)
            continue
        cummax_sym = np.maximum.accumulate(equity)
        drawdowns_sym = (cummax_sym - equity) / np.where(cummax_sym == 0, 1, cummax_sym)
        max_dd_by_symbol[sym] = float(np.nanmax(drawdowns_sym)) if drawdowns_sym.size > 0 else 0.0
        final_balance_by_symbol[sym] = float(equity[-1])

    # Resultados por símbolo
    results = {}
    for sym in symbols:
        results[sym] = {
            'df': symbol_data[sym] if keep_df else None,
            'trades': trades[sym],             # lista de floats (compatible con main)
            'final_balance': final_balance_by_symbol[sym],
            'num_signals': len(trades[sym]),
            'proportion_winners': (np.nan if len(trades[sym])==0
                                   else (np.sum(np.array(trades[sym])>0.0)/len(trades[sym]))),
            'max_dd': max_dd_by_symbol[sym]
        }

    # Resumen global
    trade_log_df = pd.DataFrame(trade_log_rows)
    results["__PORTFOLIO__"] = {
        'df': None,
        'trades': all_trades,
        'final_balance': final_balance,
        'num_signals': num_signals_executed,
        'proportion_winners': proportion_winners,
        'max_dd': max_dd_portfolio,
        'sim_balance_history': sim_balance_history,
        'trade_log': trade_log_df
    }

    return results