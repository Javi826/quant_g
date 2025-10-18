import numpy as np
import pandas as pd
from collections import defaultdict

# ============================
# CONFIGURACIÓN
# ============================
MIN_PRICE = 0.0001
INITIAL_BALANCE = 10000
ORDER_AMOUNT = 100
COMISION = 0.05


# ============================
# FUNCIONES AUXILIARES
# ============================

def get_price_at_time(df, target_time):
    """Obtiene el precio de cierre en un timestamp específico."""
    mask = df.index <= target_time
    if mask.any():
        return df.loc[mask, 'close'].iloc[-1]
    return None


def check_tp_sl_hit(row, buy_price, tp_price, sl_price):
    """
    Verifica si se alcanzó TP o SL en una vela.
    Retorna: (hit, exit_reason, exec_price)
    """
    if tp_price and row['high'] >= tp_price:
        if sl_price and row['low'] <= sl_price:
            # Ambos tocados - usar high_time y low_time si existen
            if 'high_time' in row and 'low_time' in row:
                if row['low_time'] <= row['high_time']:
                    return True, 'SL', sl_price
                else:
                    return True, 'TP', tp_price
            # Si no hay timestamps intrabar, asumir SL primero (conservador)
            return True, 'SL', sl_price
        else:
            return True, 'TP', tp_price
    
    if sl_price and row['low'] <= sl_price:
        return True, 'SL', sl_price
    
    return False, None, None


def calculate_sharpe_ratio(equity_series):
    """Calcula el Sharpe ratio anualizado."""
    if len(equity_series) < 2:
        return np.nan
    
    returns = equity_series.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    
    # Estimar períodos por año basado en el intervalo mediano
    time_diffs = equity_series.index.to_series().diff().dropna()
    median_period_hours = time_diffs.median().total_seconds() / 3600
    periods_per_year = (365 * 24) / median_period_hours if median_period_hours > 0 else 252
    
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(equity_series):
    """Calcula el máximo drawdown."""
    if len(equity_series) == 0:
        return 0.0
    
    cummax = equity_series.cummax()
    drawdown = (cummax - equity_series) / cummax
    return drawdown.max()


# ============================
# CLASE POSITION
# ============================

class Position:
    """Representa una posición abierta."""
    
    def __init__(self, symbol, buy_time, buy_price, qty, sell_after_time, 
                 tp_price=None, sl_price=None, commission_buy=0.0):
        self.symbol = symbol
        self.buy_time = buy_time
        self.buy_price = buy_price
        self.qty = qty
        self.sell_after_time = sell_after_time
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.commission_buy = commission_buy
        self.closed = False


# ============================
# FUNCIONES DE GESTIÓN DE POSICIONES
# ============================

def open_position(symbol, df, signal_time, order_amount, sell_after_candles,
                  tp_pct=0.0, sl_pct=0.0, comi_factor=0.0):
    """Abre una nueva posición."""
    buy_price = df.loc[signal_time, 'close']
    qty = order_amount / buy_price
    commission_buy = order_amount * comi_factor
    
    # Calcular tiempo de venta (sell_after_candles velas después)
    future_idx = df.index.get_loc(signal_time) + sell_after_candles
    if future_idx >= len(df):
        future_idx = len(df) - 1
    sell_after_time = df.index[future_idx]
    
    # Calcular precios TP y SL
    tp_price = buy_price * (1 + tp_pct / 100) if tp_pct > 0 else None
    sl_price = buy_price * (1 - sl_pct / 100) if sl_pct > 0 else None
    
    return Position(
        symbol=symbol,
        buy_time=signal_time,
        buy_price=buy_price,
        qty=qty,
        sell_after_time=sell_after_time,
        tp_price=tp_price,
        sl_price=sl_price,
        commission_buy=commission_buy
    )


def close_position(position, sell_time, sell_price, exit_reason, comi_factor):
    """Cierra una posición y retorna el trade."""
    commission_sell = (position.qty * sell_price) * comi_factor
    profit = (sell_price - position.buy_price) * position.qty - position.commission_buy - commission_sell
    
    trade = {
        'symbol': position.symbol,
        'buy_time': position.buy_time,
        'buy_price': position.buy_price,
        'sell_time': sell_time,
        'sell_price': sell_price,
        'qty': position.qty,
        'profit': profit,
        'exit_reason': exit_reason,
        'commission_buy': position.commission_buy,
        'commission_sell': commission_sell
    }
    
    position.closed = True
    return trade, profit


def check_position_exit(position, df, current_time):
    """
    Verifica si una posición debe cerrarse.
    Retorna: (should_close, sell_time, sell_price, exit_reason)
    """
    # Obtener rango de velas desde compra hasta ahora
    buy_loc = df.index.get_loc(position.buy_time)
    current_loc = df.index.get_loc(current_time)
    
    # Revisar velas entre compra y ahora para TP/SL
    for i in range(buy_loc + 1, current_loc + 1):
        candle_time = df.index[i]
        row = df.iloc[i]
        
        hit, exit_reason, exec_price = check_tp_sl_hit(
            row, position.buy_price, position.tp_price, position.sl_price
        )
        
        if hit:
            return True, candle_time, exec_price, exit_reason
    
    # Verificar si alcanzó sell_after
    if current_time >= position.sell_after_time:
        sell_price = df.loc[position.sell_after_time, 'close']
        return True, position.sell_after_time, sell_price, 'SELL_AFTER'
    
    return False, None, None, None


# ============================
# FUNCIÓN PRINCIPAL DE BACKTESTING
# ============================

def run_simple_backtest(
    dfs_dict,
    sell_after_candles,
    tp_pct=0.0,
    sl_pct=0.0,
    initial_balance=10000,
    order_amount=100,
    comi_pct=0.05
):
    """
    Motor de backtesting simple con pandas.
    
    Args:
        dfs_dict: Diccionario {symbol: DataFrame} con columnas ['close', 'signal', 'high', 'low']
        sell_after_candles: Número de velas después de las cuales cerrar posición
        tp_pct: Porcentaje de Take Profit (0 = desactivado)
        sl_pct: Porcentaje de Stop Loss (0 = desactivado)
        initial_balance: Balance inicial
        order_amount: Monto por orden
        comi_pct: Comisión en porcentaje
    
    Returns:
        Diccionario con estructura compatible con run_grid_backtest
    """
    # Validación básica
    if not dfs_dict:
        raise ValueError("dfs_dict no puede estar vacío")
    
    # Inicialización
    comi_factor = comi_pct / 100
    cash = float(initial_balance)
    symbols = list(dfs_dict.keys())
    
    # Estructuras para tracking
    open_positions = []
    trade_log = []
    trades_by_symbol = defaultdict(list)
    
    # Balance history
    balance_history = {'timestamp': [pd.Timestamp.now()], 'balance': [initial_balance]}
    
    # Crear timeline unificado de todas las señales
    all_signals = []
    for symbol, df in dfs_dict.items():
        signal_times = df[df['signal'] == 1].index
        for sig_time in signal_times:
            all_signals.append((sig_time, symbol))
    
    # Ordenar señales por tiempo
    all_signals.sort(key=lambda x: x[0])
    
    # Crear timeline global de todos los timestamps
    all_times = sorted(set(pd.concat([df.index.to_series() for df in dfs_dict.values()])))
    
    # ============================
    # BUCLE PRINCIPAL
    # ============================
    
    signal_idx = 0
    num_signals_executed = 0
    
    for current_time in all_times:
        # 1. Verificar y cerrar posiciones existentes
        positions_to_remove = []
        
        for pos in open_positions:
            df = dfs_dict[pos.symbol]
            should_close, sell_time, sell_price, exit_reason = check_position_exit(
                pos, df, current_time
            )
            
            if should_close:
                trade, profit = close_position(pos, sell_time, sell_price, exit_reason, comi_factor)
                trade_log.append(trade)
                trades_by_symbol[pos.symbol].append(profit)
                cash += pos.qty * sell_price - trade['commission_sell']
                positions_to_remove.append(pos)
        
        # Remover posiciones cerradas
        for pos in positions_to_remove:
            open_positions.remove(pos)
        
        # 2. Ejecutar nuevas señales (solo si no hay posiciones abiertas)
        if not open_positions:
            while signal_idx < len(all_signals) and all_signals[signal_idx][0] == current_time:
                signal_time, symbol = all_signals[signal_idx]
                signal_idx += 1
                
                # Verificar si tenemos suficiente cash
                if cash >= order_amount:
                    pos = open_position(
                        symbol=symbol,
                        df=dfs_dict[symbol],
                        signal_time=signal_time,
                        order_amount=order_amount,
                        sell_after_candles=sell_after_candles,
                        tp_pct=tp_pct,
                        sl_pct=sl_pct,
                        comi_factor=comi_factor
                    )
                    
                    open_positions.append(pos)
                    cash -= (order_amount + pos.commission_buy)
                    num_signals_executed += 1
                    break  # Solo una posición a la vez
        
        # 3. Actualizar balance history
        positions_value = sum(
            pos.qty * dfs_dict[pos.symbol].loc[current_time, 'close']
            for pos in open_positions
        )
        
        balance_history['timestamp'].append(current_time)
        balance_history['balance'].append(cash + positions_value)
    
    # ============================
    # CERRAR POSICIONES RESTANTES
    # ============================
    
    for pos in open_positions:
        if not pos.closed:
            df = dfs_dict[pos.symbol]
            last_time = df.index[-1]
            last_price = df.loc[last_time, 'close']
            
            trade, profit = close_position(pos, last_time, last_price, 'FORCED_LAST', comi_factor)
            trade_log.append(trade)
            trades_by_symbol[pos.symbol].append(profit)
            cash += pos.qty * last_price - trade['commission_sell']
    
    # ============================
    # CALCULAR MÉTRICAS
    # ============================
    
    # Convertir balance history a Series
    balance_series = pd.Series(
        balance_history['balance'],
        index=balance_history['timestamp']
    )
    
    # Métricas de portfolio
    final_balance = float(balance_series.iloc[-1])
    max_dd = calculate_max_drawdown(balance_series)
    sharpe = calculate_sharpe_ratio(balance_series)
    
    # Métricas de trades
    all_trades = [p for lst in trades_by_symbol.values() for p in lst]
    proportion_winners = np.mean([1 if p > 0 else 0 for p in all_trades]) if all_trades else np.nan
    
    # ============================
    # CONSTRUIR RESULTADOS
    # ============================
    
    results = {
        "__PORTFOLIO__": {
            'trades': all_trades,
            'final_balance': final_balance,
            'num_signals': num_signals_executed,
            'proportion_winners': proportion_winners,
            'max_dd': max_dd,
            'sim_balance_history': {
                'timestamp': balance_history['timestamp'],
                'balance': balance_history['balance']
            },
            'trade_log': pd.DataFrame(trade_log),
            'sharpe': sharpe
        }
    }
    
    return results


