"""
Test con múltiples símbolos: comparación Manual vs Función
Datos hardcodeados con múltiples símbolos y señales
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
from datetime import timedelta
from backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE, COMISION

# =====================================================================
# PARÁMETROS
# =====================================================================
SELL_AFTER = 5
TP_PCT = 3.0
SL_PCT = 2.0
ORDER_AMOUNT = 100

# =====================================================================
# DATOS HARDCODEADOS
# =====================================================================
def create_controlled_test():
    """
    Crea datos de prueba con múltiples símbolos.
    """
    # ---------------------
    # Símbolo 1 (15 días)
    # ---------------------
    dates1 = pd.to_datetime([
        "2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05",
        "2024-01-06","2024-01-07","2024-01-08","2024-01-09","2024-01-10",
        "2024-01-11","2024-01-12","2024-01-13","2024-01-14","2024-01-15"
    ])
    open1 = list(range(100, 115))
    close1 = list(range(100, 115))
    high1 = [c+1 for c in close1]
    low1 = [c-1 for c in close1]
    signal1 = [1,1,1,0,0,0,0,0,1,0,0,0,0,0,0]
    df1 = pd.DataFrame({
        "open": open1, "close": close1, "high": high1, "low": low1, "signal": signal1
    }, index=dates1)

    # ---------------------
    # Símbolo 2 (15 días)
    # ---------------------
    open2 = list(range(200, 215))
    close2 = list(range(200, 215))
    high2 = [c+1 for c in close2]
    low2 = [c-1 for c in close2]
    signal2 = [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0]
    df2 = pd.DataFrame({
        "open": open2, "close": close2, "high": high2, "low": low2, "signal": signal2
    }, index=dates1)

    # ---------------------
    # Símbolo 3 (15 días)
    # ---------------------
    open3 = list(range(300, 315))
    close3 = list(range(300, 315))
    high3 = [c+1 for c in close3]
    low3 = [c-1 for c in close3]
    signal3 = [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0]
    df3 = pd.DataFrame({
        "open": open3, "close": close3, "high": high3, "low": low3, "signal": signal3
    }, index=dates1)

    return {"SYM1": df1, "SYM2": df2, "SYM3": df3}


def df_to_ohlcv_arrays(dfs_dict):
    """Convierte DataFrames a formato requerido por run_grid_backtest."""
    ohlcv_arrays = {}
    for symbol, df in dfs_dict.items():
        timestamps = df.index.to_numpy().astype('datetime64[ns]')
        
        ohlcv_arrays[symbol] = {
            'ts': timestamps,
            'open': df['open'].to_numpy(dtype=np.float64),
            'close': df['close'].to_numpy(dtype=np.float64),
            'high': df['high'].to_numpy(dtype=np.float64),
            'low': df['low'].to_numpy(dtype=np.float64),
            'signal': df['signal'].to_numpy(dtype=np.int32),  # 1=long, -1=short, 0=nada
            'high_time': timestamps.view('int64'),  # ← CAMBIO: usar .view() en lugar de .value
            'low_time': timestamps.view('int64')    # ← CAMBIO: usar .view() en lugar de .value
        }
    return ohlcv_arrays


# =====================================================================
# CÁLCULO MANUAL (UNA POSICIÓN A LA VEZ - PORTFOLIO LEVEL)
# =====================================================================
def manual_backtest_portfolio(dfs_dict, sell_after, tp_pct, sl_pct, 
                              initial_balance, order_amount, comi_percent):
    """
    Backtest manual: replica comportamiento de run_backtest_loop.
    
    Lógica:
    - Si NO hay posiciones abiertas, buscar señales en timestamp actual
    - Abrir TODAS las señales del mismo timestamp
    - Esperar a que TODAS cierren antes de buscar nuevas
    """
    comm_factor = comi_percent / 100.0
    cash = float(initial_balance)
    trades = []
    
    # 1. CREAR ÍNDICE DE TIEMPO UNIFICADO (UNION de todos los timestamps)
    all_times = set()
    for df in dfs_dict.values():
        all_times.update(df.index)
    all_times = sorted(all_times)
    
    print(f"\n[MANUAL] Unified timeline: {len(all_times)} timestamps")
    
    # 2. ITERAR POR TIEMPO
    open_positions = []
    
    for current_time in all_times:
        
        # === CERRAR POSICIONES EXPIRADAS ===
        positions_to_close = []
        for pos in open_positions:
            df = pos['df']
            
            # Buscar índice actual en el DataFrame de este símbolo
            try:
                current_idx = df.index.get_loc(current_time)
            except KeyError:
                continue  # Este símbolo no tiene datos en este timestamp
            
            # Chequear TP/SL/TIMEOUT
            if current_idx < len(df):
                high = df['high'].iloc[current_idx]
                low = df['low'].iloc[current_idx]
                
                exec_price = None
                exit_reason = None
                
                if high >= pos['tp_price']:
                    exec_price = pos['tp_price']
                    exit_reason = 'TP'
                elif low <= pos['sl_price']:
                    exec_price = pos['sl_price']
                    exit_reason = 'SL'
                elif current_idx >= pos['sell_after_idx']:
                    exec_price = float(df['close'].iloc[current_idx])
                    exit_reason = 'SELL_AFTER'
                
                if exec_price is not None:
                    comm_sell = pos['qty'] * exec_price * comm_factor
                    cash += pos['qty'] * exec_price - comm_sell
                    
                    profit = (exec_price - pos['buy_price']) * pos['qty'] - pos['commission_buy'] - comm_sell
                    
                    trades.append({
                        'symbol': pos['symbol'],
                        'buy_idx': pos['buy_idx'],
                        'buy_time': pos['buy_time'],
                        'sell_idx': current_idx,
                        'sell_time': current_time,
                        'buy_price': pos['buy_price'],
                        'sell_price': exec_price,
                        'qty': pos['qty'],
                        'commission_buy': pos['commission_buy'],
                        'commission_sell': comm_sell,
                        'profit': profit,
                        'exit_reason': exit_reason
                    })
                    
                    positions_to_close.append(pos)
        
        # Eliminar posiciones cerradas
        for pos in positions_to_close:
            open_positions.remove(pos)
        
        # === BUSCAR NUEVAS SEÑALES (solo si NO hay posiciones abiertas) ===
        if not open_positions:
            # Buscar señales en todos los símbolos para este timestamp
            for symbol, df in dfs_dict.items():
                if current_time not in df.index:
                    continue
                
                buy_idx = df.index.get_loc(current_time)
                
                if df['signal'].iloc[buy_idx] == 1:  # Señal LONG
                    # Abrir posición
                    buy_price = float(df['open'].iloc[buy_idx])
                    qty = order_amount / buy_price
                    comm_buy = order_amount * comm_factor
                    
                    if cash < (order_amount + comm_buy):
                        continue  # No hay suficiente cash
                    
                    cash -= (order_amount + comm_buy)
                    
                    tp_price = buy_price * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else float('inf')
                    sl_price = buy_price * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else float('-inf')
                    sell_after_idx = min(buy_idx + sell_after, len(df) - 1)
                    
                    open_positions.append({
                        'symbol': symbol,
                        'df': df,
                        'buy_idx': buy_idx,
                        'buy_time': current_time,
                        'buy_price': buy_price,
                        'qty': qty,
                        'commission_buy': comm_buy,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'sell_after_idx': sell_after_idx
                    })
    
    final_balance = cash
    total_profit = sum(t['profit'] for t in trades)
    return trades, final_balance, total_profit


# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST: MANUAL vs FUNCIÓN (Portfolio-level: 1 posición a la vez)")
    print("="*70)
    
    # Crear datos
    dfs_dict = create_controlled_test()
    ohlcv_arrays = df_to_ohlcv_arrays(dfs_dict)
    
    # ============== CÁLCULO MANUAL ==============
    print("\n[MANUAL BACKTEST]")
    manual_trades, final_balance_manual, total_profit_manual = manual_backtest_portfolio(
        dfs_dict=dfs_dict,
        sell_after=SELL_AFTER,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        comi_percent=COMISION
    )
    
    print(f"\nTotal trades: {len(manual_trades)}")
    print("\nTrades executed:")
    for i, t in enumerate(manual_trades, 1):
        print(f"{i}. {t['symbol']}: Buy ${t['buy_price']:.2f} → Sell ${t['sell_price']:.2f} | "
              f"Profit: ${t['profit']:.2f} | Reason: {t['exit_reason']}")
    
    print(f"\n💰 Balance final: ${final_balance_manual:.2f}")
    print(f"📈 Profit total: ${total_profit_manual:.2f}")
    
    # ============== FUNCIÓN AUTOMÁTICA ==============
    print("\n[FUNCIÓN BACKTEST]")
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=SELL_AFTER,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        order_amount=ORDER_AMOUNT
    )
    
    portfolio = results['__PORTFOLIO__']
    trade_log = portfolio['trade_log']
    
    final_balance_func = portfolio['final_balance']
    num_trades_func = portfolio['num_signals']
    profit_func = sum(portfolio['trades']) if portfolio['trades'] else 0
    
    print(f"\nTotal trades: {num_trades_func}")
    print("\nTrades executed:")
    for i, (_, row) in enumerate(trade_log.iterrows(), 1):
        print(f"{i}. {row['symbol']}: Buy ${row['buy_price']:.2f} → Sell ${row['sell_price']:.2f} | "
              f"Profit: ${row['profit']:.2f} | Reason: {row['exit_reason']}")
    
    print(f"\n💰 Balance final: ${final_balance_func:.2f}")
    print(f"📈 Profit total: ${profit_func:.2f}")
    
    # ============== COMPARACIÓN ==============
    print("\n" + "="*70)
    print("COMPARACIÓN DETALLADA")
    print("="*70)
    
    # Comparar número de trades
    print(f"\nNúmero de trades:")
    print(f"  Manual: {len(manual_trades)}")
    print(f"  Función: {num_trades_func}")
    print(f"  Iguales: {'✅' if len(manual_trades) == num_trades_func else '❌'}")
    
    # Comparar balance
    diff_balance = abs(final_balance_manual - final_balance_func)
    print(f"\nBalance final:")
    print(f"  Manual: ${final_balance_manual:.6f}")
    print(f"  Función: ${final_balance_func:.6f}")
    print(f"  Diferencia: ${diff_balance:.8f}")
    
    # Comparar profit
    diff_profit = abs(total_profit_manual - profit_func)
    print(f"\nProfit total:")
    print(f"  Manual: ${total_profit_manual:.6f}")
    print(f"  Función: ${profit_func:.6f}")
    print(f"  Diferencia: ${diff_profit:.8f}")
    
    # Veredicto
    tolerance = 1e-5
    if diff_balance < tolerance and diff_profit < tolerance and len(manual_trades) == num_trades_func:
        print("\n" + "="*70)
        print("✅✅✅ TEST PASADO ✅✅✅")
        print("Manual == Función")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌❌❌ TEST FALLIDO ❌❌❌")
        print("Manual != Función")
        print("="*70)
        
        # Debugging
        if len(manual_trades) != num_trades_func:
            print(f"\n⚠️  Diferente número de trades: {len(manual_trades)} vs {num_trades_func}")
        
        if diff_balance >= tolerance:
            print(f"\n⚠️  Balance difiere por: ${diff_balance:.8f}")
        
        if diff_profit >= tolerance:
            print(f"\n⚠️  Profit difiere por: ${diff_profit:.8f}")