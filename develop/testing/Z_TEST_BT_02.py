"""
Test TP/SL Intrabar: comparación Manual vs Función
Test específico para verificar detección correcta de TP/SL dentro de la misma vela
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
SELL_AFTER = 2
ORDER_AMOUNT = 850  # ← CAMBIO: 100 en lugar de 10000 (para que alcance el capital)
TP_PCT = 3.0
SL_PCT = 2.0

print(f"\n📊 CONFIGURACIÓN:")
print(f"   Initial balance: ${INITIAL_BALANCE}")
print(f"   Order amount: ${ORDER_AMOUNT}")
print(f"   Commission: {COMISION}%")

# =====================================================================
# DATOS CONTROLADOS PARA TP/SL INTRABAR
# =====================================================================
print("\n" + "="*70)
print("TEST: TP/SL INTRABAR - Manual vs Función")
print("="*70)

dates = pd.date_range("2024-01-01", periods=6, freq='D')

# Hardcodeamos intrabar con horas distintas
# Vela 4: high=104 (alcanza TP=103), low=97 (alcanza SL=98)
# high_time < low_time → TP primero
high_time = np.array([
    dates[0] + pd.Timedelta(hours=10),
    dates[1] + pd.Timedelta(hours=11),
    dates[2] + pd.Timedelta(hours=11),
    dates[3] + pd.Timedelta(hours=9),
    dates[4] + pd.Timedelta(hours=10),  # TP time (10:00)
    dates[5] + pd.Timedelta(hours=15)
], dtype='datetime64[ns]')

low_time = np.array([
    dates[0] + pd.Timedelta(hours=9),
    dates[1] + pd.Timedelta(hours=10),
    dates[2] + pd.Timedelta(hours=12),
    dates[3] + pd.Timedelta(hours=11),
    dates[4] + pd.Timedelta(hours=14),  # SL time (14:00) - después
    dates[5] + pd.Timedelta(hours=14)
], dtype='datetime64[ns]')

ohlcv_arrays = {
    "SYM_TEST": {
        'ts': dates.to_numpy().astype('datetime64[ns]'),
        'open': np.array([100, 100, 100, 100, 100, 100], dtype=np.float64),
        'close': np.array([100, 100, 100, 100, 100, 100], dtype=np.float64),
        'high': np.array([100, 100, 100, 100, 104, 100], dtype=np.float64),  # Vela 4: alcanza TP
        'low':  np.array([100, 100, 100, 100, 97, 100], dtype=np.float64),   # Vela 4: alcanza SL
        'signal': np.array([1, 0, 0, 0, 0, 0], dtype=np.int32),
        'high_time': high_time.view('int64'),
        'low_time': low_time.view('int64')
    }
}

print(f"\nTest setup:")
print(f"- Buy at vela 0: price = 100")
print(f"- TP = {100 * (1 + TP_PCT/100):.2f} (+{TP_PCT}%)")
print(f"- SL = {100 * (1 - SL_PCT/100):.2f} (-{SL_PCT}%)")
print(f"- Vela 4: high=104 (alcanza TP), low=97 (alcanza SL)")
print(f"- high_time=10:00, low_time=14:00 → TP primero")

# =====================================================================
# FUNCIÓN MANUAL TP/SL INTRABAR
# =====================================================================
def generate_manual_trades(ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount, comi_pct, initial_balance):
    """
    Cálculo manual con detección de TP/SL intrabar y validación de cash.
    """
    trades = []
    cash = float(initial_balance)
    
    for symbol, data in ohlcv_arrays.items():
        close = data['close']
        open_prices = data['open']
        high = data['high']
        low = data['low']
        ts = data['ts']
        signals = data['signal']
        high_time = data['high_time']
        low_time = data['low_time']
        
        n = len(close)
        
        for i in range(n):
            if signals[i] != 1:
                continue
            
            # Validar cash suficiente
            buy_price = float(open_prices[i])
            comm_buy = order_amount * comi_pct / 100.0
            
            if cash < (order_amount + comm_buy):
                print(f"   ⚠️  Insuficiente cash: ${cash:.2f} < ${order_amount + comm_buy:.2f}")
                continue
            
            # Comprar en open price
            units = order_amount / buy_price
            cash -= (order_amount + comm_buy)
            
            tp_price = buy_price * (1 + tp_pct / 100)
            sl_price = buy_price * (1 - sl_pct / 100)
            
            # Rango de búsqueda: desde vela actual hasta sell_after
            sell_idx = min(i + sell_after, n - 1)
            
            # Default: vender en sell_idx
            exec_price = float(close[sell_idx])
            exec_time = ts[sell_idx]
            exit_reason = 'SELL_AFTER'
            
            # Buscar TP/SL intrabar
            for j in range(i, sell_idx + 1):
                tp_hit = high[j] >= tp_price
                sl_hit = low[j] <= sl_price
                
                if tp_hit or sl_hit:
                    if tp_hit and sl_hit:
                        # Ambos en misma vela → usar timestamps
                        ht = high_time[j]
                        lt = low_time[j]
                        
                        if ht <= lt:
                            exec_price = tp_price
                            exec_time = np.datetime64(ht, 'ns')
                            exit_reason = 'TP'
                        else:
                            exec_price = sl_price
                            exec_time = np.datetime64(lt, 'ns')
                            exit_reason = 'SL'
                    elif tp_hit:
                        exec_price = tp_price
                        exec_time = np.datetime64(high_time[j], 'ns')
                        exit_reason = 'TP'
                    elif sl_hit:
                        exec_price = sl_price
                        exec_time = np.datetime64(low_time[j], 'ns')
                        exit_reason = 'SL'
                    
                    break  # Salir del loop
            
            # Calcular profit
            comm_sell = units * exec_price * comi_pct / 100.0
            profit = units * (exec_price - buy_price) - comm_buy - comm_sell
            
            # Actualizar cash
            cash += units * exec_price - comm_sell
            
            trades.append({
                "symbol": symbol,
                "buy_time": ts[i],
                "buy_price": buy_price,
                "sell_time": exec_time,
                "sell_price": exec_price,
                "profit": profit,
                "exit_reason": exit_reason
            })
    
    return pd.DataFrame(trades), cash


# =====================================================================
# EJECUCIÓN MANUAL
# =====================================================================
print("\n[MANUAL BACKTEST]")
manual_trades, final_cash = generate_manual_trades(
    ohlcv_arrays=ohlcv_arrays,
    sell_after=SELL_AFTER,
    tp_pct=TP_PCT,
    sl_pct=SL_PCT,
    order_amount=ORDER_AMOUNT,
    comi_pct=COMISION,
    initial_balance=INITIAL_BALANCE
)

if not manual_trades.empty:
    print("\nTrades ejecutados:")
    for i, row in manual_trades.iterrows():
        print(f"{i+1}. {row['symbol']}: Buy ${row['buy_price']:.2f} → Sell ${row['sell_price']:.2f} | "
              f"Profit: ${row['profit']:.2f} | Reason: {row['exit_reason']}")
    
    manual_profit = manual_trades['profit'].sum()
    manual_balance = final_cash
    print(f"\n💰 Balance final: ${manual_balance:.2f}")
    print(f"📈 Profit total: ${manual_profit:.2f}")
else:
    print("No trades executed")
    manual_balance = INITIAL_BALANCE
    manual_profit = 0

# =====================================================================
# EJECUCIÓN FUNCIÓN
# =====================================================================
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

if not trade_log.empty:
    print("\nTrades ejecutados:")
    for i, row in trade_log.iterrows():
        print(f"{i+1}. {row['symbol']}: Buy ${row['buy_price']:.2f} → Sell ${row['sell_price']:.2f} | "
              f"Profit: ${row['profit']:.2f} | Reason: {row['exit_reason']}")
    
    func_profit = trade_log['profit'].sum()
    func_balance = portfolio['final_balance']
    print(f"\n💰 Balance final: ${func_balance:.2f}")
    print(f"📈 Profit total: ${func_profit:.2f}")
else:
    print("No trades executed")
    func_balance = INITIAL_BALANCE
    func_profit = 0

# =====================================================================
# COMPARACIÓN
# =====================================================================
print("\n" + "="*70)
print("COMPARACIÓN DETALLADA")
print("="*70)

# Número de trades
num_manual = len(manual_trades)
num_func = len(trade_log)
print(f"\nNúmero de trades:")
print(f"  Manual: {num_manual}")
print(f"  Función: {num_func}")
print(f"  Iguales: {'✅' if num_manual == num_func else '❌'}")

# Balance
diff_balance = abs(manual_balance - func_balance)
print(f"\nBalance final:")
print(f"  Manual: ${manual_balance:.6f}")
print(f"  Función: ${func_balance:.6f}")
print(f"  Diferencia: ${diff_balance:.8f}")

# Profit
diff_profit = abs(manual_profit - func_profit)
print(f"\nProfit total:")
print(f"  Manual: ${manual_profit:.6f}")
print(f"  Función: ${func_profit:.6f}")
print(f"  Diferencia: ${diff_profit:.8f}")

# Exit reason
if not manual_trades.empty and not trade_log.empty:
    manual_reason = manual_trades.iloc[0]['exit_reason']
    func_reason = trade_log.iloc[0]['exit_reason']
    print(f"\nExit reason:")
    print(f"  Manual: {manual_reason}")
    print(f"  Función: {func_reason}")
    print(f"  Iguales: {'✅' if manual_reason == func_reason else '❌'}")

# Veredicto
tolerance = 1e-5
if diff_balance < tolerance and diff_profit < tolerance and num_manual == num_func:
    print("\n" + "="*70)
    print("✅✅✅ TEST PASADO ✅✅✅")
    print("Manual == Función (TP/SL intrabar correcto)")
    print("="*70)
else:
    print("\n" + "="*70)
    print("❌❌❌ TEST FALLIDO ❌❌❌")
    print("Manual != Función")
    print("="*70)
    
    if num_manual != num_func:
        print(f"\n⚠️  Diferente número de trades: {num_manual} vs {num_func}")
    
    if diff_balance >= tolerance:
        print(f"\n⚠️  Balance difiere por: ${diff_balance:.8f}")
    
    if diff_profit >= tolerance:
        print(f"\n⚠️  Profit difiere por: ${diff_profit:.8f}")
    
    if not manual_trades.empty and not trade_log.empty:
        if manual_trades.iloc[0]['exit_reason'] != trade_log.iloc[0]['exit_reason']:
            print(f"\n⚠️  Exit reason diferente: {manual_trades.iloc[0]['exit_reason']} vs {trade_log.iloc[0]['exit_reason']}")