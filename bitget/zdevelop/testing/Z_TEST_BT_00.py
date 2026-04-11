"""
Test simplificado: 1 símbolo, comparación Manual vs Función
El cálculo manual se adapta automáticamente a los datos hardcodeados
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
from datetime import timedelta
from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, COMISION

# ============== PARÁMETROS ==============
ORDER_AMOUNT = 200

# ============== DATOS DE PRUEBA (MODIFICAR AQUÍ) ==============
base_time = pd.Timestamp('2024-01-01 00:00:00')
timestamps = [base_time + timedelta(hours=i) for i in range(8)]

ohlcv_data = {
    'BTC': {
        'ts': np.array(timestamps, dtype='datetime64[ns]'),
        'close': np.array([100, 100, 100, 100, 100, 100, 100, 100], dtype=np.float64),
        'open': np.array([100, 100, 100, 100, 100, 100, 100, 100], dtype=np.float64),  # ← AÑADIDO
        'high': np.array([100, 105, 100, 100, 100, 100, 100, 100], dtype=np.float64),
        'low': np.array([100, 99.5, 100, 100, 100, 100, 100, 100], dtype=np.float64),
        'signal': np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),  # ← CAMBIADO: int32 (1=long, -1=short, 0=nada)
        'high_time': np.array([ts.value for ts in timestamps], dtype=np.int64),  # ← CAMBIADO: int64 nanoseconds
        'low_time': np.array([ts.value for ts in timestamps], dtype=np.int64)    # ← CAMBIADO: int64 nanoseconds
    }
}

# PARÁMETROS BACKTEST
sell_after = 5
tp_pct = 6.0
sl_pct = 5.0

# ============== CÁLCULO MANUAL FUNCIONALIZADO ==============
def manual_backtest(signals, sell_after, close_prices, open_prices, high_prices, low_prices,
                    initial_balance, order_amount, comi_percent, tp_pct=0.0, sl_pct=0.0):
    """
    Cálculo manual LONG-only (adaptado a la lógica de tu función).
    """
    comm_factor = comi_percent / 100.0
    cash = float(initial_balance)
    trades = []
    position_open = None

    signal_indices = np.where(signals == 1)[0]  # ← Solo LONG (signal=1)
    n = len(close_prices)

    for t in range(n):
        # Ejecutar compra si hay señal y no hay posición abierta
        if t in signal_indices and position_open is None:
            # TU FUNCIÓN USA OPEN PRICE
            buy_price = float(open_prices[t])
            qty = order_amount / buy_price
            comm_buy = order_amount * comm_factor
            cash -= (order_amount + comm_buy)

            tp_price = buy_price * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else float('inf')
            sl_price = buy_price * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else float('-inf')

            position_open = {
                'buy_idx': t,
                'buy_price': buy_price,
                'qty': qty,
                'commission_buy': comm_buy,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'sell_after_idx': min(t + sell_after, n - 1)
            }

        # Cerrar posición si hay alguna abierta
        if position_open is not None:
            sell_idx = t
            high = high_prices[t]
            low = low_prices[t]

            # Determinar si se alcanza TP o SL
            exec_price = None
            exit_reason = None

            if high >= position_open['tp_price']:
                exec_price = position_open['tp_price']
                exit_reason = 'TP'
            elif low <= position_open['sl_price']:
                exec_price = position_open['sl_price']
                exit_reason = 'SL'
            elif t >= position_open['sell_after_idx']:
                exec_price = float(close_prices[t])
                exit_reason = 'SELL_AFTER'

            if exec_price is not None:
                comm_sell = position_open['qty'] * exec_price * comm_factor
                cash += position_open['qty'] * exec_price - comm_sell

                profit = (exec_price - position_open['buy_price']) * position_open['qty'] - \
                         position_open['commission_buy'] - comm_sell

                trades.append({
                    'buy_idx': position_open['buy_idx'],
                    'sell_idx': t,
                    'buy_price': position_open['buy_price'],
                    'sell_price': exec_price,
                    'qty': position_open['qty'],
                    'commission_buy': position_open['commission_buy'],
                    'commission_sell': comm_sell,
                    'profit': profit,
                    'exit_reason': exit_reason
                })

                position_open = None

    final_balance = cash
    total_profit = sum(t['profit'] for t in trades)
    return trades, final_balance, total_profit


# ============== EXTRACCIÓN AUTOMÁTICA DE DATOS ==============
symbol = list(ohlcv_data.keys())[0]
data = ohlcv_data[symbol]
close_prices = data['close']
open_prices = data['open']
signals = data['signal']

# Ejecutar cálculo manual
manual_trades, final_balance_manual, total_profit_manual = manual_backtest(
    signals=signals,
    sell_after=sell_after,
    close_prices=close_prices,
    open_prices=open_prices,
    high_prices=data['high'],
    low_prices=data['low'],
    initial_balance=INITIAL_BALANCE,
    order_amount=ORDER_AMOUNT,
    comi_percent=COMISION,
    tp_pct=tp_pct,
    sl_pct=sl_pct
)

print("\n[MANUAL]")
if manual_trades:
    for t in manual_trades:
        print(f"Señal detectada en índice: {t['buy_idx']}")
        print(f"Precio compra (t={t['buy_idx']})      : ${t['buy_price']:.2f}")
        print(f"Precio venta (t={t['sell_idx']})       : ${t['sell_price']:.2f}")
        print(f"Exit reason              : {t['exit_reason']}")
        print(f"Profit                   : ${t['profit']:.6f}\n")
else:
    print("No trades executed")

print(f"Balance final            : ${final_balance_manual:.6f}")
print(f"Profit total             : ${total_profit_manual:.6f}")

# ============== FUNCIÓN AUTOMÁTICA ==============
print("\n[FUNCIÓN]")
results = run_grid_backtest(
    ohlcv_arrays=ohlcv_data,
    sell_after=sell_after,
    tp_pct=tp_pct,
    sl_pct=sl_pct,
    order_amount=ORDER_AMOUNT  # ← AÑADIDO: parámetro faltante
)

portfolio = results['__PORTFOLIO__']

final_balance_func = portfolio['final_balance']
num_trades_func = portfolio['num_signals']
profit_func = sum(portfolio['trades']) if portfolio['trades'] else 0

print(f"Num trades               : {num_trades_func}")
print(f"Profit                   : ${profit_func:.6f}")
print(f"Balance                  : ${final_balance_func:.6f}")

# Mostrar detalles de trades
if 'trade_log' in portfolio and not portfolio['trade_log'].empty:
    print("\nTrade details:")
    print(portfolio['trade_log'][['symbol', 'buy_price', 'sell_price', 'profit', 'exit_reason']])

# ============== COMPARACIÓN ==============
print("\n" + "=" * 60)
print("COMPARACIÓN")
print("=" * 60)

diff_balance = abs(final_balance_manual - final_balance_func)
diff_profit = abs(total_profit_manual - profit_func)

print(f"Balance: Manual=${final_balance_manual:.6f}  Función=${final_balance_func:.6f}  Diff=${diff_balance:.8f}")
print(f"Profit:  Manual=${total_profit_manual:.6f}      Función=${profit_func:.6f}      Diff=${diff_profit:.8f}")

tolerance = 1e-5
if diff_balance < tolerance and diff_profit < tolerance:
    print("\n✅✅✅ TEST PASADO ✅✅✅")
    print("Manual == Función")
else:
    print("\n✗✗✗ TEST FALLIDO ✗✗✗")
    print(f"Manual != Función (tolerancia: {tolerance})")
    
    # Debugging info
    print("\n[DEBUG INFO]")
    print(f"Manual trades: {len(manual_trades)}")
    print(f"Function trades: {num_trades_func}")
    
    if abs(len(manual_trades) - num_trades_func) > 0:
        print("⚠️  Different number of trades executed!")