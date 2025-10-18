import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ZX_compute_BT import run_grid_backtest as run_original
from ZZX_DRAFT1 import run_grid_backtest as run_changed

# =====================================================================
# PARÁMETROS
# =====================================================================
SELF_AFTER = 2
INITIAL_BALANCE = 10000
ORDER_AMOUNT = 100
COMI_PCT = 0.1
TP_PCT = 3.0
SL_PCT = 2.0

# =====================================================================
# =====================================================================
# DATOS CONTROLADOS PARA TP/SL INTRABAR
# =====================================================================
import pandas as pd
import numpy as np

dates = pd.date_range("2024-01-01", periods=6, freq='D')

# Precios por vela
close = [100, 100, 100, 100, 100, 100]
high  = [100, 100, 100, 100, 104, 100]  
low   = [100, 100, 100, 100, 97, 100]       

signal = [1, 0, 0, 0, 0, 0]

# Timestamps intrabarra simulados (dentro de la misma vela)
dates = pd.date_range("2024-01-01", periods=6, freq='D')

# Hardcodeamos intrabar con horas distintas
high_time = np.array([
    dates[0] + pd.Timedelta(hours=10),
    dates[1] + pd.Timedelta(hours=11),
    dates[2] + pd.Timedelta(hours=11),  # TP
    dates[3] + pd.Timedelta(hours=9),
    dates[4] + pd.Timedelta(hours=14),
    dates[5] + pd.Timedelta(hours=15)
], dtype='datetime64[ns]')

low_time = np.array([
    dates[0] + pd.Timedelta(hours=9),
    dates[1] + pd.Timedelta(hours=10),
    dates[2] + pd.Timedelta(hours=12),  # SL antes de TP
    dates[3] + pd.Timedelta(hours=11),
    dates[4] + pd.Timedelta(hours=10),
    dates[5] + pd.Timedelta(hours=14)
], dtype='datetime64[ns]')

ohlcv_arrays = {
    "SYM_TEST": {
        'ts': dates.to_numpy(),  # solo fechas de referencia
        'close': np.array([100, 100, 100, 107, 100, 100], dtype=np.float64),
        'high': np.array([100, 100, 100, 100, 104, 100], dtype=np.float64),
        'low':  np.array([100, 100, 100, 100, 97, 100], dtype=np.float64),
        'signal': np.array([1, 0, 0, 0, 0, 0], dtype=int),
        'high_time': high_time,
        'low_time': low_time
    }
}


df = pd.DataFrame({
    "close": close,
    "high": high,
    "low": low,
    "signal": signal,
    "high_time": high_time,
    "low_time": low_time
}, index=dates)




# =====================================================================
# FUNCIÓN MANUAL TP/SL INTRABAR
# =====================================================================
def generate_manual_trades(ohlcv_arrays):
    trades = []
    for symbol, data in ohlcv_arrays.items():
        close, high, low, ts, signals = data['close'], data['high'], data['low'], data['ts'], data['signal']
        high_time, low_time = data['high_time'], data['low_time']
        n = len(close)

        for i, sig in enumerate(signals):
            if sig != 1 or i + SELF_AFTER >= n:
                continue

            buy_price = close[i]
            units = ORDER_AMOUNT / buy_price
            tp_price = buy_price * (1 + TP_PCT / 100)
            sl_price = buy_price * (1 - SL_PCT / 100)
            sell_idx = i + SELF_AFTER
            sell_price = close[sell_idx]
            exec_time = ts[sell_idx]

            # --- Revisar intrabar TP/SL usando lógica robusta ---
            intravela_detected = False
            chosen_idx = None
            exit_reason = None

            for j in range(i + 1, sell_idx + 1):
                tp_hit = high[j] >= tp_price
                sl_hit = low[j] <= sl_price

                if tp_hit or sl_hit:
                    if tp_hit and sl_hit:
                        # Ambos ocurren en la misma vela → usar timestamps
                        if high_time[j] <= low_time[j]:
                            chosen_idx = j
                            sell_price = tp_price
                            exec_time = high_time[j]
                            exit_reason = 'TP'
                        else:
                            chosen_idx = j
                            sell_price = sl_price
                            exec_time = low_time[j]
                            exit_reason = 'SL'
                    elif tp_hit:
                        chosen_idx = j
                        sell_price = tp_price
                        exec_time = high_time[j]
                        exit_reason = 'TP'
                    elif sl_hit:
                        chosen_idx = j
                        sell_price = sl_price
                        exec_time = low_time[j]
                        exit_reason = 'SL'

                    sell_idx = chosen_idx
                    intravela_detected = True
                    break  # Se ejecuta la salida intrabar, salir del bucle

            # Calcular profit neto
            profit = units * (sell_price - buy_price) - buy_price*units*COMI_PCT/100 - sell_price*units*COMI_PCT/100

            trades.append({
                "symbol": symbol,
                "buy_time": ts[i],
                "sell_time": exec_time,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "profit": profit,
                "exit_reason": exit_reason
            })

    return pd.DataFrame(trades)



# =====================================================================
# FUNCIÓN PARA BALANCE Y MÉTRICAS MANUAL
# =====================================================================
def compute_manual_metrics(trade_log, initial_balance=INITIAL_BALANCE):
    if trade_log.empty:
        return initial_balance, 0, 0, np.nan, 0.0
    sim_balance = trade_log['profit'].cumsum() + initial_balance
    balance_final = sim_balance.iloc[-1]
    num_trades = len(trade_log)
    prop_winners = (trade_log['profit'] > 0).mean()
    returns = sim_balance.pct_change().fillna(0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    max_dd = (sim_balance.cummax() - sim_balance).max()
    return balance_final, num_trades, prop_winners, sharpe, max_dd

# =====================================================================
# UTILIDAD PARA GENERAR SIM_BALANCE_HISTORY
# =====================================================================
def ensure_balance_history(results, initial_balance=INITIAL_BALANCE):
    portfolio = results['__PORTFOLIO__']
    if 'sim_balance_history' not in portfolio:
        timestamps, balances = [], []
        balance = initial_balance
        trades = portfolio['trade_log'].sort_values('sell_time')
        for _, trade in trades.iterrows():
            timestamps.append(trade['sell_time'])
            balance += trade['profit']
            balances.append(balance)
        portfolio['sim_balance_history'] = {"timestamp": timestamps, "balance": balances}
    return results

# =====================================================================
# MÉTRICAS AUTOMÁTICAS CON RIESGO
# =====================================================================
def results_to_df(results):
    records = []
    trade_log = results['__PORTFOLIO__']['trade_log']
    for _, row in trade_log.iterrows():
        records.append({'Net_Gain': row['profit'], 'Num_Signals': 1, 'Num_Wins': 1 if row['profit'] > 0 else 0})
    return pd.DataFrame(records)

def compute_auto_metrics_with_risk(grid_df, initial_capital=INITIAL_BALANCE):
    df = grid_df.copy()
    df["Balance"] = initial_capital + df["Net_Gain"].cumsum()
    balance_final = df["Balance"].iloc[-1]
    num_trades = len(df)
    win_ratio = (df["Num_Wins"] / df["Num_Signals"]).mean()
    returns = df["Balance"].pct_change().fillna(0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    max_dd = (df["Balance"].cummax() - df["Balance"]).max()
    return balance_final, num_trades, win_ratio, sharpe, max_dd

# =====================================================================
# EJECUCIÓN DEL TEST
# =====================================================================
manual_trades = generate_manual_trades(ohlcv_arrays)
manual_balance_final, manual_num, manual_win, manual_sharpe, manual_dd = compute_manual_metrics(manual_trades)

results_orig = ensure_balance_history(run_original(
    ohlcv_arrays, SELF_AFTER, TP_PCT, SL_PCT, INITIAL_BALANCE, ORDER_AMOUNT, COMI_PCT))
results_changed = ensure_balance_history(run_changed(
    ohlcv_arrays, SELF_AFTER, TP_PCT, SL_PCT, INITIAL_BALANCE, ORDER_AMOUNT, COMI_PCT))

grid_results_df = results_to_df(results_orig)
grid_changed_df = results_to_df(results_changed)
auto_balance_final, auto_num, auto_win, auto_sharpe, auto_dd = compute_auto_metrics_with_risk(grid_results_df)
changed_balance_final, changed_num, changed_win, changed_sharpe, changed_dd = compute_auto_metrics_with_risk(grid_changed_df)

metrics_table = pd.DataFrame({
    'Métrica': ['Balance final', 'Número de trades', 'Proporción ganadores', 'Sharpe', 'Max drawdown'],
    'Manual': [manual_balance_final, manual_num, manual_win, manual_sharpe, manual_dd],
    'Original': [auto_balance_final, auto_num, auto_win, auto_sharpe, auto_dd],
    'Cambiada': [changed_balance_final, changed_num, changed_win, changed_sharpe, changed_dd]
})

print("\n📊 Comparación de métricas: Manual vs Original vs Cambiada")
print(metrics_table)
