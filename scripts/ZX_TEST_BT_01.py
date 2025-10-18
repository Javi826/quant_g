import os
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
import numpy as np
import pandas as pd
from datetime import datetime

# Importar la implementación NumPy
from ZX_compute_BT import run_grid_backtest as run_original
from ZZX_DRAFT5 import run_grid_backtest as run_changed
SELF_AFTER =5
# =====================================================================
# DATOS EJEMPLO
# =====================================================================
def create_controlled_test():
    # ---------------------
    # Símbolo 1 (20 días)
    # ---------------------
    dates1 = pd.to_datetime([
        "2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05",
        "2024-01-06","2024-01-07","2024-01-08","2024-01-09","2024-01-10",
        "2024-01-11","2024-01-12","2024-01-13","2024-01-14","2024-01-15"
    ])
    close1 = list(range(100, 115))
    high1 = [c+1 for c in close1]
    low1 = [c-1 for c in close1]
    signal1 = [1,1,1,0,0,0,0,0,1,0,0,0,0,0,0]
    df1 = pd.DataFrame({"close": close1, "high": high1, "low": low1, "signal": signal1}, index=dates1)

    # ---------------------
    # Símbolo 2 (20 días)
    # ---------------------
    close2 = list(range(200, 215))
    high2 = [c+1 for c in close2]
    low2 = [c-1 for c in close2]
    signal2 = [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0]
    df2 = pd.DataFrame({"close": close2, "high": high2, "low": low2, "signal": signal2}, index=dates1)

    # ---------------------
    # Símbolo 3 (20 días)
    # ---------------------
    close3 = list(range(300, 315))
    high3 = [c+1 for c in close3]
    low3 = [c-1 for c in close3]
    signal3 = [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0]
    df3 = pd.DataFrame({"close": close3, "high": high3, "low": low3, "signal": signal3}, index=dates1)

    # ---------------------
    # Símbolo 4 (15 días, distinto inicio)
    # ---------------------
    dates4 = pd.to_datetime([
        "2024-01-05","2024-01-06","2024-01-07","2024-01-08","2024-01-09",
        "2024-01-10","2024-01-11","2024-01-12","2024-01-13","2024-01-14",
        "2024-01-15","2024-01-16","2024-01-17","2024-01-18","2024-01-19"
    ])
    close4 = list(range(400, 415))
    high4 = [c+1 for c in close4]
    low4 = [c-1 for c in close4]
    signal4 = [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    df4 = pd.DataFrame({"close": close4, "high": high4, "low": low4, "signal": signal4}, index=dates4)

    # ---------------------
    # Símbolo 5 (15 días, empieza como SYM1/SYM2/SYM3 pero termina antes)
    # ---------------------
    dates5 = pd.to_datetime([
        "2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05",
        "2024-01-06","2024-01-07","2024-01-08","2024-01-09","2024-01-10",
        "2024-01-11","2024-01-12","2024-01-13","2024-01-14","2024-01-15"
    ])
    close5 = list(range(500, 515))
    high5 = [c+1 for c in close5]
    low5 = [c-1 for c in close5]
    signal5 = [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0]
    df5 = pd.DataFrame({"close": close5, "high": high5, "low": low5, "signal": signal5}, index=dates5)

    return {"SYM1": df1, "SYM2": df2, "SYM3": df3, "SYM4": df4, "SYM5": df5}


# =====================================================================
# UTILIDADES PARA CONVERSIÓN DE DATOS
# =====================================================================
def df_to_ohlcv_arrays(dfs_dict):
    ohlcv_arrays = {}
    for symbol, df in dfs_dict.items():
        ohlcv_arrays[symbol] = {
            'ts': df.index.to_numpy().astype('datetime64[ns]'),
            'close': df['close'].to_numpy(),
            'high': df['high'].to_numpy(),
            'low': df['low'].to_numpy(),
            'signal': df['signal'].to_numpy(),
        }
    return ohlcv_arrays

# =====================================================================
# FUNCIÓN PARA GENERAR SIM_BALANCE_HISTORY
# =====================================================================
def ensure_balance_history(results, initial_balance=10000):
    portfolio = results['__PORTFOLIO__']
    if 'sim_balance_history' not in portfolio:
        timestamps = []
        balances = []
        balance = initial_balance
        trades = portfolio['trade_log'].sort_values('sell_time')
        for _, trade in trades.iterrows():
            timestamps.append(trade['sell_time'])
            balance += trade['profit']
            balances.append(balance)
        portfolio['sim_balance_history'] = {
            "timestamp": timestamps,
            "balance": balances
        }
    return results

# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    dfs_dict = create_controlled_test()
    ohlcv_arrays = df_to_ohlcv_arrays(dfs_dict)

    print("\n🚀 Ejecutando backtest con versión ORIGINAL...")
    results_orig = run_original(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=SELF_AFTER,
        tp_pct=3.0,
        sl_pct=2.0,
        initial_balance=10000,
        order_amount=1000,
        comi_pct=0.1
    )
    results_orig = ensure_balance_history(results_orig)

    print("\n🚀 Ejecutando backtest con versión CAMBIADA (1 trade a la vez)...")
    results_changed = run_changed(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=SELF_AFTER,
        tp_pct=3.0,
        sl_pct=2.0,
        initial_balance=10000,
        order_amount=1000,
        comi_pct=0.1
    )
    results_changed = ensure_balance_history(results_changed)

    # Comparar trade_logs
    trade_log_orig = results_orig['__PORTFOLIO__']['trade_log']
    trade_log_changed = results_changed['__PORTFOLIO__']['trade_log']

    print("\n✅ Trades ORIGINAL:")
    print(trade_log_orig)
    print(f"💰 Balance final ORIGINAL: {results_orig['__PORTFOLIO__']['sim_balance_history']['balance'][-1]:.2f}")
    print(f"📈 Total trades ejecutados ORIGINAL: {len(trade_log_orig)}")

    print("\n✅ Trades CAMBIADO (1 trade a la vez):")
    print(trade_log_changed)
    print(f"💰 Balance final CAMBIADO: {results_changed['__PORTFOLIO__']['sim_balance_history']['balance'][-1]:.2f}")
    print(f"📈 Total trades ejecutados CAMBIADO: {len(trade_log_changed)}")

    # Comparación rápida con pandas
    df_compare = pd.DataFrame({
        'symbol_orig': trade_log_orig['symbol'],
        'buy_time_orig': trade_log_orig['buy_time'],
        'sell_time_orig': trade_log_orig['sell_time'],
        'profit_orig': trade_log_orig['profit'],
        'symbol_changed': trade_log_changed['symbol'],
        'buy_time_changed': trade_log_changed['buy_time'],
        'sell_time_changed': trade_log_changed['sell_time'],
        'profit_changed': trade_log_changed['profit']
    })
    print("\n📊 Comparación de trades ORIGINAL vs CAMBIADO:")
    print(df_compare)

    import numpy as np
    import pandas as pd
    
    # =====================================================================
    # Comparación de métricas con columna de igualdad y número de trades
    # =====================================================================
    metrics_orig = results_orig['__PORTFOLIO__']
    metrics_changed = results_changed['__PORTFOLIO__']
    
    # Función de comparación con tolerancia
    def compare_values(a, b, tol=1e-8):
        if a is None or b is None:
            return False
        try:
            return np.isclose(a, b, atol=tol)
        except:
            return a == b
    
    num_trades_orig = len(trade_log_orig)
    num_trades_changed = len(trade_log_changed)
    
    metrics_table = pd.DataFrame({
        "Métrica": [
            "Balance final",
            "Proporción ganadores",
            "Sharpe",
            "Max drawdown"
        ],
        "Original": [
            metrics_orig.get('final_balance', metrics_orig['sim_balance_history']['balance'][-1]),
            metrics_orig.get('proportion_winners', trade_log_orig[trade_log_orig['profit'] > 0].shape[0]/num_trades_orig),
            metrics_orig.get('sharpe', np.nan),
            metrics_orig.get('max_dd', np.nan)
        ],
        "Cambiada": [
            metrics_changed.get('final_balance', metrics_changed['sim_balance_history']['balance'][-1]),
            metrics_changed.get('proportion_winners', trade_log_changed[trade_log_changed['profit'] > 0].shape[0]/num_trades_changed),
            metrics_changed.get('sharpe', np.nan),
            metrics_changed.get('max_dd', np.nan)
        ],
        "Trades Original": [num_trades_orig]*4,
        "Trades Cambiada": [num_trades_changed]*4
    })
    
    # Columna para ver si los valores son iguales
    metrics_table["Igual"] = [
        compare_values(o, c) for o, c in zip(metrics_table["Original"], metrics_table["Cambiada"])
    ]
    
    print("\n📊 Comparación de métricas entre versiones con verificación de igualdad y número de trades:")
    print(metrics_table)



