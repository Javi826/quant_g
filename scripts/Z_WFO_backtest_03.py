# === FILE: Z_WFO_backtest.py ===
# ---------------------------------
import os
import time
import numpy as np
import pandas as pd
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_results, save_filtered_symbols
from Z_add_signals_03 import add_indicators_03, explosive_signal_03
#from ZZX_DRAFT2 import walk_forward_optimization
from Z_WFO import walk_forward_optimization

start_time   = time.time()
SAVE_SYMBOLS = False
STRATEGY     = "entropy"
N_JOBS       = 1
ANCHORED     = True 

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2023_UPTO"
DATE_MIN            = "2025-01-03"
TIMEFRAME           = '4H'
MIN_VOL_USDT        = 50_000

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------

SELL_AFTER_LIST     = [10,15,20,25]
ENTROPY_MAX_LIST    = [0.6,0.8,1.0,1.2,1.4,1.6]
ACCEL_SPAN_LIST     = [10,15,20]

TP_PCT_LIST         = [0,5,10]
SL_PCT_LIST         = [0,5,10]

# =============================================================================
# =============================================================================
# SELL_AFTER_LIST    = [28,50]
# ENTROPY_MAX_LIST   = [0.8,1.0]
# ACCEL_SPAN_LIST    = [8,10]
# 
# TP_PCT_LIST        = [0,5]
# SL_PCT_LIST        = [3,5]
# =============================================================================
# =============================================================================

param_names  = ['SELL_AFTER', 'ENTROPY_MAX', 'ACCEL_SPAN', 'TP_PCT', 'SL_PCT']
param_ranges = {name: globals()[f"{name}_LIST"] for name in param_names}

# -----------------------------------------------------------------------------
# DATA LOADING AND FILTERING
# -----------------------------------------------------------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]

ohlcv_data, filtered_symbols = filter_symbols(
    symbols,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50,
    date_min=DATE_MIN
)

save_filtered_symbols(filtered_symbols, strategy=STRATEGY, timeframe=TIMEFRAME, save_symbols=SAVE_SYMBOLS)

# -----------------------------------------------------------------------------
# UNIFIED ARRAYS: ohlcv_arr
# -----------------------------------------------------------------------------
ohlcv_arr = {}
for sym, df in ohlcv_data.items():
    ohlcv_arr[sym] = {
        'ts': df.index.values.astype('datetime64[ns]'),
        'open': df['open'].to_numpy(dtype=np.float64),
        'high': df['high'].to_numpy(dtype=np.float64),
        'low': df['low'].to_numpy(dtype=np.float64),
        'close': df['close'].to_numpy(dtype=np.float64),
        'volume_quote': df['volume_quote'].to_numpy(dtype=np.float64) if 'volume_quote' in df.columns else np.zeros(len(df))
    }

# -----------------------------------------------------------------------------
# 🚀 1️⃣ WALK-FORWARD OPTIMIZATION
# -----------------------------------------------------------------------------
best_params_wfo = walk_forward_optimization(
    data_dict=ohlcv_arr,
    param_ranges=param_ranges,
    length_train_set=1750,
    pct_train_set=0.8,
    anchored=ANCHORED,
    com_ewm=0.5,
    n_jobs=N_JOBS  
)

# Redondear parámetros numéricos
for name in param_names:
    val = best_params_wfo.get(name)
    if isinstance(val, (int, float)) and not str(name).endswith("_MAX"):
        best_params_wfo[name] = int(round(val))

# -----------------------------------------------------------------------------
# 🧪 2️⃣ FINAL BACKTEST CON PARÁMETROS ÓPTIMOS
# -----------------------------------------------------------------------------
ohlcv_arrays = {}
for sym, arrs in ohlcv_arr.items():
    entropia, accel = add_indicators_03(arrs['close'], m_accel=best_params_wfo['ACCEL_SPAN'])
    signal = explosive_signal_03(entropia, accel, entropia_max=best_params_wfo['ENTROPY_MAX'], live=False)
    ohlcv_arrays[sym] = {**arrs, 'signal': signal}

final_results = run_grid_backtest(
    ohlcv_arrays,
    sell_after=best_params_wfo['SELL_AFTER'],
    tp_pct=best_params_wfo['TP_PCT'],
    sl_pct=best_params_wfo['SL_PCT'],
    initial_balance=INITIAL_BALANCE,
    order_amount=ORDER_AMOUNT,
    comi_pct=0.05
)

# -----------------------------------------------------------------------------
# PROCESAR RESULTADOS
# -----------------------------------------------------------------------------
port = final_results.get("__PORTFOLIO__", {})
net_gain = np.sum(port.get('trades', [])) if len(port.get('trades', [])) > 0 else 0.0
net_gain_pct = (net_gain / INITIAL_BALANCE) * 100.0 if INITIAL_BALANCE != 0 else np.nan
dd_pct = port.get('max_dd', 0.0) * 100.0
sharpe_ratio = float(port.get('sharpe', np.nan))
num_signals = int(port.get('num_signals', 0))

summary = {
    **best_params_wfo,
    "symbol": "__PORTFOLIO__",
    "Net_Gain": float(net_gain),
    "Net_Gain_pct": float(net_gain_pct),
    "Final_Balance": float(port.get('final_balance', INITIAL_BALANCE)),
    "DD_pct": float(dd_pct),
    "Sharpe": float(sharpe_ratio),
    "Num_Signals": num_signals
}

grid_results_df = pd.DataFrame([summary])

# -----------------------------------------------------------------------------
# PRINT EXECUTION INFO
# -----------------------------------------------------------------------------
print('\n🔹EXECUTION      :')
print(f"TIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"DATE_MIN         : {DATE_MIN}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")
print(f"ENTROPY_MAX_LIST = {ENTROPY_MAX_LIST}")
print(f"ACCEL_SPAN_LIST  = {ACCEL_SPAN_LIST}")
print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}\n")

max_len = max(len(k) for k in param_names) + len("_LIST")
print('🎯 BEST WFO PARAMS:')
for k in param_names:
    v = best_params_wfo.get(k)
    print(f"{k + '_LIST':<{max_len}} = [{v}]")

# -----------------------------------------------------------------------------
# 💾 SAVE RESULTS
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'), grid_results_df, filename=f"WFO_backtest_{DATA_FOLDER}_{TIMEFRAME}.xlsx", save=False)
df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
