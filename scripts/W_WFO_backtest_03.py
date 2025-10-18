# Z_WFO_backtest_parallel.py (MAIN)
import os
import time
import numpy as np
import pandas as pd
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols
from Z_add_signals_03 import add_indicators_03, explosive_signal_03
from tools.ZX_WFO import walk_forward_optimization
from tools.ZX_st_tools import prepare_ohlcv_arrays,compile_grid_results
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT,COMISION
start_time = time.time()

# -----------------------------
# WFO SETTINGS
# -----------------------------
ANCHORED            = True
LENGTH_TRAIN_SET    = 400
APROXIMATION        = 1.5
N_JOBS              = -1 
# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2023_ISOLD"
TIMEFRAME           = '4H'
MIN_VOL_USDT        = 50_000

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------
SELL_AFTER_LIST     = [1,5,10,15,20,25,30,35]
ENTROPY_MAX_LIST    = [0.2,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6]
ACCEL_SPAN_LIST     = [5,10,12,15,17,20]

TP_PCT_LIST         = [0,5,10]
SL_PCT_LIST         = [0,5,10]
# =============================================================================
# =============================================================================
# SELL_AFTER_LIST    = [20,25]
# ENTROPY_MAX_LIST   = [0.4,0.6]
# ACCEL_SPAN_LIST    = [5,10]
# 
# TP_PCT_LIST        = [0,5]
# SL_PCT_LIST        = [0,5]
# =============================================================================
# =============================================================================

param_names  = ['SELL_AFTER', 'ENTROPY_MAX', 'ACCEL_SPAN', 'TP_PCT', 'SL_PCT']
param_ranges = {name: globals()[f"{name}_LIST"] for name in param_names}

# -----------------------------------------------------------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]
ohlcv_data, filtered_symbols = filter_symbols(
    symbols, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME,
    data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50)

ohlcv_arr = prepare_ohlcv_arrays(ohlcv_data)

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

def strategy_builder_entropy(params, base_arrays):
    ohlcv_arrays = {}
    for sym, arrs in base_arrays.items():
        entropia, accel   = add_indicators_03(arrs['close'], m_accel=params.get('ACCEL_SPAN', 5))
        signal            = explosive_signal_03(entropia, accel, entropia_max=params.get('ENTROPY_MAX', 1.0), live=False)
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}
    return ohlcv_arrays

def backtest_runner_default(ohlcv_arrays, params):
    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params.get('SELL_AFTER'),
        tp_pct=params.get('TP_PCT'),
        sl_pct=params.get('SL_PCT'),
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        comi_pct=COMISION
    )
    results["__PORTFOLIO__"]["initial_balance"] = INITIAL_BALANCE
    return results

def evaluate_fn(params, base_arrays):
    
    ohlcv_arrays = strategy_builder_entropy(params, base_arrays)
    results      = backtest_runner_default(ohlcv_arrays, params)
    
    return metric_fn_default(results), params

def metric_fn_default(results):
    
    port            = results.get("__PORTFOLIO__", {})   
    sharpe_ratio    = float(port.get('sharpe', np.nan))
    metric_score    = sharpe_ratio
    return metric_score

# -----------------------------------------------------------------------------
# WFO
# -----------------------------------------------------------------------------
best_params_wfo = walk_forward_optimization(
    ohlcv_arr=ohlcv_arr,
    param_ranges=param_ranges,
    evaluate_fn=evaluate_fn,
    length_train_set=LENGTH_TRAIN_SET,
    pct_train_set=0.8,
    anchored=ANCHORED
)

for name in param_names:
    val = best_params_wfo.get(name)
    if isinstance(val, (int, float)) and not str(name).endswith("_MAX"):
        best_params_wfo[name] = int(round(val))

# -----------------------------------------------------------------------------
# BACKTESTING WITH BEST PARAMS
# -----------------------------------------------------------------------------
ohlcv_arrays = {}
for sym, arrs in ohlcv_arr.items():
    entropia, accel   = add_indicators_03(arrs['close'], m_accel=best_params_wfo['ACCEL_SPAN'])
    signal            = explosive_signal_03(entropia, accel, entropia_max=best_params_wfo['ENTROPY_MAX'], live=False)
    ohlcv_arrays[sym] = {**arrs, 'signal': signal}

final_results = run_grid_backtest(
    ohlcv_arrays,
    sell_after=best_params_wfo['SELL_AFTER'],
    tp_pct=best_params_wfo['TP_PCT'],
    sl_pct=best_params_wfo['SL_PCT'],
    initial_balance=INITIAL_BALANCE,
    order_amount=ORDER_AMOUNT,
    comi_pct=COMISION
)

# -----------------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------------
param_values_tuple = tuple(best_params_wfo[name] for name in param_names)
grid_results_list  = [(param_values_tuple, final_results)]
grid_records       = compile_grid_results(grid_results_list, param_names, INITIAL_BALANCE)
grid_results_df    = pd.DataFrame(grid_records)


print('\n🎯==WFO_backtest==🎯')
print(f"TIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"ANCHORED         : {ANCHORED}")
print(f"LENGTH_TRAIN_SET : {LENGTH_TRAIN_SET}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")
print(f"ENTROPY_MAX_LIST = {ENTROPY_MAX_LIST}")
print(f"ACCEL_SPAN_LIST  = {ACCEL_SPAN_LIST}")
print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}\n")

df_portfolio, mi_series = report_backtesting(df=grid_results_df,parameters=param_names,initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
