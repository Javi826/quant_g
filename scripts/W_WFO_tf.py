# Z_WFO_backtest_multi_tf.py (MAIN - Multi-Timeframe Strategy)
import os
import time
import numpy as np
import pandas as pd
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, final_prints
from tools.ZX_WFO_tf import walk_forward_optimization_tf
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from Z_add_signals_tf import explosive_signal_tf
#from Z_add_signals_tf import explosive_signal_tf_short

start_time        = time.time()
N_JOBS            = -1
STRATEGY          = "trends_tf"
# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER       = "data/crypto_2023_IS"
TIMEFRAME_MAJOR   = '1Dutc'
TIMEFRAME_MINOR   = '4H'
ORDER_AMOUNT      = 5_000
MIN_VOL_USDT      = 10_000_000

# -----------------------------------------------------------------------------
# WFO SETTINGS
# -----------------------------------------------------------------------------
ANCHORED          = True
YEARS_TRAIN       = 1.0

# Calculamos LENGTH_TRAIN_SET según el timeframe MENOR
if TIMEFRAME_MINOR == '1H':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 24)
elif TIMEFRAME_MINOR == '4H':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 6)
elif TIMEFRAME_MINOR == '6Hutc':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 4)
elif TIMEFRAME_MINOR == '12Hutc':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 2)
elif TIMEFRAME_MINOR == '1Dutc':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365)

# -----------------------------------------------------------------------------
# GRID 
# -----------------------------------------------------------------------------
SELL_AFTER_LIST     = [0]
LOOKBACK_MAJOR_LIST = [1,2,3,4]      
LOOKBACK_MINOR_LIST = [1,2,3,4] 

TP_PCT_LIST         = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10]
SL_PCT_LIST         = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10]

# =============================================================================
# SELL_AFTER_LIST     = [0]
# LOOKBACK_MAJOR_LIST = [1]      
# LOOKBACK_MINOR_LIST = [1] 
# 
# TP_PCT_LIST         = [1.5]
# SL_PCT_LIST         = [1.5]
# =============================================================================

param_names  = ['SELL_AFTER', 'LOOKBACK_MAJOR', 'LOOKBACK_MINOR', 'TP_PCT', 'SL_PCT']
param_ranges = {name: globals()[f"{name}_LIST"] for name in param_names}

# -----------------------------------------------------------------------------
# CARGA Y FILTRADO DE DATOS
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
symbols_major = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MAJOR}.parquet")]

ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50)
ohlcv_data_major, filtered_major = filter_symbols(symbols_major, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MAJOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50)
common_symbols  = list(set(filtered_minor).intersection(filtered_major))

ohlcv_data_minor = {s: ohlcv_data_minor[s] for s in common_symbols}
ohlcv_data_major = {s: ohlcv_data_major[s] for s in common_symbols}
ohlcv_arr_minor  = prepare_ohlcv_arrays(ohlcv_data_minor)
ohlcv_arr_major  = prepare_ohlcv_arrays(ohlcv_data_major)

print(f"✅ {len(common_symbols)} common in both TIMEFRAMES")

# -----------------------------------------------------------------------------
# FUNCIONES PARA WFO
# -----------------------------------------------------------------------------
def strategy_builder(params, base_arrays_minor, base_arrays_major):

    ohlcv_arrays = {}
    
    for sym in base_arrays_minor.keys():
        if sym not in base_arrays_major:
            continue
            
        arr_minor = base_arrays_minor[sym]
        arr_major = base_arrays_major[sym]
        
        signal = explosive_signal_tf(
            high_mayor=arr_major['high'],
            close_mayor=arr_major['close'],
            high_menor=arr_minor['high'],
            close_menor=arr_minor['close'],
            lookback_mayor=params.get('LOOKBACK_MAJOR'),
            lookback_menor=params.get('LOOKBACK_MINOR'),
            live=False
        )
        
        ohlcv_arrays[sym] = {**arr_minor, 'signal': signal}
    
    return ohlcv_arrays

def backtest_runner_default(ohlcv_arrays, params):

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params.get('SELL_AFTER'),
        tp_pct=params.get('TP_PCT'),
        sl_pct=params.get('SL_PCT'),
        order_amount=ORDER_AMOUNT
    )
    results["__PORTFOLIO__"]["initial_balance"] = INITIAL_BALANCE
    return results

def evaluate_fn(params, base_arrays):

    base_arrays_minor, base_arrays_major = base_arrays   
    ohlcv_arrays = strategy_builder(params, base_arrays_minor, base_arrays_major)
    results      = backtest_runner_default(ohlcv_arrays, params)
    
    return metric_fn_default(results), params

def metric_fn_default(results):

    port            = results.get("__PORTFOLIO__", {}) 
    sharpe_ratio    = float(port.get('sharpe'))
    net_gain        = np.sum(port.get('trades'))
    net_gain_pct    = (net_gain / INITIAL_BALANCE) * 100.0 
    dd_pct          = float(port.get('max_dd')) * 100.0
    
    metric_score    = (net_gain_pct - 1*dd_pct)
    metric_score    = net_gain_pct 
    #metric_score    = sharpe_ratio
    
    return metric_score

# -----------------------------------------------------------------------------
# WFO MULTI-TIMEFRAME
# -----------------------------------------------------------------------------
best_params_wfo = walk_forward_optimization_tf(
    ohlcv_arr_minor=ohlcv_arr_minor,
    ohlcv_arr_major=ohlcv_arr_major,
    param_ranges=param_ranges,
    evaluate_fn=evaluate_fn,
    length_train_set=LENGTH_TRAIN_SET,
    pct_train_set=0.8,
    anchored=ANCHORED,
    n_jobs=N_JOBS
)

# ROUNDING PARAMETERS
for name in param_names:
    val = best_params_wfo.get(name)
    if isinstance(val, (int, float)) and not str(name).endswith("_MAX"):
        best_params_wfo[name] = int(round(val))
# -----------------------------------------------------------------------------
# BACKTESTING CON MEJORES PARÁMETROS
# -----------------------------------------------------------------------------
ohlcv_arrays = {}
for sym in ohlcv_arr_minor.keys():
    if sym not in ohlcv_arr_major:
        continue
        
    arr_minor = ohlcv_arr_minor[sym]
    arr_major = ohlcv_arr_major[sym]
    
    signal = explosive_signal_tf(
        high_mayor=arr_major['high'],
        close_mayor=arr_major['close'],
        high_menor=arr_minor['high'],
        close_menor=arr_minor['close'],
        lookback_mayor=best_params_wfo['LOOKBACK_MAJOR'],
        lookback_menor=best_params_wfo['LOOKBACK_MINOR'],
        live=False
    )
    
    ohlcv_arrays[sym] = {**arr_minor, 'signal': signal}

final_results = run_grid_backtest(
    ohlcv_arrays,
    sell_after=best_params_wfo['SELL_AFTER'],
    tp_pct=best_params_wfo['TP_PCT'],
    sl_pct=best_params_wfo['SL_PCT'],
    order_amount=ORDER_AMOUNT
)

# -----------------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------------
param_values_tuple = tuple(best_params_wfo[name] for name in param_names)
grid_results_list  = [(param_values_tuple, final_results)]
grid_records       = compile_grid_results(grid_results_list, param_names, INITIAL_BALANCE)
grid_results_df    = pd.DataFrame(grid_records)

final_prints(f"🎯 WFO_{STRATEGY} 🎯", DATA_FOLDER, f"{TIMEFRAME_MAJOR}/{TIMEFRAME_MINOR}", MIN_VOL_USDT, ORDER_AMOUNT, param_names, [param_ranges[name] for name in param_names])
df_portfolio, mi_series = report_backtesting(grid_results_df, param_names, DATA_FOLDER, INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")