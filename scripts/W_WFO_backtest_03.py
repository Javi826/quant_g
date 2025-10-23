# Z_WFO_backtest_parallel_adapted.py (MAIN)
import os
import time
import numpy as np
import pandas as pd
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols,final_prints
from Z_add_signals_03 import explosive_signal_03  # solo usamos la nueva función
from tools.ZX_WFO import walk_forward_optimization
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results
from ZX_compute_BT import run_grid_backtest, MIN_PRICE,INITIAL_BALANCE

start_time = time.time()

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2023_IS"
TIMEFRAME           = '1H'
MIN_VOL_USDT        = 50_000
ORDER_AMOUNT        = 200
N_JOBS              = -1

# -----------------------------------------------------------------------------
# WFO SETTINGS
# -----------------------------------------------------------------------------
ANCHORED            = True
YEARS_TRAIN         = 1  
if TIMEFRAME == '1H':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 24)   
elif TIMEFRAME == '4H':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 6)    
elif TIMEFRAME == '1D':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365)        

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------
SELL_AFTER_LIST     = [0,5,10,15,20,25,30]
LOOKBACK_LIST       = [1,2,3,4,5,6,7]  
      
TP_PCT_LIST         = [0,5,10,15,20]
SL_PCT_LIST         = [5,10,15,20]

param_names = ['SELL_AFTER', 'LOOKBACK','TP_PCT', 'SL_PCT']
param_ranges = {name: globals()[f"{name}_LIST"] for name in param_names}

# -----------------------------------------------------------------------------
# CARGA Y FILTRADO DE DATOS
# -----------------------------------------------------------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]
ohlcv_data, filtered_symbols = filter_symbols(
    symbols, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME,
    data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50
)

ohlcv_arr = prepare_ohlcv_arrays(ohlcv_data)

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------
def strategy_builder(params, base_arrays):
    ohlcv_arrays = {}
    for sym, arrs in base_arrays.items():
        signal = explosive_signal_03(
            high=arrs['high'],       
            close=arrs['close'], 
            lookback=params.get('LOOKBACK'),
            live=False
        )
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}
    return ohlcv_arrays

def backtest_runner_default(ohlcv_arrays, params):
    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params.get('SELL_AFTER'),
        tp_pct=params.get('TP_PCT'),
        sl_pct=params.get('SL_PCT')
    )
    results["__PORTFOLIO__"]["initial_balance"] = INITIAL_BALANCE
    return results

def evaluate_fn(params, base_arrays):
    ohlcv_arrays = strategy_builder(params, base_arrays)
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
    signal = explosive_signal_03(
        high=arrs['high'],
        close=arrs['close'], 
        lookback=best_params_wfo['LOOKBACK'],
        live=False
    )
    ohlcv_arrays[sym] = {**arrs, 'signal': signal}

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

final_prints(strategy="🎯 WFO_backtest 🎯", data_folder=DATA_FOLDER, timeframe=TIMEFRAME, min_vol_usdt=MIN_VOL_USDT, order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=[param_ranges[name] for name in param_names])

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names,data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
