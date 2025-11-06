# Z_WFO_backtest_parallel_adapted.py (MAIN)
import os
import time
import numpy as np
import pandas as pd
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols,final_prints
from tools.ZX_WFO import walk_forward_optimization
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results
from ZX_compute_BT import run_grid_backtest, MIN_PRICE,INITIAL_BALANCE
from Z_add_signals_en import explosive_signal_99 

start_time = time.time()
STRATEGY            ="candle_pair"
N_JOBS              = -1

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2023_IS"
TIMEFRAME           = '4H'
ORDER_AMOUNT        = 5000
MIN_VOL_USDT        = 10_000_000

# -----------------------------------------------------------------------------
# WFO SETTINGS
# -----------------------------------------------------------------------------
ANCHORED            = True
YEARS_TRAIN         = 1.0  
if TIMEFRAME == '1H':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 24)   
elif TIMEFRAME == '4H':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 6)  
elif TIMEFRAME == '6Hutc':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 4)    
elif TIMEFRAME == '12Hutc':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365 * 2)   
elif TIMEFRAME == '1Dutc':
    LENGTH_TRAIN_SET = int(YEARS_TRAIN * 365)        

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------

SELL_AFTER_LIST    = [0]
ENTROPY_MAX_LIST   = [0.2,0.4,0.6,0.8,1.0,1.2,1.4]
ACCEL_SPAN_LIST    = [5,10,15,20,25,30,35]

TP_PCT_LIST        = [5,10,15,20,25]
SL_PCT_LIST        = [5,10,15,20,25]


param_names    = ['SELL_AFTER', 'ENTROPY_MAX', 'ACCEL_SPAN', 'TP_PCT', 'SL_PCT']
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
        
        signal = explosive_signal_99(arrs['close'],
                                     m_accel=params.get('ACCEL_SPAN'),
                                     entropia_max=params.get('ENTROPY_MAX'),
                                     live=False)
        
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}
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
    ohlcv_arrays = strategy_builder(params, base_arrays)
    results      = backtest_runner_default(ohlcv_arrays, params)
    return metric_fn_default(results), params

def metric_fn_default(results):
    port            = results.get("__PORTFOLIO__", {}) 
    sharpe_ratio    = float(port.get('sharpe'))
    net_gain        = np.sum(port.get('trades', []))
    net_gain_pct    = (net_gain / INITIAL_BALANCE) * 100.0 
    dd_pct = float(port.get('max_dd', 0.0)) * 100.0
    
    metric_score    = (net_gain_pct - 2*dd_pct)
    #metric_score    = sharpe_ratio
   
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
# BACKTESTING WITH BEST PARAMS (coherente con el WFO)
# -----------------------------------------------------------------------------
ohlcv_arrays = {}
for sym, arrs in ohlcv_arr.items():
    # Construir la señal con los mejores parámetros encontrados por el WFO
    signal = explosive_signal_99(
        arrs['close'],
        m_accel=best_params_wfo['ACCEL_SPAN'],
        entropia_max=best_params_wfo['ENTROPY_MAX'],
        live=False
    )
    
    ohlcv_arrays[sym] = {**arrs, 'signal': signal}

# Ejecutar el backtest con esos parámetros
final_results = run_grid_backtest(
    ohlcv_arrays,
    sell_after=best_params_wfo['SELL_AFTER'],
    tp_pct=best_params_wfo['TP_PCT'],
    sl_pct=best_params_wfo['SL_PCT'],
    order_amount=ORDER_AMOUNT
)


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

final_prints(strategy=f"🎯 WFO_{STRATEGY} 🎯", data_folder=DATA_FOLDER, timeframe=TIMEFRAME, min_vol_usdt=MIN_VOL_USDT, order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=[param_ranges[name] for name in param_names])

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names,data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
