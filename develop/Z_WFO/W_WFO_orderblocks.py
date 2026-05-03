# Z_WFO_backtest_multi_tf.py (MAIN - Multi-Timeframe Strategy)
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "signals")))
import numpy as np
import pandas as pd
from shared.utils.analysis import report_backtesting
from shared.utils.utils import filter_symbols, final_prints
from shared.tools.wfo import walk_forward_optimization
from shared.utils.torque import prepare_ohlcv_arrays, compile_grid_results
from shared.backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from signals.add_signals_orderblocks import orderblocks_long
from signals.add_signals_orderblocks import orderblocks_short

start_time        = time.time()
N_JOBS            = -1
STRATEGY          = "orderblocks"
# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
SPLIT_MODE        = "expanding"
SPLIT_BASE        = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER       = os.path.join(SPLIT_BASE, "IS",  "crypto_2021-01_2025-04_IS")
DATA_FOLDER       = "../data/crypto_2026_OOS"
TIMEFRAME_MINOR   = '4H'
ORDER_AMOUNT      = 400
MIN_VOL_USDT      = 10_000_000

# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------
SELL_AFTER_LIST      = [0]  
LOOKBACK_LIST        = [50,100,150]
TOLERANCE_LIST       = [10,20,30,40] 
IMPULSE_LIST         = [0.01,0.1,1.0]

TP_PCT_LIST          = [2,3,4,5]
SL_PCT_LIST          = [6,7,8,9,10]

LOOKBACK_LIST        = [70]
TOLERANCE_LIST       = [20] 
IMPULSE_LIST         = [0.2]

TP_PCT_LIST          = [4]
SL_PCT_LIST          = [11]
param_names    = ['SELL_AFTER','LOOKBACK','TOLERANCE','IMPULSE','TP_PCT','SL_PCT']
param_ranges   = {name: globals()[f"{name}_LIST"] for name in param_names}

# -----------------------------------------------------------------------------
# WFO SETTINGS
# -----------------------------------------------------------------------------
ANCHORED          = False
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
# CARGA Y FILTRADO DE DATOS
# -----------------------------------------------------------------------------
symbols_minor    = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50)
ohlcv_arr_minor  = prepare_ohlcv_arrays(ohlcv_data_minor)

# -----------------------------------------------------------------------------
# FUNCIONES PARA WFO
# -----------------------------------------------------------------------------
def strategy_builder(params, base_arrays_minor):

    ohlcv_arrays = {}   
    for sym in base_arrays_minor.keys():         
        arr_minor = base_arrays_minor[sym]
      
        signals = orderblocks_short(
            arr=arr_minor,
            lookback=params.get('LOOKBACK'),
            tolerance=params.get('TOLERANCE'),
            impulse=params.get('IMPULSE'),
            live_trading=False
        )
      
        ohlcv_arrays[sym] = {**arr_minor, 'signal': signals}
    
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

    base_arrays_minor = base_arrays   
    ohlcv_arrays = strategy_builder(params, base_arrays_minor)
    results      = backtest_runner_default(ohlcv_arrays, params)
    
    return metric_fn_default(results), params

def metric_fn_default(results):

    port            = results.get("__PORTFOLIO__", {}) 
    sharpe_ratio    = float(port.get('sharpe'))
    net_gain        = np.sum(port.get('trades'))
    net_gain_pct    = (net_gain / INITIAL_BALANCE) * 100.0 
    dd_pct          = float(port.get('max_dd')) * 100.0
    
    metric_score    = net_gain_pct 
    #metric_score    = (net_gain_pct - 1*dd_pct)
    #metric_score    = sharpe_ratio
    
    return metric_score

# -----------------------------------------------------------------------------
# WFO MULTI-TIMEFRAME
# -----------------------------------------------------------------------------
best_params_wfo = walk_forward_optimization(
    ohlcv_arr=ohlcv_arr_minor,
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
        
    arr_minor = ohlcv_arr_minor[sym]
    
    signals = orderblocks_long(
        arr=arr_minor,
        lookback=best_params_wfo['LOOKBACK'],
        tolerance=best_params_wfo['TOLERANCE'],
        impulse=best_params_wfo['IMPULSE'],
        live_trading=False
    )
  
    ohlcv_arrays[sym] = {**arr_minor, 'signal': signals}

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

final_prints(f"🏃‍♂️ WFO_{STRATEGY} 🏃‍♂️", DATA_FOLDER, f"{TIMEFRAME_MINOR}", MIN_VOL_USDT, ORDER_AMOUNT, param_names, [param_ranges[name] for name in param_names])
df_portfolio, mi_series = report_backtesting(grid_results_df, param_names, DATA_FOLDER, INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")