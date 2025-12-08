# === FILE: main_MONTECARLO_functional_sharpe_no_cache_adapted.py ===
# -----------------------------------------------------------
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from utils.ZX_analysis import report_montecarlo
from utils.ZX_utils import filter_symbols, final_prints
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from tools.ZX_st_tools import extract_ohlcv_from_path, compile_MC_results,get_n_obs
from tools.ZX_optimize_MCf_tf import generate_multiple_paths, derive_major_from_minor
from Z_add_signals_trends import trends_tf_long
from Z_add_signals_trends import trends_tf_short

DTYPE             = np.float32
start_time        = time.time()
N_JOBS            = -1
STRATEGY          = "trends_tf"
# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER       = "data/crypto_2023_IS"
TIMEFRAME_MAJOR   = '1Dutc'
TIMEFRAME_MINOR   = '4H'
ORDER_AMOUNT      = 5_000
MIN_VOL_USDT      = 10_000_000

# -----------------------------------------------------------------------------
# MONTE CARLO SETTINGS
# -----------------------------------------------------------------------------
FINAL_N_PATHS = 50

if TIMEFRAME_MINOR == '1H':
    FINAL_N_OBS_PER_PATH = 4320
elif TIMEFRAME_MINOR == '4H':
    FINAL_N_OBS_PER_PATH = 1080
elif TIMEFRAME_MINOR == '6Hutc':
    FINAL_N_OBS_PER_PATH = 720
elif TIMEFRAME_MINOR == '12Hutc':
    FINAL_N_OBS_PER_PATH = 360
elif TIMEFRAME_MINOR == '1Dutc':
    FINAL_N_OBS_PER_PATH = 180

TS_INDEX = np.arange(FINAL_N_OBS_PER_PATH).astype('datetime64[ns]')

# -----------------------------------------------------------------------------
# GRID 
# -----------------------------------------------------------------------------
SELL_AFTER_LIST     = [0]  
LOOKBACK_MINOR_LIST = [1,2,3,4,5,6] 
N_CONSECUTIVE_LIST  = [1,2,3,4,5,6]
FACTOR_LIST         = [0.2,0.4,0.6,0.8,1.0]

TP_PCT_LIST         = [5,10,15,20,30]
SL_PCT_LIST         = [5,10,20]
# =============================================================================
# SELL_AFTER_LIST     = [0]
# LOOKBACK_MAJOR_LIST = [2]      
# LOOKBACK_MINOR_LIST = [1] 
# 
# TP_PCT_LIST         = [15]
# SL_PCT_LIST         = [30]
# =============================================================================

param_names     = ['SELL_AFTER','LOOKBACK_MINOR','N_CONSECUTIVE','FACTOR','TP_PCT','SL_PCT']
lists_for_grid  = [globals()[name + "_LIST"] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]

ohlcv_data_minor, filtered_minor = filter_symbols(
    symbols_minor,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME_MINOR,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50
)
# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def generate_paths_for_all_symbols_functional(ohlcv_data, n_paths, n_obs, raw_columns=[]):
    paths_per_symbol = {}
    for symbol, df_hist in ohlcv_data.items():
        arr_paths = generate_multiple_paths(df_hist, n_paths=n_paths, n_obs=n_obs, raw_columns=raw_columns)
        if arr_paths is not None and arr_paths.shape[0] > 0:
            paths_per_symbol[symbol] = arr_paths
    return paths_per_symbol

def process_path_IDX(path_idx, paths_minor, paths_major, param_dict_list):
    all_results = []
    for param_dict in param_dict_list:
        ohlcv_arrays_minor = extract_ohlcv_from_path(paths_minor, path_idx, dtype=DTYPE)
        ohlcv_arrays_major = extract_ohlcv_from_path(paths_major, path_idx, dtype=DTYPE)

        if len(ohlcv_arrays_minor) == 0 or len(ohlcv_arrays_major) == 0:
            continue

        for sym in ohlcv_arrays_minor.keys():

            arr_minor = ohlcv_arrays_minor[sym]
            arr_major = ohlcv_arrays_major[sym]
        
            arr_minor['ts'] = pd.date_range("2023-01-01", periods=len(arr_minor["close"]), freq=tf_to_pandas_freq(TIMEFRAME_MINOR))
            arr_major['ts'] = pd.date_range("2023-01-01", periods=len(arr_major["close"]), freq=tf_to_pandas_freq(TIMEFRAME_MAJOR))
        
            signal = trends_tf_long(
                arr_major=arr_major,
                arr_minor=arr_minor,
                lookback_minor=param_dict.get('LOOKBACK_MINOR'),
                n_consecutive=param_dict.get('N_CONSECUTIVE'),
                factor=param_dict.get('FACTOR'),
                backtest=True
            )
        
            arr_minor['signal'] = np.asarray(signal, dtype=DTYPE)


        result = run_grid_backtest(
            ohlcv_arrays_minor,
            sell_after=param_dict.get('SELL_AFTER'),
            tp_pct=param_dict.get('TP_PCT'),
            sl_pct=param_dict.get('SL_PCT'),
            order_amount=ORDER_AMOUNT
        )

        portfolio_record = compile_MC_results(result, param_dict, path_idx, INITIAL_BALANCE, dtype=DTYPE)
        all_results.append(portfolio_record)

    return all_results


def parallel_with_progress(tasks, desc: str, n_jobs: int = N_JOBS):
    with tqdm_joblib(tqdm(total=len(tasks), desc=desc)):
        return Parallel(n_jobs=n_jobs)(tasks)

# -----------------------------------------------------------------------------
# GENERATE PATHS FOR MINOR TIMEFRAME AND DERIVE MAJOR
# -----------------------------------------------------------------------------
start_paths_time = time.time()
paths_minor      = generate_paths_for_all_symbols_functional(ohlcv_data_minor, n_paths=FINAL_N_PATHS, n_obs=FINAL_N_OBS_PER_PATH)
if TIMEFRAME_MINOR == '1H':
    factor = 24
elif TIMEFRAME_MINOR == '4H':
    factor = 6
elif TIMEFRAME_MINOR == '6Hutc':
    factor =  4    
elif TIMEFRAME_MINOR == '12Hutc':
    factor =  2   

paths_major = {sym: derive_major_from_minor(paths_minor[sym], factor=factor) for sym in paths_minor.keys()}

end_paths_time = time.time()
print(f"\n🕒 Paths generation + derivation: {end_paths_time - start_paths_time:.2f} seconds")

# -----------------------------------------------------------------------------
# EVALUATE MONTE CARLO PATHS
# -----------------------------------------------------------------------------
start_eval_time = time.time()
results_list    = parallel_with_progress(
    [delayed(process_path_IDX)(path_idx, paths_minor, paths_major, param_dict_list)
     for path_idx in range(FINAL_N_PATHS)],
    desc="\n🔄 Evaluating Paths_IDX"
)
end_eval_time = time.time()
print(f"\n🕒 Paths evaluation: {end_eval_time - start_eval_time:.2f} seconds")

all_results  = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

# -----------------------------------------------------------------------------
# SUMMARY / REPORT
# -----------------------------------------------------------------------------
final_prints(f"🎲 MC_{STRATEGY} 🎲", DATA_FOLDER, f"{TIMEFRAME_MAJOR}/{TIMEFRAME_MINOR}", min_vol_usdt=MIN_VOL_USDT, order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=lists_for_grid)
df_summary = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
