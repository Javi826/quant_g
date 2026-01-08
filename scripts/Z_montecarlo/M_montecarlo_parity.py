# === FILE: main_MONTECARLO_ ===
# -----------------------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
from tools.ZX_optimize_MCf_tf import generate_paths_for_all_symbols_functional
from Z_signals.add_signals_parity import parity_long
from Z_signals.add_signals_parity import parity_short

DTYPE               = np.float32
start_time          = time.time()
N_JOBS              = -1
STRATEGY            = "parity"
# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER         = "../data/crypto_2024_IS_short"
#DATA_FOLDER         = "../data/crypto_2024_short_IS"
TIMEFRAME_MINOR     = '30m'
ORDER_AMOUNT        = 80
MIN_VOL_USDT        = 5_000_000

# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------
SELL_AFTER_LIST      = [0]  
LOOKBACK_LIST        = [100,150]
TOLERANCE_LIST       = [20,25,30] 
MA_PERIOD_LIST       = [25,50]

TP_PCT_LIST          = [1,2,3]
SL_PCT_LIST          = [8,9,10]

param_names     = ['SELL_AFTER','LOOKBACK','TOLERANCE','MA_PERIOD','TP_PCT','SL_PCT']
lists_for_grid  = [globals()[name + "_LIST"] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

# -----------------------------------------------------------------------------
# MONTE CARLO SETTINGS
# -----------------------------------------------------------------------------
FINAL_N_PATHS        = 100
FINAL_N_OBS_PER_PATH = get_n_obs(TIMEFRAME_MINOR)
TS_INDEX             = np.arange(FINAL_N_OBS_PER_PATH).astype('datetime64[ns]')

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor,min_vol_usdt=MIN_VOL_USDT,timeframe=TIMEFRAME_MINOR,data_folder=DATA_FOLDER,min_price=MIN_PRICE,vol_window=50)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def process_path_IDX(path_idx, paths_minor, param_dict_list):
    all_results = []
    for param_dict in param_dict_list:
        ohlcv_arrays_minor = extract_ohlcv_from_path(paths_minor, path_idx, dtype=DTYPE)

        for sym in ohlcv_arrays_minor.keys():

            arr_minor = ohlcv_arrays_minor[sym]
 
            signals = parity_short(
                arr_minor,
                lookback=param_dict.get('LOOKBACK'),
                tolerance=param_dict.get('TOLERANCE'),
                ma_period=param_dict.get('MA_PERIOD'),
                live_trading=False
            )

            arr_minor['signal'] = np.asarray(signals, dtype=DTYPE)

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
# GENERATE & EVALUATE PATHS FOR MINOR TIMEFRAME
# -----------------------------------------------------------------------------
paths_minor  = generate_paths_for_all_symbols_functional(ohlcv_data_minor,n_paths=FINAL_N_PATHS,n_obs=FINAL_N_OBS_PER_PATH,raw_columns=[])
results_list = parallel_with_progress([delayed(process_path_IDX)(i, paths_minor, param_dict_list) for i in range(FINAL_N_PATHS)], desc="\n🔄 Evaluating Paths_IDX")
all_results  = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

# -----------------------------------------------------------------------------
# SUMMARY / REPORT
# -----------------------------------------------------------------------------
final_prints(f"🎲 MC_{STRATEGY} 🎲", DATA_FOLDER, f"{TIMEFRAME_MINOR}", min_vol_usdt=MIN_VOL_USDT, order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=lists_for_grid)
df_summary = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
