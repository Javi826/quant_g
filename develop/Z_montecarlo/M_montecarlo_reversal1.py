# === FILE: main_MONTECARLO ===
# -----------------------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
import time
import logging
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import product
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from shared.backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from shared.utils.utils import filter_symbols, save_filtered_symbols, final_prints
from shared.utils.analysis import report_montecarlo
from shared.utils.st_tools import extract_ohlcv_from_path, compile_MC_results, get_n_obs
from shared.tools.optimize_MCf_tf import generate_paths_for_all_symbols_functional
from signals.add_signals_reversal import reversal_long
from signals.add_signals_reversal import reversal_short

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(message)s"))

for name in ("BOT_batch", "BOT_trading"):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(handler)
    log.propagate = False

DTYPE      = np.float32
start_time = time.time()
N_JOBS     = -1
STRATEGY   = "reversal"
MY_SYMBOLS = True

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER     = "../../BOT_batch/data/crypto_2026_OOS"
#DATA_FOLDER     = "../../BOT_batch/data/crypto_2022_IS"
TIMEFRAME_MINOR = '4H'
ORDER_AMOUNT    = 80
MIN_VOL_USDT    = 1_800_000

# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------
SELL_AFTER_LIST = [0]
LOOKBACK_LIST   = [3, 4, 5]
MA_PERIOD_LIST  = [50]
TOLERANCE_LIST  = [20,25,30]
TP_PCT_LIST     = [2,3,4,5]
SL_PCT_LIST     = [6,7,8,9,10]

SELL_AFTER_LIST = [0]
LOOKBACK_LIST   = [4]
MA_PERIOD_LIST  = [50]
TOLERANCE_LIST  = [20]
TP_PCT_LIST     = [4]
SL_PCT_LIST     = [6]

param_names     = ['SELL_AFTER', 'LOOKBACK', 'TOLERANCE', 'MA_PERIOD', 'TP_PCT', 'SL_PCT']
param_ranges    = {name: globals()[f"{name}_LIST"] for name in param_names}
lists_for_grid  = [param_ranges[name] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

# -----------------------------------------------------------------------------
# MONTE CARLO SETTINGS
# -----------------------------------------------------------------------------
FINAL_N_PATHS        = 2000
FINAL_N_OBS_PER_PATH = get_n_obs(TIMEFRAME_MINOR)
TS_INDEX             = np.arange(FINAL_N_OBS_PER_PATH).astype('datetime64[ns]')

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50, my_symbols=MY_SYMBOLS)
save_filtered_symbols(filtered_minor, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR, save_symbols=False)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def process_path_IDX(path_idx, paths_minor, param_dict_list):
    all_results = []
    for param_dict in param_dict_list:
        ohlcv_arrays_minor = extract_ohlcv_from_path(paths_minor, path_idx, dtype=DTYPE)

        for sym in ohlcv_arrays_minor.keys():
            arr_minor = ohlcv_arrays_minor[sym]
            signals = reversal_short(
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

# -----------------------------------------------------------------------------
# GENERATE & EVALUATE PATHS
# -----------------------------------------------------------------------------
paths_minor  = generate_paths_for_all_symbols_functional(ohlcv_data_minor, n_paths=FINAL_N_PATHS, n_obs=FINAL_N_OBS_PER_PATH, raw_columns=[])
tasks        = [delayed(process_path_IDX)(i, paths_minor, param_dict_list) for i in range(FINAL_N_PATHS)]

with tqdm_joblib(tqdm(desc="🔄 Evaluating Paths_IDX", total=len(tasks))) as progress:
    results_list = Parallel(n_jobs=N_JOBS)(tasks)

all_results  = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

# -----------------------------------------------------------------------------
# SUMMARY / REPORT
# -----------------------------------------------------------------------------
final_prints(f"🎲 MC_{STRATEGY} 🎲", DATA_FOLDER, TIMEFRAME_MINOR, MIN_VOL_USDT, ORDER_AMOUNT, param_names, lists_for_grid)
df_summary = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")