# === FILE: main_MONTECARLO_funcional_sharpe_no_cache_adapted.py ===
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
from utils.ZX_utils import filter_symbols,final_prints
from ZX_compute_BT import run_grid_backtest, MIN_PRICE,INITIAL_BALANCE
from tools.ZX_st_tools import extract_ohlcv_from_path, compile_MC_results
from tools.ZX_optimize_MCf import generate_multiple_paths
from Z_add_signals_scalping import scalping_long

start_time = time.time()
DTYPE               = np.float32
STRATEGY            ="scalping"
N_JOBS              = -1

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2025_scalping_IS"
TIMEFRAME           = '5m'
ORDER_AMOUNT        = 400
MIN_VOL_USDT        = 10_000_000

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------

SELL_AFTER_LIST = [0]

RSI_LIST        = [15,20,25]
ADX_LIST        = [25,30,35]
LOOKBACK_LIST   = [10,20,30,40,50]
TOLERANCE_LIST  = [2,5,10,15]

TP_PCT_LIST     = [2.0,2.5,3.0,3.5,4.0,4.5,5.0]
SL_PCT_LIST     = [2.0,2.5,3.0,3.5,4.0,4.5,5.0]

# -----------------------------
# MONTECARLO SETTINGS
# -----------------------------
FINAL_N_PATHS        = 100

if TIMEFRAME == '5m':
    FINAL_N_OBS_PER_PATH = 34560
elif TIMEFRAME == '15m':
    FINAL_N_OBS_PER_PATH = 17280
elif TIMEFRAME == '30m':
    FINAL_N_OBS_PER_PATH = 8640
elif TIMEFRAME == '1H':
    FINAL_N_OBS_PER_PATH = 4320
elif TIMEFRAME == '4H':
    FINAL_N_OBS_PER_PATH = 1080
elif TIMEFRAME == '6Hutc':
    FINAL_N_OBS_PER_PATH = 720
elif TIMEFRAME == '12Hutc':
    FINAL_N_OBS_PER_PATH = 360
elif TIMEFRAME == '1Dutc':
    FINAL_N_OBS_PER_PATH = 180

    
TS_INDEX    = np.arange(FINAL_N_OBS_PER_PATH).astype('datetime64[ns]')
param_names = ['SELL_AFTER','RSI','ADX','LOOKBACK','TOLERANCE','TP_PCT','SL_PCT']

lists_for_grid  = [globals()[name + "_LIST"] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]


# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def generate_paths_for_all_symbols_funcional(ohlcv_data, n_paths, n_obs, raw_columns=[]):
    paths_per_symbol = {}
    for symbol, df_hist in ohlcv_data.items():
        arr_paths = generate_multiple_paths(df_hist, n_paths=n_paths, n_obs=n_obs, raw_columns=raw_columns)
        if arr_paths is not None and arr_paths.shape[0] > 0:
            paths_per_symbol[symbol] = arr_paths
    return paths_per_symbol


def process_path_IDX(path_idx, paths_per_symbol, param_dict_list):
    all_results = []

    for param_dict in param_dict_list:
       
        ohlcv_arrays = extract_ohlcv_from_path(paths_per_symbol, path_idx, dtype=DTYPE)

        for sym, arrs in ohlcv_arrays.items():
      
            signal = scalping_long(
                arrs,
                rsi_max=param_dict.get('RSI'),
                adx_min=param_dict.get('ADX'),
                lookback=param_dict.get('LOOKBACK'),
                tolerance=param_dict.get('TOLERANCE'),
                live_trading=False
                 )
        
            arrs['signal'] = np.asarray(signal, dtype=DTYPE)


        result = run_grid_backtest(
            ohlcv_arrays,
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

# -----------------------------
# SYMBOLS / DATA
# -----------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]

ohlcv_data, filtered_symbols = filter_symbols(symbols,min_vol_usdt=MIN_VOL_USDT,timeframe=TIMEFRAME,data_folder=DATA_FOLDER,min_price=MIN_PRICE,vol_window=50,my_symbols=True)

# -----------------------------
# GENERAR PATHS
# -----------------------------
start_paths_time = time.time()
paths_per_symbol = generate_paths_for_all_symbols_funcional(
    ohlcv_data,
    n_paths=FINAL_N_PATHS,
    n_obs=FINAL_N_OBS_PER_PATH,
    raw_columns=[]
)
valid_symbols = [s for s, arr in paths_per_symbol.items() if arr is not None and len(arr) > 0]
end_paths_time = time.time()
print(f"\n🕒 Paths generation: {end_paths_time - start_paths_time:.2f} segundos")

# -----------------------------
# EVALUAR Paths_IDX
# -----------------------------
start_eval_time = time.time()
results_list    = parallel_with_progress(
    [delayed(process_path_IDX)(path_idx, paths_per_symbol, param_dict_list)
     for path_idx in range(FINAL_N_PATHS)],
    desc="\n🔄 Evaluating Paths_IDX"
)
end_eval_time = time.time()
print(f"\n🕒 Paths evaluation: {end_eval_time - start_eval_time:.2f} segundos")

all_results  = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

# -----------------------------
# SUMMARY / REPORT
# -----------------------------
final_prints(f"🎲 MC_{STRATEGY} 🎲", DATA_FOLDER, TIMEFRAME, MIN_VOL_USDT, ORDER_AMOUNT,param_names, lists_for_grid)

df_summary = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
