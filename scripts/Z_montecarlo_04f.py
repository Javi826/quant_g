# === FILE: main_MONTECARLO_funcional_sharpe_no_cache.py ===
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
from utils.ZX_utils import filter_symbols
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
from ZX_optimize_MCf import generate_multiple_paths
from Z_add_signals_04 import add_indicators, explosive_signal

DTYPE = np.float32
start_time = time.time()

# -----------------------------
# MONTECARLO SETTINGS
# -----------------------------
FINAL_N_PATHS          = 200
FINAL_N_OBS_PER_PATH   = 3000
TS_INDEX               = np.arange(FINAL_N_OBS_PER_PATH).astype('datetime64[ns]')

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FOLDER            = "data/crypto_2023_UPTO"
DATE_MIN               = "2025-01-03"
TIMEFRAME              = '4H'
MIN_VOL_USDT           = 800_000
N_JOBS                 = -1

# -----------------------------
# GRID DE PARÁMETROS
# -----------------------------
SELL_AFTER_LIST        = [15,20,25,30,35]

DOJI_LIST              = [True, False]
HAMMER_LIST            = [True, False]
SHOOTING_STAR_LIST     = [True, False]
BULLISH_ENGULFING_LIST = [True, False]
BEARISH_ENGULFING_LIST = [True, False]
PIERCING_LINE_LIST     = [True, False]
DARK_CLOUD_COVER_LIST  = [True, False]

TP_PCT_LIST            = [0,15]
SL_PCT_LIST            = [0,15]

# =============================================================================
# =============================================================================
# SELL_AFTER_LIST        = [20]
# DOJI_LIST              = [False]
# HAMMER_LIST            = [False]
# SHOOTING_STAR_LIST     = [False]
# BULLISH_ENGULFING_LIST = [True]
# BEARISH_ENGULFING_LIST = [False]
# PIERCING_LINE_LIST     = [False]
# DARK_CLOUD_COVER_LIST  = [False]
# 
# TP_PCT_LIST            = [0]
# SL_PCT_LIST            = [0]
# =============================================================================
# =============================================================================

param_names = [
    'SELL_AFTER', 'DOJI', 'HAMMER', 'SHOOTING_STAR',
    'BULLISH_ENGULFING', 'BEARISH_ENGULFING',
    'PIERCING_LINE', 'DARK_CLOUD_COVER',
    'TP_PCT', 'SL_PCT'
]
lists_for_grid  = [globals()[name + "_LIST"] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

# -----------------------------
# PATHS / OUTPUT
# -----------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = SCRIPT_DIR
OUTPUT_FILE = os.path.join(RESULTS_DIR, f"montecarlo_indicators_crypto_{TIMEFRAME}.xlsx")

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
        ohlcv_arrays = {}
        for sym, arr_paths in paths_per_symbol.items():
            if path_idx >= arr_paths.shape[0]:
                continue

            arr = arr_paths[path_idx]  # (n_obs, n_features)
            open_ = arr[:, 0].astype(DTYPE)
            low_  = arr[:, 1].astype(DTYPE)
            high_ = arr[:, 2].astype(DTYPE)
            close = arr[:, 3].astype(DTYPE)

            # Indicadores y señales sin cache
            df_dummy = pd.DataFrame({'open': open_, 'high': high_, 'low': low_, 'close': close})
            df_ind   = add_indicators(df_dummy.copy())
            pattern_flags = [
                param_dict['DOJI'],
                param_dict['HAMMER'],
                param_dict['SHOOTING_STAR'],
                param_dict['BULLISH_ENGULFING'],
                param_dict['BEARISH_ENGULFING'],
                param_dict['PIERCING_LINE'],
                param_dict['DARK_CLOUD_COVER']
            ]
            df_signal = explosive_signal(df_ind, pattern_flags, live=False)
            signal    = np.asarray(df_signal['signal'], dtype=bool)

            ohlcv_arrays[sym] = {
                'ts': TS_INDEX,
                'open': open_,
                'low': low_,
                'high': high_,
                'close': close,
                'signal': signal
            }

        if not ohlcv_arrays:
            continue

        try:
            result = run_grid_backtest(
                ohlcv_arrays,
                sell_after=param_dict['SELL_AFTER'],
                initial_balance=INITIAL_BALANCE,
                order_amount=ORDER_AMOUNT,
                tp_pct=param_dict['TP_PCT'],
                sl_pct=param_dict['SL_PCT'],
                comi_pct=0.05
            )
        except Exception as e:
            continue

        portfolio = result["__PORTFOLIO__"]
        trades = np.asarray(portfolio['trades'], dtype=DTYPE) if portfolio['trades'] else np.array([], dtype=DTYPE)
        final_balance = float(portfolio['final_balance'])
        num_signals   = portfolio['num_signals']
        win_ratio     = portfolio['proportion_winners']
        max_dd        = portfolio['max_dd']
        sharpe        = float(portfolio.get('sharpe', np.nan))

        portfolio_record = {
            **param_dict,
            "path_index": path_idx,
            "symbol": "__PORTFOLIO__",
            "Net_Gain": np.sum(trades) if trades.size > 0 else 0.0,
            "Net_Gain_pct": (np.sum(trades)/INITIAL_BALANCE*100.0) if trades.size > 0 else 0.0,
            "Num_Signals": num_signals,
            "Win_Ratio": win_ratio,
            "DD": max_dd*100 if isinstance(max_dd,(int,float)) else np.nan,
            "Portfolio_Final_Balance": final_balance,
            "Portfolio_Num_Signals": num_signals,
            "Sharpe": sharpe
        }
        all_results.append(portfolio_record)
    return all_results

def parallel_with_progress(tasks, desc: str, n_jobs: int = N_JOBS):
    with tqdm_joblib(tqdm(total=len(tasks), desc=desc)):
        return Parallel(n_jobs=n_jobs)(tasks)

# -----------------------------
# SYMBOLS / DATA
# -----------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]
ohlcv_data, filtered_symbols, removed_symbols = filter_symbols(
    symbols,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50,
    date_min=DATE_MIN
)

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
end_paths_time = time.time()
print(f"\n🕒 Paths generation: {end_paths_time - start_paths_time:.2f} segundos")

# -----------------------------
# EVALUAR PATHS_IDX
# -----------------------------
start_eval_time = time.time()
results_list = parallel_with_progress(
    [delayed(process_path_IDX)(path_idx, paths_per_symbol, param_dict_list)
     for path_idx in range(FINAL_N_PATHS)],
    desc="\n🔁 Evaluating Paths_IDX"
)
end_eval_time = time.time()
print(f"\n🕒 Paths evaluation: {end_eval_time - start_eval_time:.2f} segundos")

all_results  = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

# -----------------------------
# SUMMARY / REPORT
# -----------------------------
print(f"TIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"DATE_MIN         : {DATE_MIN}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")

df_summary = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
