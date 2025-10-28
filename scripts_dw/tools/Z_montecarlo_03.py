# === FILE: main_MONTECARLO.py ===
# --------------------------------
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_analysis import report_montecarlo
from ZX_utils import filter_symbols
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
from ZX_optimize_MC import generate_paths_for_symbol, optimize_for_symbol
#from ZZX_DRAFT2 import generate_paths_for_symbol, optimize_for_symbol

from Z_add_signals_03 import add_indicators, explosive_signal

DTYPE = np.float32
start_time = time.time()

# -----------------------------
# MONTECARLO
# -----------------------------
OPTUNA_N_PATHS       = 100
FINAL_N_PATHS        = 100
FINAL_N_OBS_PER_PATH = 1000
FINAL_N_SUBSTEPS     = 10
TS_INDEX             = np.arange(FINAL_N_OBS_PER_PATH).astype('datetime64[ns]')

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FOLDER         = "data/crypto_2023_highlow_UPTO"
DATE_MIN            = "2025-01-03"
TIMEFRAME           = '1H'
MIN_VOL_USDT        = 120_000
N_JOBS              = -1

# -----------------------------
# GRID 
# -----------------------------
SELL_AFTER_LIST     = [10,15,20]
ENTROPY_MAX_LIST    = [1.0,1.5,2.0]
ACCEL_SPAN_LIST     = [5,10,15]

TP_PCT_LIST         = [0,5]
SL_PCT_LIST         = [0,5]

# =============================================================================
# =============================================================================
# SELL_AFTER_LIST    = [15]
# ENTROPY_MAX_LIST   = [1.5]
# ACCEL_SPAN_LIST    = [5]
# 
# TP_PCT_LIST        = [0]
# SL_PCT_LIST        = [0]
# =============================================================================
# =============================================================================
param_names     = ['SELL_AFTER', 'ENTROPY_MAX', 'ACCEL_SPAN', 'TP_PCT', 'SL_PCT']
lists_for_grid  = [globals()[name + "_LIST"] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

# -----------------------------
# PATHS / OUTPUT
# -----------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = SCRIPT_DIR
OUTPUT_FILE = os.path.join(RESULTS_DIR, f"montecarlo_entropy_crypto_{TIMEFRAME}.xlsx")

# -----------------------------
# CACHE DE INDICADORES GENÉRICO
# -----------------------------
INDICATOR_CACHE = {}

def cached_add_indicators(close, m_accel=5):
    close = close.astype(DTYPE, copy=False)
    key   = (hash(close.data.tobytes()), m_accel)
    if key in INDICATOR_CACHE:
        return INDICATOR_CACHE[key]
    entropia, accel = add_indicators(close, m_accel)
    entropia = np.asarray(entropia, dtype=DTYPE)
    accel    = np.asarray(accel, dtype=DTYPE)
    INDICATOR_CACHE[key] = (entropia, accel)
    return entropia, accel

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def generate_paths_for_all_symbols(ohlcv_data, best_params_dict, n_paths, n_obs, n_substeps, base_seed=42):
    paths_per_symbol = {}
    for symbol, df_hist in ohlcv_data.items():
        params = best_params_dict.get(symbol, None)
        if not params:
            continue
        paths_array = generate_paths_for_symbol(
            df_hist,
            n_paths=n_paths,
            n_obs=n_obs,
            n_substeps=n_substeps,
            vol_scale=params["vol_scale"],
            jump_prob_per_substep=params["jump_prob_per_substep"],
            jump_mu=params["jump_mu"],
            jump_sigma=params["jump_sigma"],
            min_price=MIN_PRICE,
            timeframe=TIMEFRAME,
            base_seed=42
        )
        if paths_array is not None:
            paths_array = np.asarray(paths_array, dtype=DTYPE)
        paths_per_symbol[symbol] = paths_array
    return paths_per_symbol

def process_path_IDX(path_idx, paths_per_symbol, param_dict_list):
    all_results = []

    for param_dict in param_dict_list:
        ohlcv_arrays = {}

        for sym, arr_paths in paths_per_symbol.items():
            if path_idx >= arr_paths.shape[0]:
                continue

            # Tomamos un path completo contiguo
            arr = arr_paths[path_idx]  # forma (n_obs, n_features)
            open_  = arr[:, 0].astype(DTYPE)
            low_   = arr[:, 1].astype(DTYPE)
            high_  = arr[:, 2].astype(DTYPE)
            close  = arr[:, 3].astype(DTYPE)

            # Indicadores y señales
            entropia, accel = cached_add_indicators(close, m_accel=param_dict.get('ACCEL_SPAN', 5))
            signal = explosive_signal(entropia, accel, entropia_max=param_dict.get('ENTROPY_MAX', 1.0), live=False)
            signal = np.asarray(signal, dtype=DTYPE)

            ohlcv_arrays[sym] = {
                'ts': TS_INDEX,
                'open': open_,
                'low': low_,
                'high': high_,
                'close': close,
                'signal': signal
            }

        if len(ohlcv_arrays) == 0:
            continue

        try:
            result = run_grid_backtest(
                ohlcv_arrays,
                sell_after=param_dict.get('SELL_AFTER', 10),
                initial_balance=INITIAL_BALANCE,
                order_amount=ORDER_AMOUNT,
                tp_pct=param_dict.get('TP_PCT', 0),
                sl_pct=param_dict.get('SL_PCT', 0),
                comi_pct=0.05
            )
        except Exception as e:
            all_results.append({**param_dict,
                                "path_index": path_idx,
                                "symbol": "__PORTFOLIO__",
                                "Net_Gain": np.nan,
                                "Net_Gain_pct": np.nan,
                                "Num_Signals": np.nan,
                                "Win_Ratio": np.nan,
                                "DD": np.nan,
                                "Portfolio_Final_Balance": np.nan,
                                "Portfolio_Num_Signals": np.nan,
                                "error": str(e)})
            continue

        portfolio = result["__PORTFOLIO__"]
        trades = np.asarray(portfolio['trades'], dtype=DTYPE) if portfolio['trades'] else np.array([], dtype=DTYPE)
        final_balance = np.float64(portfolio['final_balance'])
        num_signals   = portfolio['num_signals']
        win_ratio     = portfolio['proportion_winners']
        max_dd        = portfolio['max_dd']

        portfolio_record = {**param_dict,
                            "path_index": path_idx,
                            "symbol": "__PORTFOLIO__",
                            "Net_Gain": np.sum(trades) if trades.size > 0 else 0.0,
                            "Net_Gain_pct": (np.sum(trades) / INITIAL_BALANCE * 100.0) if trades.size > 0 else 0.0,
                            "Num_Signals": num_signals,
                            "Win_Ratio": win_ratio,
                            "DD": max_dd * 100 if isinstance(max_dd, (int, float)) else np.nan,
                            "Portfolio_Final_Balance": final_balance,
                            "Portfolio_Num_Signals": num_signals,
                            "error": None}

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
# OPTUNA PARAMS
# -----------------------------
start_opt_time = time.time()
results = parallel_with_progress(
    [delayed(optimize_for_symbol)(s, ohlcv_data, n_trials=50, n_paths=OPTUNA_N_PATHS,
                                  n_obs=FINAL_N_OBS_PER_PATH, n_substeps=FINAL_N_SUBSTEPS,
                                  min_price=MIN_PRICE, timeframe=TIMEFRAME, base_seed=42)
     for s in filtered_symbols],
    desc="\n🔁 Optimizing MC params"
)
best_params_dict = {symbol: params for symbol, params, score in results}
end_opt_time     = time.time()
print(f"\n🕒 OPTUNA: {end_opt_time - start_opt_time:.2f} segundos")

# -----------------------------
# GENERAR PATHS
# -----------------------------

paths_per_symbol = generate_paths_for_all_symbols(
    ohlcv_data, best_params_dict,
    n_paths=FINAL_N_PATHS,
    n_obs=FINAL_N_OBS_PER_PATH,
    n_substeps=FINAL_N_SUBSTEPS,
    base_seed=42
)
valid_symbols = [s for s, arr in paths_per_symbol.items() if arr is not None and arr.size > 0]

# --- SCORE UNICO DE COMPARACION ---
# =============================================================================
# unicoprint_score = summary_score_all_paths(
#     ohlcv_data,
#     n_paths=OPTUNA_N_PATHS,
#     n_obs=FINAL_N_OBS_PER_PATH,
#     n_substeps=FINAL_N_SUBSTEPS,
#     base_seed=42,
#     DTYPE=DTYPE
# )
# 
# print(f"\n🟢 UNICO SCORE MONTECARLO: {unicoprint_score['score_total']:.4f}")
# print("🔹 Métricas promedio por símbolo:")
# print(f"Mean : {unicoprint_score['sim_mean']:.4f}")
# print(f"Std  : {unicoprint_score['sim_std']:.4f}")
# print(f"Skew : {unicoprint_score['sim_skew']:.4f}")
# print(f"Kurt : {unicoprint_score['sim_kurt']:.4f}")
# print(f"ACF  : {unicoprint_score['sim_acf']:.4f}")
# print(f"KS   : {unicoprint_score['sim_ks']:.4f}")
# print(f"Wass : {unicoprint_score['sim_wass']:.4f}")
# =============================================================================

# -----------------------------
# EVALUAR Paths_IDX
# -----------------------------
start_paths_time = time.time()
results_list = parallel_with_progress(
    [delayed(process_path_IDX)(path_idx, paths_per_symbol, param_dict_list)
     for path_idx in range(FINAL_N_PATHS)],
    desc="\n🔁 Evaluating Paths_IDX"
)
end_paths_time = time.time()
print(f"\n🕒 Paths_IDX: {end_paths_time - start_paths_time:.2f} segundos")

all_results  = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

# -----------------------------
# SUMMARY / REPORT
# -----------------------------
print(f"TIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"DATE_MIN         : {DATE_MIN}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")
print(f"ENTROPY_MAX_LIST = {ENTROPY_MAX_LIST}")
print(f"ACCEL_SPAN_LIST  = {ACCEL_SPAN_LIST}")
print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}")

df_summary = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
