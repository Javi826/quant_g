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
#from ZZX_DRAFT2 import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
from ZX_optimize_MCf import generate_multiple_paths
from Z_add_signals_03 import add_indicators_03, explosive_signal_03

DTYPE = np.float32
start_time = time.time()

# -----------------------------
# MONTECARLO SETTINGS
# -----------------------------
FINAL_N_PATHS        = 100
FINAL_N_OBS_PER_PATH = 3000
TS_INDEX             = np.arange(FINAL_N_OBS_PER_PATH).astype('datetime64[ns]')

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FOLDER         = "data/crypto_2023_UPTO"
DATE_MIN            = "2023-01-03"
TIMEFRAME           = '4H'
MIN_VOL_USDT        = 500_000
N_JOBS              = -1

# -----------------------------
# GRID 
# -----------------------------
SELL_AFTER_LIST     = [10,15,20,25]
ENTROPY_MAX_LIST    = [0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]
ACCEL_SPAN_LIST     = [10,15,20]

TP_PCT_LIST         = [0,5,10]
SL_PCT_LIST         = [0,5,10]

# =============================================================================
SELL_AFTER_LIST    = [16]
ENTROPY_MAX_LIST   = [1.4]
ACCEL_SPAN_LIST    = [15]

TP_PCT_LIST        = [10]
SL_PCT_LIST        = [0]
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
# FUNCIONES AUXILIARES
# -----------------------------
def generate_paths_for_all_symbols_funcional(ohlcv_data, n_paths, n_obs, raw_columns=[]):
    """
    Genera múltiples paths para todos los símbolos usando arrays vectorizados.
    Devuelve un dict: {symbol: np.ndarray (n_paths, n_obs, n_features)}.
    """
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
            entropia, accel = add_indicators_03(close, m_accel=param_dict.get('ACCEL_SPAN',5))
            entropia = np.asarray(entropia, dtype=DTYPE)
            accel    = np.asarray(accel, dtype=DTYPE)
            signal   = explosive_signal_03(entropia, accel, entropia_max=param_dict.get('ENTROPY_MAX',1.0), live=False)
            signal   = np.asarray(signal, dtype=DTYPE)

            ohlcv_arrays[sym] = {
                'ts': TS_INDEX,
                'open': open_,
                'low':  low_,
                'high': high_,
                'close': close,
                'signal': signal
            }

        if len(ohlcv_arrays) == 0:
            continue

        try:
            result = run_grid_backtest(
                ohlcv_arrays,
                sell_after=param_dict.get('SELL_AFTER',10),
                tp_pct=param_dict.get('TP_PCT',0),
                sl_pct=param_dict.get('SL_PCT',0),
                initial_balance=INITIAL_BALANCE,
                order_amount=ORDER_AMOUNT,
                comi_pct=0.05
            )
        except Exception as e:
            all_results.append({
                **param_dict,
                "path_index": path_idx,
                "symbol": "__PORTFOLIO__",
                "Net_Gain": np.nan,
                "Net_Gain_pct": np.nan,
                "Num_Signals": np.nan,
                "Win_Ratio": np.nan,
                "DD": np.nan,
                "Portfolio_Final_Balance": np.nan,
                "Portfolio_Num_Signals": np.nan,
                "Sharpe": np.nan,
                "error": str(e)
            })
            continue

        portfolio = result["__PORTFOLIO__"]
        trades = np.asarray(portfolio['trades'], dtype=DTYPE) if portfolio['trades'] else np.array([], dtype=DTYPE)
        final_balance = np.float64(portfolio['final_balance'])
        num_signals   = portfolio['num_signals']
        win_ratio     = portfolio['proportion_winners']
        max_dd        = portfolio['max_dd']
        sharpe        = float(portfolio.get('sharpe', np.nan))  # Agregado Sharpe

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
            "Sharpe": sharpe,
            "error": None
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

ohlcv_data, filtered_symbols = filter_symbols(
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
valid_symbols = [s for s, arr in paths_per_symbol.items() if arr is not None and len(arr) > 0]
end_paths_time = time.time()
print(f"\n🕒 Paths generation: {end_paths_time - start_paths_time:.2f} segundos")

# -----------------------------
# EVALUAR Paths_IDX
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
print(f"\nTIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"DATE_MIN         : {DATE_MIN}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")
print(f"ENTROPY_MAX_LIST = {ENTROPY_MAX_LIST}")
print(f"ACCEL_SPAN_LIST  = {ACCEL_SPAN_LIST}")
print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}\n")

df_summary = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
