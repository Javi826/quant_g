# === FILE: main_MONTECARLO_.py ===
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
from utils.ZX_utils import filter_symbols,final_prints
from ZX_compute_BT import run_grid_backtest, MIN_PRICE,INITIAL_BALANCE
from tools.ZX_st_tools import extract_ohlcv_from_path, compile_MC_results,get_n_obs
from tools.ZX_optimize_MC import optimize_for_symbol, generate_paths_for_symbol,evaluate_synthetic_vs_real
from Z_add_signals_scalping import scalping_long

start_time = time.time()
DTYPE               = np.float32
STRATEGY            ="scalping"
N_JOBS              = -1

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2025_scalping_IS"
TIMEFRAME_MINOR     = '15m'
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

param_names     = ['SELL_AFTER','RSI','ADX','LOOKBACK','TOLERANCE','TP_PCT','SL_PCT']
lists_for_grid  = [globals()[name + "_LIST"] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

# -----------------------------
# MONTECARLO SETTINGS
# -----------------------------
FINAL_N_PATHS        = 100
FINAL_N_OBS_PER_PATH = get_n_obs(TIMEFRAME_MINOR)    
TS_INDEX             = np.arange(FINAL_N_OBS_PER_PATH).astype('datetime64[ns]')

# -----------------------------
# SYMBOLS / DATA
# -----------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_symbols = filter_symbols(symbols,min_vol_usdt=MIN_VOL_USDT,timeframe=TIMEFRAME_MINOR,data_folder=DATA_FOLDER,min_price=MIN_PRICE,vol_window=50,my_symbols=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

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
    
# -----------------------------------------------------------------------------
# OPTUNA
# -----------------------------------------------------------------------------  
# Diccionario para guardar los parámetros optimizados por símbolo
optimized_params_per_symbol = {}
metrics_per_symbol = {}

for sym in filtered_symbols:
    # Optimización de parámetros MC con Optuna
    symbol, best_params, best_score = optimize_for_symbol(
        symbol=sym,
        ohlcv_data=ohlcv_data_minor,
        n_trials=30,               # número de pruebas de Optuna
        n_paths=FINAL_N_PATHS,     # cuántos paths generar
        n_obs=FINAL_N_OBS_PER_PATH,# longitud de cada path
        n_substeps=4,              # subpasos por vela
        min_price=MIN_PRICE,
        timeframe=TIMEFRAME_MINOR,
        base_seed=42
    )
    
    optimized_params_per_symbol[sym] = best_params
    print(f"{sym} -> Mejores parámetros MC: {best_params}, Score: {best_score:.2f}")

    # Generar paths sintéticos con los parámetros óptimos
    df_hist = ohlcv_data_minor[sym]
    paths_best = generate_paths_for_symbol(
        df_hist,
        n_paths=FINAL_N_PATHS,
        n_obs=FINAL_N_OBS_PER_PATH,
        n_substeps=4,
        vol_scale=best_params["vol_scale"],
        min_price=MIN_PRICE,
        jump_prob_per_substep=best_params["jump_prob_per_substep"],
        jump_mu=best_params["jump_mu"],
        jump_sigma=best_params["jump_sigma"],
        timeframe=TIMEFRAME_MINOR,
        base_seed=42
    )

    # Evaluar métricas sintético vs histórico
    score, metrics = evaluate_synthetic_vs_real(df_hist, paths_best)
    metrics_per_symbol[sym] = metrics
    print(f"{sym} -> Score: {score:.2f}")
    metric_names = ["mean", "std", "skew", "kurt", "acf", "ks", "wasserstein"]
    for name, val in zip(metric_names, metrics):
        print(f"   {name}: {val:.2f}")


# -----------------------------------------------------------------------------
# GENERATE & EVALUATE PATHS FOR MINOR TIMEFRAME
# -----------------------------------------------------------------------------
paths_minor = {}
for sym in filtered_symbols:
    df_hist = ohlcv_data_minor[sym]
    params = optimized_params_per_symbol[sym]
    
    paths_minor[sym] = generate_paths_for_symbol(
        df_hist,
        n_paths=FINAL_N_PATHS,
        n_obs=FINAL_N_OBS_PER_PATH,
        n_substeps=4,
        vol_scale=params["vol_scale"],
        min_price=MIN_PRICE,
        jump_prob_per_substep=params["jump_prob_per_substep"],
        jump_mu=params["jump_mu"],
        jump_sigma=params["jump_sigma"],
        timeframe=TIMEFRAME_MINOR,
        base_seed=42
    )
results_list = parallel_with_progress([delayed(process_path_IDX)(i, paths_minor, param_dict_list) for i in range(FINAL_N_PATHS)], desc="\n🔄 Evaluating Paths_IDX")
all_results  = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

# -----------------------------
# SUMMARY / REPORT
# -----------------------------
final_prints(f"🎲 MC_{STRATEGY} 🎲", DATA_FOLDER, TIMEFRAME_MINOR, MIN_VOL_USDT, ORDER_AMOUNT,param_names, lists_for_grid)

df_summary = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")