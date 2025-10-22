# === FILE: main_BACKTESTING_patterns_full_signal.py ===
# ---------------------------------
import os
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_results, save_filtered_symbols
from Z_add_signals_99 import explosive_signal_99  

start_time = time.time()
SAVE_SYMBOLS = False
STRATEGY = "patterns"
N_JOBS = -1

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER           = "data/crypto_2023_IS"
TIMEFRAME             = '4H'
MIN_VOL_USDT          = 1600_000_000

# -----------------------------------------------------------------------------
# GRID DE PARÁMETROS
# -----------------------------------------------------------------------------
SELL_AFTER_LIST       = [1200]

TP_PCT_LIST           = [1000]
SL_PCT_LIST           = [1000]

param_names = ['SELL_AFTER','TP_PCT', 'SL_PCT']
lists_for_grid = [globals()[name + "_LIST"] for name in param_names]

# -----------------------------------------------------------------------------
# CARGA Y FILTRADO DE DATOS
# -----------------------------------------------------------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]

ohlcv_data, filtered_symbols = filter_symbols(symbols,min_vol_usdt=MIN_VOL_USDT,timeframe=TIMEFRAME,data_folder=DATA_FOLDER,min_price=MIN_PRICE,vol_window=50)

save_filtered_symbols(filtered_symbols, strategy=STRATEGY, timeframe=TIMEFRAME, save_symbols=SAVE_SYMBOLS)

ohlcv_arr = prepare_ohlcv_arrays(ohlcv_data)

# -----------------------------------------------------------------------------
# FUNCIÓN DE PROCESO PARA UNA COMBINACIÓN
# -----------------------------------------------------------------------------
def process_combo(comb):
    params = dict(zip(param_names, comb))
    ohlcv_arrays_combo = {}

    for sym, arrs in ohlcv_arr.items():
        signal = explosive_signal_99(
            high=arrs['high'],       
            close=arrs['close'],    
            live=False  # solo compra en el primer timestamp
        )

        ohlcv_arrays_combo[sym] = {**arrs, 'signal': signal}

    results = run_grid_backtest(
        ohlcv_arrays_combo,
        sell_after=params['SELL_AFTER'],
        tp_pct=params['TP_PCT'],
        sl_pct=params['SL_PCT']
    )
    return comb, results


# -----------------------------------------------------------------------------
# BACKTESTING PARALELIZADO
# -----------------------------------------------------------------------------
all_combinations = list(product(*lists_for_grid))
with tqdm_joblib(tqdm(desc="🔁 Backtesting Grid... \n", total=len(all_combinations))) as progress:
    grid_results_list = Parallel(n_jobs=N_JOBS)(
        delayed(process_combo)(comb) for comb in all_combinations
    )

# -----------------------------------------------------------------------------
# COMPILAR RESULTADOS A DATAFRAME
# -----------------------------------------------------------------------------
grid_records    = compile_grid_results(grid_results_list, param_names, INITIAL_BALANCE)
grid_results_df = pd.DataFrame(grid_records)

# -----------------------------------------------------------------------------
# SAVE RESULTS + TIMING
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'),grid_results_df,filename=f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME}.xlsx",save=False)

print(f'\n🥇==Grid_backtest {STRATEGY}==🥇')
print(f"\nTIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")


print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}\n")

df_portfolio, mi_series = report_backtesting(df=grid_results_df,parameters=param_names,initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
