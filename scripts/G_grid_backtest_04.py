# === FILE: main_BACKTESTING.py ===
# ---------------------------------
import os
import time
import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT,COMISION
from tools.ZX_st_tools import prepare_ohlcv_arrays,compile_grid_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_results, save_filtered_symbols
from Z_add_signals_04 import add_indicators_04, explosive_signal_04

start_time          = time.time()
SAVE_SYMBOLS        = False
STRATEGY            ="patterns"
N_JOBS              =-1
# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER            = "data/crypto_2023_ISOLD"
TIMEFRAME              = '4H'
MIN_VOL_USDT           = 50_000

# -----------------------------------------------------------------------------
# GRID DE PARÁMETROS
# -----------------------------------------------------------------------------
SELL_AFTER_LIST        = [15,20,25,30]
ENTROPIA_MAX_LIST      = [0.5,1.0,1.5,2.0]

DOJI_LIST              = [True, False]
HAMMER_LIST            = [True, False]
SHOOTING_STAR_LIST     = [True, False]
BULLISH_ENGULFING_LIST = [True, False]
BEARISH_ENGULFING_LIST = [True, False]
PIERCING_LINE_LIST     = [True, False]
DARK_CLOUD_COVER_LIST  = [True, False]

TP_PCT_LIST            = [0,10,15]
SL_PCT_LIST            = [0,10,15]

# =============================================================================
SELL_AFTER_LIST        = [20,30]
ENTROPIA_MAX_LIST      = [1.0,2.0]

DOJI_LIST              = [True,False]
HAMMER_LIST            = [True]
SHOOTING_STAR_LIST     = [True]
BULLISH_ENGULFING_LIST = [True]
BEARISH_ENGULFING_LIST = [True]
PIERCING_LINE_LIST     = [False]
DARK_CLOUD_COVER_LIST  = [True]

TP_PCT_LIST            = [0,5]
SL_PCT_LIST            = [0,6]
# =============================================================================

param_names = [
    'SELL_AFTER', 'ENTROPIA_MAX', 'DOJI', 'HAMMER', 'SHOOTING_STAR',
    'BULLISH_ENGULFING', 'BEARISH_ENGULFING',
    'PIERCING_LINE', 'DARK_CLOUD_COVER',
    'TP_PCT', 'SL_PCT'
]
lists_for_grid = [globals()[name + "_LIST"] for name in param_names]

# -----------------------------------------------------------------------------
# CARGA Y FILTRADO DE DATOS
# -----------------------------------------------------------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]

ohlcv_data, filtered_symbols = filter_symbols(
    symbols,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50
)

save_filtered_symbols(filtered_symbols, strategy=STRATEGY, timeframe=TIMEFRAME, save_symbols=SAVE_SYMBOLS)
# -----------------------------------------------------------------------------
# FUNCIÓN DE PROCESO PARA UNA COMBINACIÓN
# -----------------------------------------------------------------------------
def process_combo(comb):
    params = dict(zip(param_names, comb))

    ohlcv_arrays_combo = {}  # Creamos arrays para esta combinación

    for sym in ohlcv_data.keys():
        df = ohlcv_data[sym].copy()

        df_ind = add_indicators_04(df)

        pattern_flags = [
            params['DOJI'],
            params['HAMMER'],
            params['SHOOTING_STAR'],
            params['BULLISH_ENGULFING'],
            params['BEARISH_ENGULFING'],
            params['PIERCING_LINE'],
            params['DARK_CLOUD_COVER']
        ]

        df_signal = explosive_signal_04(
            df_ind, 
            pattern_flags, 
            entropia_max=params['ENTROPIA_MAX'], 
            live=False
        )

        arr = prepare_ohlcv_arrays({sym: df_signal})[sym]
        arr['signal'] = df_signal['signal'].to_numpy(dtype=bool)
        ohlcv_arrays_combo[sym] = arr

    results = run_grid_backtest(
        ohlcv_arrays_combo,
        sell_after=params['SELL_AFTER'],
        tp_pct=params['TP_PCT'],
        sl_pct=params['SL_PCT'],
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        comi_pct=COMISION
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
save_results(grid_results_df.to_dict('records'), grid_results_df,filename=f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME}.xlsx", save=False)

print(f"\nTIMEFRAME         : {TIMEFRAME}")
print(f"MIN_VOL_USDT      : {MIN_VOL_USDT}")
print(f"SELL_AFTER_LIST   = {SELL_AFTER_LIST}")
print(f"ENTROPIA_MAX_LIST = {ENTROPIA_MAX_LIST}\n")

df_portfolio, mi_series = report_backtesting(df=grid_results_df,parameters=param_names,initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
