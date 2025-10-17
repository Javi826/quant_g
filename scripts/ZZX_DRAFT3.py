# === FILE: main_BACKTESTING.py ===
# ---------------------------------
import os
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT,COMISION
from ZZX_DRAFT2 import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT,COMISION
from tools.ZX_st_tools import prepare_ohlcv_arrays,compile_grid_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_results, save_filtered_symbols
from Z_add_signals_03 import add_indicators_03, explosive_signal_03

start_time         = time.time()
SAVE_SYMBOLS       = False
STRATEGY           ="entropy"
N_JOBS             =-1
# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2023_ISOLD"
TIMEFRAME           = '4H'
MIN_VOL_USDT        = 50_000

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------

SELL_AFTER_LIST     = [5,10,15,20,25,30,35]
ENTROPY_MAX_LIST    = [0.2,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6]
ACCEL_SPAN_LIST     = [5,10,12,15,17,20]

TP_PCT_LIST         = [0,5,10,15]
SL_PCT_LIST         = [0,5,10,15]

# =============================================================================
SELL_AFTER_LIST    = [20,25,30]
ENTROPY_MAX_LIST   = [0.4,0.6,0.8]
ACCEL_SPAN_LIST    = [5,10,15]

TP_PCT_LIST        = [0,5]
SL_PCT_LIST        = [0,5]
# =============================================================================

param_names    = ['SELL_AFTER', 'ENTROPY_MAX', 'ACCEL_SPAN', 'TP_PCT', 'SL_PCT']
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
ohlcv_arr = prepare_ohlcv_arrays(ohlcv_data)

# -----------------------------------------------------------------------------
# FUNCIÓN DE PROCESO PARA UNA COMBINACIÓN
# -----------------------------------------------------------------------------
def process_combo(comb):
    params       = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym, arrs in ohlcv_arr.items():
        entropia, accel   = add_indicators_03(arrs['close'], m_accel=params.get('ACCEL_SPAN', 5))
        signal            = explosive_signal_03(entropia, accel, entropia_max=params.get('ENTROPY_MAX', 1.0), live=False)
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params['SELL_AFTER'],
        tp_pct=params['TP_PCT'],
        sl_pct=params['SL_PCT'],
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        comi_pct=COMISION
    )
    return comb, results

# -----------------------------------------------------------------------------
# BACKTESTING PARALIZADO
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
save_results(grid_results_df.to_dict('records'), grid_results_df, filename=f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME}.xlsx",save=False)
print('\n🥇==Grid_backtest==🥇')
print(f"TIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")
print(f"ENTROPY_MAX_LIST = {ENTROPY_MAX_LIST}")
print(f"ACCEL_SPAN_LIST  = {ACCEL_SPAN_LIST}")
print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}\n")

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, initial_capital=INITIAL_BALANCE)

import pandas as pd

# Lista para guardar todos los trade_log
all_trade_logs = []

for _, result in grid_results_list:
    trade_log_df = result['__PORTFOLIO__']['trade_log']
    all_trade_logs.append(trade_log_df)

# Concatenar todos los trade_log en un solo DataFrame
all_trades_df = pd.concat(all_trade_logs, ignore_index=True)

# Tipos de cierre posibles
all_exit_types = ['TP', 'SL', 'SELL_AFTER', 'FORCED_LAST']

# Contar trades por exit_reason e incluir los que son 0
trade_counts = all_trades_df['exit_reason'].value_counts().reindex(all_exit_types, fill_value=0)

print("\n📊 Número total de trades por tipo de cierre (todas las combinaciones):")
print(trade_counts)



elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
