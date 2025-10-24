# === FILE: main_BACKTESTING.py ===
# ---------------------------------
import os
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results, save_all_trades_to_excel, save_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_filtered_symbols, final_prints
from Z_add_signals_03 import explosive_signal_multi_tf

start_time = time.time()
SAVE_SYMBOLS = False
STRATEGY = "patterns"
N_JOBS = -1

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER       = "data/crypto_2023_IS"
TIMEFRAME_MENOR   = '4H'

TIMEFRAME_MAYOR   = '1D'

ORDER_AMOUNT = 5_000
MIN_VOL_USDT = 10_000_000

# -----------------------------------------------------------------------------
# GRID DE PARÁMETROS
# -----------------------------------------------------------------------------
SELL_AFTER_LIST     = [0]
LOOKBACK_MAYOR_LIST = [1,2,3,4,5]      
LOOKBACK_MENOR_LIST = [1,2,3,4,5] 
TP_PCT_LIST         = [5,10,15,20,30,40,100]
SL_PCT_LIST         = [5,10]

param_names = ['SELL_AFTER','LOOKBACK_MAYOR','LOOKBACK_MENOR','TP_PCT','SL_PCT']
lists_for_grid = [SELL_AFTER_LIST, LOOKBACK_MAYOR_LIST, LOOKBACK_MENOR_LIST, TP_PCT_LIST, SL_PCT_LIST]

# -----------------------------------------------------------------------------
# CARGA Y FILTRADO DE DATOS
# -----------------------------------------------------------------------------
symbols_menor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MENOR}.parquet")]
symbols_mayor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MAYOR}.parquet")]

# Filtrado por volumen y precio para cada timeframe
ohlcv_data_menor, filtered_menor = filter_symbols(
    symbols_menor,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME_MENOR,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50
)

ohlcv_data_mayor, filtered_mayor = filter_symbols(
    symbols_mayor,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME_MAYOR,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50
)


common_symbols = list(set(filtered_menor).intersection(filtered_mayor))

ohlcv_data_menor = {s: ohlcv_data_menor[s] for s in common_symbols}
ohlcv_data_mayor = {s: ohlcv_data_mayor[s] for s in common_symbols}

save_filtered_symbols(common_symbols, strategy=STRATEGY, timeframe=TIMEFRAME_MENOR, save_symbols=SAVE_SYMBOLS)

ohlcv_arr_menor = prepare_ohlcv_arrays(ohlcv_data_menor)
ohlcv_arr_mayor = prepare_ohlcv_arrays(ohlcv_data_mayor)


# -----------------------------------------------------------------------------
# FUNCIÓN DE PROCESO PARA UNA COMBINACIÓN
# -----------------------------------------------------------------------------
def process_combo(comb):
    params = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym in ohlcv_arr_menor.keys():
        arr_menor = ohlcv_arr_menor[sym]
        arr_mayor = ohlcv_arr_mayor[sym]

        signal = explosive_signal_multi_tf(
            high_mayor=arr_mayor['high'],
            close_mayor=arr_mayor['close'],
            high_menor=arr_menor['high'],
            close_menor=arr_menor['close'],
            lookback_mayor=params['LOOKBACK_MAYOR'],
            lookback_menor=params['LOOKBACK_MENOR'],
            live=False
        )

        ohlcv_arrays[sym] = {**arr_menor, 'signal': signal}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params['SELL_AFTER'],
        tp_pct=params['TP_PCT'],
        sl_pct=params['SL_PCT'],
        order_amount=ORDER_AMOUNT
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
grid_records = compile_grid_results(grid_results_list, param_names, INITIAL_BALANCE)
grid_results_df = pd.DataFrame(grid_records)

# -----------------------------------------------------------------------------
# SAVE RESULTS + TIMING
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'), grid_results_df, filename=f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME_MENOR}.xlsx", save=False)
save_all_trades_to_excel(grid_results_list, param_names, filename=f"all_trades_{TIMEFRAME_MENOR}.xlsx", save=False)

final_prints(strategy=f" 🥇Grid_Backest {STRATEGY} 🥇", data_folder=DATA_FOLDER, timeframe=TIMEFRAME_MENOR, min_vol_usdt=MIN_VOL_USDT, order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=lists_for_grid)

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
