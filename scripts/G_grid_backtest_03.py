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
from Z_add_signals_03 import explosive_signal_03,explosive_signal_99


start_time = time.time()
SAVE_SYMBOLS = False
STRATEGY = "trends_tf"
N_JOBS = -1

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER       = "data/crypto_OOS"
DATA_FOLDER       = "data/crypto_2021_OOS"
DATA_FOLDER       = "data/crypto_2022_OOS"
#DATA_FOLDER       = "data/crypto_2023_IS"
TIMEFRAME_MAJOR   = '1D'
TIMEFRAME_MINOR   = '4H'

ORDER_AMOUNT      = 5_000
MIN_VOL_USDT      = 10_000_000

# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------
SELL_AFTER_LIST     = [0]
LOOKBACK_MAJOR_LIST = [1,2,3]      
LOOKBACK_MINOR_LIST = [1,2,3] 

TP_PCT_LIST         = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10]
SL_PCT_LIST         = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10]

SELL_AFTER_LIST     = [0]
LOOKBACK_MAJOR_LIST = [2]      
LOOKBACK_MINOR_LIST = [1] 

TP_PCT_LIST         = [3.5]
SL_PCT_LIST         = [2.0]

param_names = ['SELL_AFTER','LOOKBACK_MAJOR','LOOKBACK_MINOR','TP_PCT','SL_PCT']
lists_for_grid = [SELL_AFTER_LIST, LOOKBACK_MAJOR_LIST, LOOKBACK_MINOR_LIST, TP_PCT_LIST, SL_PCT_LIST]

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
symbols_major = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MAJOR}.parquet")]

ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50)
ohlcv_data_major, filtered_major = filter_symbols(symbols_major, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MAJOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50)

common_symbols = list(set(filtered_minor).intersection(filtered_major))

ohlcv_data_minor = {s: ohlcv_data_minor[s] for s in common_symbols}
ohlcv_data_major = {s: ohlcv_data_major[s] for s in common_symbols}

save_filtered_symbols(common_symbols, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR, save_symbols=SAVE_SYMBOLS)

ohlcv_arr_minor = prepare_ohlcv_arrays(ohlcv_data_minor)
ohlcv_arr_major = prepare_ohlcv_arrays(ohlcv_data_major)

# -----------------------------------------------------------------------------
# FUNCTION TO PROCESS ONE PARAMETER COMBINATION
# -----------------------------------------------------------------------------
def process_combo(comb):
    params = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym in ohlcv_arr_minor.keys():
        arr_minor = ohlcv_arr_minor[sym]
        arr_major = ohlcv_arr_major[sym]

        signal = explosive_signal_03(
            high_mayor=arr_major['high'],
            close_mayor=arr_major['close'],
            high_menor=arr_minor['high'],
            close_menor=arr_minor['close'],
            lookback_mayor=params['LOOKBACK_MAJOR'],
            lookback_menor=params['LOOKBACK_MINOR'],
            live=False
        )

        ohlcv_arrays[sym] = {**arr_minor, 'signal': signal}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params['SELL_AFTER'],
        tp_pct=params['TP_PCT'],
        sl_pct=params['SL_PCT'],
        order_amount=ORDER_AMOUNT
    )
    return comb, results

# -----------------------------------------------------------------------------
# PARALLELIZED BACKTESTING
# -----------------------------------------------------------------------------
all_combinations = list(product(*lists_for_grid))
with tqdm_joblib(tqdm(desc="🔁 Backtesting Grid... \n", total=len(all_combinations))) as progress:
    grid_results_list = Parallel(n_jobs=N_JOBS)(
        delayed(process_combo)(comb) for comb in all_combinations
    )

# -----------------------------------------------------------------------------
# COMPILE RESULTS INTO DATAFRAME
# -----------------------------------------------------------------------------
grid_records = compile_grid_results(grid_results_list, param_names, INITIAL_BALANCE)
grid_results_df = pd.DataFrame(grid_records)

# -----------------------------------------------------------------------------
# SAVE RESULTS + EXECUTION TIME
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'), grid_results_df, filename=f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME_MINOR}.xlsx", save=False)
save_all_trades_to_excel(grid_results_list, param_names, filename=f"all_trades_{TIMEFRAME_MINOR}.xlsx", save=False)

final_prints(strategy=f" 🥇Grid_Backtest {STRATEGY} 🥇", data_folder=DATA_FOLDER, timeframe=TIMEFRAME_MINOR, min_vol_usdt=MIN_VOL_USDT, order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=lists_for_grid)

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
