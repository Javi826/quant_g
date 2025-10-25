# === FILE: main_BACKTESTING.py ===
# ---------------------------------
import os
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, MIN_PRICE,INITIAL_BALANCE
#from ZZX_DRAFT1 import run_grid_backtest, MIN_PRICE,INITIAL_BALANCE
from tools.ZX_st_tools import prepare_ohlcv_arrays,compile_grid_results,save_all_trades_to_excel,save_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_filtered_symbols,final_prints
from Z_add_signals_01 import explosive_signal_08

start_time         = time.time()
SAVE_SYMBOLS       = False
STRATEGY           ="candle_pair"
N_JOBS             =-1
# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_OOS"
DATA_FOLDER         = "data/crypto_2022_OOS"
#DATA_FOLDER         = "data/crypto_2023_IS"
TIMEFRAME           = '1D'
ORDER_AMOUNT        = 500
MIN_VOL_USDT        = 50_000
ORDER_AMOUNT        = 5_000
MIN_VOL_USDT        = 10_000_000

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------

SELL_AFTER_LIST     = [0]
LOOKBACK_LIST       = [5,10,20,30,40,50,60]
BODY_TOLERANCE_LIST = [5,10,15,20,25]
LOW_TOLERANCE_LIST  = [5,10,15,20,25]

TP_PCT_LIST         = [5,10,20,30,50,100]
SL_PCT_LIST         = [5,10,15,20]

SELL_AFTER_LIST     = [0]
LOOKBACK_LIST       = [10]
BODY_TOLERANCE_LIST = [20]
LOW_TOLERANCE_LIST  = [5]

TP_PCT_LIST         = [100]
SL_PCT_LIST         = [5]


param_names    = ['SELL_AFTER', 'LOOKBACK', 'BODY_TOLERANCE','LOW_TOLERANCE', 'TP_PCT', 'SL_PCT']
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
    params       = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym, arrs in ohlcv_arr.items():
        open_array = arrs["open"]
        close_array = arrs["close"]
        low_array = arrs["low"]

        signal = explosive_signal_08(
            open_prices=open_array,
            close_prices=close_array,
            low_prices=low_array,
            lookback=params["LOOKBACK"],
            body_tolerance=params["BODY_TOLERANCE"] / 100,  
            low_tolerance=params["LOW_TOLERANCE"] / 100,   
            live=False,
        )
        
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params['SELL_AFTER'],
        tp_pct=params['TP_PCT'],
        sl_pct=params['SL_PCT'],
        order_amount=ORDER_AMOUNT
        
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
save_all_trades_to_excel(grid_results_list, param_names, filename=f"all_trades_{TIMEFRAME}.xlsx", save=False)

final_prints(strategy=f" 🥇 Grid_Backest {STRATEGY} 🥇", data_folder=DATA_FOLDER, timeframe=TIMEFRAME, min_vol_usdt=MIN_VOL_USDT, order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=lists_for_grid)

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names,data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
