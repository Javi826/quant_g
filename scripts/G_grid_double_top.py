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
from utils.ZX_utils import filter_symbols, save_filtered_symbols, final_prints,save_equity_to_excel
from Z_add_signals_double_top import double_top_short
from Z_add_signals_double_top import double_top_long

start_time   = time.time()
SAVE_SYMBOLS = False
STRATEGY     = "double_top_long_4H"
N_JOBS       = -1

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_OOS"
#DATA_FOLDER         = "data/crypto_2021_OOS"
#DATA_FOLDER         = "data/crypto_2022_OOS"
#DATA_FOLDER         = "data/crypto_2023_IS"
TIMEFRAME_MINOR     = '4H'

ORDER_AMOUNT        = 80
MIN_VOL_USDT        = 10_000_000


# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------
SELL_AFTER_LIST      = [0]  
LOOKBACK_MINOR_LIST  = [2,3,4,5] 
PRICE_TOLERANCE_LIST = [10,20,30,40] 
TREND_TH_LIST        = [10,20,30,40] 

TP_PCT_LIST          = [3,4,5,6,7,8,9]
SL_PCT_LIST          = [3,4,5,6,7,8,9]

# =============================================================================
# SELL_AFTER_LIST      = [0]  
# LOOKBACK_MINOR_LIST  = [2] 
# PRICE_TOLERANCE_LIST = [10] 
# TREND_TH_LIST        = [10] 
# 
# TP_PCT_LIST          = [5]
# SL_PCT_LIST          = [10]
# =============================================================================


param_names    = ['SELL_AFTER','LOOKBACK_MINOR','PRICE_TOLERANCE','TREND_TH','TP_PCT','SL_PCT']
param_ranges   = {name: globals()[f"{name}_LIST"] for name in param_names}
lists_for_grid = [param_ranges[name] for name in param_names]

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50)

save_filtered_symbols(filtered_minor, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR, save_symbols=SAVE_SYMBOLS)

ohlcv_arr_minor = prepare_ohlcv_arrays(ohlcv_data_minor)

# -----------------------------------------------------------------------------
# FUNCTION TO PROCESS ONE PARAMETER COMBINATION
# -----------------------------------------------------------------------------
def process_combo(comb):
    params = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym in ohlcv_arr_minor.keys():
        arr_minor = ohlcv_arr_minor[sym]

        signals = double_top_long(
            arr_minor,
            lookback_minor=params['LOOKBACK_MINOR'],
            price_tolerance=params['PRICE_TOLERANCE'], 
            trend_th=params['TREND_TH'], 
            live_trading=False
        )

        ohlcv_arrays[sym] = {**arr_minor, 'signal': signals}

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
with tqdm_joblib(tqdm(desc="🔄 Backtesting Grid... \n", total=len(all_combinations))) as progress:
    grid_results_list = Parallel(n_jobs=N_JOBS)(
        delayed(process_combo)(comb) for comb in all_combinations
    )

# -----------------------------------------------------------------------------
# COMPILE RESULTS INTO DATAFRAME
# -----------------------------------------------------------------------------
grid_records    = compile_grid_results(grid_results_list, param_names, INITIAL_BALANCE)
grid_results_df = pd.DataFrame(grid_records)

# -----------------------------------------------------------------------------
# SAVE RESULTS + EXECUTION TIME
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'), grid_results_df, f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME_MINOR}.xlsx", save=False)
save_all_trades_to_excel(grid_results_list, param_names,f"all_trades_{TIMEFRAME_MINOR}.xlsx", save=False)
save_equity_to_excel(grid_results_list,"brief_equities", INITIAL_BALANCE,STRATEGY,save_file=False)

final_prints(f" 🥇Grid_{STRATEGY} 🥇", DATA_FOLDER, f"{TIMEFRAME_MINOR}", MIN_VOL_USDT, ORDER_AMOUNT, param_names, lists_for_grid)

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
