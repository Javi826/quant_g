# === FILE: main_grid_flag ===
# -----------------------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results, save_all_trades_to_excel, save_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_filtered_symbols, final_prints, save_equity_to_excel
from signals.add_signals_flag import flag_long
from signals.add_signals_flag import flag_short

start_time   = time.time()
SAVE_SYMBOLS = True
MY_SYMBOLS   = False
STRATEGY     = "flag_short_4H_OOSx"
N_JOBS       = -1

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER         = "../data/crypto_OOS_2025"
DATA_FOLDER         = "../data/crypto_OOS_2026"
DATA_FOLDER         = "../data/crypto_2022_OOS"
#DATA_FOLDER         = "../data/crypto_2022_IS"
TIMEFRAME_MINOR     = '4H'

ORDER_AMOUNT        = 80
MIN_VOL_USDT        = 1_000_000

# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------

# =============================================================================
# #4HLONG-9(2025) 
# SELL_AFTER_LIST    = [0]  
# LOOKBACK_LIST      = [15]
# IMPULSE_LIST       = [5]
# FLAG_LIST          = [40]
# MA_PERIOD_LIST     = [50]
# 
# TP_PCT_LIST        = [4]
# SL_PCT_LIST        = [10]
# =============================================================================

# =============================================================================
# #4HSHORT-23(2025) 
# SELL_AFTER_LIST    = [0]  
# LOOKBACK_LIST      = [10]
# IMPULSE_LIST       = [3]
# FLAG_LIST          = [50]
# MA_PERIOD_LIST     = [50]
# 
# TP_PCT_LIST        = [3]
# SL_PCT_LIST        = [9]
# =============================================================================


#1HSHORT-9 
SELL_AFTER_LIST    = [0]  
LOOKBACK_LIST      = [10]
IMPULSE_LIST       = [3]
FLAG_LIST          = [50]
MA_PERIOD_LIST     = [50]

TP_PCT_LIST        = [3]
SL_PCT_LIST        = [9]


param_names = ['SELL_AFTER','LOOKBACK','IMPULSE','FLAG','MA_PERIOD','TP_PCT','SL_PCT']
param_ranges   = {name: globals()[f"{name}_LIST"] for name in param_names}
lists_for_grid = [param_ranges[name] for name in param_names]

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50, my_symbols=MY_SYMBOLS)
save_filtered_symbols(filtered_minor, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR, save_symbols=SAVE_SYMBOLS)

ohlcv_arr_minor = prepare_ohlcv_arrays(ohlcv_data_minor)
print(f"Symbols ({len(ohlcv_arr_minor)}): {sorted(ohlcv_arr_minor.keys())}")
# -----------------------------------------------------------------------------
# FUNCTION TO PROCESS ONE PARAMETER COMBINATION
# -----------------------------------------------------------------------------
def process_combo(comb):
    params = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym in ohlcv_arr_minor.keys():
        arr_minor = ohlcv_arr_minor[sym]

        signals = flag_short(
                arr_minor,
                lookback=params['LOOKBACK'],
                impulse=params['IMPULSE'],
                flag=params['FLAG'],
                ma_period=params['MA_PERIOD'],
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
save_all_trades_to_excel(grid_results_list, param_names, f"all_trades_{STRATEGY}.xlsx", strategy_name=STRATEGY, save=False)
save_equity_to_excel(grid_results_list, "../brief_equities", INITIAL_BALANCE, STRATEGY, save_file=False)

final_prints(f" 🥇 Grid_{STRATEGY} 🥇", DATA_FOLDER, f"{TIMEFRAME_MINOR}", MIN_VOL_USDT, ORDER_AMOUNT, param_names, lists_for_grid)

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")