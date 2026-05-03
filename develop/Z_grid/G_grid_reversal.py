# === FILE: main_grid===
# -----------------------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "signals")))
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from joblib import Parallel, delayed
from shared.backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from shared.utils.torque import prepare_ohlcv_arrays, compile_grid_results, save_all_trades_to_csv, save_results,save_equity_to_excel
from shared.utils.analysis import report_backtesting

from shared.utils.utils import filter_symbols, save_filtered_symbols, final_prints
from signals.add_signals_reversal import reversal_long
from signals.add_signals_reversal import reversal_short
import logging
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(message)s"))

logger_bot = logging.getLogger("BOT_batch")
logger_bot.setLevel(logging.DEBUG)
if not logger_bot.handlers:
    logger_bot.addHandler(handler)
logger_bot.propagate = False

start_time   = time.time()
SAVE_SYMBOLS = False
MY_SYMBOLS   = True
STRATEGY     = "reversal_long_1H_OOS"
N_JOBS       = -1

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SPLIT_MODE = "expanding"
SPLIT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER = os.path.join(SPLIT_BASE, "OOS", "crypto_2026-03_2026-04_OOS")
TIMEFRAME_MINOR     = '6Hutc'

ORDER_AMOUNT        = 80
MIN_VOL_USDT        = 3_000_000

# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------

SELL_AFTER_LIST      = [16]
LOOKBACK_LIST        = [6]
MA_PERIOD_LIST       = [25]
TOLERANCE_LIST       = [30]

TP_PCT_LIST          = [4]
SL_PCT_LIST          = [7.5]


param_names    = ['SELL_AFTER','LOOKBACK','TOLERANCE','MA_PERIOD','TP_PCT','SL_PCT']
param_ranges   = {name: globals()[f"{name}_LIST"] for name in param_names}
lists_for_grid = [param_ranges[name] for name in param_names]

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50,my_symbols=MY_SYMBOLS)

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

        signals = reversal_short(
            arr_minor,
            lookback=params['LOOKBACK'],
            tolerance=params['TOLERANCE'],
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
# COMPILE RESULTS INTO DATAFRAMEFalse
# -----------------------------------------------------------------------------
grid_records    = compile_grid_results(grid_results_list, param_names, INITIAL_BALANCE)
grid_results_df = pd.DataFrame(grid_records)

# -----------------------------------------------------------------------------
# SAVE RESULTS + EXECUTION TIME
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'), grid_results_df, f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME_MINOR}.xlsx", save=False)
save_all_trades_to_csv(grid_results_list, param_names, f"all_trades_{STRATEGY}.xlsx", strategy_name=STRATEGY, save=False)
save_equity_to_excel(grid_results_list,"../brief_equities", INITIAL_BALANCE,STRATEGY,save_file=False)

final_prints(f" 🥇 Grid_{STRATEGY} 🥇", DATA_FOLDER, f"{TIMEFRAME_MINOR}", MIN_VOL_USDT, ORDER_AMOUNT, param_names, lists_for_grid)

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
