# === FILE: G_grid_entropy.py ===
# ---------------------------------
import os
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, MIN_PRICE,INITIAL_BALANCE
from tools.ZX_st_tools import prepare_ohlcv_arrays,compile_grid_results,save_all_trades_to_excel,save_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_filtered_symbols, final_prints,save_equity_to_excel
from Z_add_signals_scalping import scalping_long
from Z_add_signals_scalping import scalping_short

start_time         = time.time()
SAVE_SYMBOLS       = False
STRATEGY           ="scalping"
N_JOBS             =-1
# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2025_scalping_OOS"
#DATA_FOLDER         = "data/crypto_2023_IS"
TIMEFRAME_MINOR     = '15m'
ORDER_AMOUNT        = 400
MIN_VOL_USDT        = 10_000_000

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------

SELL_AFTER_LIST = [0]

RSI_LIST        = [30,40,50]
ADX_LIST        = [25,30,35]
LOOKBACK_LIST   = [10,20,30,40,50]
TORELANCE_LIST  = [2,5,10,15]

TP_PCT_LIST     = [2.0,2.5,3.0,3.5,4.0,4.5,5.0]
SL_PCT_LIST     = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,7.5,10]

SELL_AFTER_LIST = [0]

RSI_LIST        = [30]
ADX_LIST        = [35]
LOOKBACK_LIST   = [50]
TORELANCE_LIST  = [2]

TP_PCT_LIST     = [4.5]
SL_PCT_LIST     = [2]


param_names = ['SELL_AFTER','RSI','ADX','LOOKBACK','TORELANCE','TP_PCT','SL_PCT']

lists_for_grid = [globals()[name + "_LIST"] for name in param_names]

# -----------------------------------------------------------------------------
# CARGA Y FILTRADO DE DATOS
# -----------------------------------------------------------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]

ohlcv_data, filtered_symbols = filter_symbols(symbols,min_vol_usdt=MIN_VOL_USDT,timeframe=TIMEFRAME_MINOR,data_folder=DATA_FOLDER,min_price=MIN_PRICE,vol_window=50,my_symbols=True)

save_filtered_symbols(filtered_symbols, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR, save_symbols=SAVE_SYMBOLS)
ohlcv_arr = prepare_ohlcv_arrays(ohlcv_data)


# -----------------------------------------------------------------------------
# FUNCIÓN DE PROCESO PARA UNA COMBINACIÓN
# -----------------------------------------------------------------------------
def process_combo(comb):
    params       = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym, arrs in ohlcv_arr.items():
        signal = scalping_long(
            arrs,
            rsi_max=params.get('RSI'),
            adx_min=params.get('ADX'),
            lookback=params.get('LOOKBACK'),
            tolerance=params.get('TORELANCE'),
            live_trading=False
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

final_prints(f" 🥇 Grid_{STRATEGY} 🥇", DATA_FOLDER, f"{TIMEFRAME_MINOR}", MIN_VOL_USDT, ORDER_AMOUNT, param_names, lists_for_grid)

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
