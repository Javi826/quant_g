# === FILE: main_BACKTESTING_EMA_RSI_ATR.py ===
# ---------------------------------
import os
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT, COMISION
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_results, save_filtered_symbols
from Z_add_signals_03I import explosive_signal_03I  # nueva función EMA+RSI+ATR

start_time         = time.time()
SAVE_SYMBOLS       = False
STRATEGY           = "EMA_RSI_ATR"
N_JOBS             = -1

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER         = "data/crypto_2023_ISOLD"
TIMEFRAME           = '4H'
MIN_VOL_USDT        = 50_000

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------
SELL_AFTER_LIST     = [1,5,10,15]
EMA_FAST_LIST       = [5,10,15,20]
EMA_SLOW_LIST       = [50,100,200]
RSI_PERIOD_LIST     = [10,15,20]
ATR_PERIOD_LIST     = [10,15,20]
TP_PCT_LIST         = [0,5,10]
SL_PCT_LIST         = [0,5,10]

param_names    = ['SELL_AFTER','EMA_FAST','EMA_SLOW','RSI_PERIOD','ATR_PERIOD','TP_PCT','SL_PCT']
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
        # señal EMA+RSI+ATR
        signal, atr = explosive_signal_03I(
            close=arrs['close'],
            high=arrs['high'],
            low=arrs['low'],
            ema_fast=params.get('EMA_FAST'),
            ema_slow=params.get('EMA_SLOW'),
            rsi_period=params.get('RSI_PERIOD'),
            atr_period=params.get('ATR_PERIOD')
        )
        ohlcv_arrays[sym] = {**arrs, 'signal': signal, 'atr': atr}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params['SELL_AFTER'],
        tp_pct=params['TP_PCT'],      # porcentaje de take-profit
        sl_pct=params['SL_PCT'],      # múltiplo del ATR
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
save_results(grid_results_df.to_dict('records'), grid_results_df, filename=f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME}_EMA_RSI_ATR.xlsx", save=False)
print('\n🥇==Grid_backtest EMA+RSI+ATR==🥇')
print(f"TIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")
print(f"EMA_FAST_LIST    = {EMA_FAST_LIST}")
print(f"EMA_SLOW_LIST    = {EMA_SLOW_LIST}")
print(f"RSI_PERIOD_LIST  = {RSI_PERIOD_LIST}")
print(f"ATR_PERIOD_LIST  = {ATR_PERIOD_LIST}")
print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}\n")

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
