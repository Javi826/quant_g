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
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
#from ZZX_DRAFT2 import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
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
DATA_FOLDER         = "data/crypto_2023_UPTO"
DATE_MIN            = "2025-01-03"
TIMEFRAME           = '4H'
MIN_VOL_USDT        = 50_000

# -----------------------------------------------------------------------------
# GRID: 
# -----------------------------------------------------------------------------

SELL_AFTER_LIST     = [10,15,20,25]
ENTROPY_MAX_LIST    = [0.6,0.8,1.0,1.2,1.4,1.6]
ACCEL_SPAN_LIST     = [10,15,20]

TP_PCT_LIST         = [0,5,10]
SL_PCT_LIST         = [0,5,10]

# =============================================================================
# =============================================================================
# SELL_AFTER_LIST    = [15]
# ENTROPY_MAX_LIST   = [1.6]
# ACCEL_SPAN_LIST    = [20]
# 
# TP_PCT_LIST        = [0]
# SL_PCT_LIST        = [0]
# =============================================================================
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
    vol_window=50,
    date_min=DATE_MIN
)

save_filtered_symbols(filtered_symbols, strategy=STRATEGY, timeframe=TIMEFRAME, save_symbols=SAVE_SYMBOLS)

ohlcv_base = {}
for sym, df in ohlcv_data.items():
    ohlcv_base[sym] = {
        'ts': df.index.values.astype('datetime64[ns]'),
        'open': df['open'].to_numpy(dtype=np.float64),
        'high': df['high'].to_numpy(dtype=np.float64),
        'low': df['low'].to_numpy(dtype=np.float64),
        'close': df['close'].to_numpy(dtype=np.float64),
        'volume_quote': df['volume_quote'].to_numpy(dtype=np.float64) if 'volume_quote' in df.columns else np.zeros(len(df))
    }

# -----------------------------------------------------------------------------
# FUNCIÓN DE PROCESO PARA UNA COMBINACIÓN
# -----------------------------------------------------------------------------
def process_combo(comb):
    params       = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym, arrs in ohlcv_base.items():
        entropia, accel   = add_indicators_03(arrs['close'], m_accel=params.get('ACCEL_SPAN', 5))
        signal            = explosive_signal_03(entropia, accel, entropia_max=params.get('ENTROPY_MAX', 1.0), live=False)
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params.get('SELL_AFTER', 10),
        tp_pct=params.get('TP_PCT', 0),
        sl_pct=params.get('SL_PCT', 0),
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        comi_pct=0.05
    )
    return comb, results

# -----------------------------------------------------------------------------
# GENERAR TODAS LAS COMBINACIONES DEL GRID
# -----------------------------------------------------------------------------
all_combinations = list(product(*lists_for_grid))

# -----------------------------------------------------------------------------
# BACKTESTING PARALIZADO
# -----------------------------------------------------------------------------
with tqdm_joblib(tqdm(desc="🔁 Backtesting Grid... \n", total=len(all_combinations))) as progress:
    grid_results_list = Parallel(n_jobs=N_JOBS)(
        delayed(process_combo)(comb) for comb in all_combinations
    )

# -----------------------------------------------------------------------------
# COMPILAR RESULTADOS A DATAFRAME
# -----------------------------------------------------------------------------
grid_records = []
for comb, results in grid_results_list:
    port = results.get("__PORTFOLIO__", None)
    if port is None:
        continue

    net_gain      = np.sum(port['trades']) if len(port.get('trades', [])) > 0 else 0.0
    net_gain_pct  = (net_gain / INITIAL_BALANCE) * 100.0 if INITIAL_BALANCE != 0 else np.nan
    num_signals   = int(port.get('num_signals', 0))
    num_trades    = len(port.get('trades', []))
    win_ratio     = port.get('proportion_winners', np.nan)
    dd_pct        = port.get('max_dd', 0.0) * 100.0
    final_balance = float(port.get('final_balance', INITIAL_BALANCE))
    avg_trade     = np.nan if num_trades == 0 else np.mean(port['trades'])
    median_trade  = np.nan if num_trades == 0 else np.median(port['trades'])
    sharpe_ratio  = float(port.get('sharpe', np.nan))  # ADDED SHARPE

    row = {param: value for param, value in zip(param_names, comb)}
    row.update({
        "symbol": "__PORTFOLIO__",
        "Net_Gain": float(net_gain),
        "Net_Gain_pct": float(net_gain_pct),
        "Final_Balance": final_balance,
        "Num_Signals": num_signals,
        "Num_Trades": num_trades,
        "Win_Ratio": float(win_ratio) if not pd.isna(win_ratio) else np.nan,
        "Avg_Trade": float(avg_trade) if not pd.isna(avg_trade) else np.nan,
        "Median_Trade": float(median_trade) if not pd.isna(median_trade) else np.nan,
        "DD_pct": float(dd_pct),
        "Sharpe": sharpe_ratio  # ADDED SHARPE
    })
    grid_records.append(row)

grid_results_df = pd.DataFrame(grid_records, columns=[*param_names,"symbol", "Net_Gain", "Net_Gain_pct", "Final_Balance","Num_Signals", "Num_Trades", "Win_Ratio", "Avg_Trade", "Median_Trade", "DD_pct", "Sharpe"])  # ADDED SHARPE

# -----------------------------------------------------------------------------
# SAVE RESULTS + TIMING
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'), grid_results_df, filename=f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME}.xlsx",save=False)

print(f"\nTIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"DATE_MIN         : {DATE_MIN}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")
print(f"ENTROPY_MAX_LIST = {ENTROPY_MAX_LIST}")
print(f"ACCEL_SPAN_LIST  = {ACCEL_SPAN_LIST}")
print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}\n")

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
