# === FILE: main_BACKTESTING_dual.py ===
# --------------------------------------
import os
import time
import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

# Importa ambas versiones de backtest
from ZX_compute_BT import run_grid_backtest as run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
from backtest_cy import run_grid_backtest_cy

from ZX_analysis import report_backtesting
from ZX_utils import filter_symbols, save_results, save_filtered_symbols
from Z_add_signals_03 import add_indicators, explosive_signal

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
DATA_FOLDER  = "data/crypto_2023_highlow_UPTO"
DATE_MIN     = "2025-01-03"
TIMEFRAME    = '4H'
MIN_VOL_USDT = 500_000
SAVE_SYMBOLS = False

SELL_AFTER_LIST   = [10,20]
ENTROPY_MAX_LIST  = [0.4,0.6]
ACCEL_SPAN_LIST   = [5,10]
TP_PCT_LIST       = [0,5]
SL_PCT_LIST       = [0,5]

param_names    = ['SELL_AFTER', 'ENTROPY_MAX', 'ACCEL_SPAN', 'TP_PCT', 'SL_PCT']
lists_for_grid = [globals()[name + "_LIST"] for name in param_names]

# ---------------------------
# CARGA Y FILTRADO DE DATOS
# ---------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]

ohlcv_data, filtered_symbols, removed_symbols = filter_symbols(
    symbols,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50,
    date_min=DATE_MIN
)

save_filtered_symbols(filtered_symbols, strategy="generic", timeframe=TIMEFRAME, save_symbols=SAVE_SYMBOLS)

ohlcv_base = {}
for sym, df in ohlcv_data.items():
    ohlcv_base[sym] = {
        'ts': df.index.values.astype('datetime64[ns]'),
        'open': df['open'].to_numpy(dtype=np.float64),
        'high': df['high'].to_numpy(dtype=np.float64),
        'low': df['low'].to_numpy(dtype=np.float64),
        'close': df['close'].to_numpy(dtype=np.float64),
        'volume': df['volume'].to_numpy(dtype=np.float64) if 'volume' in df.columns else np.zeros(len(df))
    }

# ---------------------------
# FUNCIONES DE PROCESO
# ---------------------------
def process_combo(comb, version='py'):
    params       = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym, arrs in ohlcv_base.items():
        entropia, accel   = add_indicators(arrs['close'], m_accel=params.get('ACCEL_SPAN', 5))
        signal            = explosive_signal(entropia, accel, entropia_max=params.get('ENTROPY_MAX', 1.0), live=False)
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}

    if version == 'py':
        return comb, run_grid_backtest(
            ohlcv_arrays,
            sell_after=params.get('SELL_AFTER', 10),
            initial_balance=INITIAL_BALANCE,
            order_amount=ORDER_AMOUNT,
            tp_pct=params.get('TP_PCT', 0),
            sl_pct=params.get('SL_PCT', 0),
            comi_pct=0.05
        )
    else:
        return comb, run_grid_backtest_cy(
            ohlcv_arrays,
            sell_after=params.get('SELL_AFTER', 10),
            initial_balance=INITIAL_BALANCE,
            order_amount=ORDER_AMOUNT,
            tp_pct=params.get('TP_PCT', 0),
            sl_pct=params.get('SL_PCT', 0),
            comi_pct=0.05
        )

# ---------------------------
# GENERAR GRID
# ---------------------------
all_combinations = list(product(*lists_for_grid))

# ---------------------------
# BACKTEST PARALIZADO PYTHON
# ---------------------------
start_time = time.time()
with tqdm_joblib(tqdm(desc="🔁 Backtesting Python Grid...", total=len(all_combinations))) as progress:
    py_results_list = Parallel(n_jobs=-1)(
        delayed(process_combo)(comb, version='py') for comb in all_combinations
    )
elapsed_py = int(time.time() - start_time)

# ---------------------------
# BACKTEST PARALIZADO CYTHON
# ---------------------------
start_time = time.time()
with tqdm_joblib(tqdm(desc="🔁 Backtesting Cython Grid...", total=len(all_combinations))) as progress:
    cy_results_list = Parallel(n_jobs=-1)(
        delayed(process_combo)(comb, version='cy') for comb in all_combinations
    )
elapsed_cy = int(time.time() - start_time)

# ---------------------------
# VALIDACIÓN DE RESULTADOS
# ---------------------------
for ((comb_py, res_py), (comb_cy, res_cy)) in zip(py_results_list, cy_results_list):
    assert comb_py == comb_cy, "Combinaciones diferentes"
    final_py = res_py["__PORTFOLIO__"]["final_balance"]
    final_cy = res_cy["__PORTFOLIO__"]["final_balance"]
    if not np.isclose(final_py, final_cy, atol=1e-12):
        print(f"⚠️ Diferencia detectada en {comb_py}: PY={final_py}, CY={final_cy}")

print(f"✅ Backtests completados. Tiempo PY: {elapsed_py}s, Tiempo CY: {elapsed_cy}s")
