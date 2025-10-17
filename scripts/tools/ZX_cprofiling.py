import time
import cProfile
import pstats
import os
import numpy as np
from itertools import product
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
#from ZZX_DRAFT2 import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
from utils.ZX_utils import filter_symbols
from Z_add_signals_03 import add_indicators_03, explosive_signal_03
from collections import defaultdict
from joblib import Parallel, delayed

# ==============================
# Configuración idéntica a main_BACKTESTING.py
# ==============================
DATA_FOLDER  = "data/crypto_2023_UPTO"
DATE_MIN     = "2025-01-03"
TIMEFRAME    = '4H'
MIN_VOL_USDT = 500_000

SELL_AFTER_LIST    = [20,30,40,50]
ENTROPY_MAX_LIST   = [0.6,0.8,1.0,2.0]
ACCEL_SPAN_LIST    = [5,10,15]

TP_PCT_LIST        = [0,5,10]
SL_PCT_LIST        = [0,5,10]

# ==============================
# Cargar y filtrar símbolos
# ==============================
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

# ==============================
# Wrapper para profiling de una combinación
# ==============================
def run_profiled(sell_after, entropy_max, accel_span, tp_pct, sl_pct):
    ohlcv_arrays = {}
    for sym, arrs in ohlcv_base.items():
        entropia, accel = add_indicators_03(arrs['close'], m_accel=accel_span)
        signal = explosive_signal_03(entropia, accel, entropia_max=entropy_max, live=False)
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=sell_after,
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        comi_pct=0.05
    )
    return results

# ==============================
# Ejecutar profiling para todas las combinaciones y acumular resultados
# ==============================
all_combinations = list(product(
    SELL_AFTER_LIST,
    ENTROPY_MAX_LIST,
    ACCEL_SPAN_LIST,
    TP_PCT_LIST,
    SL_PCT_LIST
))

# Diccionario para acumular stats
accumulated_stats = defaultdict(lambda: {'calls': 0, 'time_total': 0.0, 'time_cum': 0.0})


# ==============================
# Lista de funciones que queremos trackear
# ==============================
local_functions = [
    "run_grid_backtest",
    "get_price_at_int",
    "prepare_data",
    "close_position",
    "close_expired_positions",
    "detect_intrabar_exit",
    "build_results_dict",
    "compute_post_backtest_metrics",
    "update_sim_balance",
    "execute_signal",
    "process_signals_for_timestamp",
    "close_all_remaining_positions",
    "initialize_backtest_structures",
    "run_backtest_loop"
]

# ==============================
# Wrapper de profiling para un worker
# ==============================
def profiled_worker(comb):
    sell_after, entropy_max, accel_span, tp_pct, sl_pct = comb

    profiler = cProfile.Profile()
    profiler.enable()
    # ==============================
    # Ejecutamos la función principal
    # ==============================
    ohlcv_arrays = {}
    for sym, arrs in ohlcv_base.items():
        entropia, accel = add_indicators_03(arrs['close'], m_accel=accel_span)
        signal = explosive_signal_03(entropia, accel, entropia_max=entropy_max, live=False)
        ohlcv_arrays[sym] = {**arrs, 'signal': signal}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=sell_after,
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        comi_pct=0.05
    )
    profiler.disable()

    # ==============================
    # Extraer stats de cProfile y filtrar solo funciones de interés
    # ==============================
    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    worker_stats = defaultdict(lambda: {'calls': 0, 'time_total': 0.0, 'time_cum': 0.0})
    for func_tuple, func_stats in stats.stats.items():
        filename, line, func_name = func_tuple
        if func_name in local_functions:
            cc, nc, tt, ct, callers = func_stats
            worker_stats[func_name]['calls'] += nc
            worker_stats[func_name]['time_total'] += tt
            worker_stats[func_name]['time_cum'] += ct

    return comb, results, worker_stats

# ==============================
# Ejecutar paralelizado y consolidar stats
# ==============================
all_combinations = list(product(SELL_AFTER_LIST, ENTROPY_MAX_LIST, ACCEL_SPAN_LIST, TP_PCT_LIST, SL_PCT_LIST))
accumulated_stats = defaultdict(lambda: {'calls': 0, 'time_total': 0.0, 'time_cum': 0.0})

# ==============================
# Medir tiempo total de ejecución
# ==============================
start_time = time.time()

grid_results_list = Parallel(n_jobs=-1)(delayed(profiled_worker)(comb) for comb in all_combinations)

elapsed_time = time.time() - start_time

# ==============================
# Combinar stats de todos los workers
# ==============================
accumulated_stats = defaultdict(lambda: {'calls': 0, 'time_total': 0.0, 'time_cum': 0.0})
for comb, results, worker_stats in grid_results_list:
    for fn, values in worker_stats.items():
        accumulated_stats[fn]['calls'] += values['calls']
        accumulated_stats[fn]['time_total'] += values['time_total']
        accumulated_stats[fn]['time_cum'] += values['time_cum']

# ==============================
# Mostrar resultados consolidados
# ==============================
total_time_cum = sum(values['time_cum'] for values in accumulated_stats.values())

print(f"\n🏁 Total execution time: {int(elapsed_time//3600)} h {(int(elapsed_time)%3600)//60} min {int(elapsed_time)%60} s\n")
print(f"{'Función':<30} {'Llamadas totales':>15} {'Tiempo total':>15} {'Tiempo cumul.':>15} {'% Total':>10}")
print("-"*95)
for fn, values in sorted(accumulated_stats.items(), key=lambda x: x[1]['time_cum'], reverse=True):
    pct_total = (values['time_cum'] / total_time_cum * 100) if total_time_cum > 0 else 0
    print(f"{fn:<30} {values['calls']:>15} {values['time_total']:15.6f} {values['time_cum']:15.6f} {pct_total:10.0f} %")


