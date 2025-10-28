import time
import os
import io
from itertools import product
from collections import defaultdict
from joblib import Parallel, delayed
from line_profiler import LineProfiler  # pip install line_profiler

from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
import ZX_compute_BT as bt_mod
from utils.ZX_utils import filter_symbols
from Z_add_signals_01 import explosive_signal_01
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results

# ==============================
# Configuración
# ==============================
DATA_FOLDER  = "data/crypto_2023_IS"
TIMEFRAME    = '4H'
MIN_VOL_USDT = 500_000
ORDER_AMOUNT = 100

SELL_AFTER_LIST    = [20,30,40,50]
ENTROPY_MAX_LIST   = [0.6,0.8,1.0,2.0]
ACCEL_SPAN_LIST    = [5,10,15]
TP_PCT_LIST        = [0,5,10]
SL_PCT_LIST        = [0,5,10]

# ==============================
# Cargar y filtrar símbolos
# ==============================
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]
ohlcv_data, filtered_symbols = filter_symbols(
    symbols,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50
)
ohlcv_base = prepare_ohlcv_arrays(ohlcv_data)

# ==============================
# Lista de funciones que queremos perfilar por línea
# ==============================
local_functions = [
    "run_grid_backtest", "get_price_at_int", "prepare_data", "close_position",
    "close_expired_positions", "detect_intrabar_exit", "build_results_dict",
    "compute_post_backtest_metrics", "update_sim_balance", "execute_signal",
    "process_signals_for_timestamp", "initialize_backtest_structures", "run_backtest_loop"
]

# ==============================
# Worker con line profiler
# ==============================
def profiled_worker(comb):
    sell_after, entropy_max, accel_span, tp_pct, sl_pct = comb

    lp = LineProfiler()
    # Añadimos funciones de interés
    try: lp.add_function(run_grid_backtest)
    except: pass
    try: lp.add_function(explosive_signal_01)
    except: pass
    for name in local_functions:
        if hasattr(bt_mod, name):
            try:
                lp.add_function(getattr(bt_mod, name))
            except:
                pass

    def worker_run():
        ohlcv_arrays = {}
        for sym, arrs in ohlcv_base.items():
            signal = explosive_signal_01(arrs['close'], m_accel=accel_span, entropia_max=entropy_max, live=False)
            ohlcv_arrays[sym] = {**arrs, 'signal': signal}
        results = run_grid_backtest(
            ohlcv_arrays,
            sell_after=sell_after,
            order_amount=ORDER_AMOUNT,
            tp_pct=tp_pct,
            sl_pct=sl_pct
        )
        return results

    lp_wrapper = lp(worker_run)
    start = time.time()
    results = lp_wrapper()
    elapsed = time.time() - start

    # Extraer datos del profiler
    data = []
    for (fn, lineno, name), stats in lp.code_map.items():
        times = stats.timings
        for line_no, nhits, time_ns in times:
            data.append({
                'file': fn,
                'func': name,
                'line': line_no,
                'hits': nhits,
                'time': time_ns / 1e6  # convertir a milisegundos
            })
    return data, elapsed


# ==============================
# Ejecutar todas las combinaciones (puedes ajustar n_jobs)
# ==============================
all_combinations = list(product(
    SELL_AFTER_LIST,
    ENTROPY_MAX_LIST,
    ACCEL_SPAN_LIST,
    TP_PCT_LIST,
    SL_PCT_LIST
))

N_JOBS = 1  # usa 1 para un perfil consolidado (recomendado)

start_time = time.time()
all_data = Parallel(n_jobs=N_JOBS)(delayed(profiled_worker)(comb) for comb in all_combinations)
elapsed_total = time.time() - start_time

# ==============================
# Consolidar resultados de todos los workers
# ==============================
merged = defaultdict(lambda: {'time': 0.0, 'hits': 0})
for worker_data, _ in all_data:
    for rec in worker_data:
        key = (rec['file'], rec['func'], rec['line'])
        merged[key]['time'] += rec['time']
        merged[key]['hits'] += rec['hits']

# Calcular total y top 10
total_time = sum(v['time'] for v in merged.values())
top10 = sorted(merged.items(), key=lambda x: x[1]['time'], reverse=True)[:10]

# ==============================
# Mostrar top 10 líneas más lentas
# ==============================
print(f"\n🏁 Tiempo total ejecución: {elapsed_total:.2f} s\n")
print(f"{'Archivo':40} {'Función':25} {'Línea':>6} {'Tiempo (ms)':>12} {'% Total':>9} {'Hits':>8}")
print("-"*95)
for (fn, func, line), v in top10:
    pct = (v['time'] / total_time * 100) if total_time > 0 else 0
    print(f"{os.path.basename(fn):40} {func:25} {line:6d} {v['time']:12.2f} {pct:9.2f} {v['hits']:8d}")
