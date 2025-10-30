# ============================================================
# TEST DE SINCRONIZACIÓN Y SEÑALES PARA EXPLOSIVE_SIGNAL_TF
# ============================================================

import os
import time
import numpy as np
import pandas as pd
from itertools import product
from utils.ZX_utils import filter_symbols
from ZX_compute_BT import MIN_PRICE
from tools.ZX_optimize_MCf_tf import generate_multiple_paths, derive_major_from_minor
from Z_add_signals_tf import explosive_signal_tf

# -----------------------------------------------------------
# CONFIGURACIÓN BÁSICA (igual que el main)
# -----------------------------------------------------------
DTYPE             = np.float32
DATA_FOLDER       = "data/crypto_2023_IS"
TIMEFRAME_MAJOR   = "1Dutc"
TIMEFRAME_MINOR   = "12Hutc"
MIN_VOL_USDT      = 100_000_000
FINAL_N_PATHS     = 2
FINAL_N_OBS_PER_PATH = 360  # para 12Hutc
N_JOBS = -1

# -----------------------------------------------------------
# CARGA Y FILTRA DATOS
# -----------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER)
                 if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]

ohlcv_data_minor, filtered_minor = filter_symbols(
    symbols_minor,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME_MINOR,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50
)

if not ohlcv_data_minor:
    raise ValueError("No se cargaron símbolos, revisa el filtro de volumen o la carpeta de datos.")

symbol = list(ohlcv_data_minor.keys())[0]
df_hist = ohlcv_data_minor[symbol]
print(f"\n📊 Usando símbolo: {symbol}, {len(df_hist)} velas")

# -----------------------------------------------------------
# GENERACIÓN DE MONTECARLO Y DERIVACIÓN DE MAYOR
# -----------------------------------------------------------
paths_minor = generate_multiple_paths(df_hist, n_paths=FINAL_N_PATHS, n_obs=FINAL_N_OBS_PER_PATH)
print(f"✅ paths_minor.shape = {paths_minor.shape}")

factor = 2  # 12H → 1D
paths_major = derive_major_from_minor(paths_minor, factor=factor)
print(f"✅ paths_major.shape = {paths_major.shape}")

# -----------------------------------------------------------
# SELECCIONAR UN PATH Y EXTRAER ARRAYS
# -----------------------------------------------------------
path_idx = 0
arr_minor = paths_minor[path_idx]
arr_major = paths_major[path_idx]

# Extraemos columnas
open_menor, low_menor, high_menor, close_menor = arr_minor[:, 0], arr_minor[:, 1], arr_minor[:, 2], arr_minor[:, 3]
open_mayor, low_mayor, high_mayor, close_mayor = arr_major[:, 0], arr_major[:, 1], arr_major[:, 2], arr_major[:, 3]

ts_menor = pd.to_datetime(arr_minor[:, 6], unit='s')
ts_mayor = pd.to_datetime(arr_major[:, 6], unit='s')

print("\n🕒 Rangos de tiempo:")
print(f"Minor: {ts_menor.min()} → {ts_menor.max()}  ({len(ts_menor)} velas)")
print(f"Major: {ts_mayor.min()} → {ts_mayor.max()}  ({len(ts_mayor)} velas)")

# -----------------------------------------------------------
# TEST DE SEÑALES
# -----------------------------------------------------------
lookback_mayor = 1
lookback_menor = 1

signal = explosive_signal_tf(
    high_mayor=high_mayor,
    close_mayor=close_mayor,
    high_menor=high_menor,
    close_menor=close_menor,
    lookback_mayor=lookback_mayor,
    lookback_menor=lookback_menor,
    index_mayor=ts_mayor,
    index_menor=ts_menor,
    live=False
)

# -----------------------------------------------------------
# VALIDACIONES / PRINTS DETALLADOS
# -----------------------------------------------------------
print("\n📈 Señales generadas:")
print(f"Total = {len(signal)} | Señales = {np.sum(signal)} | Porcentaje = {100*np.mean(signal):.2f}%")

print("\n🔍 Muestras de señal (índices y timestamps donde signal==1):")
idxs = np.where(signal == 1)[0]
if len(idxs) > 0:
    for idx in idxs[:10]:
        print(f"  idx={idx}  ts={ts_menor[idx]}  close_menor={close_menor[idx]:.2f}")
else:
    print("❌ Ninguna señal encontrada")

# -----------------------------------------------------------
# DEPURACIÓN ADICIONAL (comprobar coincidencias mayor-menor)
# -----------------------------------------------------------
print("\n🧠 Depuración de sincronización:")
for i in range(0, len(ts_menor), len(ts_menor)//5):
    ts_minor_now = ts_menor[i]
    idx_major = np.where(ts_mayor + pd.Timedelta(hours=24) <= ts_minor_now)[0]
    last_idx = idx_major[-1] if len(idx_major) > 0 else None
    print(f"Minor[{i}]={ts_minor_now}, Última major cerrada idx={last_idx}")

print("\n✅ Test finalizado correctamente.")
