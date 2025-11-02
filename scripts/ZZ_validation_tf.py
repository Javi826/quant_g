import numpy as np
import pandas as pd
from Z_add_signals_tf import (
    signal_minor_tf,
    signal_major_tf,
    explosive_signal_tf,
    get_last_closed_major_bar
)

# ==========================================================
# 🔹 Datos del timeframe mayor (1D)
# ==========================================================
major_data = [
    ["2025-10-18", 100, 110, 100],
    ["2025-10-19", 100, 105, 100],
    ["2025-10-20", 100, 102, 100],
    ["2025-10-21", 100, 103, 100],
    ["2025-10-22", 100, 100, 100],
]

ts_mayor = pd.to_datetime([row[0] for row in major_data])
high_mayor = np.array([row[1] for row in major_data], dtype=float)
close_mayor = np.array([row[2] for row in major_data], dtype=float)
low_mayor = np.array([row[3] for row in major_data], dtype=float)

arr_major = {
    "ts": ts_mayor,
    "high": high_mayor,
    "close": close_mayor,
    "low": low_mayor
}

# ==========================================================
# 🔹 Datos del timeframe menor (4H)
# ==========================================================
minor_data = [
    ["2025-10-19 00:00", 100, 100, 100],
    ["2025-10-19 04:00", 100, 100, 100],
    ["2025-10-19 08:00", 100, 100, 100],
    ["2025-10-19 12:00", 100, 100, 100],
    ["2025-10-19 16:00", 100, 100, 100],
    ["2025-10-19 20:00", 100, 100, 100],
    ["2025-10-20 00:00", 100, 100, 100],
    ["2025-10-20 04:00", 100, 100, 100],
    ["2025-10-20 08:00", 100, 100, 100],
    ["2025-10-20 12:00", 100, 100, 100],
    ["2025-10-20 16:00", 100, 100, 100],
    ["2025-10-20 20:00", 100, 100, 100],
    ["2025-10-21 00:00", 100, 100, 100],
    ["2025-10-21 04:00", 100, 100, 100],
    ["2025-10-21 08:00", 100, 100, 100],
    ["2025-10-21 12:00", 100, 100, 100],
    ["2025-10-21 16:00", 100, 100, 100],
    ["2025-10-21 20:00", 100, 100, 100],
    ["2025-10-22 00:00", 100, 100, 100],
    ["2025-10-22 04:00", 90, 100, 100],
    ["2025-10-22 08:00", 100, 110, 100],
    ["2025-10-22 12:00", 100, 111, 100],
    ["2025-10-22 16:00", 100, 112, 100],
    ["2025-10-22 20:00", 100, 110, 100],
]

ts_menor = pd.to_datetime([row[0] for row in minor_data])
high_menor = np.array([row[1] for row in minor_data], dtype=float)
close_menor = np.array([row[2] for row in minor_data], dtype=float)
low_menor = np.array([row[3] for row in minor_data], dtype=float)

arr_minor = {
    "ts": ts_menor,
    "high": high_menor,
    "close": close_menor,
    "low": low_menor
}

# ==========================================================
# ⚙️ Parámetros
# ==========================================================
LOOKBACK_MENOR = 2
N_CONSECUTIVE  = 2
FACTOR_TRIGGER = 1.10

# ==========================================================
# 🚀 Calcular señales reales con las funciones
# ==========================================================
signal_minor_array = signal_minor_tf(arr_minor, LOOKBACK_MENOR, N_CONSECUTIVE, FACTOR_TRIGGER)
signal_major_array = signal_major_tf(arr_major, N_CONSECUTIVE)
signals_comb = explosive_signal_tf(arr_minor, arr_major, LOOKBACK_MENOR, N_CONSECUTIVE, FACTOR_TRIGGER, backtest=True)

# ==========================================================
# 🧾 Mostrar resultados
# ==========================================================
print("\n=== Asociaciones ===")
print(f"{'TS_MAJOR':<15} {'SIGNAL_MAJOR':<13} {'TS_MINOR':<20} {'SIGNAL_MINOR':<13} {'SIGNAL_COMBINED':<15}")
print("-" * 76)

for i in range(len(ts_menor)):
    idx_major = get_last_closed_major_bar(ts_mayor, ts_menor[i])
    major_ts = ts_mayor[idx_major].strftime("%Y-%m-%d") if idx_major is not None else "None"
    sig_major = int(signal_major_array[idx_major]) if idx_major is not None else 0
    sig_minor = int(signal_minor_array[i])
    sig_comb = int(signals_comb[i])

    print(f"{major_ts:<15} {sig_major:<13} {str(ts_menor[i]):<20} {sig_minor:<13} {sig_comb:<15}")
