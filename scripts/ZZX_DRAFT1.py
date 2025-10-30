#!/usr/bin/env python3
# test_explosive_signal_1D_4H_minimo.py
# Test autocontenido para explosive_signal_tf con 3 timestamps 1D y sus 4H correspondientes

import numpy as np
import pandas as pd

def get_last_closed_major_bar(ts_mayor, ts_minor_now):
    # Una barra 1D con timestamp "2025-10-20" cierra al final del día 20
    # Es decir, está cerrada para cualquier ts_minor >= "2025-10-21 00:00"
    # Entonces buscamos la última barra mayor cuyo timestamp + 1 día <= ts_minor_now
    
    ts_mayor_close = ts_mayor + pd.Timedelta(days=1)
    mask           = ts_mayor_close <= ts_minor_now
    indices        = np.where(mask)[0]
    return indices[-1] if len(indices) > 0 else None

def explosive_signal_tf(
    high_mayor, close_mayor,
    high_menor, close_menor,
    lookback_mayor=1, lookback_menor=2,
    index_mayor=None, index_menor=None,
    live=False
):
    # --- Aseguramos que los timestamps sean pd.Timestamp ---
    ts_mayor = pd.to_datetime(index_mayor)
    ts_menor = pd.to_datetime(index_menor)

    high_mayor = np.array(high_mayor)
    close_mayor = np.array(close_mayor)
    high_menor = np.array(high_menor)
    close_menor = np.array(close_menor)
    
    n_minor = len(close_menor)
    n_major = len(close_mayor)
    
    final_signal = np.zeros(n_minor, dtype=int)
    signal_minor_array = np.zeros(n_minor, dtype=int)
    signal_major_array = np.zeros(n_major, dtype=int)
    last_major_idx_per_minor = np.array([None]*n_minor, dtype=object)
    
    # Calculamos señales para cada barra major
    for j in range(n_major):
        if j < lookback_mayor:
            signal_major_array[j] = 0
        else:
            close_major = close_mayor[j]
            highs_major = high_mayor[j - lookback_mayor:j]
            signal_major_array[j] = 1 if close_major > np.max(highs_major) else 0
    
    # Calculamos señales para cada barra minor y la final combinada
    for i in range(1, n_minor):
        ts_minor_now = ts_menor[i]
        
        # SEÑAL MINOR
        if i - 1 < lookback_menor:
            signal_minor = 0
        else:
            close_prev = close_menor[i - 1]
            highs_prev = high_menor[i - 1 - lookback_menor:i - 1]
            signal_minor = 1 if close_prev > np.max(highs_prev) else 0
        signal_minor_array[i] = signal_minor
        
        # Última barra major cerrada antes del minor actual
        idx_major = get_last_closed_major_bar(ts_mayor, ts_minor_now)
        last_major_idx_per_minor[i] = idx_major
        
        if idx_major is None:
            sig_major_for_this_minor = 0
        else:
            sig_major_for_this_minor = int(signal_major_array[idx_major])
        
        # SEÑAL FINAL (ambas deben ser 1)
        final_signal[i] = 1 if (signal_minor == 1 and sig_major_for_this_minor == 1) else 0
    
    return {
        "final_signal_minor": final_signal,
        "signal_major_array": signal_major_array,
        "signal_minor_array": signal_minor_array,
        "last_major_idx_per_minor": last_major_idx_per_minor,
        "ts_major": ts_mayor,
        "ts_minor": ts_menor
    }

if __name__ == "__main__":
    # --- Timeframe mayor 1D (solo 3 días) ---
    major_data = [
        ["2025-10-20", 100, 100],
        ["2025-10-21", 100, 150],
        ["2025-10-22", 100, 100],
    ]
    
    # Convertir a arrays como antes
    ts_mayor = pd.to_datetime([row[0] for row in major_data])
    high_mayor = np.array([row[1] for row in major_data], dtype=float)
    close_mayor = np.array([row[2] for row in major_data], dtype=float)

    # --- Timeframe menor 4H ---
    minor_data = [
        ["2025-10-20 04:00", 100, 100],
        ["2025-10-20 08:00", 100, 100],
        ["2025-10-20 12:00", 100, 100],
        ["2025-10-20 16:00", 100, 100],
        ["2025-10-20 20:00", 100, 100],
        ["2025-10-21 00:00", 100, 100],
        ["2025-10-21 04:00", 100, 100],
        ["2025-10-21 08:00", 100, 100],
        ["2025-10-21 12:00", 100, 100],
        ["2025-10-21 16:00", 100, 100],
        ["2025-10-21 20:00", 100, 105],
        ["2025-10-22 00:00", 100, 100],
        ["2025-10-22 04:00", 100, 100],
        ["2025-10-22 08:00", 100, 100],
        ["2025-10-22 12:00", 100, 100],
        ["2025-10-22 16:00", 100, 100],
        ["2025-10-22 20:00", 100, 100],
    ]
    
    # Convertir a numpy arrays separados si quieres usarlo igual que antes
    ts_menor = pd.to_datetime([row[0] for row in minor_data])
    high_menor = np.array([row[1] for row in minor_data], dtype=float)
    close_menor = np.array([row[2] for row in minor_data], dtype=float)



    # Ejecutamos la función
    res = explosive_signal_tf(
        high_mayor=high_mayor, close_mayor=close_mayor,
        high_menor=high_menor, close_menor=close_menor,
        lookback_mayor=1, lookback_menor=1,
        index_mayor=ts_mayor, index_menor=ts_menor
    )

    final_signal_minor = res["final_signal_minor"]
    signal_major_array = res["signal_major_array"]
    signal_minor_array = res["signal_minor_array"]
    last_major_idx_per_minor = res["last_major_idx_per_minor"]
    ts_major = res["ts_major"]
    ts_minor = res["ts_minor"]

    # --- PRINT 1: Major ---
    print("=== Señales MAJOR (1D) ===")
    for j, ts in enumerate(ts_major):
        print(f"{j:02d} | {ts.strftime('%Y-%m-%d')} | signal_major = {int(signal_major_array[j])}")
    print()

    # --- PRINT 2: Minor ---
    print("=== Señales MINOR (4H) ===")
    for i, ts in enumerate(ts_minor):
        print(f"{i:02d} | {ts.strftime('%Y-%m-%d %H:%M')} | signal_minor = {int(signal_minor_array[i])}")
    print()

    # --- PRINT 3: Asociaciones correctas (major primero) ---
    print("=== Asociaciones (major timestamp, minor timestamp, signal_major, signal_minor, signal_combinada) ===")
    for i, ts_minor_val in enumerate(ts_minor):
        idx_major = last_major_idx_per_minor[i]
        if idx_major is None:
            ts_major_str = "None"
            sig_major = 0
        else:
            ts_major_val = ts_major[int(idx_major)]
            ts_major_str = ts_major_val.strftime('%Y-%m-%d')
            sig_major = int(signal_major_array[int(idx_major)])
        ts_minor_str = ts_minor_val.strftime('%Y-%m-%d %H:%M')
        sig_minor = int(signal_minor_array[i])
        sig_comb = int(final_signal_minor[i])
        print(f"major_ts={ts_major_str:10s} | minor_ts={ts_minor_str} | sig_major={sig_major} | sig_minor={sig_minor} | sig_comb={sig_comb}")
