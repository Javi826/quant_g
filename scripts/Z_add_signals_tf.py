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
        
    
    return final_signal  # en vez del diccionario

