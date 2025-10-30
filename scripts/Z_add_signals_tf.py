import numpy as np
import pandas as pd

def get_last_closed_major_bar(ts_mayor, ts_minor_now):
    """
    Retorna el índice de la última vela del major que cerró 
    ANTES de ts_minor_now
    """
    mask = ts_mayor < ts_minor_now
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return None
    
    return indices[-1]


def explosive_signal_tf(
    high_mayor, close_mayor,
    high_menor, close_menor,
    lookback_mayor=3, lookback_menor=3,
    index_mayor=None, index_menor=None,
    live=False
):
    high_mayor = np.array(high_mayor)
    close_mayor = np.array(close_mayor)
    high_menor = np.array(high_menor)
    close_menor = np.array(close_menor)
    ts_mayor = np.array(index_mayor)
    ts_menor = np.array(index_menor)
    
    n_minor = len(close_menor)
    final_signal = np.zeros(n_minor, dtype=int)
    
    for i in range(1, n_minor):
        ts_minor_now = ts_menor[i]
        
        # SEÑAL MINOR
        if i - 1 < lookback_menor:
            signal_minor = 0
        else:
            close_prev = close_menor[i - 1]
            highs_prev = high_menor[i - 1 - lookback_menor:i - 1]
            signal_minor = 1 if close_prev > np.max(highs_prev) else 0
        
        # SEÑAL MAJOR
        idx_major = get_last_closed_major_bar(ts_mayor, ts_minor_now)
        
        if idx_major is None or idx_major < lookback_mayor:
            signal_major = 0
        else:
            close_major = close_mayor[idx_major]
            highs_major = high_mayor[idx_major - lookback_mayor:idx_major]
            signal_major = 1 if close_major > np.max(highs_major) else 0
        
        # SEÑAL FINAL
        final_signal[i] = 1 if (signal_minor == 1 and signal_major == 1) else 0
    
    return final_signal

