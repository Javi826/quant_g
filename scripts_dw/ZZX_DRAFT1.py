# === FILE: add_signals03.py ===
# ---------------------------------
import numpy as np

def explosive_signal_tf(high_mayor, close_mayor, high_menor, close_menor, lookback_mayor, lookback_menor, live=False):
    # Señales individuales
    signal_mayor = np.zeros_like(close_mayor, dtype=np.int8)
    signal_menor = np.zeros_like(close_menor, dtype=np.int8)

    # Timeframe mayor
    n_mayor = len(close_mayor)
    for i in range(lookback_mayor, n_mayor):
        window = high_mayor[i-lookback_mayor:i]
        if np.all(window[:-1] > window[1:]):
            max_window = np.max(window)
            if close_mayor[i] > max_window:
                signal_mayor[i] = 1

    # Timeframe menor
    n_menor = len(close_menor)
    for i in range(lookback_menor, n_menor):
        window = high_menor[i-lookback_menor:i]
        if np.all(window[:-1] > window[1:]):
            max_window = np.max(window)
            if close_menor[i] > max_window:
                signal_menor[i] = 1

    # Combinación multi-timeframe
    factor = len(close_menor) // len(close_mayor)
    signal_final = np.zeros_like(close_menor, dtype=np.int8)

    for i, s_menor in enumerate(signal_menor):
        idx_mayor = i // factor
        if idx_mayor < len(signal_mayor) and s_menor == 1 and signal_mayor[idx_mayor] == 1:
            signal_final[i] = 1

    # Shift 
    if not live:
        signal_shifted = np.empty_like(signal_final)
        signal_shifted[0] = 0
        signal_shifted[1:] = signal_final[:-1]
        signal_final = signal_shifted

    return signal_final