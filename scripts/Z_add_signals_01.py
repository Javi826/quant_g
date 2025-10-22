# === FILE: add_signals03.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
from numba import njit
from utils.ZX_indicators import rolling_entropy_numba,delta_numba,second_diff,ewm_numba
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")



@njit
def explosive_signal_01(close, m_accel=5, entropia_max=2.0, live=False):

    # --- Cálculo de indicadores ---
    delta     = delta_numba(close)
    entropia  = rolling_entropy_numba(delta, 5, 10)
    accel_raw = second_diff(close)
    accel     = ewm_numba(accel_raw, m_accel)

    # --- Señal explosiva ---
    signal = (entropia < entropia_max) & (accel > 0)

    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = False
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted

    # Convertir booleanos a enteros 0/1
    return signal.astype(np.int8)

