# === FILE: add_signals03.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
from numba import njit
from utils.ZX_indicators import rolling_entropy_numba,delta_numba
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


@njit
def signal_99_long(close, entropia_max, live=False):

    delta     = delta_numba(close)
    entropia  = rolling_entropy_numba(delta, 5, 10)
    signal    = (entropia < entropia_max) 

    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = False
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted

    return signal.astype(np.int8)

@njit
def signal_99_short(close, entropia_max, live=False):

    delta     = delta_numba(close)
    entropia  = rolling_entropy_numba(delta, 5, 10)
    signal    = (entropia > entropia_max)

    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = False
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted

    return (-signal.astype(np.int8))
