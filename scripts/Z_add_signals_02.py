# === FILE: add_signals03.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
from numba import njit
from utils.ZX_indicators import rolling_entropy_numba,delta_numba,second_diff,ewm_numba
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

def explosive_signal_02(close, sma_fast, sma_slow, live=False):

    fast_ma = ewm_numba(close, sma_fast)
    slow_ma = ewm_numba(close, sma_slow)
    
    signal = np.zeros_like(close, dtype=np.int8)
    
    
    for i in range(1, len(close)):
        if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
            signal[i] = 1
        else:
            signal[i] = 0  

    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = 0
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted

    return signal
