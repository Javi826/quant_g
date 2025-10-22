# === FILE: add_signals04.py ===
# ---------------------------------
import warnings
import logging
import pandas as pd
import numpy as np
from utils.ZX_indicators import rolling_entropy_numba, delta_numba

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def explosive_signal_03(high, close, lookback=3, live=False):
    
    signal = np.zeros_like(close, dtype=np.int8)
    n = len(close)

    for i in range(lookback, n):
        window = high[i - lookback:i]  
        
        if np.all(window[:-1] > window[1:]):
            
            max_window = np.max(window)
            if close[i] > max_window:
                signal[i] = 1  

    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = 0
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted

    return signal