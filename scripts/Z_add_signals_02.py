# === FILE: add_signals02.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
from numba import njit
from utils.ZX_indicators import rolling_entropy_numba,delta_numba,second_diff,ewm_numba
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
from ZX_indicators import ema_numpy, rsi_numpy, macd_numpy, atr_numpy, obv_numpy, bollinger_bands_numpy
from ZX_indicators import sma_numpy, stochastic_oscillator, ichimoku_numpy, adx_numpy

def explosive_signal_02(close, sma_fast, sma_slow, live=False):


    signal = np.zeros_like(close, dtype=np.int8)
    
 

    return signal
