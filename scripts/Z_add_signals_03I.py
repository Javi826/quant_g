# === FILE: add_signals03.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
from numba import njit
from utils.ZX_indicators import atr_numba,rsi_numba,ema_numba
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


@njit
def explosive_signal_03I(close, high, low, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14):
    ema50 = ema_numba(close, ema_fast)
    ema200 = ema_numba(close, ema_slow)
    rsi = rsi_numba(close, rsi_period)
    atr = atr_numba(high, low, close, atr_period)

    # Señal long: EMA50>EMA200, close>EMA50, RSI>50
    signal = (ema50 > ema200) & (close > ema50) & (rsi > 50)

    # Shift para evitar lookahead
    signal_shifted = np.empty_like(signal)
    signal_shifted[0] = False
    signal_shifted[1:] = signal[:-1]
    return signal_shifted, atr

