import heapq
import logging
import warnings
import numpy as np
import pandas as pd
from numba import njit
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# -----------------------------
# CÁLCULO DE GANANCIAS NETAS (igual)
# -----------------------------

@njit
def second_diff(close):
    n = len(close)
    accel_raw = np.zeros(n)
    for i in range(2, n):
        accel_raw[i] = close[i] - 2*close[i-1] + close[i-2]
    return accel_raw


@njit
def delta_numba(close):
    n = len(close)
    delta = np.empty(n)
    delta[0] = 0.0
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    return delta

@njit
def rolling_entropy_numba(delta, window=5, bins=10):
    n = len(delta)
    entropia  = np.zeros(n)
    delta_min = delta.min()
    delta_max = delta.max()
    hist = np.zeros(bins)  # reusar array

    for i in range(n):
        start = max(0, i - window + 1)
        hist[:] = 0.0  # resetear histograma
        for j in range(start, i + 1):
            bin_idx = int((delta[j] - delta_min) / (delta_max - delta_min + 1e-9) * bins)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist[bin_idx] += 1
        s = hist.sum()
        e = 0.0
        for k in range(bins):
            if hist[k] > 0:
                p = hist[k] / s
                e -= p * np.log2(p)
        entropia[i] = e
    return entropia


@njit
def ewm_numba(x, span):
    n = len(x)
    alpha = 2 / (span + 1)
    ewm = np.empty(n)
    ewm[0] = x[0]
    for i in range(1, n):
        ewm[i] = alpha * x[i] + (1 - alpha) * ewm[i - 1]
    return ewm

@njit
def add_indicators_arrays(close, m_accel=5):
    delta = delta_numba(close)
    entropia = rolling_entropy_numba(delta, 5, 10)
    accel_raw = second_diff(close)
    accel = ewm_numba(accel_raw, m_accel)
    return entropia, accel


@njit
def explosive_signal_arrays(entropia, accel, entropia_max=2.0, live=False):
    signal = (entropia < entropia_max) & (accel > 0)
    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = False
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted
    return signal

