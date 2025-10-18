# === FILE: indicators.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
from numba import njit
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# === EMA =====
# ---------------------------------
@njit
def ema_numba(arr, period):
    ema = np.empty_like(arr)
    alpha = 2 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, arr.size):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    return ema

# === RSI =====
# ---------------------------------
@njit
def rsi_numba(prices, period=14):
    deltas = prices[1:] - prices[:-1]
    rsi = np.empty_like(prices)
    rsi[:period] = 50
    avg_gain = np.mean(np.clip(deltas[:period], 0, np.inf))
    avg_loss = -np.mean(np.clip(deltas[:period], -np.inf, 0))
    rs = avg_gain / (avg_loss + 1e-8)
    rsi[period] = 100 - (100 / (1 + rs))
    for i in range(period+1, prices.size):
        gain = max(deltas[i-1],0)
        loss = -min(deltas[i-1],0)
        avg_gain = (avg_gain*(period-1) + gain)/period
        avg_loss = (avg_loss*(period-1) + loss)/period
        rs = avg_gain / (avg_loss + 1e-8)
        rsi[i] = 100 - (100 / (1 + rs))
    return rsi

# ===ATR =====
# ---------------------------------
@njit
def atr_numba(high, low, close, period=14):
    # Calcular True Range (TR)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))

    # Preasignar array de ATR
    atr = np.empty(close.shape[0], dtype=np.float64)

    # ATR inicial como promedio de los primeros 'period' TR
    atr[0] = np.mean(tr[:period])

    # Calcular ATR exponencial suavizado
    for i in range(1, tr.size):
        if i < period:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = (atr[i-1]*(period-1) + tr[i])/period

    # Ajustar el último valor
    atr[-1] = atr[-2]  # Opcional, para evitar NaNs si es necesario

    return atr


# === ENTROPY =====
# ---------------------------------
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

# === ACELERATION =====
# ---------------------------------

@njit
def second_diff(close):
    n = len(close)
    accel_raw = np.zeros(n)
    for i in range(2, n):
        accel_raw[i] = close[i] - 2*close[i-1] + close[i-2]
    return accel_raw

# === EWM =====
# ---------------------------------

@njit
def ewm_numba(x, span):
    n = len(x)
    alpha = 2 / (span + 1)
    ewm = np.empty(n)
    ewm[0] = x[0]
    for i in range(1, n):
        ewm[i] = alpha * x[i] + (1 - alpha) * ewm[i - 1]
    return ewm



import numpy as np
import pandas as pd

# =========================
# ENTROPÍA RODANTE
# =========================
def rolling_entropy_pandas(delta, window=5, bins=10):
    delta_series = pd.Series(delta)
    
    def entropy_func(x):
        hist, _ = np.histogram(x, bins=bins, range=(delta_series.min(), delta_series.max()))
        prob = hist / hist.sum()
        prob = prob[prob > 0]  # evitar log(0)
        return -np.sum(prob * np.log2(prob))
    
    return delta_series.rolling(window, min_periods=1).apply(entropy_func, raw=True).to_numpy()


# =========================
# SEGUNDA DIFERENCIA
# =========================
def second_diff_pandas(close):
    close_series = pd.Series(close)
    return close_series.diff().diff().fillna(0).to_numpy()  # diff dos veces


# =========================
# EWM
# =========================
def ewm_pandas(x, span):
    x_series = pd.Series(x)
    return x_series.ewm(span=span, adjust=False).mean().to_numpy()
