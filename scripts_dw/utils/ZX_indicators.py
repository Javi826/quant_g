# === FILE: indicators.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
from numba import njit
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# === SMA =====
# ---------------------------------

@njit
def sma_numba(x, span):
    n = len(x)
    sma = np.empty(n)
    
    # Para las primeras barras donde no hay suficientes datos
    for i in range(span - 1):
        sma[i] = np.mean(x[:i+1])
    
    # SMA "normal" a partir de span
    cum_sum = np.sum(x[:span])
    sma[span - 1] = cum_sum / span
    
    for i in range(span, n):
        cum_sum = cum_sum - x[i - span] + x[i]
        sma[i] = cum_sum / span
    
    return sma


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

# === NUMPY =====
# ---------------------------------

import numpy as np

# 1. Media Móvil Simple (SMA)
def sma_numpy(prices, period):
    sma = np.convolve(prices, np.ones(period)/period, mode='same')
    return sma

# 2. Media Móvil Exponencial (EMA)
def ema_numpy(prices, period):
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    return ema

# 3. RSI
def rsi_numpy(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='same')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='same')
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 4. MACD
def macd_numpy(close, fast=12, slow=26, signal=9):
    ema_fast = ema_numpy(close, fast)
    ema_slow = ema_numpy(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema_numpy(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# 5. Bandas de Bollinger
def bollinger_bands_numpy(close, period=20, std_dev=2):
    sma = sma_numpy(close, period)
    std = np.zeros_like(close)
    for i in range(period-1, len(close)):
        std[i] = np.std(close[i-period+1:i+1])
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

# 6. ATR (Average True Range)
def atr_numpy(high, low, close, period=14):
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0]-low[0])
    atr = np.convolve(tr, np.ones(period)/period, mode='same')
    return atr

# 7. Estocástico (Slow %K y %D)
def stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    lowest_low = np.zeros_like(close)
    highest_high = np.zeros_like(close)
    for i in range(k_period-1, len(close)):
        lowest_low[i] = np.min(low[i-k_period+1:i+1])
        highest_high[i] = np.max(high[i-k_period+1:i+1])
    slowk = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    slowd = np.convolve(slowk, np.ones(d_period)/d_period, mode='same')
    return slowk, slowd

# 8. OBV (On-Balance Volume)
def obv_numpy(close, volume):
    obv = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv

# 9. Ichimoku Kinko Hyo
def ichimoku_numpy(high, low, close):
    # Tenkan-sen (9 períodos)
    high_9 = np.zeros_like(close)
    low_9 = np.zeros_like(close)
    for i in range(8, len(close)):
        high_9[i] = np.max(high[i-8:i+1])
        low_9[i] = np.min(low[i-8:i+1])
    tenkan = (high_9 + low_9)/2

    # Kijun-sen (26 períodos)
    high_26 = np.zeros_like(close)
    low_26 = np.zeros_like(close)
    for i in range(25, len(close)):
        high_26[i] = np.max(high[i-25:i+1])
        low_26[i] = np.min(low[i-25:i+1])
    kijun = (high_26 + low_26)/2

    # Senkou Span A (promedio de Tenkan y Kijun, desplazado 26)
    senkou_a = np.zeros_like(close)
    senkou_a[26:] = (tenkan[26:] + kijun[26:])/2

    # Senkou Span B (52 períodos desplazado 26)
    high_52 = np.zeros_like(close)
    low_52 = np.zeros_like(close)
    for i in range(51, len(close)):
        high_52[i] = np.max(high[i-51:i+1])
        low_52[i] = np.min(low[i-51:i+1])
    senkou_b = np.zeros_like(close)
    senkou_b[26:] = (high_52[26:] + low_52[26:])/2

    return tenkan, kijun, senkou_a, senkou_b

# 10. ADX (Average Directional Index)
def adx_numpy(high, low, close, period=14):
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0]-low[0])
    atr = np.convolve(tr, np.ones(period)/period, mode='same')
    plus_di = 100 * np.convolve(plus_dm, np.ones(period)/period, mode='same') / (atr[1:] + 1e-9)
    minus_di = 100 * np.convolve(minus_dm, np.ones(period)/period, mode='same') / (atr[1:] + 1e-9)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = np.convolve(dx, np.ones(period)/period, mode='same')
    adx = np.insert(adx, 0, 0)  # Ajuste de longitud
    return adx

