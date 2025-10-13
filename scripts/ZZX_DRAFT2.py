#!/usr/bin/env python3
# compare_latest_signal.py
# Compara la última señal entre versión "numba-like" y "pandas"

import numpy as np
import pandas as pd

# ---------------- Configuración ----------------
ACCEL_SPAN = 10
ENTROPIA_MAX = 1.5

# ---------------- Implementaciones numba-like ----------------
def second_diff_np(close):
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    accel_raw = np.zeros(n, dtype=np.float64)
    for i in range(2, n):
        accel_raw[i] = close[i] - 2*close[i-1] + close[i-2]
    return accel_raw

def delta_np(close):
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    delta = np.empty(n, dtype=np.float64)
    if n > 0:
        delta[0] = 0.0
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    return delta

def rolling_entropy_np(delta, window=5, bins=10):
    delta = np.asarray(delta, dtype=np.float64)
    n = len(delta)
    entropia = np.zeros(n, dtype=np.float64)
    if n == 0:
        return entropia
    delta_min = delta.min()
    delta_max = delta.max()
    hist = np.zeros(bins, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        hist[:] = 0.0
        for j in range(start, i + 1):
            denom = (delta_max - delta_min + 1e-9)
            bin_idx = int((delta[j] - delta_min) / denom * bins)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist[bin_idx] += 1.0
        s = hist.sum()
        e = 0.0
        if s > 0:
            for k in range(bins):
                if hist[k] > 0:
                    p = hist[k]/s
                    e -= p * np.log2(p)
        entropia[i] = e
    return entropia

def ewm_np(x, span):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    ewm = np.empty(n, dtype=np.float64)
    ewm[0] = x[0]
    for i in range(1, n):
        ewm[i] = alpha * x[i] + (1 - alpha) * ewm[i-1]
    return ewm

def add_indicators_numba_like(close, m_accel=5):
    delta = delta_np(close)
    entropia = rolling_entropy_np(delta, window=5, bins=10)
    accel_raw = second_diff_np(close)
    accel = ewm_np(accel_raw, m_accel)
    return entropia, accel

def explosive_signal_numba_like(entropia, accel, entropia_max=2.0, live=False):
    entropia = np.asarray(entropia, dtype=np.float64)
    accel = np.asarray(accel, dtype=np.float64)
    signal = (entropia < entropia_max) & (accel > 0.0)
    if not live:
        signal_shifted = np.empty_like(signal, dtype=np.bool_)
        if signal_shifted.size > 0:
            signal_shifted[0] = False
            if signal_shifted.size > 1:
                signal_shifted[1:] = signal[:-1]
        signal = signal_shifted
    return signal

# ---------------- Implementaciones pandas ----------------
def add_indicators_pandas(close_series, m_accel=5):
    s = pd.Series(close_series).astype(np.float64)
    n = len(s)
    delta = np.empty(n, dtype=np.float64)
    if n > 0:
        delta[0] = 0.0
    for i in range(1, n):
        delta[i] = s.iat[i] - s.iat[i-1]
    entropia = rolling_entropy_np(delta, window=5, bins=10)
    accel_raw = np.zeros(n, dtype=np.float64)
    for i in range(2, n):
        accel_raw[i] = s.iat[i] - 2*s.iat[i-1] + s.iat[i-2]
    accel = ewm_np(accel_raw, m_accel)
    return pd.Series(entropia, index=s.index), pd.Series(accel, index=s.index)

def explosive_signal_pandas(entropia_s, accel_s, entropia_max=2.0, live=False):
    ent = np.asarray(entropia_s, dtype=np.float64)
    acc = np.asarray(accel_s, dtype=np.float64)
    signal = (ent < entropia_max) & (acc > 0.0)
    if not live:
        signal_shifted = np.empty_like(signal, dtype=np.bool_)
        if signal_shifted.size > 0:
            signal_shifted[0] = False
            if signal_shifted.size > 1:
                signal_shifted[1:] = signal[:-1]
        signal = signal_shifted
    return pd.Series(signal, index=entropia_s.index)

# ---------------- Funciones auxiliares ----------------
def normalize_live_ohlcv(df):
    # Solo como ejemplo: asegurar tipo float y columnas mínimas
    df = df.copy()
    df['close'] = df['close'].astype(float)
    return df

# ---------------- Funciones de prueba ----------------
def check_latest_signal_vector(df, symbol):
    df = normalize_live_ohlcv(df)
    close_prices = df['close'].values
    entropia, accel = add_indicators_numba_like(close_prices, m_accel=ACCEL_SPAN)
    signals = explosive_signal_numba_like(entropia, accel, entropia_max=ENTROPIA_MAX, live=True)
    return signals[-1]

def check_latest_signal_pandas(df, symbol):
    df = normalize_live_ohlcv(df)
    ent, acc = add_indicators_pandas(df['close'], m_accel=ACCEL_SPAN)
    sig = explosive_signal_pandas(ent, acc, entropia_max=ENTROPIA_MAX, live=True)
    return sig.iloc[-1]

# ---------------- Test ----------------
if __name__ == "__main__":
    # Crear DataFrame de ejemplo
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'timestamp': pd.date_range("2025-01-01", periods=100, freq='T'),
        'close': rng.normal(100, 2, 100),
        'open': rng.normal(100, 2, 100),
        'high': rng.normal(102, 2, 100),
        'low': rng.normal(98, 2, 100),
        'volume': rng.integers(100, 200, 100)
    })
    
    symbol = "TEST"
    
    signal_vec = check_latest_signal_vector(df, symbol)
    signal_pd  = check_latest_signal_pandas(df, symbol)
    
    print("Última señal versión vectorizada:", signal_vec)
    print("Última señal versión pandas     :", signal_pd)
    print("Coinciden?", signal_vec == signal_pd)
