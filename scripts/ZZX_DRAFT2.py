import numpy as np
import pandas as pd
from Z_add_signals_03 import add_indicators, explosive_signal  # Numba
from copy import deepcopy

# -------------------------
# Implementaciones Pandas
# -------------------------
def delta_pandas(close):
    return pd.Series(close).diff().fillna(0).to_numpy()

def second_diff_pandas(close):
    return pd.Series(close).diff().diff().fillna(0).to_numpy()

def rolling_entropy_pandas(delta, window=5, bins=10):
    delta_series = pd.Series(delta)
    entropia = []
    for i in range(len(delta_series)):
        start = max(0, i - window + 1)
        window_slice = delta_series[start:i+1]
        hist, _ = np.histogram(window_slice, bins=bins, range=(delta.min(), delta.max()))
        probs = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist, dtype=float)
        e = -np.sum([p * np.log2(p) for p in probs if p > 0])
        entropia.append(e)
    return np.array(entropia)

def ewm_pandas(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()

def add_indicators_pandas(close, m_accel=5):
    delta = delta_pandas(close)
    entropia = rolling_entropy_pandas(delta, window=5, bins=10)
    accel_raw = second_diff_pandas(close)
    accel = ewm_pandas(accel_raw, m_accel)
    return entropia, accel

def explosive_signal_pandas(entropia, accel, entropia_max=2.0, live=False):
    signal = (entropia < entropia_max) & (accel > 0)
    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = False
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted
    return signal

# -------------------------
# DATOS DE PRUEBA
# -------------------------
np.random.seed(42)
close_prices = np.random.rand(50) * 100  # ejemplo de 50 precios aleatorios

# -------------------------
# CALCULO NUMBA
# -------------------------
entropia_numba, accel_numba = add_indicators(close_prices, m_accel=5)
signal_numba = explosive_signal(entropia_numba, accel_numba, entropia_max=2.0, live=True)

# -------------------------
# CALCULO PANDAS
# -------------------------
entropia_pd, accel_pd = add_indicators_pandas(close_prices, m_accel=5)
signal_pd = explosive_signal_pandas(entropia_pd, accel_pd, entropia_max=2.0, live=True)

# -------------------------
# COMPARACIÓN
# -------------------------
print("Entropía iguales:", np.allclose(entropia_numba, entropia_pd, atol=1e-9))
print("Aceleración iguales:", np.allclose(accel_numba, accel_pd, atol=1e-9))
print("Señales iguales:", np.array_equal(signal_numba, signal_pd))
