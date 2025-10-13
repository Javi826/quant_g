import numpy as np
import pandas as pd

# -------------------------
# DELTA
# -------------------------
def delta_pandas(close):
    close_series = pd.Series(close)
    delta = close_series.diff().fillna(0).to_numpy()
    return delta

# -------------------------
# SEGUNDA DIFERENCIA (ACELERACIÓN)
# -------------------------
def second_diff_pandas(close):
    close_series = pd.Series(close)
    accel = close_series.diff().diff().fillna(0).to_numpy()
    return accel

# -------------------------
# ENTROPÍA RODANTE
# -------------------------
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

# -------------------------
# EWM
# -------------------------
def ewm_pandas(x, span):
    x_series = pd.Series(x)
    return x_series.ewm(span=span, adjust=False).mean().to_numpy()

# -------------------------
# ADD INDICATORS
# -------------------------
def add_indicators_pandas(close, m_accel=5):
    delta = delta_pandas(close)
    entropia = rolling_entropy_pandas(delta, window=5, bins=10)
    accel_raw = second_diff_pandas(close)
    accel = ewm_pandas(accel_raw, m_accel)
    return entropia, accel

# -------------------------
# EXPLOSIVE SIGNAL
# -------------------------
def explosive_signal_pandas(entropia, accel, entropia_max=2.0, live=False):
    signal = (entropia < entropia_max) & (accel > 0)
    if not live:
        signal_shifted = np.empty_like(signal)
        signal_shifted[0] = False
        signal_shifted[1:] = signal[:-1]
        signal = signal_shifted
    return signal
