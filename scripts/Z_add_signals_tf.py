import numpy as np
import pandas as pd


# ==========================================================
# Función auxiliar
# ==========================================================
def get_last_closed_major_bar(ts_mayor, ts_minor_now):
    ts_mayor_close = ts_mayor + pd.Timedelta(days=1)
    mask = ts_mayor_close <= ts_minor_now
    indices = np.where(mask)[0]
    return indices[-1] if len(indices) > 0 else None


# ==========================================================
# Señales del timeframe menor
# ==========================================================
def signal_minor_tf(arr_minor, lookback, n_consecutive, factor):
    close = arr_minor["close"]
    high = arr_minor["high"]
    low = arr_minor["low"]

    n = len(close)
    signals = np.zeros_like(close, dtype=int)

    for i in range(lookback, n - n_consecutive):
        # Ruptura del máximo previo
        if close[i] > high[i - lookback:i].max():
            # Cierres consecutivos al alza
            if all(close[i + j + 1] > close[i + j] for j in range(n_consecutive)):
                confirm_idx = i + n_consecutive
                if confirm_idx < n and close[confirm_idx] >= low[i] * factor:
                    signals[confirm_idx] = 1

    return signals


def signal_major_tf(arr_major, n_consecutive):
    close = arr_major["close"]
    n = len(close)
    signals = np.zeros(n, dtype=int)

    for i in range(n_consecutive - 1, n):
        # Comprueba que los últimos n_consecutive cierres (incluyendo i) sean estrictamente crecientes
        if all(close[i - k] > close[i - k - 1] for k in range(n_consecutive - 1)):
            signals[i] = 1

    return signals


# ==========================================================
# Composición de señales menor + mayor
# ==========================================================
def explosive_signal_tf(arr_minor, arr_major, lookback_minor, n_consecutive, factor, backtest=True):
    ts_minor = pd.to_datetime(arr_minor["ts"])
    ts_major = pd.to_datetime(arr_major["ts"])
    

    # 1️⃣ Señales por separado
    minor_signals = signal_minor_tf(arr_minor, lookback_minor, n_consecutive, factor)
    major_signals = signal_major_tf(arr_major, n_consecutive)

    # 2️⃣ Composición final
    final_signals = np.zeros_like(minor_signals)

    for i in range(len(minor_signals)):
        if minor_signals[i] == 1:
            idx_major = get_last_closed_major_bar(ts_major, ts_minor[i])
            if idx_major is not None and major_signals[idx_major] == 1:
                final_signals[i] = 1

    # 3️⃣ Ajuste para backtest
    if backtest:
        final_signals = np.roll(final_signals, 1)
        final_signals[0] = 0

    return final_signals
