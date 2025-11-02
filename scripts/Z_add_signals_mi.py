import numpy as np
import pandas as pd


def explosive_signal_tf_minor(arr_minor, lookback, n_consecutive, factor, backtest=True):

    close = arr_minor['close']
    high  = arr_minor['high']
    low   = arr_minor['low']
    signals = np.zeros_like(close, dtype=int)
    n = len(close)

    for i in range(lookback, n - n_consecutive):
        # 1️⃣ Ruptura del high previo
        if close[i] > high[i - lookback:i].max():
            # 2️⃣ Cierres crecientes posteriores
            is_consecutive_up = all(close[i + j + 1] > close[i + j] for j in range(n_consecutive))
            if is_consecutive_up:
                # 3️⃣ Comprobación del factor sobre el low de la ruptura
                confirm_idx = i + n_consecutive
                if confirm_idx < n and close[confirm_idx] >= low[i] * factor:
                    signals[confirm_idx] = 1

    if backtest:
        signals = np.roll(signals, 1)
        signals[0] = 0

    return signals


def explosive_signal_tf_mi_basic(arr_minor, lookback, n_consecutive=2, backtest=True):

    close = arr_minor['close']
    high  = arr_minor['high']
    signals = np.zeros_like(close, dtype=int)
    n = len(close)

    for i in range(lookback, n - n_consecutive):
        # 1️⃣ Comprueba la ruptura del máximo previo
        if close[i] > high[i - lookback:i].max():
            # 2️⃣ Comprueba si los siguientes cierres son crecientes n_consecutive veces
            is_consecutive_up = all(close[i + j + 1] > close[i + j] for j in range(n_consecutive))
            if is_consecutive_up:
                # Marca la señal cuando se confirma la secuencia completa
                signals[i + n_consecutive] = 1

    # 3️⃣ En modo backtest, desplazamos la señal una vela adelante
    if backtest:
        signals = np.roll(signals, 1)
        signals[0] = 0

    return signals