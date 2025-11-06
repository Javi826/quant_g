import numpy as np


def explosive_signal_tf_minor(arr_minor, lookback, n_consecutive, factor, backtest):

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

# --- Datos de prueba ---
major_data = [
    ["2025-10-18", 100,  90, 100],   #0
    ["2025-10-19", 100,  90, 100],   #1
    ["2025-10-20", 100,  90, 100],   #2
    ["2025-10-21", 100,  90, 100],   #3
    ["2025-10-22", 100,  100, 103],   #4 ← ruptura
    ["2025-10-23", 100,  90, 110],   #5 ↑ cierre mayor
    ["2025-10-24", 100,  90, 111],   #6 ↑ cierre mayor + cumple factor
    ["2025-10-25", 100,  90, 105],   #7
]

import numpy as np

arr_minor = {
    'ts': np.array([x[0] for x in major_data]),
    'high': np.array([x[1] for x in major_data]),
    'low': np.array([x[2] for x in major_data]),
    'close': np.array([x[3] for x in major_data]),
}

lookback = 2
n_consecutive = 2
factor = 1.1  # el close final debe ser >= low de la ruptura * 1.02

signals = explosive_signal_tf_minor(arr_minor, lookback, n_consecutive, factor, backtest=False)

# Mostramos los resultados
print("Idx | Fecha        |   High |    Low |  Close | Señal")
print("-----------------------------------------------------")
for i, ts in enumerate(arr_minor['ts']):
    print(f"{i:>3} | {ts} | {arr_minor['high'][i]:>6} | {arr_minor['low'][i]:>6} | {arr_minor['close'][i]:>6} | {signals[i]}")


