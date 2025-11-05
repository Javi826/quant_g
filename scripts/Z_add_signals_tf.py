import numpy as np
import pandas as pd

# ==========================================================
# Función auxiliar para encontrar la última barra cerrada del timeframe mayor
# ==========================================================
def get_last_closed_major_bar(ts_major, ts_minor_now):
    ts_major_close = ts_major + pd.Timedelta(days=1)
    mask           = ts_major_close <= ts_minor_now
    indices        = np.where(mask)[0]
    return indices[-1] if len(indices) > 0 else None

# ==========================================================
# Función auxiliar de tendencia
# ==========================================================
def _compute_trend(close, trend_th, window, direction="up"):
    n = len(close)
    trend = np.zeros(n, dtype=bool)
    for i in range(window, n):
        price_change = (close[i] - close[i - window]) / close[i - window]
        if direction == "up":
            trend[i] = price_change > trend_th
        else:
            trend[i] = price_change < -trend_th
    return trend

# ==========================================================
# Función para aplicar desplazamiento para backtesting
# ==========================================================
def _apply_shift(signal, entry_delay=1):
    shifted = np.zeros_like(signal)
    if entry_delay <= 0:
        return signal.copy()
    shifted[entry_delay:] = signal[:-entry_delay]
    return shifted

# ==========================================================
# Señales del timeframe menor (versión 100% causal)
# ==========================================================
def signal_minor_tf_generic(arr_minor, lookback, n_consecutive, direction="up"):

    close = np.asarray(arr_minor["close"])
    high  = np.asarray(arr_minor["high"])
    low   = np.asarray(arr_minor["low"])
    n = len(close)
    signals = np.zeros(n, dtype=int)

    # MA50 causal: rolling hasta cada índice j (min_periods=1 para evitar NaNs al inicio)
    ma50 = pd.Series(close).rolling(window=50, min_periods=1).mean().to_numpy()

    # Recorremos el tiempo buscando breakouts (en i) y confirmaciones (en k)
    for i in range(lookback, n):
        if direction == "up":
            # breakout: close[i] > max(high of previous 'lookback' bars)
            prev_max = high[i - lookback:i].max() if lookback > 0 else -np.inf
            breakout = close[i] > prev_max
            # después del breakout se esperan cierres ascendentes (i+1, i+2, ...)
            is_confirming = lambda a, b: close[b] > close[a]
            signal_value = 1
            ma_condition = lambda j: close[j] >= ma50[j]
        else:  # down
            prev_min = low[i - lookback:i].min() if lookback > 0 else np.inf
            breakout = close[i] < prev_min
            is_confirming = lambda a, b: close[b] < close[a]
            signal_value = -1
            ma_condition = lambda j: close[j] <= ma50[j]

        if not breakout:
            continue

        # Ahora esperamos de forma causal n_consecutive cierres confirmatorios.
        # Si una vela rompe la cadena antes de alcanzar n_consecutive, abandonamos este breakout.
        consec = 0
        k = i
        while k + 1 < n and consec < n_consecutive:
            # comparamos la vela siguiente con la actual (k -> k+1)
            if is_confirming(k, k + 1):
                consec += 1
                k += 1
            else:
                # cadena rota: este breakout no se confirmará de forma consecutiva
                consec = 0
                break

        # Si no se alcanzó la confirmación antes del final, no hay señal
        if consec < n_consecutive:
            continue

        # k es el índice de la última vela de la secuencia consecutiva confirmatoria
        start_idx = k  # equivalente a i + n_consecutive en el enfoque original

        # A partir de start_idx, buscamos la primera vela j >= start_idx que cumpla MA (usando datos ≤ j)
        sign_set = False
        for j in range(start_idx, n):
            if ma_condition(j):
                signals[j] = signal_value
                sign_set = True
                break

        # si se encontró la señal para este breakout, seguimos buscando otros breakouts (outer loop continúa)
        # si no se encontró (p. ej. no hay MA condition antes del final), simplemente no marcamos nada para este breakout

    return signals

# ==========================================================
# Señales del timeframe mayor unificada (sin lookahead)
# ==========================================================
def signal_major_tf_generic(arr_major, trend_th=5, trend_window=20, direction="up"):
    close = np.asarray(arr_major["close"])
    n = len(close)
    signals = np.zeros(n, dtype=int)
    trend = _compute_trend(close, trend_th / 100, trend_window, direction=direction)
    signals[trend] = 1
    return signals

# ==========================================================
# Composición unificada de señales menor + mayor
# ==========================================================
def trends_tf_generic(arr_minor, arr_major, lookback_minor, n_consecutive, factor,
                      trend_th=5, trend_window=20, direction="up", backtest=True, entry_delay=1):
    ts_minor = pd.to_datetime(arr_minor["ts"])
    ts_major = pd.to_datetime(arr_major["ts"])

    minor_signals = signal_minor_tf_generic(arr_minor, lookback_minor, n_consecutive, direction=direction)
    major_signals = signal_major_tf_generic(arr_major, trend_th=trend_th, trend_window=trend_window,
                                            direction=direction)

    final_signals = np.zeros_like(minor_signals)

    for i in range(len(minor_signals)):
        if minor_signals[i] != 0:
            idx_major = get_last_closed_major_bar(ts_major, ts_minor[i])
            if idx_major is not None and major_signals[idx_major] == 1:
                final_signals[i] = minor_signals[i]

    if backtest:
        final_signals = _apply_shift(final_signals, entry_delay=entry_delay)

    return final_signals

# ==========================================================
# Interfaces públicas para LONG y SHORT
# ==========================================================

def trends_tf_long(arr_minor, arr_major, lookback_minor, n_consecutive, factor,
                   trend_th=5, trend_window=20, backtest=True):

    direction = "up"
    entry_delay = 1  # fijo para backtesting
    final_signals = trends_tf_generic(
        arr_minor=arr_minor,
        arr_major=arr_major,
        lookback_minor=lookback_minor,
        n_consecutive=n_consecutive,
        factor=factor,
        trend_th=trend_th,
        trend_window=trend_window,
        direction=direction,
        backtest=backtest,
        entry_delay=entry_delay
    )
    return final_signals


def trends_tf_short(arr_minor, arr_major, lookback_minor, n_consecutive, factor,
                    trend_th=5, trend_window=20, backtest=True):

    direction = "down"
    entry_delay = 1  # fijo para backtesting
    final_signals = trends_tf_generic(
        arr_minor=arr_minor,
        arr_major=arr_major,
        lookback_minor=lookback_minor,
        n_consecutive=n_consecutive,
        factor=factor,
        trend_th=trend_th,
        trend_window=trend_window,
        direction=direction,
        backtest=backtest,
        entry_delay=entry_delay
    )
    return final_signals

