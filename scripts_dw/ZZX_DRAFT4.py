import numpy as np
from utils.ZX_indicators import delta_numba, rolling_entropy_numba  # asegúrate de tenerlas importadas

def detect_double_top_long(
    arr,
    lookback_minor,        # tamaño de la ventana de búsqueda de patrones
    price_tolerance,
    trend_th,
    downtrend_window=20,
    fib_level=0.618,
    entropy_max=1.5,
    backtest=False
):
    # === Normalización de parámetros ===
    price_tolerance  = price_tolerance / 100.0
    trend_th         = trend_th / 100.0
    fib_level        = 0.618
    entropy_max      = entropy_max

    high, low, close = arr['high'], arr['low'], arr['close']
    n = len(close)

    # === ENTROPÍA una sola vez ===
    delta   = delta_numba(close)
    entropy = rolling_entropy_numba(delta, window=5, bins=10)
    signal  = np.zeros(n, dtype=int)

    # === Tendencia bajista ===
    downtrend = _compute_downtrend(close, trend_th, downtrend_window)

    # === Patrón doble suelo ===
    active_patterns = []
    min_candles_between_bottoms = 2

    for i in range(n):
        # --- Buscar patrones dentro de la ventana pasada ---
        if i >= lookback_minor:
            window_start = i - lookback_minor
            lows_window = low[window_start:i+1]
            highs_window = high[window_start:i+1]

            # Buscar suelos locales dentro de la ventana
            local_bottoms = _find_local_bottoms_in_window(lows_window, offset=window_start)

            # Actualizar patrones con esos suelos
            _update_active_patterns(
                local_bottoms, active_patterns, low, high, downtrend,
                price_tolerance, fib_level, min_candles_between_bottoms
            )

        # --- Comprobar rupturas de patrones activos ---
        _check_pattern_breakouts(active_patterns, close, entropy, signal, i, entropy_max)

    # === Shift causal para backtesting ===
    if backtest:
        signal = _apply_shift(signal, entry_delay=1)

    return signal


# === SUBFUNCIONES AUXILIARES ===

def _compute_downtrend(close, trend_th, window):
    n = len(close)
    downtrend = np.zeros(n, dtype=bool)
    for i in range(window, n):
        price_change = (close[i] - close[i - window]) / close[i - window]
        downtrend[i] = price_change < -trend_th
    return downtrend


def _find_local_bottoms_in_window(lows_window, offset):
    """Detecta mínimos locales dentro de una ventana."""
    idxs = []
    n = len(lows_window)
    for j in range(1, n - 1):
        if lows_window[j] < lows_window[j - 1] and lows_window[j] < lows_window[j + 1]:
            idxs.append(offset + j)
    return idxs


def _update_active_patterns(
    bottoms_idx, active_patterns, low, high, downtrend,
    price_tolerance, fib_level, min_candles_between_bottoms
):
    """Actualiza la lista de patrones activos al detectar nuevos suelos."""
    if len(bottoms_idx) < 2:
        return
    bottom2_idx = bottoms_idx[-1]
    for bottom1_idx in reversed(bottoms_idx[:-1]):
        if bottom2_idx - bottom1_idx < min_candles_between_bottoms:
            continue

        bottom1_low = low[bottom1_idx]
        bottom2_low = low[bottom2_idx]
        if bottom1_low == 0:
            continue

        price_diff = abs(bottom1_low - bottom2_low) / bottom1_low
        if price_diff > price_tolerance:
            continue

        if not downtrend[bottom1_idx]:
            continue

        peak_segment = high[bottom1_idx:bottom2_idx + 1]
        if len(peak_segment) == 0:
            continue
        peak_idx_rel = np.argmax(peak_segment)
        peak_idx = bottom1_idx + peak_idx_rel
        peak_high = peak_segment[peak_idx_rel]

        pattern_range = peak_high - bottom1_low
        if pattern_range <= 0:
            continue

        fib_level_price = bottom1_low + (pattern_range * fib_level)
        pattern = {
            "bottom1_idx": bottom1_idx,
            "bottom2_idx": bottom2_idx,
            "peak_idx": peak_idx,
            "peak_high": peak_high,
            "bottom1_low": bottom1_low,
            "fib_level_price": fib_level_price,
            "active": True
        }
        active_patterns.append(pattern)


def _check_pattern_breakouts(active_patterns, close, entropy, signal, i, entropy_max):
    """Confirma la ruptura de patrones activos según la entropía y el nivel fib."""
    if len(active_patterns) == 0:
        return
    for pat in active_patterns:
        if not pat["active"]:
            continue
        if i <= pat["bottom2_idx"]:
            continue
        if close[i] > pat["fib_level_price"] and entropy[i] <= entropy_max:
            signal[i] = 1
            pat["active"] = False


def _apply_shift(signal, entry_delay=1):
    """Desplaza las señales para mantener causalidad en backtesting."""
    n = len(signal)
    shifted = np.zeros_like(signal)
    if entry_delay < n:
        shifted[entry_delay:] = signal[:-entry_delay]
    return shifted


import numpy as np

#=========================================================
# SHORT - versión con ventana
#=========================================================

def detect_double_top_short(
    arr,
    lookback_minor,       # tamaño de la ventana para buscar el patrón
    price_tolerance,
    trend_th,
    uptrend_window=20,
    fib_level=0.618,
    entropy_max=1.5,
    backtest=False
):
    # === Normalización de parámetros ===
    price_tolerance = price_tolerance / 100.0
    trend_th        = trend_th / 100.0
    fib_level       = 0.618
    entropy_max     = entropy_max

    high, low, close = arr['high'], arr['low'], arr['close']
    n = len(close)

    # === Calcular ENTROPÍA una sola vez ===
    delta   = delta_numba(close)
    entropy = rolling_entropy_numba(delta, window=5, bins=10)
    signal  = np.zeros(n, dtype=int)

    # === Calcular tendencia alcista ===
    uptrend = _compute_uptrend(close, trend_th, uptrend_window)

    # === Detectar patrones ===
    active_patterns = []
    min_candles_between_peaks = 2

    for i in range(n):
        # --- Buscar posibles patrones dentro de la ventana ---
        if i >= lookback_minor:
            window_start = i - lookback_minor
            highs_window = high[window_start:i+1]
            lows_window  = low[window_start:i+1]

            # Detectar picos locales dentro de la ventana
            local_peaks = _find_local_peaks_in_window(highs_window, offset=window_start)

            # Actualizar lista de patrones activos
            _update_active_patterns_short_window(
                local_peaks, active_patterns, high, low, uptrend,
                price_tolerance, fib_level, min_candles_between_peaks
            )

        # --- Comprobar rupturas de patrones activos ---
        _check_pattern_breakouts_short(active_patterns, close, entropy, signal, i, entropy_max)

    # === Shift causal para backtest ===
    if backtest:
        signal = _apply_shift(signal, entry_delay=1)

    return signal


# === SUBFUNCIONES AUXILIARES ===

def _compute_uptrend(close, trend_th, window):
    """Evalúa si hay tendencia alcista previa."""
    n = len(close)
    uptrend = np.zeros(n, dtype=bool)
    for i in range(window, n):
        price_change = (close[i] - close[i - window]) / close[i - window]
        uptrend[i] = price_change > trend_th
    return uptrend


def _find_local_peaks_in_window(highs_window, offset):
    """Detecta máximos locales dentro de una ventana."""
    idxs = []
    n = len(highs_window)
    for j in range(1, n - 1):
        if highs_window[j] > highs_window[j - 1] and highs_window[j] > highs_window[j + 1]:
            idxs.append(offset + j)
    return idxs


def _update_active_patterns_short_window(
    peaks_idx, active_patterns, high, low, uptrend,
    price_tolerance, fib_level, min_candles_between_peaks
):
    """Actualiza la lista de patrones activos para doble techo."""
    if len(peaks_idx) < 2:
        return

    peak2_idx = peaks_idx[-1]
    for peak1_idx in reversed(peaks_idx[:-1]):
        # Separación mínima
        if peak2_idx - peak1_idx < min_candles_between_peaks:
            continue

        peak1_high = high[peak1_idx]
        peak2_high = high[peak2_idx]

        # Similitud de precios
        price_diff = abs(peak1_high - peak2_high) / peak1_high
        if price_diff > price_tolerance:
            continue

        # Verificar tendencia alcista previa
        if not uptrend[peak1_idx]:
            continue

        # Encontrar valle entre los picos
        valley_segment = low[peak1_idx:peak2_idx + 1]
        if len(valley_segment) == 0:
            continue
        valley_idx_rel = np.argmin(valley_segment)
        valley_idx = peak1_idx + valley_idx_rel
        valley_low = valley_segment[valley_idx_rel]

        # Nivel Fibonacci
        pattern_range = peak1_high - valley_low
        if pattern_range <= 0:
            continue
        fib_level_price = valley_low + (pattern_range * fib_level)

        pattern = {
            "peak1_idx": peak1_idx,
            "peak2_idx": peak2_idx,
            "valley_idx": valley_idx,
            "valley_low": valley_low,
            "peak1_high": peak1_high,
            "fib_level_price": fib_level_price,
            "active": True
        }
        active_patterns.append(pattern)


def _check_pattern_breakouts_short(active_patterns, close, entropy, signal, i, entropy_max):
    """Confirma la ruptura bajista de patrones activos."""
    if len(active_patterns) == 0:
        return
    for pat in active_patterns:
        if not pat["active"]:
            continue
        if i <= pat["peak2_idx"]:
            continue
        if close[i] < pat["fib_level_price"] and entropy[i] <= entropy_max:
            signal[i] = -1
            pat["active"] = False


def _apply_shift(signal, entry_delay=1):
    """Desplaza las señales una vela para mantener causalidad en backtest."""
    n = len(signal)
    shifted = np.zeros_like(signal)
    if entry_delay < n:
        shifted[entry_delay:] = signal[:-entry_delay]
    return shifted
