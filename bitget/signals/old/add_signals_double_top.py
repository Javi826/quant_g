import numpy as np

# =========================================================
# === FUNCIONES AUXILIARES COMUNES ===
# =========================================================

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


def _is_local_extreme(series, i, lookback, mode="peak"):
    window_start = i - lookback
    segment = series[window_start:i]
    if len(segment) == 0:
        return False
    if mode == "peak":
        return series[i] > np.max(segment)
    else:
        return series[i] < np.min(segment)


def _apply_shift(signal, entry_delay=1):
    
    shifted               = np.zeros_like(signal)
    shifted[entry_delay:] = signal[:-entry_delay]
    
    return shifted

# =========================================================
# === LONG 
# =========================================================

def double_top_long(
    arr,
    lookback_minor,
    price_tolerance,
    trend_th,
    downtrend_window=20,
    fib_level=0.618,
    live_trading=True
):
    # === Normalización de parámetros ===
    price_tolerance  = price_tolerance / 100.0
    trend_th         = trend_th / 100.0
    fib_level        = 0.618

    high, low, close = arr['high'], arr['low'], arr['close']
    n = len(close)

    signal  = np.zeros(n, dtype=int)

    # === Calcular tendencia bajista ===
    downtrend = _compute_trend(close, trend_th, downtrend_window, direction="down")

    # === Detectar patrones ===
    active_patterns = []
    bottoms_idx     = []
    min_candles_between_bottoms = 2

    for i in range(n):
        # Detectar suelos locales
        if i >= lookback_minor:
            if _is_local_extreme(low, i, lookback_minor, mode="bottom"):
                bottoms_idx.append(i)
                _update_active_patterns_long(
                    bottoms_idx, active_patterns, low, high, downtrend,
                    price_tolerance, fib_level, min_candles_between_bottoms
                )

        # Comprobar rupturas de patrones activos
        _check_pattern_breakouts_long(
            active_patterns, close, signal, i
        )

    # === Shift causal para backtest ===
    if not live_trading:
        signal = _apply_shift(signal, entry_delay=1)

    return signal


def _update_active_patterns_long(
    bottoms_idx, active_patterns, low, high, downtrend,
    price_tolerance, fib_level, min_candles_between_bottoms
):
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


def _check_pattern_breakouts_long(active_patterns, close, signal, i):
    if len(active_patterns) == 0:
        return
    for pat in active_patterns:
        if not pat["active"]:
            continue
        if i <= pat["bottom2_idx"]:
            continue
        if close[i] > pat["fib_level_price"]:
            signal[i] = 1
            pat["active"] = False


# =========================================================
# === SHORT 
# =========================================================

def double_top_short(
    arr,
    lookback_minor,
    price_tolerance,
    trend_th,
    uptrend_window=20,
    fib_level=0.618,
    live_trading=True
):
    # === Normalización de parámetros ===
    price_tolerance = price_tolerance / 100.0
    trend_th        = trend_th / 100.0
    fib_level       = 0.618

    high, low, close = arr['high'], arr['low'], arr['close']
    n = len(close)

    signal  = np.zeros(n, dtype=int)

    # === Calcular tendencia alcista ===
    uptrend = _compute_trend(close, trend_th, uptrend_window, direction="up")

    # === Detectar patrones ===
    active_patterns = []
    peaks_idx       = []
    min_candles_between_peaks = 2

    for i in range(n):
        # Detectar picos locales
        if i >= lookback_minor:
            if _is_local_extreme(high, i, lookback_minor, mode="peak"):
                peaks_idx.append(i)
                _update_active_patterns_short(
                    peaks_idx, active_patterns, high, low, uptrend,
                    price_tolerance, fib_level, min_candles_between_peaks
                )

        # Comprobar rupturas de patrones activos
        _check_pattern_breakouts_short(
            active_patterns, close, signal, i
        )

    # === Shift causal para backtest ===
    if not live_trading:
        signal = _apply_shift(signal, entry_delay=1)

    return signal


def _update_active_patterns_short(
    peaks_idx, active_patterns, high, low, uptrend,
    price_tolerance, fib_level, min_candles_between_peaks
):
    peak2_idx = peaks_idx[-1]
    for peak1_idx in reversed(peaks_idx[:-1]):
        if peak2_idx - peak1_idx < min_candles_between_peaks:
            continue

        peak1_high = high[peak1_idx]
        peak2_high = high[peak2_idx]
        price_diff = abs(peak1_high - peak2_high) / peak1_high
        if price_diff > price_tolerance:
            continue

        if not uptrend[peak1_idx]:
            continue

        valley_segment = low[peak1_idx:peak2_idx + 1]
        if len(valley_segment) == 0:
            continue
        valley_idx_rel = np.argmin(valley_segment)
        valley_idx = peak1_idx + valley_idx_rel
        valley_low = valley_segment[valley_idx_rel]

        pattern_range = peak1_high - valley_low
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


def _check_pattern_breakouts_short(active_patterns, close, signal, i):
    if len(active_patterns) == 0:
        return
    for pat in active_patterns:
        if not pat["active"]:
            continue
        if i <= pat["peak2_idx"]:
            continue
        if close[i] < pat["fib_level_price"]:
            signal[i] = -1
            pat["active"] = False