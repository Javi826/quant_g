import numpy as np
from utils.ZX_indicators import delta_numba, rolling_entropy_numba


def orderblocks_long(arr, lookback, tolerance,  ma_period=50, impulse=0.001,live_trading=True):

    # Obtener series como numpy
    open_ = np.asarray(arr['open'], dtype=float)
    high  = np.asarray(arr['high'], dtype=float)
    low   = np.asarray(arr['low'], dtype=float)
    close = np.asarray(arr['close'], dtype=float)

    n = len(close)
    if not (len(open_) == len(high) == len(low) == len(close)):
        raise ValueError("Todas las series deben tener la misma longitud")

    tol = tolerance / 100.0
    impulse = impulse / 100.0
    signal = np.zeros(n, dtype=int)

    # Precalcular MA simple si hay suficientes datos
    ma50 = np.full(n, np.nan)
    if n >= ma_period:
        cumsum = np.cumsum(np.insert(close, 0, 0.0))
        ma50[ma_period-1:] = (cumsum[ma_period:] - cumsum[:-ma_period]) / ma_period

    # Iterar desde left_lookback hasta el final (igual estructura que tu función)
    for t in range(lookback, n):
        # Ventana donde buscamos el OB
        win_start = t - lookback
        win_end = t  # excluyente
        ob_index = None

        # Buscar la última vela bajista en la ventana que sea seguida por un impulso alcista
        for idx in range(win_end - 1, win_start - 1, -1):  # recorrer hacia atrás
            # Asegurar que exista la vela siguiente para evaluar el impulso
            if idx + 1 >= win_end:
                continue

            # Condición vela bajista (posible order block)
            if close[idx] < open_[idx]:
                # Vela siguiente como impulso: close siguiente > high de la vela bajista * (1 + umbral)
                if close[idx + 1] > high[idx] * (1.0 + impulse) and close[idx + 1] > open_[idx + 1]:
                    ob_index = idx
                    break

        # Si no encontramos OB en la ventana, continuar
        if ob_index is None:
            continue

        # Definir zona del order block: cuerpo de la vela bajista (close -> open)
        zone_low  = min(close[ob_index], open_[ob_index])   # será close[ob_index]
        zone_high = max(close[ob_index], open_[ob_index])   # será open_[ob_index]

        # Expandir la zona con tolerancia
        zone_lower = zone_low * (1.0 - tol)
        zone_upper = zone_high * (1.0 + tol)

        # Comprobar que la vela actual "toca" la zona del order block
        touches_zone = (low[t] <= zone_upper) and (high[t] >= zone_lower)

        if not touches_zone:
            continue

        # Confirmación 1: cierre actual por encima de la parte superior del OB (zona_high)
        close_above_zone = close[t] > zone_high

        # Confirmación 2: si hay MA disponible, el cierre debe estar por encima de la MA (tendencia)
        ma_ok = True
        if not np.isnan(ma50[t]):
            ma_ok = close[t] > ma50[t]

        # Opcional: evitar señales en la misma barra que definió el OB (si ob_index == t-1)
        # Aquí permitimos señales en cualquier barra t >= left_lookback

        # Si se cumplen confirmaciones, marcar señal
        if close_above_zone and ma_ok:
            signal[t] = 1
        else:
            signal[t] = 0

    # Mismo comportamiento que tu reversal_long para backtests
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal


def orderblocks_short(arr, lookback, tolerance, ma_period=50, impulse=0.001, live_trading=True):

    # Obtener series como numpy
    open_ = np.asarray(arr['open'], dtype=float)
    high  = np.asarray(arr['high'], dtype=float)
    low   = np.asarray(arr['low'], dtype=float)
    close = np.asarray(arr['close'], dtype=float)

    n = len(close)
    if not (len(open_) == len(high) == len(low) == len(close)):
        raise ValueError("Todas las series deben tener la misma longitud")

    tol = tolerance / 100.0
    impulse = impulse / 100.0
    signal = np.zeros(n, dtype=int)

    # Precalcular MA simple si hay suficientes datos
    ma50 = np.full(n, np.nan)
    if n >= ma_period:
        cumsum = np.cumsum(np.insert(close, 0, 0.0))
        ma50[ma_period-1:] = (cumsum[ma_period:] - cumsum[:-ma_period]) / ma_period

    # Iterar desde lookback hasta el final (misma estructura que la función long)
    for t in range(lookback, n):
        # Ventana donde buscamos el OB
        win_start = t - lookback
        win_end = t  # excluyente
        ob_index = None

        # Buscar la última vela alcista en la ventana que sea seguida por un impulso bajista
        for idx in range(win_end - 1, win_start - 1, -1):  # recorrer hacia atrás
            # Asegurar que exista la vela siguiente para evaluar el impulso
            if idx + 1 >= win_end:
                continue

            # Condición vela alcista (posible order block)
            if close[idx] > open_[idx]:
                # Vela siguiente como impulso: close siguiente < low de la vela alcista * (1 - umbral)
                if close[idx + 1] < low[idx] * (1.0 - impulse) and close[idx + 1] < open_[idx + 1]:
                    ob_index = idx
                    break

        # Si no encontramos OB en la ventana, continuar
        if ob_index is None:
            continue

        # Definir zona del order block: cuerpo de la vela alcista (open -> close)
        zone_low  = min(close[ob_index], open_[ob_index])   # será open_[ob_index]
        zone_high = max(close[ob_index], open_[ob_index])   # será close[ob_index]

        # Expandir la zona con tolerancia
        zone_lower = zone_low * (1.0 - tol)
        zone_upper = zone_high * (1.0 + tol)

        # Comprobar que la vela actual "toca" la zona del order block
        touches_zone = (low[t] <= zone_upper) and (high[t] >= zone_lower)

        if not touches_zone:
            continue

        # Confirmación 1: cierre actual por debajo de la parte inferior del OB (zone_low)
        close_below_zone = close[t] < zone_low

        # Confirmación 2: si hay MA disponible, el cierre debe estar por debajo de la MA (tendencia bajista)
        ma_ok = True
        if not np.isnan(ma50[t]):
            ma_ok = close[t] < ma50[t]

        # Si se cumplen confirmaciones, marcar señal short en la barra t
        if close_below_zone and ma_ok:
            signal[t] = 1
        else:
            signal[t] = 0

    # Mismo comportamiento que tu orderblocks_long para backtests
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal
