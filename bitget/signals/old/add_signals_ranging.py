import numpy as np

def ranging_long(arr, lookback=20, tolerance=0.5, ma_period=10,
                          ranges=1.0, live_trading=True):
    """
    Detección simple de longs en mercados ranging:
    - Comprueba que la amplitud del lookback sea reducida (rango).
    - Encuentra soporte como mínimo local en el lookback.
    - Busca toque/rebote en soporte (dentro de 'tolerance' %).
    - Confirma con movimiento alcista y filtro por MA (price < ma antes, luego sube).
    """
    high  = arr['high']
    low   = arr['low']
    close = arr['close']
    n = len(close)

    tol = tolerance / 100.0
    signal = np.zeros(n, dtype=int)

    for t in range(lookback, n):
        window_high = high[t-lookback:t]
        window_low  = low[t-lookback:t]
        window_close= close[t-lookback:t]

        # 1) Comprobar que estamos en un rango estrecho: (max - min) / precio_medio < threshold
        range_pct = (window_high.max() - window_low.min()) / (window_close.mean()) * 100.0
        if range_pct > ranges:
            # No es ranging suficientemente estrecho
            continue

        # 2) Definir soporte y resistencia del rango
        support = window_low.min()
        resistance = window_high.max()

        # 3) Detectar toque al soporte (dentro de tolerance)
        #    Acceptamos que el low toque o se acerque al soporte dentro de tol.
        support_upper = support * (1 + tol)
        if low[t] <= support_upper:
            # 4) Confirmación de rebote: cierre por encima de soporte+buffer y velas alcistas
            buffer_level = support * (1 + tol/2)  # pequeño colchón
            if close[t] > buffer_level and close[t] > close[t-1]:
                # 5) Confirmación con MA: en ranging queremos que el precio vuelva hacia la media
                if t >= ma_period:
                    ma = np.mean(close[t-ma_period:t])
                    # interés: price estaba debajo de la MA (en soporte) y ahora muestra impulso hacia la media
                    if close[t-1] < ma and close[t] >= close[t-1]:
                        signal[t] = 1
                else:
                    # Si no hay suficiente historia para MA, permitir la señal basada en el rebote simple
                    signal[t] = 1

        # (Opcional) evitar entradas si close está demasiado cerca de la resistencia
        # para no entrar en la parte alta del rango:
        if signal[t] == 1 and close[t] >= resistance * (1 - 0.02):
            # cancelar señal si estamos a menos del 2% de la resistencia
            signal[t] = 0

    # Ajuste para backtesting vs live
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal

def ranging_short(arr, lookback=20, tolerance=0.5, ma_period=10,
                  ranges=1.0, live_trading=True):
    """
    Detección simple de shorts en mercados ranging:
    - Comprueba que la amplitud del lookback sea reducida (rango).
    - Encuentra resistencia como máximo local en el lookback.
    - Busca toque/rechazo en resistencia (dentro de 'tolerance' %).
    - Confirma con movimiento bajista y filtro por MA (price > ma antes, luego baja).
    """
    high  = arr['high']
    low   = arr['low']
    close = arr['close']
    n = len(close)

    tol = tolerance / 100.0
    signal = np.zeros(n, dtype=int)

    for t in range(lookback, n):
        window_high = high[t-lookback:t]
        window_low  = low[t-lookback:t]
        window_close= close[t-lookback:t]

        # 1) Comprobar que estamos en un rango estrecho: (max - min) / precio_medio < threshold
        range_pct = (window_high.max() - window_low.min()) / (window_close.mean()) * 100.0
        if range_pct > ranges:
            # No es ranging suficientemente estrecho
            continue

        # 2) Definir soporte y resistencia del rango
        support = window_low.min()
        resistance = window_high.max()

        # 3) Detectar toque a la resistencia (dentro de tolerance)
        #    Aceptamos que el high toque o se acerque a la resistencia dentro de tol.
        resistance_lower = resistance * (1 - tol)
        if high[t] >= resistance_lower:
            # 4) Confirmación de rechazo: cierre por debajo de resistencia-buffer y velas bajistas
            buffer_level = resistance * (1 - tol/2)  # pequeño colchón
            if close[t] < buffer_level and close[t] < close[t-1]:
                # 5) Confirmación con MA: en ranging queremos que el precio vuelva hacia la media
                if t >= ma_period:
                    ma = np.mean(close[t-ma_period:t])
                    # interés: price estaba encima de la MA (en resistencia) y ahora muestra impulso bajista hacia la media
                    if close[t-1] > ma and close[t] <= close[t-1]:
                        signal[t] = -1
                else:
                    # Si no hay suficiente historia para MA, permitir la señal basada en el rechazo simple
                    signal[t] = -1

        # (Opcional) evitar entradas si close está demasiado cerca del soporte
        # para no entrar en la parte baja del rango:
        if signal[t] == -1 and close[t] <= support * (1 + 0.02):
            # cancelar señal si estamos a menos del 2% del soporte
            signal[t] = 0

    # Ajuste para backtesting vs live
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal
