import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator


def orderblocks_long(arr, lookback, tolerance, impulse, live_trading=True):

    # Convertir a DataFrame para usar TA
    df = pd.DataFrame({
        'open':  arr['open'],
        'high':  arr['high'],
        'low':   arr['low'],
        'close': arr['close']
    })

    n = len(df)
    signal = np.zeros(n, dtype=int)
    tol = tolerance / 100.0
    impulse = impulse / 100.0

    # ---------------------------------------------
    # SMA50 (sin lookahead)
    # ---------------------------------------------
    df['ma50'] = SMAIndicator(df['close'], window=50).sma_indicator()

    # ---------------------------------------------
    # RSI14 (sin lookahead)
    # ---------------------------------------------
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

    # ---------------------------------------------
    # ADX(14), +DI , -DI (sin lookahead)
    # ---------------------------------------------
    adx_ind = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )

    df['adx']      = adx_ind.adx()
    df['plus_di']  = adx_ind.adx_pos()
    df['minus_di'] = adx_ind.adx_neg()

    # ---------------------------------------------
    # Búsqueda de Order Block
    # ---------------------------------------------
    open_  = df['open' ].values
    high   = df['high' ].values
    low    = df['low'  ].values
    close  = df['close'].values

    ma50    = df['ma50'   ].values
    rsi     = df['rsi'    ].values
    adx     = df['adx'    ].values
    plus_di = df['plus_di'].values
    minus_di=df['minus_di'].values

    for t in range(lookback, n):

        win_start = t - lookback
        win_end   = t
        ob_index  = None

        # ---------------------------------------------
        # Buscar última vela bajista con impulso alcista
        # ---------------------------------------------
        for idx in range(win_end - 1, win_start - 1, -1):

            if idx + 1 >= win_end:
                continue

            # vela bajista
            if close[idx] < open_[idx]:

                # impulso alcista en la siguiente
                if close[idx + 1] > high[idx] * (1.0 + impulse) and \
                   close[idx + 1] > open_[idx + 1]:
                    ob_index = idx
                    break

        if ob_index is None:
            continue

        zone_low  = min(close[ob_index], open_[ob_index])
        zone_high = max(close[ob_index], open_[ob_index])

        zone_lower = zone_low  * (1.0 - tol)
        zone_upper = zone_high * (1.0 + tol)

        # ¿Toca la zona hoy?
        touches_zone = (low[t] <= zone_upper) and (high[t] >= zone_lower)
        if not touches_zone:
            continue

        close_above_zone = close[t] > zone_high

        # ---------------------------------------------
        # Filtros
        # ---------------------------------------------
        ma_ok  = (not np.isnan(ma50[t])) and (close[t] > ma50[t])
        rsi_ok = (not np.isnan(rsi[t]))  and (rsi[t] > 50)

        adx_ok = (not np.isnan(adx[t]) or np.isnan(plus_di[t]) or np.isnan(minus_di[t]))
        if adx_ok:
            adx_ok = (adx[t] > 20) and (plus_di[t] > minus_di[t])

        # Señal final
        if close_above_zone and ma_ok and rsi_ok and adx_ok:
            signal[t] = 1

    # ---------------------------------------------
    # Mover señal para backtest (evita look-ahead)
    # ---------------------------------------------
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal



def orderblocks_short(arr, lookback, tolerance, impulse=0.1, live_trading=True):

    # Convertir a DataFrame para usar TA
    df = pd.DataFrame({
        'open':  arr['open'],
        'high':  arr['high'],
        'low':   arr['low'],
        'close': arr['close']
    })

    n = len(df)
    signal = np.zeros(n, dtype=int)
    tol = tolerance / 100.0
    impulse = impulse / 100.0

    # ---------------------------------------------
    # SMA50 (sin lookahead)
    # ---------------------------------------------
    df['ma50'] = SMAIndicator(df['close'], window=50).sma_indicator()

    # ---------------------------------------------
    # RSI14 (sin lookahead)
    # ---------------------------------------------
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

    # ---------------------------------------------
    # ADX(14), +DI , -DI (sin lookahead)
    # ---------------------------------------------
    adx_ind = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )

    df['adx']      = adx_ind.adx()
    df['plus_di']  = adx_ind.adx_pos()
    df['minus_di'] = adx_ind.adx_neg()

    # ---------------------------------------------
    # Búsqueda de Order Block (inverso para SHORT)
    # ---------------------------------------------
    open_  = df['open' ].values
    high   = df['high' ].values
    low    = df['low'  ].values
    close  = df['close'].values

    ma50    = df['ma50'   ].values
    rsi     = df['rsi'    ].values
    adx     = df['adx'    ].values
    plus_di = df['plus_di'].values
    minus_di= df['minus_di'].values

    for t in range(lookback, n):

        win_start = t - lookback
        win_end   = t
        ob_index  = None

        # ---------------------------------------------
        # Buscar última vela alcista con impulso bajista
        # ---------------------------------------------
        for idx in range(win_end - 1, win_start - 1, -1):

            if idx + 1 >= win_end:
                continue

            # vela alcista
            if close[idx] > open_[idx]:

                # impulso bajista en la siguiente
                if close[idx + 1] < low[idx] * (1.0 - impulse) and \
                   close[idx + 1] < open_[idx + 1]:
                    ob_index = idx
                    break

        if ob_index is None:
            continue

        zone_low  = min(close[ob_index], open_[ob_index])
        zone_high = max(close[ob_index], open_[ob_index])

        zone_lower = zone_low  * (1.0 - tol)
        zone_upper = zone_high * (1.0 + tol)

        # ¿Toca la zona hoy?
        touches_zone = (low[t] <= zone_upper) and (high[t] >= zone_lower)
        if not touches_zone:
            continue

        close_below_zone = close[t] < zone_low

        # ---------------------------------------------
        # Filtros (inversos para SHORT)
        # ---------------------------------------------
        ma_ok  = (not np.isnan(ma50[t])) and (close[t] < ma50[t])
        rsi_ok = (not np.isnan(rsi[t]))  and (rsi[t] < 50)

        adx_ok = (not np.isnan(adx[t]) or np.isnan(plus_di[t]) or np.isnan(minus_di[t]))
        if adx_ok:
            adx_ok = (adx[t] > 20) and (minus_di[t] > plus_di[t])

        # Señal final (short = -1)
        if close_below_zone and ma_ok and rsi_ok and adx_ok:
            signal[t] = -1

    # ---------------------------------------------
    # Mover señal para backtest (evita look-ahead)
    # ---------------------------------------------
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal
