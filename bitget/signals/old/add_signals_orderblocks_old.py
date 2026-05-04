import numpy as np
import pandas as pd
from numba import njit
from ta.trend import ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator


@njit(cache=True)
def _orderblocks_long_core(open_, high, low, close, ma50, rsi, adx, plus_di, minus_di,
                            n, lookback, tol, impulse):
    signal = np.zeros(n, dtype=np.int32)

    for t in range(lookback, n):
        win_start = t - lookback
        win_end   = t
        ob_index  = -1

        for idx in range(win_end - 1, win_start - 1, -1):
            if idx + 1 >= win_end:
                continue
            if close[idx] < open_[idx]:
                if close[idx + 1] > high[idx] * (1.0 + impulse) and \
                   close[idx + 1] > open_[idx + 1]:
                    ob_index = idx
                    break

        if ob_index == -1:
            continue

        zone_low  = min(close[ob_index], open_[ob_index])
        zone_high = max(close[ob_index], open_[ob_index])

        zone_lower = zone_low  * (1.0 - tol)
        zone_upper = zone_high * (1.0 + tol)

        touches_zone = (low[t] <= zone_upper) and (high[t] >= zone_lower)
        if not touches_zone:
            continue

        close_above_zone = close[t] > zone_high

        ma_ok  = (not np.isnan(ma50[t]))    and (close[t] > ma50[t])
        rsi_ok = (not np.isnan(rsi[t]))     and (rsi[t] > 50)
        adx_ok = (not np.isnan(adx[t]))     and \
                 (not np.isnan(plus_di[t]))  and \
                 (not np.isnan(minus_di[t])) and \
                 (adx[t] > 20)               and \
                 (plus_di[t] > minus_di[t])

        if close_above_zone and ma_ok and rsi_ok and adx_ok:
            signal[t] = 1

    return signal


@njit(cache=True)
def _orderblocks_short_core(open_, high, low, close, ma50, rsi, adx, plus_di, minus_di,
                             n, lookback, tol, impulse):
    signal = np.zeros(n, dtype=np.int32)

    for t in range(lookback, n):
        win_start = t - lookback
        win_end   = t
        ob_index  = -1

        for idx in range(win_end - 1, win_start - 1, -1):
            if idx + 1 >= win_end:
                continue
            if close[idx] > open_[idx]:
                if close[idx + 1] < low[idx] * (1.0 - impulse) and \
                   close[idx + 1] < open_[idx + 1]:
                    ob_index = idx
                    break

        if ob_index == -1:
            continue

        zone_low  = min(close[ob_index], open_[ob_index])
        zone_high = max(close[ob_index], open_[ob_index])

        zone_lower = zone_low  * (1.0 - tol)
        zone_upper = zone_high * (1.0 + tol)

        touches_zone = (low[t] <= zone_upper) and (high[t] >= zone_lower)
        if not touches_zone:
            continue

        close_below_zone = close[t] < zone_low

        ma_ok  = (not np.isnan(ma50[t]))    and (close[t] < ma50[t])
        rsi_ok = (not np.isnan(rsi[t]))     and (rsi[t] < 50)
        adx_ok = (not np.isnan(adx[t]))     and \
                 (not np.isnan(plus_di[t]))  and \
                 (not np.isnan(minus_di[t])) and \
                 (adx[t] > 20)               and \
                 (minus_di[t] > plus_di[t])

        if close_below_zone and ma_ok and rsi_ok and adx_ok:
            signal[t] = -1

    return signal


def _compute_indicators(arr):
    df = pd.DataFrame({
        'open':  arr['open'],
        'high':  arr['high'],
        'low':   arr['low'],
        'close': arr['close'],
    })
    df['ma50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['rsi']  = RSIIndicator(df['close'], window=14).rsi()

    adx_ind      = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx']      = adx_ind.adx()
    df['plus_di']  = adx_ind.adx_pos()
    df['minus_di'] = adx_ind.adx_neg()

    return df


def orderblocks_long(arr, lookback, tolerance, impulse, live_trading=True):
    df = _compute_indicators(arr)
    n  = len(df)

    signal = _orderblocks_long_core(
        open_    = np.ascontiguousarray(df['open'].values,     dtype=np.float64),
        high     = np.ascontiguousarray(df['high'].values,     dtype=np.float64),
        low      = np.ascontiguousarray(df['low'].values,      dtype=np.float64),
        close    = np.ascontiguousarray(df['close'].values,    dtype=np.float64),
        ma50     = np.ascontiguousarray(df['ma50'].values,     dtype=np.float64),
        rsi      = np.ascontiguousarray(df['rsi'].values,      dtype=np.float64),
        adx      = np.ascontiguousarray(df['adx'].values,      dtype=np.float64),
        plus_di  = np.ascontiguousarray(df['plus_di'].values,  dtype=np.float64),
        minus_di = np.ascontiguousarray(df['minus_di'].values, dtype=np.float64),
        n        = n,
        lookback = lookback,
        tol      = tolerance / 100.0,
        impulse  = impulse / 100.0,
    )

    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal


def orderblocks_short(arr, lookback, tolerance, impulse=0.1, live_trading=True):
    df = _compute_indicators(arr)
    n  = len(df)

    signal = _orderblocks_short_core(
        open_    = np.ascontiguousarray(df['open'].values,     dtype=np.float64),
        high     = np.ascontiguousarray(df['high'].values,     dtype=np.float64),
        low      = np.ascontiguousarray(df['low'].values,      dtype=np.float64),
        close    = np.ascontiguousarray(df['close'].values,    dtype=np.float64),
        ma50     = np.ascontiguousarray(df['ma50'].values,     dtype=np.float64),
        rsi      = np.ascontiguousarray(df['rsi'].values,      dtype=np.float64),
        adx      = np.ascontiguousarray(df['adx'].values,      dtype=np.float64),
        plus_di  = np.ascontiguousarray(df['plus_di'].values,  dtype=np.float64),
        minus_di = np.ascontiguousarray(df['minus_di'].values, dtype=np.float64),
        n        = n,
        lookback = lookback,
        tol      = tolerance / 100.0,
        impulse  = impulse / 100.0,
    )

    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal