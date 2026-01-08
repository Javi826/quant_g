import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

def doji_long(arr, lookback, tolerance, rsi_period=14, rsi_threshold=30, live_trading=True):

    open_price = arr['open']
    close = arr['close']
    low = arr['low']
    n = len(close)
    
    tolerance = tolerance / 100
    signal = np.zeros(n, dtype=int)
    
    # Convertir a pandas Series si es necesario
    close_series = pd.Series(close) if not isinstance(close, pd.Series) else close
    
    # Calcular RSI
    rsi_indicator = RSIIndicator(close=close_series, window=rsi_period)
    rsi = rsi_indicator.rsi().values
    
    for i in range(lookback, n):
        # Verificar que las últimas 'lookback' velas son rojas
        previous_candles = range(i - lookback, i)
        all_red = all(close[j] < open_price[j] for j in previous_candles)
        
        # Verificar que los mínimos son decrecientes
        lows_decreasing = all(low[j] > low[j+1] for j in range(i - lookback, i - 1))
        
        if all_red and lows_decreasing:
            # Calcular tamaño de la vela actual
            candle_size = abs(close[i] - open_price[i])
            reference_price = open_price[i]
            
            # Verificar si la vela es muy pequeña (dentro de tolerancia)
            if candle_size <= reference_price * tolerance:
                
                # Confirmación con RSI (sobreventa)
                #if not np.isnan(rsi[i]) and rsi[i] < rsi_threshold:
                    signal[i] = 1
    
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0
    
    return signal


def doji_short(arr, lookback, tolerance, rsi_period=14, rsi_threshold=70, live_trading=True):

    open_price = arr['open']
    close = arr['close']
    high = arr['high']
    n = len(close)
    
    tolerance = tolerance / 100
    signal = np.zeros(n, dtype=int)
    
    # Convertir a pandas Series si es necesario
    close_series = pd.Series(close) if not isinstance(close, pd.Series) else close
    
    # Calcular RSI
    rsi_indicator = RSIIndicator(close=close_series, window=rsi_period)
    rsi = rsi_indicator.rsi().values
    
    for i in range(lookback, n):
        # Verificar que las últimas 'lookback' velas son verdes
        previous_candles = range(i - lookback, i)
        all_green = all(close[j] > open_price[j] for j in previous_candles)
        
        # Verificar que los máximos son crecientes
        highs_increasing = all(high[j] < high[j+1] for j in range(i - lookback, i - 1))
        
        if all_green and highs_increasing:
            # Calcular tamaño de la vela actual
            candle_size = abs(close[i] - open_price[i])
            reference_price = open_price[i]
            
            # Verificar si la vela es muy pequeña (dentro de tolerancia)
            if candle_size <= reference_price * tolerance:
                
                # Confirmación con RSI (sobrecompra)
                #if not np.isnan(rsi[i]) and rsi[i] > rsi_threshold:
                    signal[i] = -1
    
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0
    
    return signal