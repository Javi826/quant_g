import numpy as np
import pandas as pd
import ta

def scalping_long(
    arr,
    rsi_max=20,     
    adx_min=30,
    lookback=20,
    tolerance=0.2,     
    live_trading=False
):

    high  = pd.Series(arr["high"]).astype(float)
    low   = pd.Series(arr["low"]).astype(float)
    close = pd.Series(arr["close"]).astype(float)
    
    # === Indicadores fijos ===
    ema = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    rsi = ta.momentum.RSIIndicator(close, window=3).rsi()
    adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=5).adx()
    
    n = len(close)
    signals = np.zeros(n, dtype=int)
    start = max(50, lookback)  # el mayor entre EMA(50) y lookback
    
    # Convertir tolerance a decimal (ej: 20 -> 0.20)
    tolerance_decimal = tolerance / 100.0
    
    for i in range(start, n):
        if pd.isna(ema.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(adx.iloc[i]):
            continue
        
        # Condiciones originales
        cond_precio = close.iloc[i] > ema.iloc[i]
        cond_rsi = rsi.iloc[i] <= rsi_max
        cond_adx = adx.iloc[i] >= adx_min
        
        # Nueva condición: detectar soporte
        # Soporte = mínimo de los últimos 'lookback' periodos (excluyendo vela actual)
        soporte = low.iloc[i-lookback:i].min()
        
        # La vela actual "toca" soporte si su low está cerca del soporte
        precio_low_actual = low.iloc[i]
        distancia_relativa = abs(precio_low_actual - soporte) / soporte
        cond_soporte = distancia_relativa <= tolerance_decimal
        
        # Señal cuando se cumplen todas las condiciones
        if cond_precio and cond_rsi and cond_adx and cond_soporte:
            signals[i] = 1
    
    # Evitar lookahead en backtest
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0
    
    return signals

def scalping_short(
    arr,
    rsi_max=80,     
    adx_min=30,
    lookback=20,
    tolerance=0.2,     
    live_trading=False
):
    high  = pd.Series(arr["high"]).astype(float)
    low   = pd.Series(arr["low"]).astype(float)
    close = pd.Series(arr["close"]).astype(float)
    
    # === Indicadores fijos ===
    ema = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    rsi = ta.momentum.RSIIndicator(close, window=3).rsi()
    adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=5).adx()
    
    n = len(close)
    signals = np.zeros(n, dtype=int)
    start = max(50, lookback)  # el mayor entre EMA(50) y lookback
    
    # Convertir tolerance a decimal (ej: 20 -> 0.20)
    tolerance_decimal = tolerance / 100.0
    
    for i in range(start, n):
        if pd.isna(ema.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(adx.iloc[i]):
            continue
        
        # Condiciones inversas para short
        cond_precio = close.iloc[i] < ema.iloc[i]
        cond_rsi = rsi.iloc[i] >= rsi_max
        cond_adx = adx.iloc[i] >= adx_min
        
        # Nueva condición: detectar resistencia
        # Resistencia = máximo de los últimos 'lookback' periodos (excluyendo vela actual)
        resistencia = high.iloc[i-lookback:i].max()
        
        # La vela actual "toca" resistencia si su high está cerca de la resistencia
        precio_high_actual = high.iloc[i]
        distancia_relativa = abs(precio_high_actual - resistencia) / resistencia
        cond_resistencia = distancia_relativa <= tolerance_decimal
        
        # Señal cuando se cumplen todas las condiciones
        if cond_precio and cond_rsi and cond_adx and cond_resistencia:
            signals[i] = -1
    
    # Evitar lookahead en backtest
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0
    
    return signals