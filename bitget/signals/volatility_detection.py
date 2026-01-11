import numpy as np
import pandas as pd

def detect_volatility_ll(arr, atr_period=14, chaos_percentile=90):
    """
    Detecta CHAOS: ATR% > percentil rolling
    
    Simple y efectivo.
    """
    high  = arr['high']
    low   = arr['low']
    close = arr['close']
    n = len(close)
    
    LOOKBACK = 200
    
    # 1. True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # 2. ATR
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period:i])
    
    # 3. ATR normalizado (% del precio)
    atr_pct = (atr / close) * 100
    
    # 4. Filtro: solo percentil rolling, sin MIN_VOL_PCT
    vol_filter = np.ones(n, dtype=np.int8)
    start_idx = atr_period + LOOKBACK
    
    for i in range(start_idx, n):
        window = atr_pct[i - LOOKBACK:i]
        window = window[~np.isnan(window)]
        
        if len(window) == 0:
            continue
            
        threshold = np.percentile(window, chaos_percentile)
        
        if atr_pct[i] > threshold:
            vol_filter[i] = 0
    
    return vol_filter

def detect_volatility(arr, atr_period=14, chaos_percentile=90):
    """
    Detecta régimen de volatilidad: STABLE (1) o CHAOS (0)
    Usa ventana rolling de 200 velas en lugar de histórico completo.
    """
    high = arr['high']
    low = arr['low']
    close = arr['close']
    n = len(close)
    
    LOOKBACK = 200
    
    # Calcular True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calcular ATR
    atr = np.zeros(n)
    atr[:atr_period] = np.nan
    
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period:i])
    
    # Filtro con ventana rolling
    vol_filter = np.ones(n, dtype=np.int8)
    start_idx  = atr_period + LOOKBACK
    
    for i in range(start_idx, n):
        # Últimas LOOKBACK velas (no todo el histórico)
        window_atr = atr[i - LOOKBACK:i]
        window_atr = window_atr[~np.isnan(window_atr)]
        
        if len(window_atr) > 0:
            threshold = np.percentile(window_atr, chaos_percentile)
            
            if atr[i] > threshold:
                vol_filter[i] = 0
    
    return vol_filter