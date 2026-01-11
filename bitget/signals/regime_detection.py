import numpy as np
import pandas as pd
from ta.trend import ADXIndicator

def detect_regime(arr, adx_threshold=25, adx_period=14, live_trading=True):
    """
    Detecta régimen de mercado con dirección:
    0 = RANGING, 1 = UPTREND, 2 = DOWNTREND
    
    Parameters:
    -----------
    arr : dict
        Dictionary con keys: 'open', 'high', 'low', 'close' (numpy arrays)
    adx_threshold : float
        Threshold para clasificar trending (default: 25)
    adx_period : int
        Periodo para calcular ADX (default: 14)
    live_trading : bool
        Si False, aplica shift de 1 vela para backtest (default: True)
    
    Returns:
    --------
    regimes : np.array
        Array de int: 0 = RANGING, 1 = UPTREND, 2 = DOWNTREND
    """
    
    high  = pd.Series(arr['high'], dtype=np.float64)
    low   = pd.Series(arr['low'], dtype=np.float64)
    close = pd.Series(arr['close'], dtype=np.float64)
    
    adx_ind  = ADXIndicator(high=high, low=low, close=close, window=adx_period)
    adx      = adx_ind.adx().to_numpy()
    plus_di  = adx_ind.adx_pos().to_numpy()
    minus_di = adx_ind.adx_neg().to_numpy()
    
    regimes = np.zeros(len(close), dtype=np.int8)  # RANGING por defecto
    
    trending_mask  = adx > adx_threshold
    uptrend_mask   = trending_mask & (plus_di > minus_di)
    downtrend_mask = trending_mask & (minus_di > plus_di)
    
    regimes[uptrend_mask]   = 1    # UPTREND
    regimes[downtrend_mask] = 2  # DOWNTREND
    
    regimes = np.nan_to_num(regimes, nan=0.0).astype(np.int8)
    
    if not live_trading:
        regimes = np.roll(regimes, 1)
        regimes[0] = 0
    
    return regimes