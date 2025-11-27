import numpy as np
import pandas as pd
import ta

def scalping_long(arr, ema_short=35, ema_long=50, lookback=3, touch_tolerance=0.01, live_trading=False):

    high  = pd.Series(arr['high'])
    low   = pd.Series(arr['low'])
    close = pd.Series(arr['close'])
    
   
    ema_s = ta.trend.EMAIndicator(close, window=ema_short).ema_indicator()
    ema_l = ta.trend.EMAIndicator(close, window=ema_long).ema_indicator()
    
    n = len(close)
    signals = np.zeros(n, dtype=int)
    
    for i in range(lookback, n):
        
        all_above = True
        for j in range(i - lookback, i):
            if not (low.iloc[j] > ema_s.iloc[j] and low.iloc[j] > ema_l.iloc[j]):
                all_above = False
                break
        
        if not all_above:
            continue
             
        touches_short = low.iloc[i] >= ema_s.iloc[i] * (1 - touch_tolerance) and low.iloc[i] <= ema_s.iloc[i] * (1 + touch_tolerance)
        touches_long = low.iloc[i] >= ema_l.iloc[i] * (1 - touch_tolerance) and low.iloc[i] <= ema_l.iloc[i] * (1 + touch_tolerance)
        
        if touches_short or touches_long:
            signals[i] = 1
    
    # SHIFT para evitar lookahead bias
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0
    
    return signals