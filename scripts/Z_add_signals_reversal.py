import numpy as np

def reversal_long(arr, left_lookback, tolerance, live_trading=True):
    high  = arr['high']
    low   = arr['low']
    close = arr['close']
    n = len(high)
    tolerance = tolerance / 100

    signal = np.zeros(n, dtype=int)
    trend_changed = False
    last_break_level = None

    for t in range(left_lookback, n):
        
        recent_highs = high[t-left_lookback:t]

        # Comprobar tendencia bajista
        is_bearish = all(recent_highs[i] > recent_highs[i+1] for i in range(left_lookback-1))
        if is_bearish:
            trend_changed = False  # aún bajista
        
        last_high = recent_highs[-1]
        if not trend_changed and high[t] > last_high:
            trend_changed = True
            last_break_level = last_high

        # Entrada long: toque del último máximo roto con tolerancia
        if trend_changed and last_break_level is not None:
            upper = last_break_level * (1 + tolerance)
            lower = last_break_level * (1 - tolerance)
            
            # Confirma toque del nivel roto
            if low[t] <= upper and high[t] >= lower:
                
                # Confirmación cierre por encima del máximo roto
                if close[t] > last_break_level:
                    
                    # Confirmación de tendencia con MA50
                    if t >= 200:  # asegurarse de tener suficientes datos
                        ma50 = np.mean(close[t-200:t])
                        if close[t] > ma50:
                            signal[t] = 1
                    else:
                        # Si no hay suficientes barras para MA50, opcionalmente no generar señal
                        signal[t] = 0

    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0

    return signal

def reversal_short(arr, left_lookback, tolerance, live_trading=True):
    high  = arr['high']
    low   = arr['low']
    close = arr['close']
    n = len(high)
    tolerance = tolerance / 100
    signal = np.zeros(n, dtype=int)
    trend_changed = False
    last_break_level = None
    for t in range(left_lookback, n):

        recent_lows = low[t-left_lookback:t]
       
        is_bullish = all(recent_lows[i] < recent_lows[i+1] for i in range(left_lookback-1))
        if is_bullish:
            trend_changed = False  

        last_low = recent_lows[-1]
        if not trend_changed and low[t] < last_low:
            trend_changed = True
            last_break_level = last_low
        if trend_changed and last_break_level is not None:
            upper = last_break_level * (1 + tolerance)
            lower = last_break_level * (1 - tolerance)

            
            if low[t] <= upper and high[t] >= lower:
             
                if close[t] < last_break_level:
                
                    if t >= 200: 
                        ma50 = np.mean(close[t-200:t])
                        if close[t] < ma50:
                            signal[t] = -1
                    else:
                       
                        signal[t] = 0
    if not live_trading:
        signal = np.roll(signal, 1)
        signal[0] = 0
    return signal
