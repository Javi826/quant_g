import numpy as np

def parity_long(arr, lookback, tolerance, ma_period, live_trading=True):
    opens  = arr['open']
    closes = arr['close']
    n      = len(closes)
    
    signals    = np.zeros(n, dtype=np.int8)
    body_sizes = np.abs(closes - opens)
    is_red     = closes < opens
    is_green   = closes > opens
    
    for i in range(lookback, n):
        for j in range(1, lookback):
            if i - j - 1 < 0:
                break
            
            idx_red1 = i - j - 1
            idx_green = i - j
            
            if is_red[idx_red1] and is_green[idx_green]:
                size_red1 = body_sizes[idx_red1]
                size_green = body_sizes[idx_green]
                
                if size_red1 == 0:
                    continue
                
                diff_green_red1 = abs(size_green - size_red1) / size_red1 * 100
                
                if diff_green_red1 <= tolerance:
                    for k in range(idx_green + 1, i):
                        if is_red[k]:
                            size_red2 = body_sizes[k]
                            close_red1 = closes[idx_red1]
                            close_red2 = closes[k]
                            
                            diff_red2_red1 = abs(size_red2 - size_red1) / size_red1 * 100
                            diff_close = abs(close_red2 - close_red1) / abs(close_red1) * 100
                            
                            if diff_red2_red1 <= tolerance and diff_close <= tolerance:
                                # Confirmación de tendencia con MA
                                if i >= ma_period:
                                    ma = np.mean(closes[i-ma_period:i])
                                    if closes[i] > ma:
                                        signals[i] = 1
                                else:
                                    # Si no hay suficientes datos, no genera señal
                                    signals[i] = 0
                                break
                    
                    if signals[i] == 1:
                        break
    
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0  
    
    return signals

def parity_short(arr, lookback, tolerance, ma_period, live_trading=True):
    opens  = arr['open']
    closes = arr['close']
    n      = len(closes)
    
    signals    = np.zeros(n, dtype=np.int8)
    body_sizes = np.abs(closes - opens)
    is_red     = closes < opens
    is_green   = closes > opens
    
    for i in range(lookback, n):
        for j in range(1, lookback):
            if i - j - 1 < 0:
                break
            
            idx_green1 = i - j - 1
            idx_red = i - j
            
            if is_green[idx_green1] and is_red[idx_red]:
                size_green1 = body_sizes[idx_green1]
                size_red = body_sizes[idx_red]
                
                if size_green1 == 0:
                    continue
                
                diff_red_green1 = abs(size_red - size_green1) / size_green1 * 100
                
                if diff_red_green1 <= tolerance:
                    for k in range(idx_red + 1, i):
                        if is_green[k]:
                            size_green2 = body_sizes[k]
                            close_green1 = closes[idx_green1]
                            close_green2 = closes[k]
                            
                            diff_green2_green1 = abs(size_green2 - size_green1) / size_green1 * 100
                            diff_close = abs(close_green2 - close_green1) / abs(close_green1) * 100
                            
                            if diff_green2_green1 <= tolerance and diff_close <= tolerance:
                                # Confirmación de tendencia con MA
                                if i >= ma_period:
                                    ma = np.mean(closes[i-ma_period:i])
                                    if closes[i] < ma:
                                        signals[i] = -1
                                else:
                                    # Si no hay suficientes datos, no genera señal
                                    signals[i] = 0
                                break
                    
                    if signals[i] == -1:
                        break
    
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0  
    
    return signals