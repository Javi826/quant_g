import numpy as np


def flag_long(
    ohlcv_arr,
    lookback=50,
    impulse=5,
    flag=40,
    ma_period=50,
    live_trading=False
):
    """
    Detects bullish flag pattern (continuation pattern).
    
    Logic:
    1. Find strong upward impulse (low to high movement)
    2. Consolidation near the impulse high (flag)
    3. Breakout above consolidation confirming continuation
    4. Price above MA confirmation
    
    Parameters:
    -----------
    ohlcv_arr : dict
        Dictionary with 'open', 'high', 'low', 'close' arrays
    lookback : int
        Period to search for impulse + flag pattern
    impulse : int
        Minimum percentage for impulse (e.g., 5 for 5%)
    flag : int
        Maximum percentage for flag range relative to flag_low (e.g., 40 for 40%)
    ma_period : int
        Moving average period for trend confirmation
    live_trading : bool
        If True, uses only data up to i-1 (causal for live)
        If False, can use data up to i (backtesting)
    
    Returns:
    --------
    signals : np.ndarray
        Array of 0s and 1s where 1 indicates long entry signal
    """
    close = ohlcv_arr['close']
    high = ohlcv_arr['high']
    low = ohlcv_arr['low']
    n = len(close)
    
    signals = np.zeros(n, dtype=np.int32)
    
    # Convert percentages to decimals
    impulse_min = impulse / 100.0
    flag_max_range = flag / 100.0
    
    # Flag period (last 30-40% of lookback for consolidation)
    flag_period = max(3, int(lookback * 0.35))
    
    min_bars = max(lookback + 5, ma_period)
    
    # Breakout threshold relaxation (98% of flag high)
    breakout_threshold = 0.98
    
    for i in range(min_bars, n):
        # Define windows
        lookback_start = i - lookback
        flag_start_idx = i - flag_period + 1
        flag_end_idx = i
        
        if lookback_start < 0:
            continue
        
        # 1. DETECT IMPULSE: measure from lowest low to highest high in lookback
        impulse_low = np.min(low[lookback_start:flag_start_idx])
        impulse_high = np.max(high[lookback_start:flag_start_idx])
        
        if impulse_low <= 0:
            continue
        
        impulse_change = (impulse_high - impulse_low) / impulse_low
        
        # Must have strong upward impulse
        if impulse_change < impulse_min:
            continue
        
        # 2. CHECK FLAG: consolidation with small range near impulse high
        flag_high = np.max(high[flag_start_idx:flag_end_idx + 1])
        flag_low = np.min(low[flag_start_idx:flag_end_idx + 1])
        
        if flag_low <= 0:
            continue
        
        flag_range = (flag_high - flag_low) / flag_low
        
        # Flag range must be relatively small
        if flag_range > flag_max_range:
            continue
        
        # 3. CHECK BREAKOUT: price breaks above or near flag high
        current_close = close[flag_end_idx]
        breakout_level = flag_high * breakout_threshold
        
        # Relaxed breakout condition: close above 98% of flag high
        if current_close <= breakout_level:
            continue
        
        # 4. MA CONFIRMATION: price above moving average
        if i >= ma_period:
            ma = np.mean(close[i - ma_period:i])
            if current_close > ma:
                signals[i] = 1
    
    # SHIFT to avoid lookahead in backtesting
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0
    
    return signals

import numpy as np

def flag_short(
    ohlcv_arr,
    lookback=50,
    impulse=5,
    flag=40,
    ma_period=50,
    live_trading=False
):
    """
    Detects bearish flag pattern (continuation pattern).
    
    Logic:
    1. Find strong downward impulse (high to low movement)
    2. Consolidation near the impulse low (flag)
    3. Breakout below consolidation confirming continuation
    4. Price below MA confirmation
    
    Parameters:
    -----------
    ohlcv_arr : dict
        Dictionary with 'open', 'high', 'low', 'close' arrays
    lookback : int
        Period to search for impulse + flag pattern
    impulse : int
        Minimum percentage for impulse (e.g., 5 for 5%)
    flag : int
        Maximum percentage for flag range relative to flag_high (e.g., 40 for 40%)
    ma_period : int
        Moving average period for trend confirmation
    live_trading : bool
        If True, uses only data up to i-1 (causal for live)
        If False, can use data up to i (backtesting)
    
    Returns:
    --------
    signals : np.ndarray
        Array of 0s and -1s where -1 indicates short entry signal
    """
    close = ohlcv_arr['close']
    high = ohlcv_arr['high']
    low = ohlcv_arr['low']
    n = len(close)
    
    signals = np.zeros(n, dtype=np.int32)
    
    # Convert percentages to decimals
    impulse_min = impulse / 100.0
    flag_max_range = flag / 100.0
    
    # Flag period (last 30-40% of lookback for consolidation)
    flag_period = max(3, int(lookback * 0.35))
    
    min_bars = max(lookback + 5, ma_period)
    
    # Breakout threshold relaxation (102% of flag low)
    breakout_threshold = 1.02
    
    for i in range(min_bars, n):
        # Define windows
        lookback_start = i - lookback
        flag_start_idx = i - flag_period + 1
        flag_end_idx = i
        
        if lookback_start < 0:
            continue
        
        # 1. DETECT IMPULSE: measure from highest high to lowest low in lookback
        impulse_high = np.max(high[lookback_start:flag_start_idx])
        impulse_low = np.min(low[lookback_start:flag_start_idx])
        
        if impulse_high <= 0:
            continue
        
        impulse_change = (impulse_high - impulse_low) / impulse_high
        
        # Must have strong downward impulse
        if impulse_change < impulse_min:
            continue
        
        # 2. CHECK FLAG: consolidation with small range near impulse low
        flag_high = np.max(high[flag_start_idx:flag_end_idx + 1])
        flag_low = np.min(low[flag_start_idx:flag_end_idx + 1])
        
        if flag_high <= 0:
            continue
        
        flag_range = (flag_high - flag_low) / flag_high
        
        # Flag range must be relatively small
        if flag_range > flag_max_range:
            continue
        
        # 3. CHECK BREAKOUT: price breaks below or near flag low
        current_close = close[flag_end_idx]
        breakout_level = flag_low * breakout_threshold
        
        # Relaxed breakout condition: close below 102% of flag low
        if current_close >= breakout_level:
            continue
        
        # 4. MA CONFIRMATION: price below moving average
        if i >= ma_period:
            ma = np.mean(close[i - ma_period:i])
            if current_close < ma:
                signals[i] = -1
    
    # SHIFT to avoid lookahead in backtesting
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0
    
    return signals