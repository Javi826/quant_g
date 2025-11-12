import numpy as np

def detect_plane(arr, live_trading=True):
    n = len(arr['close'])
    signals = np.ones(n, dtype=np.int8)
    
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0
    
    return signals


