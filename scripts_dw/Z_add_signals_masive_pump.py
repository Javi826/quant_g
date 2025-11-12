import numpy as np

import numpy as np

def generate_signals_from_array(data, n=5, pct_volume_increase=0.03, max_close_change=0.02):

    close = np.array(data['close'])
    quote_volume = np.array(data['volume_quote'])
    
    signals = np.zeros_like(close)
    
    for i in range(2 * n, len(close)):
        # Ventana anterior y actual de volumen
        prev_window_volume = quote_volume[i - 2*n:i - n]
        curr_window_volume = quote_volume[i - n:i]
        
        # Ventana actual de precios
        curr_window_close = close[i - n:i]
        
        # Promedio de volumen de cada ventana
        prev_volume_mean = np.mean(prev_window_volume)
        curr_volume_mean = np.mean(curr_window_volume)
                
        # Cambio máximo del precio de cierre dentro de la ventana actual
        min_close = np.min(curr_window_close)
        max_close = np.max(curr_window_close)
        close_change = (max_close - min_close) / min_close
        
        # Generar señal si el volumen aumentó suficiente y el precio se mantuvo estable
        if curr_volume_mean > prev_volume_mean * (1 + pct_volume_increase) and close_change <= max_close_change:
            signals[i] = 1
    
    # Evitar lookahead
    signals = np.roll(signals, 1)
    signals[0] = 0
    
    return signals
