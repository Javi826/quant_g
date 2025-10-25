# === FILE: add_signals03.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

def explosive_signal_08(open_prices, close_prices, low_prices, lookback=3,
                        body_tolerance=0.1, low_tolerance=0.002, live=False):


    n = len(close_prices)
    signal = np.zeros(n, dtype=np.int8)

    # Guardamos los mínimos de las velas rojas que formaron paridades válidas
    parity_lows = []

    for i in range(1, n):
        # 1️⃣ Condición: vela verde seguida de roja
        prev_green = close_prices[i-1] > open_prices[i-1]
        curr_red = close_prices[i] < open_prices[i]

        # 2️⃣ Cuerpos de las velas
        body_prev = abs(close_prices[i-1] - open_prices[i-1])
        body_curr = abs(close_prices[i] - open_prices[i])

        if body_prev == 0:
            continue

        # Similitud de cuerpo
        body_similar = abs(body_curr - body_prev) / body_prev <= body_tolerance

        # 3️⃣ Similitud de mínimos
        low_prev = low_prices[i-1]
        low_curr = low_prices[i]
        low_similar = abs(low_curr - low_prev) / low_prev <= low_tolerance

        # ✅ Si cumple todas las condiciones, guardamos el mínimo de la roja
        if prev_green and curr_red and body_similar and low_similar:
            parity_lows.append((i, low_curr))

        # 🔍 Verificar si la vela actual toca el mínimo de alguna paridad reciente
        recent_parities = [low for j, low in parity_lows if i - j <= lookback]

        if recent_parities:
            #if low_prices[i] <= min(recent_parities):
            if any(low_prices[i] <= parity_low for parity_low in recent_parities):
                signal[i] = 1

    # 🕒 Shift si no es modo live
    if not live:
        shifted = np.zeros_like(signal)
        shifted[1:] = signal[:-1]
        signal = shifted

    return signal

import numpy as np
from numba import njit

@njit
def explosive_signal_01(open_prices, close_prices, low_prices, lookback,
                        body_tolerance, low_tolerance, live=False):
    n = len(close_prices)
    signal = np.zeros(n, dtype=np.int8)
    
    # Pre-calculamos todas las condiciones
    is_green = close_prices > open_prices
    is_red = close_prices < open_prices
    bodies = np.abs(close_prices - open_prices)
    
    # Lista para guardar paridades (índice, mínimo)
    parity_indices = []
    parity_lows_vals = []
    
    for i in range(1, n):
        # 1️⃣ Condición: vela verde seguida de roja
        prev_green = is_green[i-1]
        curr_red = is_red[i]
        
        # 2️⃣ Cuerpos de las velas
        body_prev = bodies[i-1]
        body_curr = bodies[i]
        
        if body_prev == 0:
            continue
        
        # Similitud de cuerpo
        body_similar = abs(body_curr - body_prev) / body_prev <= body_tolerance
        
        # 3️⃣ Similitud de mínimos
        low_prev = low_prices[i-1]
        low_curr = low_prices[i]
        low_similar = abs(low_curr - low_prev) / low_prev <= low_tolerance
        
        # ✅ Si cumple todas las condiciones, guardamos el mínimo de la roja
        if prev_green and curr_red and body_similar and low_similar:
            parity_indices.append(i)
            parity_lows_vals.append(low_curr)
        
        # 🔍 Verificar si la vela actual toca el mínimo de alguna paridad reciente
        min_parity = np.inf
        for j in range(len(parity_indices)):
            if i - parity_indices[j] <= lookback:
                if parity_lows_vals[j] < min_parity:
                    min_parity = parity_lows_vals[j]
        
        if min_parity != np.inf:
            if low_prices[i] <= min_parity:
                signal[i] = 1
    
    # 🕒 Shift si no es modo live
    if not live:
        shifted = np.zeros(n, dtype=np.int8)
        shifted[1:] = signal[:-1]
        signal = shifted
    
    return signal