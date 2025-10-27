import numpy as np

def explosive_signal_pr(open_prices, 
                        close_prices, 
                        factor,
                        body_tolerance, 
                        close_tolerance, 
                        live=False):
    lookback = 50  # ventana para patrones y promedio de cuerpos
    n = len(close_prices)
    signal = np.zeros(n, dtype=np.int8)

    first_parity = []   # roja → verde
    second_parity = []  # verde → roja

    for i in range(1, n):
        # --- Cálculo de cuerpos ---
        body_prev = abs(close_prices[i-1] - open_prices[i-1])
        body_curr = abs(close_prices[i] - open_prices[i])
        if body_prev == 0:
            continue

        # --- Promedio de cuerpos recientes usando el mismo lookback ---
        start_idx = max(0, i - lookback)
        recent_bodies = [abs(close_prices[j] - open_prices[j]) for j in range(start_idx, i)]
        avg_body = np.mean(recent_bodies) if recent_bodies else body_prev
        large_prev = body_prev >= factor * avg_body
        large_curr = body_curr >= factor * avg_body

        # --- Primer patrón: roja → verde ---
        prev_red = close_prices[i-1] < open_prices[i-1]
        curr_green = close_prices[i] > open_prices[i]

        body_similar = abs(body_curr - body_prev) / body_prev <= body_tolerance
        close_prev = close_prices[i-1]
        close_curr = close_prices[i]
        close_similar = abs(close_curr - close_prev) / close_prev <= close_tolerance

        if prev_red and curr_green and body_similar and close_similar and large_prev and large_curr:
            first_parity.append((i, close_curr))

        # --- Segundo patrón: verde → roja ---
        prev_green = close_prices[i-1] > open_prices[i-1]
        curr_red = close_prices[i] < close_prices[i-1]
        body_similar_2  = abs(body_curr - body_prev) / body_prev <= body_tolerance
        close_similar_2 = abs(close_curr - close_prev) / close_prev <= close_tolerance

        if prev_green and curr_red and body_similar_2 and close_similar_2 and large_prev and large_curr:
            second_parity.append((i, close_curr))

        # --- Generar señal ---
        recent_first = [(j, c) for j, c in first_parity if i - j <= lookback]
        recent_second = [(j, c) for j, c in second_parity if i - j <= lookback]

        if recent_first and recent_second:
            last_second_idx, _ = recent_second[-1]
            first_before_second = [c for j, c in recent_first if j < last_second_idx]
            if first_before_second:
                first_close = first_before_second[-1]
                lower = first_close * (1 - close_tolerance)
                upper = first_close * (1 + close_tolerance)
                
                # --- Condición extra: vela actual también debe ser grande ---
                is_large_signal = body_curr >= factor * avg_body

                if lower <= close_prices[i] <= upper and is_large_signal:
                    signal[i] = 1

    # Shift si no es live
    if not live:
        shifted = np.zeros_like(signal)
        shifted[1:] = signal[:-1]
        signal = shifted

    return signal
