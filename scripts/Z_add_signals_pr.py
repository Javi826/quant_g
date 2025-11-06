import numpy as np

def detect_parity_reversal_long(arr, tolerance, lookback, backtest=False):

    tolerance = tolerance / 100.0
    open_, close, low = arr['open'], arr['close'], arr['low']
    n = len(open_)
    signal = np.zeros(n, dtype=np.int8)
    
    # Historial de bloques completados
    completed_blocks = []
    
    # Estado del bloque actual
    current_block = None
    
    # Variables para tracking de paridad
    ref_body = None
    ref_low = None
    
    for i in range(n):
        # Dirección de la vela actual
        dir_i = 1 if close[i] > open_[i] else -1
        body_i = abs(close[i] - open_[i])
        
        # --- Inicializar o continuar bloque ---
        if current_block is None:
            # Primer bloque
            current_block = {
                'dir': dir_i,
                'start': i,
                'bodies': [body_i],
                'lows': [low[i]],
                'close': close[i]
            }
        elif current_block['dir'] == dir_i:
            # Continuar bloque actual
            current_block['bodies'].append(body_i)
            current_block['lows'].append(low[i])
            current_block['close'] = close[i]
        else:
            # Cambio de dirección → cerrar bloque anterior
            prev_block = {
                'dir': current_block['dir'],
                'start': current_block['start'],
                'end': i - 1,
                'body_mean': np.mean(current_block['bodies']),
                'low': min(current_block['lows']),
                'close': current_block['close']
            }
            completed_blocks.append(prev_block)
            
            # --- Buscar paridades con bloques previos ---
            # Solo buscamos en bloques YA COMPLETADOS (sin lookahead)
            if prev_block['dir'] == 1:  # Bloque verde recién completado
                # Buscar bloque rojo similar en el historial
                start_idx = max(0, len(completed_blocks) - 1 - lookback)
                for j in range(start_idx, len(completed_blocks) - 1):
                    b_j = completed_blocks[j]
                    if b_j['dir'] == -1:
                        rel_diff = abs(prev_block['body_mean'] - b_j['body_mean']) / max(prev_block['body_mean'], b_j['body_mean'])
                        if rel_diff <= tolerance:
                            # Paridad encontrada
                            ref_body = prev_block['body_mean']
                            ref_low = prev_block['low']
                            break
            
            # --- Verificar si el bloque anterior cumple condición de reversión ---
            if ref_body is not None and prev_block['dir'] == -1:
                rel_diff_ref = abs(prev_block['body_mean'] - ref_body) / max(prev_block['body_mean'], ref_body)
                rel_diff_low = abs(prev_block['close'] - ref_low) / ref_low if ref_low != 0 else float('inf')
                
                if rel_diff_ref <= tolerance and rel_diff_low <= tolerance:
                    # SEÑAL: se genera al CIERRE del bloque rojo (i-1)
                    # En livetrading, esta señal estaría disponible en la apertura de la barra i
                    signal[i - 1] = 1
                    ref_body = None
                    ref_low = None
            
            # Iniciar nuevo bloque
            current_block = {
                'dir': dir_i,
                'start': i,
                'bodies': [body_i],
                'lows': [low[i]],
                'close': close[i]
            }
    
    # --- Desplazamiento para backtest ---
    if backtest:
        shifted = np.zeros_like(signal)
        shifted[1:] = signal[:-1]
        signal = shifted
    
    return signal


def detect_parity_reversal_short(arr, tolerance, lookback, backtest=False):
    tolerance = tolerance / 100.0
    open_, close, high = arr['open'], arr['close'], arr['high']
    n = len(open_)
    signal = np.zeros(n, dtype=np.int8)
    
    # Historial de bloques completados
    completed_blocks = []
    
    # Estado del bloque actual
    current_block = None
    
    # Variables para tracking de paridad
    ref_body = None
    ref_high = None
    
    for i in range(n):
        # Dirección de la vela actual
        dir_i = 1 if close[i] > open_[i] else -1
        body_i = abs(close[i] - open_[i])
        
        # --- Inicializar o continuar bloque ---
        if current_block is None:
            # Primer bloque
            current_block = {
                'dir': dir_i,
                'start': i,
                'bodies': [body_i],
                'highs': [high[i]],
                'close': close[i]
            }
        elif current_block['dir'] == dir_i:
            # Continuar bloque actual
            current_block['bodies'].append(body_i)
            current_block['highs'].append(high[i])
            current_block['close'] = close[i]
        else:
            # Cambio de dirección → cerrar bloque anterior
            prev_block = {
                'dir': current_block['dir'],
                'start': current_block['start'],
                'end': i - 1,
                'body_mean': np.mean(current_block['bodies']),
                'high': max(current_block['highs']),
                'close': current_block['close']
            }
            completed_blocks.append(prev_block)
            
            # --- Buscar paridades con bloques previos ---
            # Solo buscamos en bloques YA COMPLETADOS (sin lookahead)
            if prev_block['dir'] == -1:  # Bloque rojo recién completado
                # Buscar bloque verde similar en el historial
                start_idx = max(0, len(completed_blocks) - 1 - lookback)
                for j in range(start_idx, len(completed_blocks) - 1):
                    b_j = completed_blocks[j]
                    if b_j['dir'] == 1:
                        rel_diff = abs(prev_block['body_mean'] - b_j['body_mean']) / max(prev_block['body_mean'], b_j['body_mean'])
                        if rel_diff <= tolerance:
                            # Paridad encontrada
                            ref_body = prev_block['body_mean']
                            ref_high = prev_block['high']
                            break
            
            # --- Verificar si el bloque anterior cumple condición de reversión ---
            if ref_body is not None and prev_block['dir'] == 1:
                rel_diff_ref = abs(prev_block['body_mean'] - ref_body) / max(prev_block['body_mean'], ref_body)
                rel_diff_high = abs(prev_block['close'] - ref_high) / ref_high if ref_high != 0 else float('inf')
                
                if rel_diff_ref <= tolerance and rel_diff_high <= tolerance:
                    # SEÑAL: se genera al CIERRE del bloque verde (i-1)
                    # En livetrading, esta señal estaría disponible en la apertura de la barra i
                    signal[i - 1] = -1
                    ref_body = None
                    ref_high = None
            
            # Iniciar nuevo bloque
            current_block = {
                'dir': dir_i,
                'start': i,
                'bodies': [body_i],
                'highs': [high[i]],
                'close': close[i]
            }
    
    # --- Desplazamiento para backtest ---
    if backtest:
        shifted = np.zeros_like(signal)
        shifted[1:] = signal[:-1]
        signal = shifted
    
    return signal
