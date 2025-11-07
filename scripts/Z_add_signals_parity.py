import numpy as np


def detect_parity_reversal_long_gpt(arr, tolerance, lookback, shift_for_execution=False):

    tolerance = tolerance / 100.0
    open_, close, low = arr['open'], arr['close'], arr['low']
    n = len(open_)
    signal = np.zeros(n, dtype=np.int8)

    completed_blocks = []
    current_block = None
    ref_body = None
    ref_low = None


    for idx in range(n):
        dir_k = 1 if close[idx] > open_[idx] else -1
        body_k = abs(close[idx] - open_[idx])
        low_k = low[idx]

        if current_block is None:
            # iniciamos primer bloque con la vela idx
            current_block = {
                'dir': dir_k,
                'start': idx,
                'bodies': [body_k],
                'lows': [low_k],
                'close': close[idx]
            }
            continue

        if current_block['dir'] == dir_k:

            current_block['bodies'].append(body_k)
            current_block['lows'].append(low_k)
            current_block['close'] = close[idx]
        else:

            prev_block = {
                'dir': current_block['dir'],
                'start': current_block['start'],
                'end': idx - 1,
                'body_mean': np.mean(current_block['bodies']),
                'low': min(current_block['lows']),
                'close': current_block['close']
            }
            completed_blocks.append(prev_block)

            # Si el bloque finalizado fue alcista, buscamos un bloque bajista previo similar
            if prev_block['dir'] == 1:
                start_idx = max(0, len(completed_blocks) - 1 - lookback)
                for j in range(start_idx, len(completed_blocks) - 1):
                    b_j = completed_blocks[j]
                    if b_j['dir'] == -1:
                        rel_diff = abs(prev_block['body_mean'] - b_j['body_mean']) / max(prev_block['body_mean'], b_j['body_mean'])
                        if rel_diff <= tolerance:
                            ref_body = prev_block['body_mean']
                            ref_low = prev_block['low']
                            break

            if ref_body is not None and prev_block['dir'] == -1:
                rel_diff_ref = abs(prev_block['body_mean'] - ref_body) / max(prev_block['body_mean'], ref_body)
                rel_diff_low = abs(prev_block['close'] - ref_low) / ref_low if ref_low != 0 else float('inf')


                exec_index = idx  
                if rel_diff_ref <= tolerance and rel_diff_low <= tolerance and exec_index < n:
                    signal[exec_index] = 1
                    ref_body = None
                    ref_low = None

            # iniciamos nuevo bloque con la vela idx (ya cerrada)
            current_block = {
                'dir': dir_k,
                'start': idx,
                'bodies': [body_k],
                'lows': [low_k],
                'close': close[idx]
            }


    return signal


def detect_parity_reversal_long(arr, tolerance, lookback, shift_for_execution=True):

    tolerance = tolerance / 100.0
    open_, close, low = arr['open'], arr['close'], arr['low']
    n = len(open_)
    signal = np.zeros(n, dtype=np.int8)
    

    completed_blocks = []
    
    current_block = None
    
    ref_body = None
    ref_low = None
    
    for i in range(n):

        dir_i = 1 if close[i] > open_[i] else -1
        body_i = abs(close[i] - open_[i])
        
        if current_block is None:

            current_block = {
                'dir': dir_i,
                'start': i,
                'bodies': [body_i],
                'lows': [low[i]],
                'close': close[i]
            }
        elif current_block['dir'] == dir_i:

            current_block['bodies'].append(body_i)
            current_block['lows'].append(low[i])
            current_block['close'] = close[i]
        else:

            prev_block = {
                'dir': current_block['dir'],
                'start': current_block['start'],
                'end': i - 1,
                'body_mean': np.mean(current_block['bodies']),
                'low': min(current_block['lows']),
                'close': current_block['close']
            }
            completed_blocks.append(prev_block)

            if prev_block['dir'] == 1:  

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

            if ref_body is not None and prev_block['dir'] == -1:
                rel_diff_ref = abs(prev_block['body_mean'] - ref_body) / max(prev_block['body_mean'], ref_body)
                rel_diff_low = abs(prev_block['close'] - ref_low) / ref_low if ref_low != 0 else float('inf')
                
                if rel_diff_ref <= tolerance and rel_diff_low <= tolerance:

                    signal[i - 1] = 1
                    ref_body = None
                    ref_low = None
            
            current_block = {
                'dir': dir_i,
                'start': i,
                'bodies': [body_i],
                'lows': [low[i]],
                'close': close[i]
            }

    if shift_for_execution:
        shifted = np.zeros_like(signal)
        shifted[1:] = signal[:-1]
        signal = shifted
    
    return signal


def detect_parity_reversal_short(arr, tolerance, lookback, shift_for_execution=True):
    tolerance = tolerance / 100.0
    open_, close, high = arr['open'], arr['close'], arr['high']
    n = len(open_)
    signal = np.zeros(n, dtype=np.int8)
    

    completed_blocks = []
    
    current_block = None
    
    ref_body = None
    ref_high = None
    
    for i in range(n):

        dir_i = 1 if close[i] > open_[i] else -1
        body_i = abs(close[i] - open_[i])
        

        if current_block is None:

            current_block = {
                'dir': dir_i,
                'start': i,
                'bodies': [body_i],
                'highs': [high[i]],
                'close': close[i]
            }
        elif current_block['dir'] == dir_i:

            current_block['bodies'].append(body_i)
            current_block['highs'].append(high[i])
            current_block['close'] = close[i]
        else:

            prev_block = {
                'dir': current_block['dir'],
                'start': current_block['start'],
                'end': i - 1,
                'body_mean': np.mean(current_block['bodies']),
                'high': max(current_block['highs']),
                'close': current_block['close']
            }
            completed_blocks.append(prev_block)
            
            if prev_block['dir'] == -1: 

                start_idx = max(0, len(completed_blocks) - 1 - lookback)
                for j in range(start_idx, len(completed_blocks) - 1):
                    b_j = completed_blocks[j]
                    if b_j['dir'] == 1:
                        rel_diff = abs(prev_block['body_mean'] - b_j['body_mean']) / max(prev_block['body_mean'], b_j['body_mean'])
                        if rel_diff <= tolerance:

                            ref_body = prev_block['body_mean']
                            ref_high = prev_block['high']
                            break
            
            if ref_body is not None and prev_block['dir'] == 1:
                rel_diff_ref = abs(prev_block['body_mean'] - ref_body) / max(prev_block['body_mean'], ref_body)
                rel_diff_high = abs(prev_block['close'] - ref_high) / ref_high if ref_high != 0 else float('inf')
                
                if rel_diff_ref <= tolerance and rel_diff_high <= tolerance:

                    signal[i - 1] = -1
                    ref_body = None
                    ref_high = None
            
            current_block = {
                'dir': dir_i,
                'start': i,
                'bodies': [body_i],
                'highs': [high[i]],
                'close': close[i]
            }
    
    if shift_for_execution:
        shifted = np.zeros_like(signal)
        shifted[1:] = signal[:-1]
        signal = shifted
    
    return signal
