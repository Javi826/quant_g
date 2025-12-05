import numpy as np

def calculate_btc_regime(btc_data, window=20, vol_threshold=2.5, trend_window=50):
    """
    Calcula el régimen de mercado basado en BTC.
    
    Parámetros:
    - btc_data: array con datos de BTC (debe tener 'close', 'high', 'low')
    - window: ventana para calcular volatilidad (default 20)
    - vol_threshold: umbral de volatilidad en % (default 2.5)
    - trend_window: ventana para determinar tendencia (default 50)
    
    Retorna:
    - Array booleano: True = régimen favorable, False = régimen desfavorable
    """
    closes = btc_data['close']
    highs = btc_data['high']
    lows = btc_data['low']
    n = len(closes)
    
    regime = np.zeros(n, dtype=bool)
    
    # Calcular volatilidad (ATR normalizado)
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - np.roll(closes, 1)),
                              np.abs(lows - np.roll(closes, 1))))
    
    for i in range(max(window, trend_window), n):
        # 1. Volatilidad controlada (ATR% promedio últimos 'window' períodos)
        atr_window = tr[i-window+1:i+1]
        atr_avg = np.mean(atr_window)
        atr_pct = (atr_avg / closes[i]) * 100
        
        # 2. Tendencia alcista (precio > MA50 y MA20 > MA50)
        ma20 = np.mean(closes[i-20:i])
        ma50 = np.mean(closes[i-trend_window:i])
        
        # 3. Momento positivo (precio subiendo en últimos 5 días)
        price_momentum = closes[i] > closes[i-5]
        
        # Régimen favorable si:
        # - Volatilidad no excesiva
        # - Tendencia alcista
        # - Momentum positivo
        vol_ok = atr_pct <= vol_threshold
        trend_ok = closes[i] > ma50 and ma20 > ma50
        momentum_ok = price_momentum
        
        regime[i] = vol_ok and trend_ok and momentum_ok
    
    return regime


def calculate_advanced_btc_regime(btc_data, lookback=20):
    """
    Régimen de mercado más sofisticado usando múltiples indicadores.
    
    Retorna:
    - Array de scores (0-4): a mayor score, mejor régimen
    """
    closes = btc_data['close']
    highs = btc_data['high']
    lows = btc_data['low']
    volumes = btc_data.get('volume', np.ones(len(closes)))
    
    n = len(closes)
    regime_score = np.zeros(n, dtype=np.int8)
    
    for i in range(max(50, lookback), n):
        score = 0
        
        # 1. Volatilidad (ATR)
        tr = np.maximum(highs[i] - lows[i],
                       np.maximum(abs(highs[i] - closes[i-1]),
                                 abs(lows[i] - closes[i-1])))
        atr = np.mean([np.maximum(highs[j] - lows[j],
                                  np.maximum(abs(highs[j] - closes[j-1]),
                                            abs(lows[j] - closes[j-1])))
                      for j in range(i-lookback+1, i+1)])
        atr_pct = (atr / closes[i]) * 100
        
        if atr_pct < 2.0:  # Volatilidad baja-media
            score += 1
        
        # 2. Tendencia (MAs alineadas)
        ma20 = np.mean(closes[i-20:i])
        ma50 = np.mean(closes[i-50:i])
        
        if closes[i] > ma20 > ma50:
            score += 1
        
        # 3. RSI (no sobrecomprado)
        deltas = np.diff(closes[i-14:i+1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            if 40 < rsi < 70:  # RSI neutral-alcista
                score += 1
        
        # 4. Volumen creciente
        vol_ma = np.mean(volumes[i-20:i])
        if volumes[i] > vol_ma * 0.8:
            score += 1
        
        regime_score[i] = score
    
    return regime_score


def parity_long(arr, lookback, tolerance, live_trading=True, 
                ohlcv_data_minor=None, use_btc_regime=False, 
                regime_mode='advanced', min_regime_score=2):
    """
    Señal parity_long con filtro opcional de régimen de mercado BTC.
    
    Parámetros adicionales:
    - ohlcv_data_minor: diccionario con todos los símbolos (debe incluir 'BTCUSDT')
    - use_btc_regime: si True, filtra señales por régimen BTC
    - regime_mode: 'simple' o 'advanced'
    - min_regime_score: score mínimo para generar señal (solo en modo 'advanced')
    """
    opens  = arr['open']
    closes = arr['close']
    n      = len(closes)
    
    signals    = np.zeros(n, dtype=np.int8)
    body_sizes = np.abs(closes - opens)
    is_red     = closes < opens
    is_green   = closes > opens
    
    # Extraer y calcular régimen de BTC si está habilitado
    btc_regime = None
    if use_btc_regime and ohlcv_data_minor is not None:
        if 'BTCUSDT' in ohlcv_data_minor:
            btc_data = ohlcv_data_minor['BTCUSDT']
            
            if regime_mode == 'simple':
                btc_regime = calculate_btc_regime(btc_data)
            else:  # advanced
                btc_regime_scores = calculate_advanced_btc_regime(btc_data)
                btc_regime        = btc_regime_scores >= min_regime_score
        else:
            print("⚠️  BTCUSDT no encontrado en ohlcv_data_minor. Ignorando filtro de régimen.")
    
    for i in range(lookback, n):
        # Verificar régimen BTC si está habilitado
        if btc_regime is not None and not btc_regime[i]:
            continue
            
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
                                if i >= 50:
                                    ma50 = np.mean(closes[i-50:i])
                                    if closes[i] > ma50:
                                        signals[i] = 1
                                break
                    
                    if signals[i] == 1:
                        break
    
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0  
    
    return signals


def parity_short(arr, lookback, tolerance, live_trading=True,
                 ohlcv_data_minor=None, use_btc_regime=False,
                 regime_mode='advanced', min_regime_score=2):
    """
    Señal parity_short con filtro opcional de régimen de mercado BTC.
    
    IMPORTANTE: Para shorts, el régimen favorable es el OPUESTO:
    - Volatilidad alta
    - Tendencia bajista
    - Momentum negativo
    
    Parámetros adicionales:
    - ohlcv_data_minor: diccionario con todos los símbolos (debe incluir 'BTCUSDT')
    - use_btc_regime: si True, filtra señales por régimen BTC
    - regime_mode: 'simple' o 'advanced'
    - min_regime_score: score mínimo para generar señal
    """
    opens  = arr['open']
    closes = arr['close']
    n      = len(closes)
    
    signals    = np.zeros(n, dtype=np.int8)
    body_sizes = np.abs(closes - opens)
    is_red     = closes < opens
    is_green   = closes > opens
    
    # Extraer y calcular régimen de BTC si está habilitado
    btc_regime = None
    if use_btc_regime and ohlcv_data_minor is not None:
        if 'BTCUSDT' in ohlcv_data_minor:
            btc_data = ohlcv_data_minor['BTCUSDT']
            
            # Para shorts, invertimos la lógica del régimen
            if regime_mode == 'simple':
                btc_regime_long = calculate_btc_regime(btc_data)
                btc_regime = ~btc_regime_long  # Invertimos: queremos régimen desfavorable para longs
            else:  # advanced
                btc_regime_scores = calculate_advanced_btc_regime(btc_data)
                # Para shorts: score BAJO es favorable (mercado bajista/volátil)
                btc_regime = btc_regime_scores <= (4 - min_regime_score)
        else:
            print("⚠️  BTCUSDT no encontrado en ohlcv_data_minor. Ignorando filtro de régimen.")
    
    for i in range(lookback, n):
        # Verificar régimen BTC si está habilitado
        if btc_regime is not None and not btc_regime[i]:
            continue
            
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
                                if i >= 50:
                                    ma50 = np.mean(closes[i-50:i])
                                    if closes[i] < ma50:
                                        signals[i] = -1
                                break
                    
                    if signals[i] == -1:
                        break
    
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0  
    
    return signals