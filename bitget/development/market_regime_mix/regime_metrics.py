"""
market_regime/regime_metrics.py

Funciones puras para calcular métricas de régimen de mercado.
Cada función recibe arrays numpy y devuelve un valor float.

Métricas:
- Hurst Exponent: persistencia de tendencia
- Efficiency Ratio: direccionalidad vs ruido
- ATR%: volatilidad normalizada
- Permutation Entropy: complejidad/aleatoriedad
"""

import numpy as np
from typing import Union


def calc_hurst(closes: np.ndarray, window: int = 100) -> float:
    """
    Calcula el exponente de Hurst usando el método R/S (Rescaled Range).
    
    Interpretación:
        H < 0.5  → Mean-reverting (anti-persistente)
        H = 0.5  → Random walk
        H > 0.5  → Trending (persistente)
    
    Args:
        closes: Array de precios de cierre
        window: Número de observaciones a usar (mínimo ~100 para estabilidad)
    
    Returns:
        float: Exponente de Hurst [0, 1]
    """
    if len(closes) < window:
        return np.nan
    
    ts = closes[-window:]
    
    # Calcular retornos
    returns = np.diff(ts) / ts[:-1]
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 20:
        return np.nan
    
    # Método R/S simplificado
    n = len(returns)
    
    # Dividir en subseries de diferentes tamaños
    max_k = min(n // 2, 50)
    if max_k < 4:
        return np.nan
    
    rs_values = []
    ns_values = []
    
    for size in range(10, max_k + 1, 2):
        num_subseries = n // size
        if num_subseries < 1:
            continue
            
        rs_list = []
        for i in range(num_subseries):
            subseries = returns[i * size:(i + 1) * size]
            
            mean_sub = np.mean(subseries)
            centered = subseries - mean_sub
            cumsum = np.cumsum(centered)
            
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(subseries, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns_values.append(size)
    
    if len(rs_values) < 3:
        return np.nan
    
    # Regresión log-log para obtener H
    log_n = np.log(ns_values)
    log_rs = np.log(rs_values)
    
    # H = pendiente de la regresión
    slope, _ = np.polyfit(log_n, log_rs, 1)
    
    # Clamp a [0, 1]
    return float(np.clip(slope, 0.0, 1.0))


def calc_efficiency_ratio(closes: np.ndarray, window: int = 14) -> float:
    """
    Calcula el Efficiency Ratio (ER) de Kaufman.
    
    ER = |Cambio neto| / Suma de cambios absolutos
    
    Interpretación:
        ER → 1: Tendencia limpia, poco ruido
        ER → 0: Mercado lateral/ruidoso
    
    Args:
        closes: Array de precios de cierre
        window: Período de lookback
    
    Returns:
        float: Efficiency Ratio [0, 1]
    """
    if len(closes) < window + 1:
        return np.nan
    
    ts = closes[-(window + 1):]
    
    # Cambio neto (direccional)
    net_change = abs(ts[-1] - ts[0])
    
    # Suma de cambios absolutos (volatilidad/ruido)
    abs_changes = np.abs(np.diff(ts))
    total_change = np.sum(abs_changes)
    
    if total_change == 0:
        return np.nan
    
    er = net_change / total_change
    
    return float(np.clip(er, 0.0, 1.0))


def calc_atr_pct(ohlc: dict, window: int = 14) -> float:
    """
    Calcula el ATR como porcentaje del precio (ATR%).
    
    ATR% = (ATR / Close) * 100
    
    Interpretación:
        Alto ATR% → Alta volatilidad
        Bajo ATR% → Baja volatilidad
    
    Args:
        ohlc: Dict con keys 'high', 'low', 'close' (arrays numpy)
        window: Período del ATR
    
    Returns:
        float: ATR como porcentaje del precio
    """
    high = ohlc['high']
    low = ohlc['low']
    close = ohlc['close']
    
    if len(close) < window + 1:
        return np.nan
    
    # True Range
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # ATR = SMA del True Range
    atr = np.mean(tr[-window:])
    
    # ATR como porcentaje del precio actual
    current_price = close[-1]
    if current_price == 0:
        return np.nan
    
    atr_pct = (atr / current_price) * 100
    
    return float(atr_pct)


def calc_permutation_entropy(closes: np.ndarray, window: int = 50, order: int = 3, delay: int = 1) -> float:
    """
    Calcula la Permutation Entropy normalizada.
    
    Mide la complejidad/aleatoriedad de la serie temporal basándose
    en los patrones ordinales de los datos.
    
    Interpretación:
        PE → 0: Serie muy predecible/estructurada
        PE → 1: Serie muy aleatoria/caótica
    
    Args:
        closes: Array de precios de cierre
        window: Número de observaciones a usar
        order: Orden de la permutación (típicamente 3-7)
        delay: Delay entre elementos (típicamente 1)
    
    Returns:
        float: Permutation Entropy normalizada [0, 1]
    """
    from math import factorial
    
    if len(closes) < window:
        return np.nan
    
    ts = np.asarray(closes[-window:], dtype=np.float64)
    n = len(ts)
    
    # Número de vectores embebidos
    n_vectors = n - (order - 1) * delay
    if n_vectors < 10:
        return np.nan
    
    # Contar patrones de permutación
    max_patterns = factorial(order)
    pattern_counts = {}
    
    for i in range(n_vectors):
        # Extraer vector de orden elementos con delay
        indices = [i + j * delay for j in range(order)]
        pattern_values = np.array([ts[idx] for idx in indices])
        
        # Obtener el patrón ordinal (ranking)
        pattern = tuple(np.argsort(pattern_values).tolist())
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    total_patterns = sum(pattern_counts.values())
    if total_patterns == 0:
        return np.nan
    
    # Calcular entropía de Shannon
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total_patterns
        if p > 0:
            entropy -= p * np.log2(p)
    
    # Normalizar por entropía máxima posible
    max_entropy = np.log2(max_patterns)
    if max_entropy == 0:
        return np.nan
    
    normalized_entropy = entropy / max_entropy
    
    return float(np.clip(normalized_entropy, 0.0, 1.0))


def calc_all_metrics(ohlc: dict, 
                     hurst_window: int = 100,
                     er_window: int = 14,
                     atr_window: int = 14,
                     pe_window: int = 50,
                     pe_order: int = 3) -> dict:
    """
    Calcula todas las métricas de régimen en un solo llamado.
    
    Args:
        ohlc: Dict con keys 'open', 'high', 'low', 'close' (arrays numpy)
        hurst_window: Ventana para Hurst
        er_window: Ventana para Efficiency Ratio
        atr_window: Ventana para ATR
        pe_window: Ventana para Permutation Entropy
        pe_order: Orden para Permutation Entropy
    
    Returns:
        dict: {
            'hurst': float,
            'efficiency_ratio': float,
            'atr_pct': float,
            'permutation_entropy': float
        }
    """
    closes = ohlc['close']
    
    return {
        'hurst': calc_hurst(closes, hurst_window),
        'efficiency_ratio': calc_efficiency_ratio(closes, er_window),
        'atr_pct': calc_atr_pct(ohlc, atr_window),
        'permutation_entropy': calc_permutation_entropy(closes, pe_window, pe_order)
    }


# =============================================================================
# FUNCIONES AUXILIARES PARA CLASIFICACIÓN DE RÉGIMEN
# =============================================================================

def classify_regime(metrics: dict) -> str:
    """
    Clasificación simple del régimen basada en las métricas.
    
    Returns:
        str: 'trending', 'mean_reverting', 'volatile', 'calm', 'chaotic', 'unknown'
    """
    h = metrics.get('hurst', np.nan)
    er = metrics.get('efficiency_ratio', np.nan)
    atr = metrics.get('atr_pct', np.nan)
    pe = metrics.get('permutation_entropy', np.nan)
    
    # Si faltan métricas, no podemos clasificar
    if any(np.isnan([h, er, atr, pe])):
        return 'unknown'
    
    # Reglas de clasificación (ajustables)
    if h > 0.55 and er > 0.5:
        return 'trending'
    elif h < 0.45 and er < 0.3:
        return 'mean_reverting'
    elif atr > 5.0 and pe > 0.8:
        return 'chaotic'
    elif atr > 4.0:
        return 'volatile'
    elif atr < 2.0 and pe < 0.7:
        return 'calm'
    else:
        return 'neutral'


if __name__ == "__main__":
    # Test básico
    np.random.seed(42)
    
    # Simular datos OHLC
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    open_ = close + np.random.randn(n) * 0.1
    
    ohlc = {
        'open': open_,
        'high': high,
        'low': low,
        'close': close
    }
    
    print("=== Test de Métricas de Régimen ===\n")
    
    metrics = calc_all_metrics(ohlc)
    
    print(f"Hurst Exponent:      {metrics['hurst']:.4f}")
    print(f"Efficiency Ratio:    {metrics['efficiency_ratio']:.4f}")
    print(f"ATR%:                {metrics['atr_pct']:.4f}")
    print(f"Permutation Entropy: {metrics['permutation_entropy']:.4f}")
    print(f"\nRégimen detectado:   {classify_regime(metrics)}")
