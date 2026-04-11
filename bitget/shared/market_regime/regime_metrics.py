"""
BOT_trading/market_regime/regime_metrics.py - LIBRARY-ONLY VERSION

Calculates market regime metrics using ONLY proven libraries.
NO fallback implementations - if library is missing, raises ImportError.

REQUIRED Dependencies:
    pip install nolds ta neurokit2 pandas

Metrics:
- Hurst Exponent: trend persistence (>0.5 = trending, <0.5 = mean-reverting)
- Efficiency Ratio: directional movement quality (1 = perfect trend, 0 = choppy)
- ATR %: volatility as percentage of price
- Permutation Entropy: price predictability (0 = deterministic, 1 = random)
"""

import numpy as np
import pandas as pd
from typing import Dict
import warnings

# Suppress warnings from external libraries
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger('BOT_trading.market_regime.regime_metrics')
import nolds
from ta.volatility import AverageTrueRange
import neurokit2 as nk


def calc_hurst(close: np.ndarray, window: int = 100) -> float:
    """
    Calculates Hurst Exponent using nolds library.
    
    Args:
        close: Array of closing prices
        window: Lookback window
    
    Returns:
        Hurst exponent (0-1). >0.5 = trending, <0.5 = mean-reverting
    """
    if len(close) < window:
        return np.nan
    
    series = close[-window:]
    
    try:
        # Use nolds library (R/S method)
        H = nolds.hurst_rs(series, nvals=None, fit='poly')
        
        # Clip to valid range [0, 1]
        return float(np.clip(H, 0.0, 1.0))
    
    except Exception as e:
        # If calculation fails (e.g., insufficient data variation)
        return np.nan


def calc_efficiency_ratio(close: np.ndarray, window: int = 14) -> float:
    """
    Calculates Kaufman's Efficiency Ratio.
    
    This is a simple metric - no library implementation needed.
    Formula: ER = |price_change| / sum(|price_changes|)
    
    Args:
        close: Array of closing prices
        window: Lookback window
    
    Returns:
        Efficiency ratio (0-1). 1 = perfect trend, 0 = choppy
    """
    if len(close) < window + 1:
        return np.nan
    
    series = close[-(window + 1):]
    
    # Net change (direction)
    net_change = abs(series[-1] - series[0])
    
    # Sum of absolute changes (volatility)
    abs_changes = np.abs(np.diff(series))
    total_change = np.sum(abs_changes)
    
    if total_change == 0:
        return 0.0  # No movement = zero efficiency
    
    er = net_change / total_change
    
    # Ensure valid range
    return float(np.clip(er, 0.0, 1.0))


def calc_atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float:
    """
    Calculates Average True Range as percentage of price using ta library.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        window: Lookback window
    
    Returns:
        ATR as percentage of current price
    """
    if len(close) < window + 1 or len(high) < window or len(low) < window:
        return np.nan
    
    try:
        # Convert to pandas Series (required by ta library)
        high_series = pd.Series(high, dtype=float)
        low_series = pd.Series(low, dtype=float)
        close_series = pd.Series(close, dtype=float)
        
        # Calculate ATR using ta library
        atr_indicator = AverageTrueRange(
            high=high_series,
            low=low_series,
            close=close_series,
            window=window,
            fillna=False
        )
        
        atr_values = atr_indicator.average_true_range()
        atr        = atr_values.iloc[-1]
        
        # Convert to percentage of current price
        current_price = close[-1]
        
        if current_price == 0 or np.isnan(current_price) or np.isnan(atr):
            return np.nan
        
        atr_pct = (atr / current_price) * 100
        
        # Sanity check (ATR% should be reasonable)
        if atr_pct < 0 or atr_pct > 100:
            return np.nan
        
        return float(atr_pct)
    
    except Exception as e:
        return np.nan


def calc_permutation_entropy(close: np.ndarray, window: int = 50, order: int = 3) -> float:
    """
    Calculates Permutation Entropy using neurokit2 library.
    
    Args:
        close: Array of closing prices
        window: Lookback window
        order: Embedding dimension (pattern length)
    
    Returns:
        Normalized entropy (0-1). 0 = deterministic, 1 = random
    """
    if len(close) < window:
        return np.nan
    
    series = close[-window:]
    
    # Check for constant/near-constant values
    if np.std(series) < 1e-8:
        return 0.0
    
    try:
        # Calculate permutation entropy using neurokit2
        pe_result = nk.entropy_permutation(series, dimension=order, delay=1)
        
        # Extract value from tuple (neurokit2 returns tuple)
        if isinstance(pe_result, tuple):
            pe = pe_result[0]
        else:
            pe = pe_result
        
        # Check if result is valid
        if pe is None or np.isnan(pe) or np.isinf(pe):
            return np.nan
        
        # Normalize to [0, 1] range
        from math import factorial
        max_entropy = np.log2(factorial(order))
        
        if max_entropy == 0:
            return np.nan
        
        pe_normalized = pe / max_entropy
        
        # Ensure valid range
        return float(np.clip(pe_normalized, 0.0, 1.0))
    
    except Exception as e:
        return np.nan


def calc_all_metrics(
    ohlc: Dict[str, np.ndarray],
    hurst_window: int = 100,
    er_window: int = 14,
    atr_window: int = 14,
    pe_window: int = 50,
    pe_order: int = 3
) -> Dict[str, float]:
    """
    Calculates all regime metrics from OHLC data using proven libraries.
    
    Args:
        ohlc: Dict with 'open', 'high', 'low', 'close' arrays
        hurst_window: Window for Hurst exponent
        er_window: Window for Efficiency Ratio
        atr_window: Window for ATR
        pe_window: Window for Permutation Entropy
        pe_order: Order for Permutation Entropy
    
    Returns:
        Dict with all metrics
    """
    close = ohlc['close']
    high = ohlc['high']
    low = ohlc['low']
    
    return {
        'hurst': calc_hurst(close, hurst_window),
        'efficiency_ratio': calc_efficiency_ratio(close, er_window),
        'atr_pct': calc_atr_pct(high, low, close, atr_window),
        'permutation_entropy': calc_permutation_entropy(close, pe_window, pe_order)
    }
