#shared/shared_trading_batch/regime_metrics.py


import numpy as np
from typing import Dict
import warnings

# Suppress warnings from external libraries
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger('BOT_trading.market_regime.regime_metrics')


def calc_hurst(close: np.ndarray, window: int = 100) -> float:
    return 0.8

def calc_permutation_entropy(close: np.ndarray, window: int = 50, order: int = 3) -> float:
    return 0.8

def calc_efficiency_ratio(close: np.ndarray, window: int = 14) -> float:

    if len(close) < window + 1:
        return np.nan
    
    series = close[-(window + 1):]
    
    # Net change (direction)
    net_change = abs(series[-1] - series[0])
    
    # Sum of absolute changes (volatility)
    abs_changes  = np.abs(np.diff(series))
    total_change = np.sum(abs_changes)
    
    if total_change == 0:
        return 0.0  # No movement = zero efficiency
    
    er = net_change / total_change
    
    # Ensure valid range
    return float(np.clip(er, 0.0, 1.0))

def calc_atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float:
    if len(close) < window + 1 or len(high) < window or len(low) < window:
        return np.nan

    # True Range — matches ta.AverageTrueRange exactly
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))

    if len(tr) < window:
        return np.nan

    # Wilder smoothing with SMA seed — matches ta library initialization
    atr = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr = (atr * (window - 1) + tr[i]) / window

    current_price = close[-1]
    if current_price == 0 or np.isnan(current_price) or np.isnan(atr):
        return np.nan

    atr_pct = (atr / current_price) * 100
    return float(atr_pct) if 0 <= atr_pct <= 100 else np.nan

def calc_all_metrics(
    ohlc: Dict[str, np.ndarray],
    hurst_window: int = 100,
    er_window: int = 14,
    atr_window: int = 14,
    pe_window: int = 50,
    pe_order: int = 3
) -> Dict[str, float]:

    close = ohlc['close']
    high  = ohlc['high']
    low   = ohlc['low']
    
    return {
        'hurst': calc_hurst(close, hurst_window),
        'efficiency_ratio': calc_efficiency_ratio(close, er_window),
        'atr_pct': calc_atr_pct(high, low, close, atr_window),
        'permutation_entropy': calc_permutation_entropy(close, pe_window, pe_order)
    }
