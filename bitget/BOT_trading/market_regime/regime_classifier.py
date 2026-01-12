"""
market_regime/regime_classifier.py

Core module for market regime classification and position sizing.
Integrates with BOT_trading's market_data infrastructure.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from market_data.api_client import _call_history_candles, to_dataframe_from_api
from market_data.data_utils import normalize_live_ohlcv, df_to_arrays_live
from market_regime.regime_metrics import calc_all_metrics
from config.settings import (
    BTC_SYMBOL,
    REGIME_LOOKBACK_BARS,
    REGIME_FAMILIES,
    REGIME_FAMILY_SIZING,
    REGIME_HURST_WINDOW,
    REGIME_ER_WINDOW,
    REGIME_ATR_WINDOW,
    REGIME_PE_WINDOW,
    REGIME_PE_ORDER,
    LOG_REGIME_DECISIONS
)

logger = logging.getLogger('BOT_trading.market_regime')


def fetch_btc_ohlcv(timeframe: str, limit: int = None) -> Optional[pd.DataFrame]:
    """
    Fetch BTC OHLCV data for regime calculation.
    
    Args:
        timeframe: Timeframe (e.g., '4H', '1H', '6Hutc')
        limit: Number of bars to fetch (default from settings)
    
    Returns:
        DataFrame with OHLCV data or None on error
    """
    if limit is None:
        limit = REGIME_LOOKBACK_BARS
    
    try:
        logger.debug(f"Fetching {limit} bars of {BTC_SYMBOL} {timeframe} for regime calculation")
        
        candles = _call_history_candles(
            symbol=BTC_SYMBOL,
            granularity=timeframe,
            limit=limit
        )
        
        if not candles:
            logger.warning(f"No candle data returned for {BTC_SYMBOL} {timeframe}")
            return None
        
        df = to_dataframe_from_api(candles)
        
        if df.empty:
            logger.warning(f"Empty DataFrame after parsing {BTC_SYMBOL} {timeframe}")
            return None
        
        logger.debug(f"Successfully fetched {len(df)} bars")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching BTC OHLCV for regime: {e}")
        return None


def calculate_regime_metrics(timeframe: str) -> Optional[Dict[str, float]]:
    """
    Calculate all regime metrics for current BTC state.
    
    Args:
        timeframe: Timeframe to analyze
    
    Returns:
        Dict with metrics or None on error
    """
    try:
        # Fetch BTC data
        df = fetch_btc_ohlcv(timeframe, limit=REGIME_LOOKBACK_BARS)
        
        if df is None or df.empty:
            logger.error("Cannot calculate regime metrics: no BTC data")
            return None
        
        # Normalize DataFrame
        df_norm = normalize_live_ohlcv(df)
        
        # Convert to arrays
        arrays = df_to_arrays_live(df_norm)
        
        # Prepare OHLC dict for metrics calculation
        ohlc = {
            'open': arrays['open'],
            'high': arrays['high'],
            'low': arrays['low'],
            'close': arrays['close']
        }
        
        # Calculate all metrics
        metrics = calc_all_metrics(
            ohlc=ohlc,
            hurst_window=REGIME_HURST_WINDOW,
            er_window=REGIME_ER_WINDOW,
            atr_window=REGIME_ATR_WINDOW,
            pe_window=REGIME_PE_WINDOW,
            pe_order=REGIME_PE_ORDER
        )
        
        # Log metrics
        logger.debug(f"Regime metrics calculated: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating regime metrics: {e}", exc_info=True)
        return None


def classify_regime(metrics: Dict[str, float]) -> str:
    """
    Classify regime into family based on metrics.
    
    Uses rule-based classification with first-match-wins logic.
    
    Args:
        metrics: Dict with calculated metrics
    
    Returns:
        Family name ('trending', 'ranging', 'volatile', or 'default')
    """
    # Check for NaN metrics
    if any(pd.isna(v) for v in metrics.values()):
        logger.warning(f"NaN values in metrics, using default family: {metrics}")
        return 'default'
    
    # First-match-wins classification
    for family_name, rules in REGIME_FAMILIES.items():
        # Empty rules = default/catch-all
        if not rules:
            continue
        
        # Check all rules for this family
        match = True
        for metric, (op, threshold) in rules.items():
            if metric not in metrics:
                match = False
                break
            
            value = metrics[metric]
            
            if pd.isna(value):
                match = False
                break
            
            if op == '>' and not (value > threshold):
                match = False
                break
            elif op == '<' and not (value < threshold):
                match = False
                break
        
        if match:
            return family_name
    
    # If no specific family matched, return default (ranging)
    for family_name, rules in REGIME_FAMILIES.items():
        if not rules:
            return family_name
    
    # Ultimate fallback
    return 'default'


def get_regime_metrics(timeframe: str) -> Optional[Dict[str, float]]:
    """
    Get current regime metrics for BTC.
    
    Args:
        timeframe: Timeframe to analyze
    
    Returns:
        Dict with metrics or None on error
    """
    return calculate_regime_metrics(timeframe)


def get_current_regime(timeframe: str) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    Get current regime family and metrics.
    
    Args:
        timeframe: Timeframe to analyze
    
    Returns:
        Tuple of (family_name, metrics_dict)
    """
    metrics = calculate_regime_metrics(timeframe)
    
    if metrics is None:
        logger.warning("Could not calculate metrics, returning default regime")
        return 'default', None
    
    family = classify_regime(metrics)
    
    if LOG_REGIME_DECISIONS:
        logger.info(
            f"Regime classified as '{family}' for {timeframe} | "
            f"Metrics: hurst={metrics.get('hurst', 0):.3f}, "
            f"er={metrics.get('efficiency_ratio', 0):.3f}, "
            f"atr%={metrics.get('atr_pct', 0):.2f}, "
            f"pe={metrics.get('permutation_entropy', 0):.3f}"
        )
    
    return family, metrics


def get_regime_multiplier(symbol: str, timeframe: str) -> float:
    """
    Get position sizing multiplier based on current regime.
    
    This is the main function called by orchestrator to adjust position size.
    
    Args:
        symbol: Trading symbol (currently unused, kept for future symbol-specific logic)
        timeframe: Timeframe being traded
    
    Returns:
        Multiplier to apply to base order_amount (0.5 to 1.5)
    """
    try:
        # Get current regime
        family, metrics = get_current_regime(timeframe)
        
        # Get multiplier for this family
        multiplier = REGIME_FAMILY_SIZING.get(family, 1.0)
        
        if LOG_REGIME_DECISIONS:
            logger.info(f"Regime sizing: {family} → {multiplier}x multiplier")
        
        return multiplier
        
    except Exception as e:
        logger.error(f"Error getting regime multiplier: {e}", exc_info=True)
        # On error, return 1.0 (no sizing adjustment)
        return 1.0


def get_regime_info(timeframe: str) -> Dict:
    """
    Get comprehensive regime information for API/dashboard.
    
    Args:
        timeframe: Timeframe to analyze
    
    Returns:
        Dict with regime info including family, metrics, multiplier
    """
    try:
        family, metrics = get_current_regime(timeframe)
        multiplier = REGIME_FAMILY_SIZING.get(family, 1.0)
        
        return {
            'timeframe': timeframe,
            'family': family,
            'multiplier': multiplier,
            'metrics': metrics or {},
            'thresholds': REGIME_FAMILIES.get(family, {}),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error getting regime info: {e}")
        return {
            'timeframe': timeframe,
            'family': 'error',
            'multiplier': 1.0,
            'metrics': {},
            'thresholds': {},
            'success': False,
            'error': str(e)
        }