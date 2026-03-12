"""
market_regime/regime_classifier.py

Core module for market regime classification and position sizing.
Integrates with BOT_trading's market_data infrastructure.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from market_data.data_utils import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from market_regime.regime_metrics import calc_all_metrics
from config.settings import REGIME_REFERENCE_SYMBOL, REGIME_FAMILIES, REGIME_GENERAL
from config.settings import REGIME_HURST_WINDOW, REGIME_ER_WINDOW, REGIME_ATR_WINDOW
from config.settings import REGIME_PE_WINDOW, REGIME_PE_ORDER
from config.settings import GLOBAL_SYSTEM_REGIME_TH1, GLOBAL_SYSTEM_REGIME_TH2

logger = logging.getLogger('BOT_trading.market_regime.regime_classifier')


def fetch_btc_ohlcv(timeframe: str, limit: int = None) -> Optional[pd.DataFrame]:
    """
    Fetch BTC OHLCV data for regime calculation.
    
    Uses the same data fetching logic as trading strategies to ensure consistency.
    
    Args:
        timeframe: Timeframe (e.g., '4H', '1H', '6Hutc')
        limit: Number of bars to fetch (not used, kept for compatibility)
    
    Returns:
        DataFrame with OHLCV data or None on error
    """
    try:
        logger.debug(f"Fetching {REGIME_REFERENCE_SYMBOL} {timeframe} data for regime calculation")
        
        # Use the same fetch function as strategies
        ohlcv_data = fetch_ohlcv_data([REGIME_REFERENCE_SYMBOL], timeframe)
        
        df = ohlcv_data.get(REGIME_REFERENCE_SYMBOL)
        
        if df is None or df.empty:
            logger.warning(f"No data returned for {REGIME_REFERENCE_SYMBOL} {timeframe}")
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
        # Fetch BTC data using same method as strategies
        df = fetch_btc_ohlcv(timeframe)
        
        if df is None or df.empty:
            logger.error("Cannot calculate regime metrics: no BTC data")
            return None
        
        # Normalize DataFrame (same as strategies)
        df_norm = normalize_live_ohlcv(df)
        
        # Convert to arrays (same as strategies)
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
    
    logger.debug(
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
        multiplier = REGIME_GENERAL.get(family, 1.0)
        
        # Logging is handled by orchestrator
        
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
        Dict with regime info including family, metrics, multiplier, BTC price/MA50/trend
    """
    try:
        family, metrics = get_current_regime(timeframe)
        multiplier = REGIME_GENERAL.get(family, 1.0)
        
        # NEW: Calculate BTC price, MA50, and trend
        btc_price = None
        btc_ma50 = None
        btc_trend = 'unknown'
        
        try:
            # Fetch BTC data
            df = fetch_btc_ohlcv(timeframe)
            
            if df is not None and not df.empty and len(df) >= 50:
                # Get current price (last close)
                btc_price = float(pd.to_numeric(df['close'].iloc[-1], errors='coerce'))
                
                # Calculate MA50
                btc_ma50 = float(pd.to_numeric(df['close'], errors='coerce').tail(50).mean())
                
                # Determine trend
                if btc_price > btc_ma50:
                    btc_trend = 'uptrend'
                else:
                    btc_trend = 'downtrend'
                
                logger.debug(
                    f"BTC trend: {btc_trend} | Price: ${btc_price:.2f} | "
                    f"MA50: ${btc_ma50:.2f}"
                )
            else:
                logger.warning("Insufficient BTC data for MA50 calculation")
                
        except Exception as e:
            logger.error(f"Error calculating BTC price/MA50: {e}")
        
        return {
            'timeframe': timeframe,
            'family': family,
            'multiplier': multiplier,
            'metrics': metrics or {},
            'thresholds': REGIME_FAMILIES.get(family, {}),
            'btc_price': btc_price,
            'btc_ma50': btc_ma50,
            'btc_trend': btc_trend,
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
            'btc_price': None,
            'btc_ma50': None,
            'btc_trend': 'unknown',
            'success': False,
            'error': str(e)
        }
def get_current_direction(timeframe: str) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Get current BTC direction (uptrend/dwtrend) based on price vs MA50.
    
    Args:
        timeframe: Timeframe to analyze
    
    Returns:
        Tuple of (direction, btc_price, btc_ma50)
        - direction: 'uptrend' or 'dwtrend'
        - btc_price: Current BTC price or None
        - btc_ma50: 50-period MA or None
    """
    try:
        df = fetch_btc_ohlcv(timeframe)
        
        if df is None or df.empty or len(df) < 50:
            logger.warning("Insufficient data for direction calculation, defaulting to uptrend")
            return 'uptrend', None, None
        
        # Get current price (last close)
        btc_price = float(pd.to_numeric(df['close'].iloc[-1], errors='coerce'))
        
        # Calculate MA50
        btc_ma50 = float(pd.to_numeric(df['close'], errors='coerce').tail(50).mean())
        
        # Determine direction
        if btc_price > btc_ma50:
            direction = 'uptrend'
        else:
            direction = 'dwtrend'
        
        logger.debug(
            f"BTC direction: {direction} | "
            f"Price: ${btc_price:.2f} | MA50: ${btc_ma50:.2f}"
        )
        
        return direction, btc_price, btc_ma50
        
    except Exception as e:
        logger.error(f"Error calculating direction: {e}")
        return 'uptrend', None, None  # Fallback
    
def get_btc_1d_filter(direction: str) -> bool:
    """
    Check if trade allowed based on BTC 1D price vs MA5.
    
    LONG: BTC > MA5 * GLOBAL_SYSTEM_REGIME_TH2
    SHORT: BTC < MA5 * GLOBAL_SYSTEM_REGIME_TH1
    """
    try:
        df = fetch_btc_ohlcv('1Dutc')
        
        if df is None or df.empty or len(df) < 5:  # ← Restaurado df.empty
            logger.warning("[REGIME0] Insufficient BTC 1D data, allowing trade")
            return True
        
        # Safe conversion (como el original)
        btc_close = float(pd.to_numeric(df['close'].iloc[-1], errors='coerce'))
        ma5 = float(pd.to_numeric(df['close'], errors='coerce').tail(5).mean())
        
        if direction == 'long':
            allowed = btc_close > ma5 * GLOBAL_SYSTEM_REGIME_TH2
            threshold = GLOBAL_SYSTEM_REGIME_TH2
        elif direction == 'short':
            allowed = btc_close < ma5 * GLOBAL_SYSTEM_REGIME_TH1
            threshold = GLOBAL_SYSTEM_REGIME_TH1
        else:
            allowed = True
            threshold = 1.00
        
        # Log decision
        status = 'ALLOW' if allowed else 'BLOCK'
        logger.info(
            f"[REGIME0] {direction.upper()}S: "
            f"BTC=${btc_close:.2f} vs MA5*{threshold:.2f}=${ma5*threshold:.2f} → {status}"
        )
        
        return allowed
        
    except Exception as e:
        logger.error(f"[REGIME0]Error-in filter: {e}")
        return False  # On error, not allow trade