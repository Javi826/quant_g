#BOT_trading/market_regime/regime_classifier.py
"""
Core module for market regime classification and position sizing.
Integrates with BOT_trading's market_data infrastructure.
"""
import logging
import pandas as pd
from typing import Dict, Optional, Tuple
from shared_batchs.regime.regime_module import calc_all_metrics, ER_WINDOW as REGIME_ER_WINDOW, ATR_WINDOW as REGIME_ATR_WINDOW
from market_data.data_utils import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from config.settings import REGIME_FAMILIES, REGIME_GENERAL, ACCOUNTS

logger = logging.getLogger('BOT_trading.market_regime.regime_classifier')

REGIME_REFERENCE_SYMBOL  = None
REGIME0_MA_PERIOD        = None
GLOBAL_SYSTEM_REGIME_TH1 = None
GLOBAL_SYSTEM_REGIME_TH2 = None
_cached_direction: str = 'uptrend'

def update_direction_cache(direction: str) -> None:
    global _cached_direction
    _cached_direction = direction

def get_cached_direction() -> str:
    return _cached_direction

def configure_regime(account_number: str) -> None:
    global REGIME0_MA_PERIOD, GLOBAL_SYSTEM_REGIME_TH1, GLOBAL_SYSTEM_REGIME_TH2, REGIME_REFERENCE_SYMBOL
    config = ACCOUNTS.get(account_number, {})
    REGIME_REFERENCE_SYMBOL  = config.get('regime_reference_symbol')
    REGIME0_MA_PERIOD        = config.get('regime01_ma_period', REGIME0_MA_PERIOD)
    GLOBAL_SYSTEM_REGIME_TH1 = config.get('regime01_short_th',  GLOBAL_SYSTEM_REGIME_TH1)
    GLOBAL_SYSTEM_REGIME_TH2 = config.get('regime01_long_th',   GLOBAL_SYSTEM_REGIME_TH2)
    
def fetch_ref_ohlcv(timeframe: str) -> Optional[pd.DataFrame]:
    """
    Fetch reference symbol OHLCV data for regime calculation.

    Uses the same data fetching logic as trading strategies to ensure consistency.

    Args:
        timeframe: Timeframe (e.g., '4H', '1H', '6Hutc')

    Returns:
        DataFrame with OHLCV data or None on error
    """
    try:
        logger.debug(f"Fetching {REGIME_REFERENCE_SYMBOL} {timeframe} data for regime calculation")

        ohlcv_data = fetch_ohlcv_data([REGIME_REFERENCE_SYMBOL], timeframe)
        df         = ohlcv_data.get(REGIME_REFERENCE_SYMBOL)

        if df is None or df.empty:
            logger.warning(f"No data returned for {REGIME_REFERENCE_SYMBOL} {timeframe}")
            return None

        logger.debug(f"Successfully fetched {len(df)} bars")
        return df

    except Exception as e:
        logger.error(f"Error fetching {REGIME_REFERENCE_SYMBOL} OHLCV for regime: {e}")
        return None


def calculate_regime_metrics(timeframe: str) -> Optional[Dict[str, float]]:
    """
    Calculate all regime metrics for current REF state.

    Args:
        timeframe: Timeframe to analyze

    Returns:
        Dict with metrics or None on error
    """
    try:
        df = fetch_ref_ohlcv(timeframe)

        if df is None or df.empty:
            logger.error(f"Cannot calculate regime metrics: no {REGIME_REFERENCE_SYMBOL} data")
            return None

        df_norm = normalize_live_ohlcv(df)
        arrays  = df_to_arrays_live(df_norm)

        ohlc = {
            'open':  arrays['open'],
            'high':  arrays['high'],
            'low':   arrays['low'],
            'close': arrays['close'],
        }

        metrics = calc_all_metrics(
                    ohlc       = ohlc,
                    er_window  = REGIME_ER_WINDOW,
                    atr_window = REGIME_ATR_WINDOW,
                )

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
    if any(pd.isna(v) for v in metrics.values()):
        logger.warning(f"NaN values in metrics, using default family: {metrics}")
        return 'default'

    for family_name, rules in REGIME_FAMILIES.items():
        if not rules:
            continue

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

    for family_name, rules in REGIME_FAMILIES.items():
        if not rules:
            return family_name

    return 'default'


def get_regime_metrics(timeframe: str) -> Optional[Dict[str, float]]:
    """Get current regime metrics for reference symbol."""
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
            f"Metrics: er={metrics.get('efficiency_ratio', 0):.3f}, "
            f"atr%={metrics.get('atr_pct', 0):.2f}"
        )

    return family, metrics

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
        multiplier      = REGIME_GENERAL.get(family, 1.0)

        return {
            'timeframe':  timeframe,
            'family':     family,
            'multiplier': multiplier,
            'metrics':    metrics or {},
            'thresholds': REGIME_FAMILIES.get(family, {}),
            'success':    True,
        }

    except Exception as e:
        logger.error(f"Error getting regime info: {e}")
        return {
            'timeframe':  timeframe,
            'family':     'error',
            'multiplier': 1.0,
            'metrics':    {},
            'thresholds': {},
            'success':    False,
            'error':      str(e),
        }


def get_ref_1d_direction() -> str:
    """
    Get current BTC macro direction based on REF 1D price vs MA.

    Uses REGIME0_MA_PERIOD and thresholds from settings (shared_config).

    Returns:
        'uptrend' | 'dwtrend'
    """
    try:
        df = fetch_ref_ohlcv('1Dutc')

        if df is None or df.empty or len(df) < REGIME0_MA_PERIOD:
            logger.warning(f"[REGIME01] Insufficient {REGIME_REFERENCE_SYMBOL} 1D data, defaulting to uptrend")
            return 'uptrend'

        ref_close = float(pd.to_numeric(df['close'].iloc[-1], errors='coerce'))
        ma        = float(pd.to_numeric(df['close'], errors='coerce').tail(REGIME0_MA_PERIOD).mean())

        if ref_close > ma * GLOBAL_SYSTEM_REGIME_TH2:
            direction = 'uptrend'
        elif ref_close < ma * GLOBAL_SYSTEM_REGIME_TH1:
            direction = 'dwtrend'
        else:
            direction = 'uptrend'  # neutral → treat as uptrend

        logger.info(
            f"[REGIME01] {REGIME_REFERENCE_SYMBOL} direction: {direction.upper()} | "
            f"{REGIME_REFERENCE_SYMBOL}=${ref_close:.2f} | MA{REGIME0_MA_PERIOD}=${ma:.2f}"
        )

        return direction

    except Exception as e:
        logger.error(f"[REGIME01] Error calculating {REGIME_REFERENCE_SYMBOL} direction: {e}")
        return 'uptrend'