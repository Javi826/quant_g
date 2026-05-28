#BOT_trading/market_regime/regime_classifier.py
"""
Core module for market regime classification.
Uses GE indicator config (config_trading_batch.py) mirroring batch logic.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from shared_trading_batch.config_trading_batch import INDICATORS, COMBINE_MODE, ANALYSIS_MODE, REGIME_TIMEFRAME_MODE
from shared_trading_batch_regime.regime_metrics import _CALC_FN
from market_data.data_utils import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from config.settings import ACCOUNTS

logger = logging.getLogger('BOT_trading.market_regime.regime_classifier')

REGIME_REFERENCE_SYMBOL = None


def configure_regime(account_number: str) -> None:
    global REGIME_REFERENCE_SYMBOL
    config                  = ACCOUNTS.get(account_number, {})
    REGIME_REFERENCE_SYMBOL = config.get('regime_reference_symbol')


# =============================================================================
# ACTIVE CONFIG
# =============================================================================

def _active_windows() -> Dict[str, int]:
    return {k: v["windows"][0] for k, v in INDICATORS.items() if v.get("enabled")}

def _active_thresholds() -> Dict[str, float]:
    return {k: v["thresholds"][0] for k, v in INDICATORS.items() if v.get("enabled")}


# =============================================================================
# CLASSIFICATION
# =============================================================================

def _is_trending(metrics: Dict[str, Optional[float]]) -> bool:
    thresholds = _active_thresholds()
    signals    = []
    for key, val in metrics.items():
        if val is None or np.isnan(val):
            continue
        signals.append(val >= thresholds[key])
    if not signals:
        return False
    return all(signals) if COMBINE_MODE == "AND" else any(signals)


def _classify(metrics: Dict[str, Optional[float]]) -> str:
    if not metrics or all(v is None for v in metrics.values()):
        return 'neutral'
    return 'trending' if _is_trending(metrics) else 'ranging'


# =============================================================================
# METRICS FROM ARRAY (no fetch — array already computed in strategy_registry)
# =============================================================================

def _calc_metrics_from_arr(arr: dict) -> Dict[str, Optional[float]]:
    """Calculate regime metrics from an already-fetched OHLCV array."""
    windows = _active_windows()
    high    = arr['high']
    low     = arr['low']
    close   = arr['close']
    metrics = {}
    for key, w in windows.items():
        try:
            val          = _CALC_FN[key](high, low, close, w)
            metrics[key] = float(val) if not np.isnan(val) else None
        except Exception as e:
            logger.warning(f"[REGIME_GE] Error computing {key}: {e}")
            metrics[key] = None
    return metrics


# =============================================================================
# METRICS FROM FETCH (for reference symbol BTC or DAILY timeframe)
# =============================================================================

def _fetch_and_calc_metrics(symbol: str, timeframe: str) -> Dict[str, Optional[float]]:
    """Fetch OHLCV and calculate regime metrics for a given symbol/timeframe."""
    try:
        ohlcv_data = fetch_ohlcv_data([symbol], timeframe)
        df         = ohlcv_data.get(symbol)
        if df is None or df.empty:
            logger.warning(f"[REGIME_GE] No data for {symbol} {timeframe}")
            return {}
        df_norm = normalize_live_ohlcv(df)
        arr     = df_to_arrays_live(df_norm)
        return _calc_metrics_from_arr(arr)
    except Exception as e:
        logger.error(f"[REGIME_GE] Error fetching {symbol} {timeframe}: {e}")
        return {}


# =============================================================================
# PUBLIC API
# =============================================================================

def get_symbol_regime(
    symbol:    str,
    timeframe: str,
    arr:       Optional[dict] = None,
) -> str:
    """
    Get regime classification for a symbol.

    Respects ANALYSIS_MODE and REGIME_TIMEFRAME_MODE from config_trading_batch.

    ANALYSIS_MODE:
        'SYMBOL' → use the symbol's own data
        'BTC'    → use REGIME_REFERENCE_SYMBOL data

    REGIME_TIMEFRAME_MODE:
        'DAILY'    → always use '1Dutc' timeframe (fetch required)
        'STRATEGY' → use strategy timeframe (reuse arr if SYMBOL mode)

    Args:
        symbol   : Trading symbol (e.g. 'BTCUSDT')
        timeframe: Strategy timeframe (e.g. '4H')
        arr      : Pre-fetched OHLCV arrays (reused when possible to avoid refetch)

    Returns:
        'trending' | 'ranging' | 'neutral'
    """
    ref_symbol   = REGIME_REFERENCE_SYMBOL if ANALYSIS_MODE == "BTC" else symbol
    ref_timeframe = "1Dutc" if REGIME_TIMEFRAME_MODE == "DAILY" else timeframe

    logger.info(
        f"[REGIME_GE] {symbol} | ref={ref_symbol} | tf={ref_timeframe} | "
        f"mode={ANALYSIS_MODE} | tf_mode={REGIME_TIMEFRAME_MODE}"
    )

    # Reuse arr only when symbol and timeframe match exactly
    can_reuse = (
        arr is not None
        and ANALYSIS_MODE == "SYMBOL"
        and REGIME_TIMEFRAME_MODE == "STRATEGY"
    )

    if can_reuse:
        metrics = _calc_metrics_from_arr(arr)
    else:
        metrics = _fetch_and_calc_metrics(ref_symbol, ref_timeframe)

    if not metrics:
        logger.warning(f"[REGIME_GE] No metrics for {ref_symbol} {ref_timeframe} — defaulting to neutral")
        return 'neutral'

    regime = _classify(metrics)

    logger.info(
        f"[REGIME_GE] {symbol} → regime={regime.upper()} | "
        + " | ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=None"
            for k, v in metrics.items()
        )
    )

    return regime

def get_regime_info_front(
    timeframe: str,
    symbol=None,
) -> Dict:
    """
    Fetch regime metrics for dashboard consumption.

    Args:
        timeframe : Timeframe to analyse (e.g. '4H', '1Dutc').
        symbol    : None → use REGIME_REFERENCE_SYMBOL (header card).
                    list → compute metrics for each symbol (symbols grid).

    Returns:
        Single-symbol mode  → {'success', 'timeframe', 'family', 'metrics'}
        Multi-symbol mode   → {'success', 'timeframe', 'symbols': {symbol: {'family', 'metrics'}}}
    """
    # ── Single-symbol mode (header) ──────────────────────────────────────────
    if symbol is None:
        target = REGIME_REFERENCE_SYMBOL
        try:
            metrics = _fetch_and_calc_metrics(target, timeframe)
            family  = _classify(metrics) if metrics else 'neutral'
            return {
                'success':   True,
                'timeframe': timeframe,
                'family':    family,
                'metrics':   metrics or {},
            }
        except Exception as e:
            logger.error(f"[REGIME_FRONT] Error for {target} {timeframe}: {e}")
            return {
                'success':   False,
                'timeframe': timeframe,
                'family':    'neutral',
                'metrics':   {},
                'error':     str(e),
            }

    # ── Multi-symbol mode (symbols grid) ────────────────────────────────────
    results = {}
    for sym in symbol:
        try:
            metrics        = _fetch_and_calc_metrics(sym, timeframe)
            family         = _classify(metrics) if metrics else 'neutral'
            results[sym]   = {
                'family':  family,
                'metrics': metrics or {},
            }
        except Exception as e:
            logger.error(f"[REGIME_FRONT] Error for {sym} {timeframe}: {e}")
            results[sym] = {
                'family':  'neutral',
                'metrics': {},
                'error':   str(e),
            }

    return {
        'success':   True,
        'timeframe': timeframe,
        'symbols':   results,
    }