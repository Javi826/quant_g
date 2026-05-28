#shared/shared_batchs/regime/regime_GE_module.py

import logging
import numpy as np
import pandas as pd
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest
from shared_trading_batch.config_trading_batch import INDICATORS, COMBINE_MODE, ANALYSIS_MODE, REGIME_TIMEFRAME_MODE
from shared_batch_regime.regime_GE_core import precompute_indicators, lookup_indicators, load_ohlcv_raw
from shared_batch_regime.regime_GE_core import load_ohlcv, is_trending, load_regime_bins_ge

logger = logging.getLogger("shared_batch.regime.regime_GE_module")
REGIME_REFERENCE = 'BTCUSDT'
# =============================================================================
# CONFIGURATION  — mirror from regime_GE.py after calibration
# =============================================================================

REGIME_ENABLED   = True
DEBUG_REGIME_LOG = False


# =============================================================================
# ACTIVE CONFIG
# =============================================================================

def _active_windows() -> dict[str, int]:
    return {k: v["windows"][0] for k, v in INDICATORS.items() if v.get("enabled")}

def _active_thresholds() -> dict[str, float]:
    return {k: v["thresholds"][0] for k, v in INDICATORS.items() if v.get("enabled")}

# =============================================================================
# INDICATOR CACHE  (per-run, keyed by symbol or (symbol, timeframe))
# =============================================================================

_indicator_cache: dict = {}


def _get_indicator_cache(symbol: str, timeframe: str | None = None) -> tuple | None:
    """Lazily load and cache indicator arrays for a symbol from crypto_full_IS."""
    key = (symbol, timeframe) if REGIME_TIMEFRAME_MODE == "STRATEGY" and timeframe else symbol
    if key not in _indicator_cache:
        df = load_ohlcv_raw(symbol, timeframe) if REGIME_TIMEFRAME_MODE == "STRATEGY" and timeframe else load_ohlcv(symbol)
        if df.empty:
            return None
        _indicator_cache[key] = precompute_indicators(df, _active_windows())
    return _indicator_cache[key]


# =============================================================================
# SIGNAL FILTERING
# =============================================================================

def _filter_signals_ge(
    signals:        np.ndarray,
    ts:             np.ndarray,
    symbol:         str,
    classification: str,
    timeframe:      str | None = None,
) -> np.ndarray:
    """
    Filter signals for a single symbol based on the GE classification.
      "ranging"  → block when trending  (market is trending = bad for ranging strategy)
      "trending" → block when ranging   (market is ranging  = bad for trending strategy)
      "neutral"  → no filter
    """
    if classification == "neutral" or not REGIME_ENABLED:
        return signals

    thresholds = _active_thresholds()
    mode       = COMBINE_MODE
    ref_sym    = "BTCUSDT" if ANALYSIS_MODE == "BTC" else symbol

    cache = _get_indicator_cache(ref_sym, timeframe)
    if cache is None:
        return signals

    ts_arr, values_arr = cache
    filtered           = signals.copy()
    tf                 = timeframe if REGIME_TIMEFRAME_MODE == "STRATEGY" else None

    for idx in np.nonzero(signals)[0]:
        indicator_values = lookup_indicators(ts_arr, values_arr, pd.Timestamp(ts[idx]), timeframe=tf)
        trending         = is_trending(indicator_values, thresholds, mode)

        if classification == "ranging" and trending:
            filtered[idx] = 0
        elif classification == "trending" and not trending:
            filtered[idx] = 0

    if DEBUG_REGIME_LOG:
        rows = []
        for idx in range(len(signals)):
            indicator_values = lookup_indicators(ts_arr, values_arr, pd.Timestamp(ts[idx]), timeframe=tf)
            trending         = is_trending(indicator_values, _active_thresholds(), COMBINE_MODE)
            rows.append({
                "timestamp":       pd.Timestamp(ts[idx]),
                "signal_baseline": int(signals[idx]),
                "signal_regime":   int(filtered[idx]),
                "regime_symbol":   ref_sym,
                "trending":        trending,
                "indicators":      str(indicator_values),
            })
        df_debug = pd.DataFrame(rows)
        df_debug_all = df_debug[df_debug["signal_baseline"] != 0]
        logger.info(f"  [REGIME COUNT] symbol={symbol} | total_signals={len(df_debug_all)}")
        df_debug = df_debug_all.iloc[:10]
        logger.info(f"\n[REGIME DEBUG] symbol={symbol} | classification={classification}\n{df_debug.to_string(index=False)}")

    return filtered


# =============================================================================
# LOAD REGIME BINS  (API compatibility with regime_module.py)
# =============================================================================

def load_regime_bins(bins_path: str, strategy_id: str) -> str:
    """
    Returns the GE classification string for a strategy.
    API-compatible wrapper around load_regime_bins_ge.
    """
    return load_regime_bins_ge(bins_path, strategy_id)


# =============================================================================
# RUN OOS BACKTEST WITH REGIME  (API compatible with regime_module.py)
# =============================================================================

def run_oos_backtest_with_regime(
    strategy_id:     str,
    ohlcv_arrays:    dict,
    signal_fn,
    signal_params:   dict,
    best_params:     dict,
    order_amount:    int,
    data_folder:     str,
    timeframe:       str,
    bins_to_filter:  str,
    initial_balance: float,
    debug_label:     str = "",
) -> tuple:
    """
    Run OOS backtest applying the GE regime filter.
    bins_to_filter: classification string returned by load_regime_bins().
    """
    ohlcv_arrays_regime: dict = {}

    for sym, arr in ohlcv_arrays.items():
        signals = signal_fn(arr, **signal_params, live_trading=False)
        logger.debug(f"  [SIGNAL CHECK] sym={sym} | signals sum={int(signals.sum())} | len={len(signals)}")
        if REGIME_ENABLED and bins_to_filter and bins_to_filter != "neutral":
            signals = _filter_signals_ge(
                signals        = signals,
                ts             = arr['ts'],
                symbol         = sym,
                classification = bins_to_filter,
                timeframe      = timeframe,
            )

        ohlcv_arrays_regime[sym] = {**arr, "signal": signals}

    result = run_grid_backtest(
        ohlcv_arrays_regime,
        sell_after   = best_params["SELL_AFTER"],
        tp_pct       = best_params["TP_PCT"],
        sl_pct       = best_params["SL_PCT"],
        order_amount = order_amount,
    )
    trades_df             = result["__PORTFOLIO__"]["trade_log"].copy()
    trades_df.columns     = trades_df.columns.str.lower().str.strip()
    trades_df["buy_time"] = pd.to_datetime(trades_df["buy_time"])
    metrics = compute_metrics(trades_df, capital=initial_balance, name=strategy_id) if len(trades_df) > 0 else None
    return trades_df, metrics