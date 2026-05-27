#shared/shared_batchs/regime/regime_GE_module.py
"""
GE Regime module — main_batch integration.
Drop-in replacement for regime_module.py.

Uses the GE regime system (ER / Hurst / ATR_norm indicators) to filter signals,
loading indicator data from crypto_full_IS.

Classification logic (from regime_GE.py):
  "ranging"  → block signals when market is trending    (filter_trending improves)
  "trending" → block signals when market is ranging     (filter_ranging improves)
  "both"     → block both trending and ranging signals
  "neutral"  → no filter applied

Configuration mirrors INDICATORS dict from regime_GE.py.
Copy the final chosen windows/thresholds/mode here after calibration.
"""
import os
import logging
import numpy as np
import pandas as pd

from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE, run_grid_backtest
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batch_develop.regime_GE_core import (
    _load_ohlcv,
    _is_trending,
    load_regime_bins_ge,
    CRYPTO_FULL_DIR,
)
from shared_trading_batch_develop.regime_metrics import (
    precompute_indicators,
    lookup_indicators,
)

logger = logging.getLogger("shared_batch.regime.regime_GE_module")

# =============================================================================
# CONFIGURATION  — mirror from regime_GE.py after calibration
# =============================================================================

REGIME_ENABLED = True

INDICATORS: dict[str, dict] = {
    "atr_norm": {
        "windows":    [10],
        "thresholds": [0.04],
        "enabled":    True,
    },
    "er": {
        "windows":    [10],
        "thresholds": [0.6],
        "enabled":    True,
    },
    "hurst": {
        "windows":    [30],
        "thresholds": [0.8],
        "enabled":    True,
    },
}

COMBINE_MODE  = "OR"
ANALYSIS_MODE = "SYMBOL"

# Kept for API compatibility with regime_module.py
REGIME_MIN_TRADES    = 50
REGIME_FAMILY_SOURCE = "strategy"
REGIME0_MA_PERIOD    = 50

# =============================================================================
# ACTIVE CONFIG
# =============================================================================

# CORRECTO
def _active_windows() -> dict[str, int]:
    return {k: v["windows"][0] for k, v in INDICATORS.items() if v.get("enabled")}

def _active_thresholds() -> dict[str, float]:
    return {k: v["thresholds"][0] for k, v in INDICATORS.items() if v.get("enabled")}

# =============================================================================
# INDICATOR CACHE  (per-run, keyed by symbol)
# =============================================================================

_indicator_cache: dict[str, tuple] = {}


def _get_indicator_cache(symbol: str) -> tuple | None:
    """Lazily load and cache indicator arrays for a symbol from crypto_full_IS."""
    if symbol not in _indicator_cache:
        df = _load_ohlcv(symbol)
        if df.empty:
            return None
        _indicator_cache[symbol] = precompute_indicators(df, _active_windows())
    return _indicator_cache[symbol]


# =============================================================================
# SIGNAL FILTERING
# =============================================================================

def _filter_signals_ge(
    signals:        np.ndarray,
    ts:             np.ndarray,
    symbol:         str,
    classification: str,
) -> np.ndarray:
    """
    Filter signals for a single symbol based on the GE classification.
      "ranging"  → block when trending  (market is trending = bad for ranging strategy)
      "trending" → block when ranging   (market is ranging  = bad for trending strategy)
      "both"     → block always
      "neutral"  → no filter
    """
    if classification == "neutral" or not REGIME_ENABLED:
        return signals

    thresholds = _active_thresholds()
    mode       = COMBINE_MODE

    cache = _get_indicator_cache("BTCUSDT" if ANALYSIS_MODE == "BTC" else symbol)
    if cache is None:
        return signals

    ts_arr, values_arr = cache
    filtered           = signals.copy()

    for idx in np.nonzero(signals)[0]:
        indicator_values = lookup_indicators(ts_arr, values_arr, pd.Timestamp(ts[idx]))
        trending         = _is_trending(indicator_values, thresholds, mode)

        if classification == "ranging" and trending:
            filtered[idx] = 0
        elif classification == "trending" and not trending:
            filtered[idx] = 0
        elif classification == "both":
            filtered[idx] = 0

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

        if REGIME_ENABLED and bins_to_filter and bins_to_filter != "neutral":
            signals = _filter_signals_ge(
                signals        = signals,
                ts             = arr['ts'],
                symbol         = sym,
                classification = bins_to_filter,
            )

        ohlcv_arrays_regime[sym] = {**arr, "signal": signals}

    result                = run_grid_backtest(
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