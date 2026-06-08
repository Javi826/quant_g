#shared/shared_batchs/regime/regime_GE_module_uptrend.py

import logging
import numpy as np
import pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batch_regime.regime_GE_core_uptrend import (
    load_ohlcv_raw,
    precompute_indicators,
    lookup_ma_batch,
    classify_market_regime,
    load_regime_bins_ge,
)

logger = logging.getLogger("shared_batch.regime.regime_GE_module_uptrend")

# =============================================================================
# CONFIGURATION  (populated by load_config_from_bins_uptrend)
# =============================================================================
REGIME_ENABLED = True
MA_WINDOW      = 3
MA_TIMEFRAME   = "1Dutc"
ANALYSIS_MODE  = "SYMBOL"

# =============================================================================
# INDICATOR CACHE  (MA over daily close, keyed by symbol)
# =============================================================================

_indicator_cache: dict = {}


def load_config_from_bins_uptrend(bins_path: str) -> None:
    """Load MA_WINDOW, MA_TIMEFRAME, ANALYSIS_MODE from a regime_bins_uptrend file."""
    global MA_WINDOW, MA_TIMEFRAME, ANALYSIS_MODE, _indicator_cache

    spec   = spec_from_file_location("regime_bins_uptrend", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "MA_WINDOW"):
        MA_WINDOW = module.MA_WINDOW
    if hasattr(module, "MA_TIMEFRAME"):
        MA_TIMEFRAME = module.MA_TIMEFRAME
    if hasattr(module, "ANALYSIS_MODE"):
        ANALYSIS_MODE = module.ANALYSIS_MODE

    _indicator_cache = {}
    logger.debug(f"[REGIME UPTREND] Config loaded — MA_WINDOW={MA_WINDOW} MA_TIMEFRAME={MA_TIMEFRAME} ANALYSIS_MODE={ANALYSIS_MODE}")


def _get_indicator_cache(symbol: str) -> tuple | None:
    """Lazily load and cache (ts_arr, ma_arr) for a symbol on daily timeframe."""
    if symbol not in _indicator_cache:
        df = load_ohlcv_raw(symbol, MA_TIMEFRAME)
        if df.empty:
            return None
        _indicator_cache[symbol] = precompute_indicators(df, MA_WINDOW)
    return _indicator_cache[symbol]


# =============================================================================
# LOAD REGIME BINS
# =============================================================================

def load_regime_bins_uptrend(bins_path: str, strategy_id: str) -> str:
    """Return the uptrend classification string for a strategy."""
    return load_regime_bins_ge(bins_path, strategy_id)


# =============================================================================
# RUN OOS BACKTEST WITH UPTREND REGIME
# =============================================================================

def run_oos_backtest_with_regime_uptrend(
    strategy_id:     str,
    ohlcv_arrays:    dict,
    signal_fn,
    signal_params:   dict,
    best_params:     dict,
    order_amount:    int,
    bins_to_filter:  str,
    initial_balance: float,
) -> tuple:
    """
    Run OOS backtest applying the uptrend/downtrend MA regime filter.

    bins_to_filter : "uptrend" | "downtrend" | "neutral"
      - "uptrend"   → keep signals only when close > MA  (uptrend regime)
      - "downtrend" → keep signals only when close < MA  (downtrend regime)
      - "neutral"   → no filter applied
    """
    ohlcv_arrays_regime: dict = {}

    for sym, arr in ohlcv_arrays.items():
        signals = signal_fn(arr, **signal_params, live_trading=False)

        if REGIME_ENABLED and bins_to_filter and bins_to_filter != "neutral":
            ref_sym   = "BTCUSDT" if ANALYSIS_MODE == "BTC" else sym
            sym_cache = _get_indicator_cache(ref_sym)

            if sym_cache is not None:
                ts_arr, ma_arr = sym_cache
                signal_idxs    = np.nonzero(signals)[0]

                if signal_idxs.size > 0:
                    signal_ts = arr['ts'][signal_idxs]
                    lookups   = lookup_ma_batch(ts_arr, ma_arr, signal_ts)

                    for i, idx in enumerate(signal_idxs):
                        close_val = float(arr['close'][idx])
                        ma_val    = float(lookups[i]) if not np.isnan(lookups[i]) else None
                        regime    = classify_market_regime(close_val, ma_val)
                        if regime != bins_to_filter:
                            signals[idx] = 0

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