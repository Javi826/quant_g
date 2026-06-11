#shared/shared_batchs/regime/regime_module.py

import logging
import numpy as np
import pandas as pd
from importlib.util import spec_from_file_location, module_from_spec
import os
from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batch_regime.regime_core import load_ohlcv_raw
from shared_batch_regime.regime_core import precompute_indicators, lookup_indicator_batch
from shared_batch_regime.regime_core import classify_market_regime

logger = logging.getLogger("shared_batch.regime.regime_module")

# =============================================================================
# CONFIGURATION  (populated by load_config_from_bins)
# =============================================================================
REGIME_ENABLED  = None
INDICATOR_CFG:  dict = {}
# =============================================================================
# INDICATOR CACHE  (MA over daily close, keyed by symbol)
# =============================================================================
_indicator_cache: dict = {}

def load_config_from_bins(bins_path: str) -> None:
    """Load indicator config from a regime_bins file. Validates MA_TIMEFRAME against REGIME_TIMEFRAME."""
    global INDICATOR_CFG, _indicator_cache

    spec   = spec_from_file_location("regime_bins", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "INDICATOR_CFG"):
        raise ValueError("❌ regime_bins file must contain INDICATOR_CFG. Re-run regime_calibration.py to regenerate.")

    INDICATOR_CFG    = module.INDICATOR_CFG
    _indicator_cache = {}
    logger.debug(f"  [regime_module] config loaded — INDICATOR_CFG={INDICATOR_CFG}")

def _get_indicator_cache(symbol: str) -> dict | None:
    if symbol not in _indicator_cache:
        df = load_ohlcv_raw(symbol)
        if df.empty:
            return None
        _indicator_cache[symbol] = precompute_indicators(df, INDICATOR_CFG)
    return _indicator_cache[symbol]

# =============================================================================
# LOAD REGIME BINS
# =============================================================================

def load_regime_bins(bins_path: str, strategy_id: str) -> list[str]:

    if not os.path.exists(bins_path):
        logger.warning(f"regime_bins file not found: {bins_path} — defaulting to neutral.")
        return []
    spec   = spec_from_file_location("regime_bins", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    bins = getattr(module, "REGIME_BINS", {})

    return bins.get(strategy_id, [])

# =============================================================================
# RUN OOS BACKTEST WITH REGIME
# =============================================================================

def run_oos_backtest_with_regime(
    strategy_id:     str,
    ohlcv_arrays:    dict,
    signal_fn,
    signal_params:   dict,
    best_params:     dict,
    order_amount:    int,
    bins_to_filter:  str | list[str],
    initial_balance: float,
) -> tuple:
    _bins_to_filter = [bins_to_filter] if isinstance(bins_to_filter, str) else bins_to_filter

    ohlcv_arrays_regime: dict = {}

    for sym, arr in ohlcv_arrays.items():
        signals = signal_fn(arr, **signal_params, live_trading=False)

        if REGIME_ENABLED and _bins_to_filter and _bins_to_filter != ["neutral"]:
            sym_cache = _get_indicator_cache(sym)

            if sym_cache is not None:
                signal_idxs = np.nonzero(signals)[0]

                if signal_idxs.size > 0:
                    signal_ts = arr['ts'][signal_idxs]
                    lookups   = {
                        key: lookup_indicator_batch(sym_cache["ts"], sym_cache[key], signal_ts)
                        for key in sym_cache if key != "ts"
                    }
                    for i, idx in enumerate(signal_idxs):
                        context = {"close": float(arr['close'][idx])}
                        for key, values in lookups.items():
                            context[key] = float(values[i]) if not np.isnan(values[i]) else None
                        if classify_market_regime(context, cfg=INDICATOR_CFG) not in _bins_to_filter:
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