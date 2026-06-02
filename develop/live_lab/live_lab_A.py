"""
validate_regime.py
------------------
Validates that regime classification produces identical results
between the batch pipeline and the trading pipeline.

For each trade in the Excel:
  - Loads OHLCV from disk (CRYPTO_FULL_DIR, 1Dutc timeframe)
  - Classifies regime using BATCH functions  (precompute_indicators + lookup_indicators + is_trending)
  - Classifies regime using TRADING functions (_calc_metrics_from_arr + _is_trending)
  - Compares and prints result
"""

import os
import sys

# =============================================================================
# PATHS — adjust if needed
# =============================================================================
BITGET_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget"))
TRADES_XLSX    = os.path.join(BITGET_ROOT, "BOT_trading", "persistence", "bot_files_00", "bot_trades_00.xlsx")
CRYPTO_FULL_DIR = os.path.join(BITGET_ROOT, "data_pipeline", "data", "04_split_OLD", "expanding", "IS", "crypto_full_IS")

# Add shared paths — order matters: signals must be before shared_batchs loads
SIGNALS_DIR             = os.path.join(BITGET_ROOT, "signals")
SHARED_DIR              = os.path.join(BITGET_ROOT, "shared")
SHARED_BATCH_REGIME_DIR = os.path.join(BITGET_ROOT, "shared", "shared_batch_regime")
SHARED_TRADING_DIR      = os.path.join(BITGET_ROOT, "shared", "shared_trading_batch_regime")
SHARED_BATCHS_DIR       = os.path.join(BITGET_ROOT, "shared", "shared_batchs")
BOT_TRADING_DIR         = os.path.join(BITGET_ROOT, "BOT_trading")

for p in [SIGNALS_DIR, SHARED_DIR, SHARED_BATCH_REGIME_DIR, SHARED_TRADING_DIR, SHARED_BATCHS_DIR, BOT_TRADING_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Force signals into sys.modules so nested imports from shared_batchs find it
import importlib.util as _ilu
for _mod, _path in [
    ("signals", os.path.join(SIGNALS_DIR, "__init__.py")),
]:
    if _mod not in sys.modules:
        _spec = _ilu.spec_from_file_location(_mod, _path)
        _m    = _ilu.module_from_spec(_spec)
        sys.modules[_mod] = _m
        _spec.loader.exec_module(_m)

# =============================================================================
# ACCOUNT CONFIG — trading pipeline params (from settings.py ACCOUNTS["E1"])
# =============================================================================
ACCOUNT_CONFIG = {
    "regime_indicators": {
        "atr_norm": {"window": 10, "threshold": 0.04, "enabled": True},
        "er":       {"window": 40, "threshold": 0.8,  "enabled": True},
    },
    "regime_combine_mode":   "OR",
    "regime_analysis_mode":  "SYMBOL",
    "regime_timeframe_mode": "DAILY",
}

REGIME_TIMEFRAME = "1Dutc"

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd

# Batch imports
from regime_GE_core import precompute_indicators, lookup_indicators, is_trending, load_ohlcv_raw

# Trading imports
from regime_metrics import _CALC_FN

# =============================================================================
# HELPERS — trading pipeline (mirrors regime_classifier.py)
# =============================================================================

def _active_windows() -> dict:
    return {k: v["window"] for k, v in ACCOUNT_CONFIG["regime_indicators"].items() if v.get("enabled")}

def _active_thresholds() -> dict:
    return {k: v["threshold"] for k, v in ACCOUNT_CONFIG["regime_indicators"].items() if v.get("enabled")}

def trading_calc_metrics(arr: dict) -> dict:
    """Mirrors _calc_metrics_from_arr in regime_classifier.py."""
    windows = _active_windows()
    high    = arr["high"]
    low     = arr["low"]
    close   = arr["close"]
    metrics = {}
    for key, w in windows.items():
        val = _CALC_FN[key](high, low, close, w)
        metrics[key] = float(val) if not np.isnan(val) else None
    return metrics

def trading_is_trending(metrics: dict) -> bool:
    """Mirrors _is_trending in regime_classifier.py."""
    thresholds = _active_thresholds()
    mode       = ACCOUNT_CONFIG["regime_combine_mode"]
    signals    = []
    for key, val in metrics.items():
        if val is None or np.isnan(val):
            continue
        signals.append(val >= thresholds[key])
    if not signals:
        return False
    return all(signals) if mode == "AND" else any(signals)

def trading_classify(metrics: dict) -> str:
    if not metrics or all(v is None for v in metrics.values()):
        return "neutral"
    return "trending" if trading_is_trending(metrics) else "ranging"

# =============================================================================
# HELPERS — batch pipeline
# =============================================================================

def batch_classify(symbol: str, open_at: pd.Timestamp) -> str:
    """Classify regime using batch pipeline functions."""
    windows    = _active_windows()
    thresholds = _active_thresholds()
    mode       = ACCOUNT_CONFIG["regime_combine_mode"]

    df = load_ohlcv_raw(symbol, REGIME_TIMEFRAME)
    if df.empty:
        return "no_data"

    ts_arr, values_arr = precompute_indicators(df, windows)
    if len(ts_arr) == 0:
        return "no_data"

    indicator_values = lookup_indicators(ts_arr, values_arr, open_at, timeframe=REGIME_TIMEFRAME)
    if all(v is None for v in indicator_values.values()):
        return "no_data"

    trending = is_trending(indicator_values, thresholds, mode)
    return "trending" if trending else "ranging"

# =============================================================================
# HELPERS — trading pipeline (full fetch from disk, mirrors get_symbol_regime)
# =============================================================================

def trading_classify_from_disk(symbol: str, open_at: pd.Timestamp) -> str:
    """
    Classify regime using trading pipeline functions.
    Fetches the full 1Dutc series up to open_at, takes the last candle.
    """
    df = load_ohlcv_raw(symbol, REGIME_TIMEFRAME)
    if df.empty:
        return "no_data"

    # Use only candles available at signal time (same as live: latest candle)
    open_at_utc = open_at.tz_localize("UTC") if open_at.tzinfo is None else open_at
    df_at_time  = df[df["ts"] < open_at_utc]
    if df_at_time.empty:
        return "no_data"

    arr = {
        "high":  df_at_time["high"].values,
        "low":   df_at_time["low"].values,
        "close": df_at_time["close"].values,
    }
    metrics = trading_calc_metrics(arr)
    return trading_classify(metrics)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{'='*80}")
    print(f"  REGIME VALIDATION — Batch vs Trading pipeline")
    print(f"  Indicators : {list(_active_windows().keys())}")
    print(f"  Windows    : {_active_windows()}")
    print(f"  Thresholds : {_active_thresholds()}")
    print(f"  Mode       : {ACCOUNT_CONFIG['regime_combine_mode']}")
    print(f"  Timeframe  : {REGIME_TIMEFRAME}")
    print(f"{'='*80}\n")

    df_trades = pd.read_excel(TRADES_XLSX)
    df_trades.columns = [c.strip().upper() for c in df_trades.columns]

    required = {"OPEN_AT", "SYMBOL", "STRATEGY"}
    missing  = required - set(df_trades.columns)
    if missing:
        print(f"ERROR: missing columns in Excel: {missing}")
        print(f"Available columns: {list(df_trades.columns)}")
        return

    df_trades["OPEN_AT"] = pd.to_datetime(df_trades["OPEN_AT"], utc=True, errors="coerce")
    df_trades = df_trades.dropna(subset=["OPEN_AT", "SYMBOL"])

    print(f"  Trades to validate: {len(df_trades)}\n")
    print(f"  {'#':<5} {'STRATEGY':<30} {'SYMBOL':<14} {'OPEN_AT':<22} {'BATCH':<10} {'TRADING':<10} {'MATCH'}")
    print(f"  {'-'*95}")

    n_match    = 0
    n_mismatch = 0
    n_nodata   = 0

    for i, row in df_trades.iterrows():
        symbol      = str(row["SYMBOL"]).strip()
        open_at     = row["OPEN_AT"]
        strategy_id = str(row["STRATEGY"]).strip()

        batch_regime   = batch_classify(symbol, open_at)
        trading_regime = trading_classify_from_disk(symbol, open_at)

        if "no_data" in (batch_regime, trading_regime):
            match_str = "⚠️  NO DATA"
            n_nodata += 1
        elif batch_regime == trading_regime:
            match_str = "✅ MATCH"
            n_match += 1
        else:
            match_str = "❌ MISMATCH"
            n_mismatch += 1

        print(f"  {i+1:<5} {strategy_id:<30} {symbol:<14} {str(open_at)[:19]:<22} "
              f"{batch_regime:<10} {trading_regime:<10} {match_str}")

    total = n_match + n_mismatch + n_nodata
    print(f"\n  {'-'*95}")
    print(f"  RESULTS: {total} trades | ✅ {n_match} match | ❌ {n_mismatch} mismatch | ⚠️  {n_nodata} no data")
    pct = n_match / (n_match + n_mismatch) * 100 if (n_match + n_mismatch) > 0 else 0
    print(f"  Match rate: {pct:.1f}%")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()