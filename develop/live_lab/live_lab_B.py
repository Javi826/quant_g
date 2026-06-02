"""
validate_signals.py
-------------------
Validates that signal generation produces identical results
between the trading pipeline (broker data) and the batch pipeline (parquet data).

For each active strategy and each symbol prints:
  - Timestamp of last candle (broker vs parquet)
  - Baseline signal (before regime filter)
  - Regime classification
  - Final signal (after regime filter)
  - Match between broker and parquet
"""

import os
import sys

# =============================================================================
# PATHS
# =============================================================================
BITGET_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget"))
CRYPTO_FULL_DIR  = os.path.join(BITGET_ROOT, "data_pipeline", "data", "04_split_OLD", "expanding", "IS", "crypto_full_IS")
SYMBOLS_LIVE_DIR = os.path.join(BITGET_ROOT, "BOT_trading", "symbols_live", "E1")

# =============================================================================
# SELECTED STRATEGIES — set to [] to run all active strategies
# =============================================================================
SELECTED_STRATEGIES = [
    "05_reversal_long_1H",
    "31_orderblocks_long_15m",
]

# =============================================================================
# ACCOUNT CONFIG
# =============================================================================
ACCOUNT_CONFIG = {
    "regime_indicators": {
        "atr_norm": {"window": 10, "threshold": 0.04, "enabled": True},
        "er":       {"window": 40, "threshold": 0.8,  "enabled": True},
    },
    "regime_combine_mode":   "OR",
    "regime_timeframe_mode": "DAILY",
    "regime_analysis_mode":  "SYMBOL",
}
REGIME_TIMEFRAME = "1Dutc"

# =============================================================================
# SYS PATH
# =============================================================================
SIGNALS_DIR             = os.path.join(BITGET_ROOT, "signals")
SHARED_DIR              = os.path.join(BITGET_ROOT, "shared")
SHARED_BATCH_REGIME_DIR = os.path.join(BITGET_ROOT, "shared", "shared_batch_regime")
SHARED_TRADING_DIR      = os.path.join(BITGET_ROOT, "shared", "shared_trading_batch_regime")
SHARED_BATCHS_DIR       = os.path.join(BITGET_ROOT, "shared", "shared_batchs")
BOT_TRADING_DIR         = os.path.join(BITGET_ROOT, "BOT_trading")

for p in [SIGNALS_DIR, SHARED_DIR, SHARED_BATCH_REGIME_DIR, SHARED_TRADING_DIR, SHARED_BATCHS_DIR, BOT_TRADING_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util as _ilu
for _mod, _path in [("signals", os.path.join(SIGNALS_DIR, "__init__.py"))]:
    if _mod not in sys.modules:
        _spec = _ilu.spec_from_file_location(_mod, _path)
        _m    = _ilu.module_from_spec(_spec)
        sys.modules[_mod] = _m
        _spec.loader.exec_module(_m)

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd

from signals.add_signals_reversal    import reversal_long, reversal_short
from signals.add_signals_parity      import parity_long, parity_short
from signals.add_signals_flag        import flag_long, flag_short
from signals.add_signals_orderblocks import orderblocks_long, orderblocks_short

from market_data.data_utils          import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from shared_batch_regime.regime_GE_core import load_ohlcv_raw, precompute_indicators, lookup_indicators, is_trending
from regime_metrics                  import _CALC_FN

from config.strategies_00            import STRATEGIES

# =============================================================================
# HELPERS — symbols
# =============================================================================

def load_symbols(strategy_id: str, timeframe: str) -> list:
    path = os.path.join(SYMBOLS_LIVE_DIR, f"symbols_live_{strategy_id}_{timeframe}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


# =============================================================================
# HELPERS — signal
# =============================================================================

def compute_signal(strat: dict, arr: dict) -> int:
    """Return raw baseline signal on last candle (0 or 1). No regime logic."""
    sid = strat["id"]
    if "reversal_long"     in sid: signals = reversal_long(arr,     lookback=strat["lookback"], tolerance=strat["tolerance"], ma_period=strat["ma_period"], live_trading=True)
    elif "reversal_short"  in sid: signals = reversal_short(arr,    lookback=strat["lookback"], tolerance=strat["tolerance"], ma_period=strat["ma_period"], live_trading=True)
    elif "parity_long"     in sid: signals = parity_long(arr,       lookback=strat["lookback"], tolerance=strat["tolerance"], ma_period=strat["ma_period"], live_trading=True)
    elif "parity_short"    in sid: signals = parity_short(arr,      lookback=strat["lookback"], tolerance=strat["tolerance"], ma_period=strat["ma_period"], live_trading=True)
    elif "flag_long"       in sid: signals = flag_long(arr,         lookback=strat["lookback"], impulse=strat["impulse"], flag=strat["flag"], ma_period=strat["ma_period"], live_trading=True)
    elif "flag_short"      in sid: signals = flag_short(arr,        lookback=strat["lookback"], impulse=strat["impulse"], flag=strat["flag"], ma_period=strat["ma_period"], live_trading=True)
    elif "orderblocks_long"  in sid: signals = orderblocks_long(arr,  lookback=strat["lookback"], tolerance=strat["tolerance"], impulse=strat["impulse"], live_trading=True)
    elif "orderblocks_short" in sid: signals = orderblocks_short(arr, lookback=strat["lookback"], tolerance=strat["tolerance"], impulse=strat["impulse"], live_trading=True)
    else: return -1
    return int(signals[-1]) if signals is not None and len(signals) > 0 else -1


# =============================================================================
# HELPERS — regime (trading pipeline)
# =============================================================================

def _active_windows() -> dict:
    return {k: v["window"] for k, v in ACCOUNT_CONFIG["regime_indicators"].items() if v.get("enabled")}

def _active_thresholds() -> dict:
    return {k: v["threshold"] for k, v in ACCOUNT_CONFIG["regime_indicators"].items() if v.get("enabled")}

def trading_get_regime(symbol: str, signal_ts: pd.Timestamp) -> tuple:
    """Trading pipeline: load 1Dutc parquet, calc metrics on candles up to signal_ts.
    Returns (regime_str, ts_of_last_daily_candle_used)."""
    df = load_ohlcv_raw(symbol, REGIME_TIMEFRAME)
    if df.empty:
        return "no_data", None
    signal_ts_utc = signal_ts.tz_localize("UTC") if signal_ts.tzinfo is None else signal_ts
    df_at         = df[df["ts"] < signal_ts_utc]
    if df_at.empty:
        return "no_data", None
    ts_regime = pd.Timestamp(df_at["ts"].iloc[-1])
    arr        = {"high": df_at["high"].values, "low": df_at["low"].values, "close": df_at["close"].values}
    windows    = _active_windows()
    thresholds = _active_thresholds()
    mode       = ACCOUNT_CONFIG["regime_combine_mode"]
    metrics    = {}
    for key, w in windows.items():
        val = _CALC_FN[key](arr["high"], arr["low"], arr["close"], w)
        metrics[key] = float(val) if not np.isnan(val) else None
    signals = []
    for key, val in metrics.items():
        if val is None: continue
        signals.append(val >= thresholds[key])
    if not signals:
        return "neutral", ts_regime
    trending = all(signals) if mode == "AND" else any(signals)
    return ("trending" if trending else "ranging"), ts_regime


# =============================================================================
# HELPERS — regime (batch pipeline)
# =============================================================================

def batch_get_regime(symbol: str, signal_ts: pd.Timestamp) -> tuple:
    """Batch pipeline: precompute_indicators + lookup_indicators + is_trending.
    Returns (regime_str, ts_of_daily_candle_used)."""
    df = load_ohlcv_raw(symbol, REGIME_TIMEFRAME)
    if df.empty:
        return "no_data", None
    windows    = _active_windows()
    thresholds = _active_thresholds()
    mode       = ACCOUNT_CONFIG["regime_combine_mode"]
    ts_arr, values_arr = precompute_indicators(df, windows)
    if len(ts_arr) == 0:
        return "no_data", None
    indicator_values, idx = lookup_indicators(ts_arr, values_arr, signal_ts, timeframe=REGIME_TIMEFRAME, return_idx=True)
    if all(v is None for v in indicator_values.values()):
        return "no_data", None
    ts_regime = pd.Timestamp(ts_arr[idx]) if idx >= 0 else None
    trending  = is_trending(indicator_values, thresholds, mode)
    return ("trending" if trending else "ranging"), ts_regime


# =============================================================================
# HELPERS — final signal after regime filter
# =============================================================================

def apply_regime_filter(strat: dict, baseline_signal: int, regime: str) -> int:
    """Apply regime bin flags from strategy config to baseline signal."""
    if baseline_signal == 0:
        return 0
    bin_key = f"regime_{regime}"
    flag    = strat.get(bin_key, 1)
    return baseline_signal if flag == 1 else 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    strategies = [s for s in STRATEGIES if s.get("active", False)]
    if SELECTED_STRATEGIES:
        strategies = [s for s in strategies if s["id"] in SELECTED_STRATEGIES]

    print(f"\n{'='*160}")
    print(f"  SIGNAL VALIDATION — Broker (trading) vs Parquet (batch)")
    print(f"  Strategies : {[s['id'] for s in strategies]}")
    print(f"  Regime TF  : {REGIME_TIMEFRAME} | Mode: {ACCOUNT_CONFIG['regime_combine_mode']}")
    print(f"{'='*160}\n")

    col = f"  {'STRATEGY':<30} {'SYMBOL':<14} | {'TS_BROKER':<22} {'BASE':>5} {'TS_REG_BROKER':<22} {'REGIME':<10} {'FINAL':>6} | {'TS_PARQUET':<22} {'BASE':>5} {'TS_REG_PARQUET':<22} {'REGIME':<10} {'FINAL':>6} | {'MATCH'}"
    print(col)
    print(f"  {'-'*155}")

    n_match    = 0
    n_mismatch = 0
    n_nodata   = 0

    for strat in strategies:
        sid       = strat["id"]
        timeframe = strat["timeframe"]
        symbols   = load_symbols(sid, timeframe)

        if not symbols:
            print(f"  {sid:<30} ⚠️  no symbols file")
            continue

        broker_data = fetch_ohlcv_data(symbols, timeframe)

        for symbol in symbols:

            # ── TRADING: broker ───────────────────────────────────────────
            df_broker = broker_data.get(symbol)
            if df_broker is None or df_broker.empty:
                print(f"  {sid:<30} {symbol:<14} | ⚠️  no broker data")
                n_nodata += 1
                continue

            df_norm      = normalize_live_ohlcv(df_broker)
            arr_trading  = df_to_arrays_live(df_norm)
            ts_trading   = pd.Timestamp(arr_trading["ts"][-1])
            base_trading = compute_signal(strat, arr_trading)
            reg_trading, ts_reg_trading = trading_get_regime(symbol, ts_trading)
            fin_trading  = apply_regime_filter(strat, base_trading, reg_trading)

            # ── BATCH: parquet ────────────────────────────────────────────
            df_parquet = load_ohlcv_raw(symbol, timeframe)
            if df_parquet.empty:
                print(f"  {sid:<30} {symbol:<14} | ⚠️  no parquet data")
                n_nodata += 1
                continue

            arr_batch  = {
                "open":   df_parquet["open"].values.astype(np.float32),
                "high":   df_parquet["high"].values.astype(np.float32),
                "low":    df_parquet["low"].values.astype(np.float32),
                "close":  df_parquet["close"].values.astype(np.float32),
                "volume": df_parquet["volume"].values.astype(np.float32) if "volume" in df_parquet.columns else np.zeros(len(df_parquet), dtype=np.float32),
                "ts":     df_parquet["ts"].values,
            }
            ts_batch   = pd.Timestamp(arr_batch["ts"][-1])
            base_batch = compute_signal(strat, arr_batch)
            reg_batch, ts_reg_batch = batch_get_regime(symbol, ts_batch)
            fin_batch  = apply_regime_filter(strat, base_batch, reg_batch)

            # ── COMPARE ───────────────────────────────────────────────────
            if base_trading == -1 or base_batch == -1:
                m_base   = "⚠️"
                m_regime = "⚠️"
                m_final  = "⚠️"
                n_nodata += 1
            else:
                m_base   = "✅" if base_trading == base_batch  else "❌"
                m_regime = "✅" if reg_trading  == reg_batch   else "❌"
                m_final  = "✅" if fin_trading  == fin_batch   else "❌"
                if m_base == "✅" and m_regime == "✅" and m_final == "✅":
                    n_match += 1
                else:
                    n_mismatch += 1

            ts_t_str = str(ts_trading)[:19]
            ts_b_str = str(ts_batch)[:19]

            ts_reg_t_str = str(ts_reg_trading)[:19] if ts_reg_trading else "—"
            ts_reg_b_str = str(ts_reg_batch)[:19]   if ts_reg_batch   else "—"

            print(
                f"  {sid:<30} {symbol:<14} | "
                f"{ts_t_str:<22} {base_trading:>5} {ts_reg_t_str:<22} {reg_trading:<10} {fin_trading:>6} | "
                f"{ts_b_str:<22} {base_batch:>5} {ts_reg_b_str:<22} {reg_batch:<10} {fin_batch:>6} | "
                f"BASE:{m_base} REGIME:{m_regime} FINAL:{m_final}"
            )

    total = n_match + n_mismatch + n_nodata
    print(f"\n  {'-'*155}")
    print(f"  RESULTS: {total} symbols | ✅ {n_match} match | ❌ {n_mismatch} mismatch | ⚠️  {n_nodata} no data")
    pct = n_match / (n_match + n_mismatch) * 100 if (n_match + n_mismatch) > 0 else 0
    print(f"  Match rate: {pct:.1f}%")
    print(f"{'='*160}\n")


if __name__ == "__main__":
    main()