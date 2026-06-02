#develop/live_lab/live_lab_B.py

import os
import sys

# =============================================================================
# PATHS
# =============================================================================
BITGET_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget"))
CRYPTO_FULL_DIR  = os.path.join(BITGET_ROOT, "data_pipeline", "data", "04_split_OLD", "expanding", "IS", "crypto_full_IS")
RAW_DIR          = os.path.join(BITGET_ROOT, "data_pipeline", "data", "01_raw")
SYMBOLS_LIVE_DIR = os.path.join(BITGET_ROOT, "BOT_trading", "symbols_live", "E1")

# =============================================================================
# SELECTED STRATEGIES — set to [] to run all active strategies
# =============================================================================
SELECTED_STRATEGIES = [
    #"05_reversal_long_1H",
   # "31_orderblocks_long_15m",
]

# =============================================================================
# ACCOUNT CONFIG
# =============================================================================
ACCOUNT_NUMBER   = "E1"
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

# --- Trading pipeline ---
from strategies.strategy_registry    import detect_signals_for_strategy
from market_regime.regime_classifier import configure_regime, get_symbol_regime
from market_regime.position_sizer    import PositionSizer
from market_data.data_utils          import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live

# --- Batch pipeline ---
from shared_batch_regime.regime_GE_core import load_ohlcv_raw, precompute_indicators, lookup_indicators, is_trending
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY

# --- Shared ---
from config.strategies_00 import STRATEGIES

# =============================================================================
# INIT
# =============================================================================
import logging
logging.getLogger("BOT_trading.strategies.registry").setLevel(logging.WARNING)
logging.getLogger("BOT_trading.market_regime.regime_classifier").setLevel(logging.WARNING)

configure_regime(ACCOUNT_NUMBER)
_position_sizer = PositionSizer(logger=None)

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
# HELPERS — signal key
# =============================================================================

def _signal_key(strat: dict) -> str:
    """Extract SIGNAL_REGISTRY key from strategy name. e.g. 'reversal_long_1H' → 'reversal_long'"""
    return "_".join(strat["name"].split("_")[:-1])


def _signal_params(strat: dict, registry_entry: dict) -> dict:
    """Extract signal params from strat dict using registry param keys."""
    return {k: strat[k] for k in registry_entry["params"]}


# =============================================================================
# HELPERS — trading pipeline signal
# =============================================================================

def trading_get_signal(strat: dict, symbol: str) -> int:
    """Call detect_signals_for_strategy (broker fetch internally). Returns 0 or 1."""
    signals = detect_signals_for_strategy(strat, [symbol], None, regime_enabled=False)
    return 1 if signals else 0


# =============================================================================
# HELPERS — batch pipeline signal
# =============================================================================

def load_ohlcv_from_dir(symbol: str, timeframe: str, data_dir: str) -> pd.DataFrame:
    """Clone of load_ohlcv_raw but reading from a custom directory."""
    path = os.path.join(data_dir, f"{symbol}_{timeframe}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if df.index.name and df.index.name.lower() in ("timestamp", "ts", "date", "time"):
        df.index.name = "ts"
        df = df.reset_index()
    rename_map = {"timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts",
                  "volume_quote": "volume", "vol": "volume"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    if "volume_base" in df.columns:
        df.drop(columns=["volume_base"], inplace=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["ts"] = df["ts"].dt.tz_localize("UTC") if df["ts"].dt.tz is None else df["ts"].dt.tz_convert("UTC")
    df.dropna(subset=["ts", "close"], inplace=True)
    df.sort_values("ts", inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def batch_get_signal(strat: dict, symbol: str) -> int:
    """Load raw parquet data and compute signal on last candle. Returns 0 or 1."""
    df = load_ohlcv_from_dir(symbol, strat["timeframe"], RAW_DIR)
    if df.empty:
        return -1
    key   = _signal_key(strat)
    entry = SIGNAL_REGISTRY.get(key)
    if entry is None:
        return -1
    arr = {
        "open":   df["open"].values.astype(np.float32),
        "high":   df["high"].values.astype(np.float32),
        "low":    df["low"].values.astype(np.float32),
        "close":  df["close"].values.astype(np.float32),
        "volume": df["volume"].values.astype(np.float32) if "volume" in df.columns else np.zeros(len(df), dtype=np.float32),
        "ts":     df["ts"].values,
    }
    params  = _signal_params(strat, entry)
    signals = entry["fn"](arr, **params, live_trading=True)
    if signals is None or len(signals) == 0:
        return 0
    last = int(signals[-1])
    return 1 if last != 0 else 0


# =============================================================================
# HELPERS — batch pipeline regime
# =============================================================================

def batch_get_regime(symbol: str, signal_ts: pd.Timestamp) -> tuple:
    """Batch pipeline: precompute_indicators + lookup_indicators + is_trending.
    Returns (regime_str, ts_of_daily_candle_used)."""
    from config.settings import ACCOUNTS
    config     = ACCOUNTS[ACCOUNT_NUMBER]
    windows    = {k: v["window"]    for k, v in config["regime_indicators"].items() if v.get("enabled")}
    thresholds = {k: v["threshold"] for k, v in config["regime_indicators"].items() if v.get("enabled")}
    mode       = config["regime_combine_mode"]

    df = load_ohlcv_from_dir(symbol, REGIME_TIMEFRAME, RAW_DIR)
    if df.empty:
        return "no_data", None

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
# HELPERS — regime filter (shared)
# =============================================================================

def apply_regime_filter(strat: dict, baseline_signal: int, regime: str) -> int:
    """Apply regime bin flags via PositionSizer. Returns 0 (blocked) or baseline_signal."""
    if baseline_signal <= 0:
        return 0
    _, meta = _position_sizer.calculate_adjusted_amount(
        base_amount   = 1.0,
        strat         = strat,
        market_regime = regime,
    )
    return baseline_signal if not meta["blocked"] else 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_test_1h = load_ohlcv_from_dir("BTCUSDT", "1H", RAW_DIR)
    df_test_1d = load_ohlcv_from_dir("BTCUSDT", REGIME_TIMEFRAME, RAW_DIR)
    print(f"RAW_DIR         : {RAW_DIR}")
    print(f"BTCUSDT 1H       last candle — raw : {df_test_1h['ts'].iloc[-1]}")
    print(f"BTCUSDT {REGIME_TIMEFRAME} last candle — raw : {df_test_1d['ts'].iloc[-1]}")

    strategies = [s for s in STRATEGIES if s.get("active", False)]
    if SELECTED_STRATEGIES:
        strategies = [s for s in strategies if s["id"] in SELECTED_STRATEGIES]

    print(f"\n{'='*160}")
    print(f"  SIGNAL VALIDATION — Broker (trading) vs Parquet (batch)")
    print(f"  Strategies : {[s['id'] for s in strategies]}")
    print(f"  Regime TF  : {REGIME_TIMEFRAME} | Account: {ACCOUNT_NUMBER}")
    print(f"{'='*160}\n")

    col = f"  {'STRATEGY':<30} {'SYMBOL':<14} | {'TS_BROKER':<22} {'BASE':>5} {'REGIME':<10} {'FINAL':>6} | {'TS_PARQUET':<22} {'BASE':>5} {'TS_REG_PARQUET':<22} {'REGIME':<10} {'FINAL':>6} | {'MATCH'}"
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

        # Pre-fetch broker data once per strategy (reused per symbol inside detect_signals_for_strategy)
        broker_data   = fetch_ohlcv_data(symbols, timeframe)
        _sample_sym   = symbols[0]
        _df_broker_s  = broker_data.get(_sample_sym)
        _df_raw_s     = load_ohlcv_from_dir(_sample_sym, timeframe, RAW_DIR)
        _df_broker_r  = fetch_ohlcv_data([_sample_sym], REGIME_TIMEFRAME).get(_sample_sym)
        _df_raw_r     = load_ohlcv_from_dir(_sample_sym, REGIME_TIMEFRAME, RAW_DIR)
        _ts_broker_s  = normalize_live_ohlcv(_df_broker_s).index[-1] if _df_broker_s is not None and not _df_broker_s.empty else "N/A"
        _ts_broker_r  = normalize_live_ohlcv(_df_broker_r).index[-1] if _df_broker_r is not None and not _df_broker_r.empty else "N/A"
        _ts_raw_s     = pd.Timestamp(_df_raw_s["ts"].iloc[-1])        if not _df_raw_s.empty  else "N/A"
        _ts_raw_r     = pd.Timestamp(_df_raw_r["ts"].iloc[-1])        if not _df_raw_r.empty  else "N/A"

        for symbol in symbols:

            # ── TRADING: broker ───────────────────────────────────────────
            df_broker = broker_data.get(symbol)
            if df_broker is None or df_broker.empty:
                print(f"  {sid:<30} {symbol:<14} | ⚠️  no broker data")
                n_nodata += 1
                continue

            df_norm    = normalize_live_ohlcv(df_broker)
            ts_trading = pd.Timestamp(df_to_arrays_live(df_norm)["ts"][-1])

            base_trading = trading_get_signal(strat, symbol)
            reg_trading  = get_symbol_regime(symbol, timeframe) if base_trading >= 0 else "no_data"
            fin_trading  = apply_regime_filter(strat, base_trading, reg_trading)

            # ── BATCH: raw ───────────────────────────────────────────────
            df_raw = load_ohlcv_from_dir(symbol, timeframe, RAW_DIR)
            if df_raw.empty:
                print(f"  {sid:<30} {symbol:<14} | ⚠️  no raw data")
                n_nodata += 1
                continue

            ts_batch                = pd.Timestamp(df_raw["ts"].iloc[-1])
            base_batch              = batch_get_signal(strat, symbol)
            reg_batch, ts_reg_batch = batch_get_regime(symbol, ts_batch)
            fin_batch               = apply_regime_filter(strat, base_batch, reg_batch)

            # ── COMPARE ───────────────────────────────────────────────────
            if base_trading == -1 or base_batch == -1:
                m_base   = "⚠️"
                m_regime = "⚠️"
                m_final  = "⚠️"
                n_nodata += 1
            else:
                m_base   = "✅" if base_trading == base_batch else "❌"
                m_regime = "✅" if reg_trading  == reg_batch  else "❌"
                m_final  = "✅" if fin_trading  == fin_batch  else "❌"
                if m_base == "✅" and m_regime == "✅" and m_final == "✅":
                    n_match += 1
                else:
                    n_mismatch += 1

            ts_t_str     = str(ts_trading)[:19]
            ts_b_str     = str(ts_batch)[:19]
            ts_reg_b_str = str(ts_reg_batch)[:19] if ts_reg_batch else "—"

            print(
                f"  {sid:<30} {symbol:<14} | "
                f"{ts_t_str:<22} {base_trading:>5} {reg_trading:<10} {fin_trading:>6} | "
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