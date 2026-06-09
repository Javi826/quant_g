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
    #"31_orderblocks_long_15m",
]

# =============================================================================
# ACCOUNT CONFIG
# =============================================================================
ACCOUNT_NUMBER = "E1"
STRATEGIES_SET = "00"

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
from market_regime.regime_classifier import get_symbol_regime
from market_regime.position_sizer    import PositionSizer
from market_data.data_utils          import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live

# --- Batch pipeline ---
from shared_batch_regime.regime_core        import REGIME_TIMEFRAME
from shared_batch_regime.regime_core        import precompute_indicators, lookup_ma_batch, classify_market_regime
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY

# --- Shared ---
from config.strategies_00        import STRATEGIES
from config.settings             import ACCOUNTS
from market_regime.regime_classifier import configure_regime

# =============================================================================
# REGIME CONFIG — from ACCOUNTS settings (single source of truth)
# =============================================================================
MA_WINDOW = ACCOUNTS[ACCOUNT_NUMBER]["regime_ma_window"]

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
# HELPERS — signal key / params
# =============================================================================

def _signal_key(strat: dict) -> str:
    """Extract SIGNAL_REGISTRY key from strategy name. e.g. 'reversal_long_1H' -> 'reversal_long'"""
    return "_".join(strat["name"].split("_")[:-1])


def _signal_params(strat: dict, registry_entry: dict) -> dict:
    """Extract signal params from strat dict using registry param keys."""
    return {k: strat[k] for k in registry_entry["params"]}


# =============================================================================
# HELPERS — trading pipeline
# =============================================================================

def trading_get_signal_and_regime(strat: dict, symbol: str, arr: dict) -> tuple[int, str]:
    """Compute baseline signal and regime independently.
    Returns (base_signal, regime)."""
    signals = detect_signals_for_strategy(strat, [symbol], None, regime_enabled=False)
    regime  = get_symbol_regime(symbol, strat["timeframe"], arr)
    base    = 1 if signals else 0
    return base, regime


# =============================================================================
# HELPERS — batch pipeline
# =============================================================================

def load_ohlcv_from_dir(symbol: str, timeframe: str, data_dir: str) -> pd.DataFrame:
    """Load OHLCV parquet from a custom directory."""
    path = os.path.join(data_dir, f"{symbol}_{timeframe}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if df.index.name and df.index.name.lower() in ("timestamp", "ts", "date", "time"):
        df.index.name = "ts"
        df = df.reset_index()
    rename_map = {
        "timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts",
        "volume_quote": "volume", "vol": "volume",
    }
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


def batch_get_signal_and_arr(strat: dict, symbol: str) -> tuple[int, dict | None]:
    """Load raw parquet and compute signal on last candle.
    Returns (signal, arr) where arr is the ohlcv array."""
    df = load_ohlcv_from_dir(symbol, strat["timeframe"], RAW_DIR)
    if df.empty:
        return -1, None
    key   = _signal_key(strat)
    entry = SIGNAL_REGISTRY.get(key)
    if entry is None:
        return -1, None
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
        return 0, arr
    return (1 if int(signals[-1]) != 0 else 0), arr


def batch_get_regime(symbol: str, signal_ts: pd.Timestamp, close_signal: float) -> tuple[str, pd.Timestamp | None]:
    """Batch pipeline: lookup MA on daily candle D-1, classify regime using strategy candle close.
    Returns (regime_str, ts_of_daily_candle_used)."""
    df = load_ohlcv_from_dir(symbol, REGIME_TIMEFRAME, RAW_DIR)
    if df.empty:
        return "no_data", None

    ts_arr, ma_arr = precompute_indicators(df, MA_WINDOW)
    if len(ts_arr) == 0:
        return "no_data", None

    signal_ts_arr = np.array([signal_ts], dtype="datetime64[ns]")
    lookups       = lookup_ma_batch(ts_arr, ma_arr, signal_ts_arr)
    ma_daily      = float(lookups[0]) if not np.isnan(lookups[0]) else None

    idx       = np.searchsorted(ts_arr, signal_ts_arr, side="right") - 1
    ts_regime = pd.Timestamp(ts_arr[idx[0]]) if idx[0] >= 0 else None

    if ma_daily is None:
        return "no_data", ts_regime

    return classify_market_regime(close_signal, ma_daily), ts_regime


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
    print(f"RAW_DIR                  : {RAW_DIR}")
    print(f"BTCUSDT 1H  last candle  : {df_test_1h['ts'].iloc[-1]}")
    print(f"BTCUSDT {REGIME_TIMEFRAME} last candle  : {df_test_1d['ts'].iloc[-1]}")
    print(f"MA_WINDOW                : {MA_WINDOW}  TF={REGIME_TIMEFRAME}")

    strategies = [s for s in STRATEGIES if s.get("active", False)]
    if SELECTED_STRATEGIES:
        strategies = [s for s in strategies if s["id"] in SELECTED_STRATEGIES]

    print(f"\n{'='*160}")
    print(f"  SIGNAL VALIDATION — Broker (trading) vs Parquet (batch)")
    print(f"  Strategies     : {[s['id'] for s in strategies]}")
    print(f"  Regime TF      : {REGIME_TIMEFRAME} | MA_W={MA_WINDOW} | Account: {ACCOUNT_NUMBER} | Set: {STRATEGIES_SET}")
    print(f"{'='*160}\n")

    col = (
        f"  {'STRATEGY':<30} {'SYMBOL':<14} | "
        f"{'TS_BROKER':<22} {'BASE':>5} {'REGIME':<10} {'FINAL':>6} | "
        f"{'TS_PARQUET':<22} {'BASE':>5} {'TS_REG_PARQUET':<22} {'REGIME':<10} {'FINAL':>6} | "
        f"{'MATCH'}"
    )
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

            df_norm    = normalize_live_ohlcv(df_broker)
            arr_trading = df_to_arrays_live(df_norm)
            ts_trading  = pd.Timestamp(arr_trading["ts"][-1])

            base_trading, reg_trading = trading_get_signal_and_regime(strat, symbol, arr_trading)
            fin_trading               = apply_regime_filter(strat, base_trading, reg_trading)

            # ── BATCH: raw ───────────────────────────────────────────────
            base_batch, arr_batch = batch_get_signal_and_arr(strat, symbol)
            if arr_batch is None:
                print(f"  {sid:<30} {symbol:<14} | ⚠️  no raw data")
                n_nodata += 1
                continue

            ts_batch                = pd.Timestamp(arr_batch["ts"][-1])
            close_batch             = float(arr_batch["close"][-1])
            reg_batch, ts_reg_batch = batch_get_regime(symbol, ts_batch, close_batch)
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