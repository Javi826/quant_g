

import os
import sys

# =============================================================================
# PATHS
# =============================================================================
BITGET_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget"))

SYMBOLS_LIVE_DIR = os.path.join(BITGET_ROOT, "BOT_trading", "symbols_live", "E1")

SIGNALS_DIR             = os.path.join(BITGET_ROOT, "signals")
SHARED_DIR              = os.path.join(BITGET_ROOT, "shared")
SHARED_BATCH_REGIME_DIR = os.path.join(BITGET_ROOT, "shared", "shared_batch_regime")
SHARED_TRADING_DIR      = os.path.join(BITGET_ROOT, "shared", "shared_trading_batch_regime")
SHARED_BATCHS_DIR       = os.path.join(BITGET_ROOT, "shared", "shared_batchs")
BOT_BATCH_DIR           = os.path.join(BITGET_ROOT, "BOT_batch_00")
BOT_TRADING_DIR         = os.path.join(BITGET_ROOT, "BOT_trading")

for p in [SIGNALS_DIR, SHARED_DIR, SHARED_BATCH_REGIME_DIR, SHARED_TRADING_DIR,
          SHARED_BATCHS_DIR, BOT_BATCH_DIR, BOT_TRADING_DIR]:
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
import logging
import warnings
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(message)s")
warnings.filterwarnings("ignore", category=UserWarning)

from shared_batch_regime.regime_core        import REGIME_TIMEFRAME, precompute_indicators, lookup_indicator_batch, classify_market_regime
from shared_batch_regime.config_paths       import DATA_FOLDER_OOS1
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.regime.regime_module     import load_regime_bins
from market_regime.position_sizer           import PositionSizer
from market_regime.regime_classifier        import configure_regime

RAW_DIR = DATA_FOLDER_OOS1

# =============================================================================
# CONFIGURATION
# =============================================================================
STRATEGY_ID    = "31_orderblocks_long_15m"
INSPECT_TS     = "2026-06-16 04:30:00"       # UTC
STRATEGIES_SET = "00"
ACCOUNT_NUMBER = "00"

# Set to a list of symbols to print raw indicator values at inspect_ts.
# Use [] to disable, or ["DOGEUSDT", "SOLUSDT"] to compare a diverging pair.
# "AUTO" will automatically show all symbols where SIG_LIVE != SIG_BATCH.
DEBUG_SYMBOLS  = "AUTO"

# =============================================================================
# INIT
# =============================================================================
configure_regime(ACCOUNT_NUMBER)
_position_sizer = PositionSizer(logger=None)

REGIME_BINS_PATH = os.path.join(
    BOT_BATCH_DIR, "strategies_files", f"regime_bins_{STRATEGIES_SET}.py"
)

# =============================================================================
# HELPERS
# =============================================================================

def load_symbols(strategy_id: str, timeframe: str) -> list:
    path = os.path.join(SYMBOLS_LIVE_DIR, f"symbols_live_{strategy_id}_{timeframe}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{symbol}_{timeframe}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if df.index.name and df.index.name.lower() in ("timestamp", "ts", "date", "time"):
        df.index.name = "ts"
        df = df.reset_index()
    rename_map = {"timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
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


def get_signal_at(strat: dict, symbol: str, inspect_ts: pd.Timestamp) -> tuple[int, int, float | None]:
    """Compute baseline signal at inspect_ts for both live_trading modes.
    Returns (signal_live, signal_batch, close_at_ts)."""
    key   = "_".join(strat["name"].split("_")[:-1])
    entry = SIGNAL_REGISTRY.get(key)
    if entry is None:
        return -1, -1, None

    df = load_ohlcv(symbol, strat["timeframe"])
    if df.empty:
        return -1, -1, None

    df_up_to = df[df["ts"] < inspect_ts]
    if df_up_to.empty:
        return -1, -1, None

    arr = {
        "open":  df_up_to["open"].values.astype(np.float64),
        "high":  df_up_to["high"].values.astype(np.float64),
        "low":   df_up_to["low"].values.astype(np.float64),
        "close": df_up_to["close"].values.astype(np.float64),
        "ts":    df_up_to["ts"].values,
    }
    if "volume" in df_up_to.columns:
        arr["volume"] = df_up_to["volume"].values.astype(np.float64)

    params        = {k: strat[k] for k in entry["params"]}
    signals_live  = entry["fn"](arr, **params, live_trading=True)
    signals_batch = entry["fn"](arr, **params, live_trading=False)

    close_val    = float(df_up_to["close"].iloc[-1])
    signal_live  = (1 if int(signals_live[-1])  != 0 else 0) if signals_live  is not None and len(signals_live)  > 0 else 0
    signal_batch = (1 if int(signals_batch[-1]) != 0 else 0) if signals_batch is not None and len(signals_batch) > 0 else 0
    return signal_live, signal_batch, close_val


def get_regime_at(symbol: str, inspect_ts: pd.Timestamp, close_val: float) -> tuple[str, float | None, float | None]:
    """Compute regime at inspect_ts. Returns (regime, close_used, ma_value)."""
    df = load_ohlcv(symbol, REGIME_TIMEFRAME)
    if df.empty:
        return "no_data", None, None

    from config.settings import ACCOUNTS
    indicator_cfg = {"ma_window": ACCOUNTS[ACCOUNT_NUMBER]["regime_ma_window"]}

    cache  = precompute_indicators(df, indicator_cfg)
    ts_arr = cache["ts"]
    if len(ts_arr) == 0:
        return "no_data", None, None

    inspect_ts_arr = np.array([inspect_ts.tz_localize(None)], dtype="datetime64[ns]")
    lookups        = lookup_indicator_batch(ts_arr, cache["ma"], inspect_ts_arr)
    ma_val         = float(lookups[0]) if not np.isnan(lookups[0]) else None

    if ma_val is None:
        return "no_data", close_val, None

    context = {"close": close_val, "ma": ma_val}
    regime  = classify_market_regime(context)
    return regime, close_val, ma_val


def apply_regime_filter(strat: dict, baseline_signal: int, regime: str) -> int:
    if baseline_signal <= 0:
        return 0
    _, meta = _position_sizer.calculate_adjusted_amount(
        base_amount   = 1.0,
        strat         = strat,
        market_regime = regime,
    )
    return baseline_signal if not meta["blocked"] else 0


def debug_indicators(strat: dict, symbol: str, inspect_ts: pd.Timestamp) -> None:
    """Print raw indicator values at inspect_ts (live idx) and inspect_ts-1candle (batch idx)
    to explain why SIG_LIVE and SIG_BATCH differ for this symbol."""
    from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY

    key   = "_".join(strat["name"].split("_")[:-1])
    entry = SIGNAL_REGISTRY.get(key)
    if entry is None:
        print(f"  [{symbol}] signal fn not found in registry")
        return

    df = load_ohlcv(symbol, strat["timeframe"])
    if df.empty:
        print(f"  [{symbol}] no OHLCV data")
        return

    df_up_to = df[df["ts"] < inspect_ts].reset_index(drop=True)
    if len(df_up_to) < 2:
        print(f"  [{symbol}] not enough data")
        return

    # Import _compute_indicators from the signal module directly
    import importlib
    sig_module = entry["fn"].__module__
    mod        = importlib.import_module(sig_module)
    if not hasattr(mod, "_compute_indicators"):
        print(f"  [{symbol}] _compute_indicators not found in {sig_module}")
        return

    arr = {
        "open":  df_up_to["open"].values.astype(np.float64),
        "high":  df_up_to["high"].values.astype(np.float64),
        "low":   df_up_to["low"].values.astype(np.float64),
        "close": df_up_to["close"].values.astype(np.float64),
    }

    open_, high, low, close, ma50, rsi, adx, plus_di, minus_di = mod._compute_indicators(arr)

    # live  → last index (inspect_ts candle)
    # batch → second-to-last index (inspect_ts - 1 candle, because of np.roll(signal, 1))
    live_idx  = len(close) - 1
    batch_idx = len(close) - 2

    def _fmt(arr, idx):
        v = arr[idx]
        return f"{v:.4f}" if not np.isnan(v) else "NaN"

    def _ok(val, condition):
        return "✓" if condition else "✗"

    print(f"\n  {'─'*90}")
    print(f"  INDICATOR DEBUG — {symbol} | live_idx={live_idx} (ts={df_up_to['ts'].iloc[live_idx]}) | batch_idx={batch_idx} (ts={df_up_to['ts'].iloc[batch_idx]})")
    print(f"  {'─'*90}")
    print(f"  {'INDICATOR':<14} {'LIVE_VAL':>12} {'LIVE_OK':>8} {'BATCH_VAL':>12} {'BATCH_OK':>9}")
    print(f"  {'─'*60}")

    c_live  = close[live_idx]
    c_batch = close[batch_idx]

    rows = [
        ("ma50",     ma50,     lambda v, c: not np.isnan(v) and c > v,      c_live,  c_batch),
        ("rsi",      rsi,      lambda v, _: not np.isnan(v) and v > 50,     c_live,  c_batch),
        ("adx",      adx,      lambda v, _: not np.isnan(v) and v > 20,     c_live,  c_batch),
        ("plus_di",  plus_di,  lambda v, _: not np.isnan(v),                c_live,  c_batch),
        ("minus_di", minus_di, lambda v, _: not np.isnan(v),                c_live,  c_batch),
    ]

    for name, arr_ind, cond, cl, cb in rows:
        vl = arr_ind[live_idx]
        vb = arr_ind[batch_idx]
        print(
            f"  {name:<14} {_fmt(arr_ind, live_idx):>12} {_ok(vl, cond(vl, cl)):>8} "
            f"{_fmt(arr_ind, batch_idx):>12} {_ok(vb, cond(vb, cb)):>9}"
        )

    # adx combined condition: adx>20 AND plus_di > minus_di
    adx_live_ok  = (not np.isnan(adx[live_idx]))  and adx[live_idx]  > 20 and plus_di[live_idx]  > minus_di[live_idx]
    adx_batch_ok = (not np.isnan(adx[batch_idx])) and adx[batch_idx] > 20 and plus_di[batch_idx] > minus_di[batch_idx]
    print(f"  {'adx_combined':<14} {'':>12} {_ok(None, adx_live_ok):>8} {'':>12} {_ok(None, adx_batch_ok):>9}")
    print(f"  {'close':<14} {c_live:>12.5g} {'':>8} {c_batch:>12.5g}")
    print(f"  {'─'*90}\n")


# =============================================================================
# DATASET COMPARISON
# =============================================================================

def debug_compare_datasets(symbols: list, inspect_ts: pd.Timestamp) -> None:
    """Compare MA and close D-1 between CRYPTO_FULL_DIR and DATA_FOLDER_OOS1 for each symbol."""
    from shared_batch_regime.config_paths import CRYPTO_FULL_DIR, DATA_FOLDER_OOS1
    from shared_batch_regime.regime_core import classify_market_regime
    from config.settings import ACCOUNTS

    indicator_cfg = {"ma_window": ACCOUNTS[ACCOUNT_NUMBER]["regime_ma_window"]}

    print(f"\n{'='*110}")
    print(f"  DATASET COMPARISON — MA source check at {inspect_ts}")
    print(f"{'='*110}")
    print(f"  {'SYMBOL':<14} {'DATASET':<14} {'DAILY_TS':<25} {'CLOSE_D1':>10} {'MA':>10} {'REGIME':<12}")
    print(f"  {'-'*100}")

    for symbol in symbols:
        for label, folder in [("CRYPTO_FULL", CRYPTO_FULL_DIR), ("OOS1", DATA_FOLDER_OOS1)]:
            path = os.path.join(folder, f"{symbol}_1Dutc.parquet")
            if not os.path.exists(path):
                print(f"  {symbol:<14} {label:<14} ⚠️  not found")
                continue

            df = pd.read_parquet(path)
            df.columns = [c.lower().strip() for c in df.columns]
            if df.index.name and df.index.name.lower() in ("timestamp", "ts", "date", "time"):
                df.index.name = "ts"
                df = df.reset_index()
            rename_map = {"timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts"}
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            df["ts"]    = pd.to_datetime(df["ts"], errors="coerce")
            df["ts"]    = df["ts"].dt.tz_localize("UTC") if df["ts"].dt.tz is None else df["ts"].dt.tz_convert("UTC")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.sort_values("ts").drop_duplicates(subset=["ts"]).dropna(subset=["close"]).reset_index(drop=True)

            cache       = precompute_indicators(df, indicator_cfg)
            inspect_arr = np.array([inspect_ts.tz_localize(None)], dtype="datetime64[ns]")
            lookups     = lookup_indicator_batch(cache["ts"], cache["ma"], inspect_arr)
            ma_val      = float(lookups[0]) if not np.isnan(lookups[0]) else None

            d1_ts    = (inspect_ts.tz_localize(None).to_datetime64().astype("datetime64[D]") - np.timedelta64(1, "D")).astype("datetime64[ns]")
            idx      = int(np.searchsorted(cache["ts"], d1_ts, side="right") - 1)
            daily_ts = str(cache["ts"][idx])[:19] if idx >= 0 else "—"

            df_before = df[df["ts"] < inspect_ts]
            close_d1  = float(df_before["close"].iloc[-1]) if not df_before.empty else None
            regime    = classify_market_regime({"close": close_d1, "ma": ma_val}) if (close_d1 and ma_val) else "—"

            close_str = f"{close_d1:.4f}" if close_d1 is not None else "—"
            ma_str    = f"{ma_val:.4f}"   if ma_val   is not None else "—"
            print(f"  {symbol:<14} {label:<14} {daily_ts:<25} {close_str:>10} {ma_str:>10} {regime:<12}")

        print(f"  {'-'*100}")

    print(f"  {'='*110}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    from config.strategies_00 import STRATEGIES

    strat = next((s for s in STRATEGIES if s["id"] == STRATEGY_ID), None)
    if strat is None:
        print(f"  ⚠️  Strategy '{STRATEGY_ID}' not found in STRATEGIES.")
        return

    inspect_ts  = pd.Timestamp(INSPECT_TS, tz="UTC")
    bins_filter = load_regime_bins(REGIME_BINS_PATH, STRATEGY_ID)
    symbols     = load_symbols(STRATEGY_ID, strat["timeframe"])

    if not symbols:
        print(f"  ⚠️  No symbols found for {STRATEGY_ID}")
        return

    print(f"\n{'='*150}")
    print(f"  HISTORICAL SIGNAL + REGIME INSPECTOR")
    print(f"  Strategy     : {STRATEGY_ID}")
    print(f"  Timestamp    : {INSPECT_TS} UTC")
    print(f"  Bins filter  : {bins_filter}")
    print(f"  Regime TF    : {REGIME_TIMEFRAME} | Account: {ACCOUNT_NUMBER} | Set: {STRATEGIES_SET}")
    print(f"{'='*150}")
    print(
        f"  {'SYMBOL':<14} {'SIG_LIVE':>9} {'SIG_BATCH':>10} {'CLOSE':>12} {'MA':>12} "
        f"{'REGIME':<12} {'PASS':>5} {'FINAL_LIVE':>11} {'FINAL_BATCH':>12}"
    )
    print(f"  {'-'*140}")

    diverging = []

    for symbol in symbols:
        sig_live, sig_batch, close_val = get_signal_at(strat, symbol, inspect_ts)

        if sig_live == -1 or close_val is None:
            print(f"  {symbol:<14} {'⚠️ no data'}")
            continue

        regime, close_used, ma_val = get_regime_at(symbol, inspect_ts, close_val)

        passes      = "—"
        final_live  = sig_live
        final_batch = sig_batch
        if regime != "no_data" and bins_filter:
            final_live  = apply_regime_filter(strat, sig_live,  regime)
            final_batch = apply_regime_filter(strat, sig_batch, regime)
            passes      = "✓" if (sig_live > 0 and final_live > 0) else ("✗" if sig_live > 0 else "—")

        close_str = f"{close_used:.5g}" if close_used is not None else "—"
        ma_str    = f"{ma_val:.5g}"     if ma_val    is not None else "—"

        print(
            f"  {symbol:<14} {sig_live:>9} {sig_batch:>10} {close_str:>12} {ma_str:>12} "
            f"{regime:<12} {passes:>5} {final_live:>11} {final_batch:>12}"
        )

        if sig_live != sig_batch:
            diverging.append(symbol)

    print(f"  {'='*140}\n")

    # --- indicator debug ---
    debug_targets = diverging if DEBUG_SYMBOLS == "AUTO" else (DEBUG_SYMBOLS or [])
    if debug_targets:
        print(f"\n{'='*90}")
        print(f"  INDICATOR DEBUG — symbols where SIG_LIVE != SIG_BATCH")
        print(f"  live_idx  = candle at inspect_ts (what live_trading=True sees)")
        print(f"  batch_idx = candle at inspect_ts - 1 (what live_trading=False sees after np.roll)")
        print(f"{'='*90}")
        for symbol in debug_targets:
            debug_indicators(strat, symbol, inspect_ts)

    debug_compare_datasets(symbols, inspect_ts)


if __name__ == "__main__":
    main()