#shared/shared_batch_regime/regime_core.py
import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
# =============================================================================
# REGIME CONSTANTS — edit here to change global behaviour
# =============================================================================

REGIME_TIMEFRAME: str = "1Dutc"  # daily timeframe for MA computation — only "1Dutc" is supported

if REGIME_TIMEFRAME != "1Dutc":
    raise ValueError(f"❌ REGIME_TIMEFRAME='{REGIME_TIMEFRAME}' is not supported. Only '1Dutc' is allowed.")

logger.debug(f"  [regime_core] REGIME_TIMEFRAME={REGIME_TIMEFRAME}")

# =============================================================================
# CONSTANTS
# =============================================================================

BINS: list[str] = ["uptrend", "dwtrend"]

BIN_CONDITIONS: list[tuple[str, callable]] = [
    ("uptrend", lambda ctx: ctx["close"] > ctx["ma"]),
    ("dwtrend", lambda ctx: ctx["close"] <= ctx["ma"]),
]

INDICATOR_COMPUTERS: dict[str, callable] = {
    "ma": lambda df, cfg: compute_ma(df["close"].values, cfg["ma_window"]),
}

assert [b for b, _ in BIN_CONDITIONS] == BINS, \
    f"BIN_CONDITIONS keys must match BINS exactly. Got {[b for b,_ in BIN_CONDITIONS]} vs {BINS}"
    

# =============================================================================
# BINS: list[str] = ["uptrend_volatile", "uptrend_quiet", "dwtrend_volatile", "dwtrend_quiet"]
# 
# BIN_CONDITIONS: list[tuple[str, callable]] = [
#     ("uptrend_volatile", lambda ctx: ctx["close"] > ctx["ma"] and ctx["atr_pct"] >  ctx["atr_threshold"]),
#     ("uptrend_quiet",    lambda ctx: ctx["close"] > ctx["ma"] and ctx["atr_pct"] <= ctx["atr_threshold"]),
#     ("dwtrend_volatile", lambda ctx: ctx["close"] <= ctx["ma"] and ctx["atr_pct"] >  ctx["atr_threshold"]),
#     ("dwtrend_quiet",    lambda ctx: ctx["close"] <= ctx["ma"] and ctx["atr_pct"] <= ctx["atr_threshold"]),
# ]
# 
# INDICATOR_COMPUTERS: dict[str, callable] = {
#     "ma":      lambda df, cfg: compute_ma(df["close"].values, cfg["ma_window"]),
#     "atr_pct": lambda df, cfg: compute_atr_pct(df, cfg["atr_period"]),
# }
# 
# =============================================================================
# =============================================================================
# HELPERS
# =============================================================================

def pct_improvement(val: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (val - base) / abs(base) * 100


def compute_ma(close: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average over close prices. Returns NaN for first window-1 values."""
    result = np.full(len(close), np.nan)
    for i in range(window - 1, len(close)):
        result[i] = close[i - window + 1: i + 1].mean()
    return result


def classify_market_regime(context: dict, cfg: dict | None = None) -> str:
    if cfg:
        context = {**context, **cfg}
    required_indicators = list(INDICATOR_COMPUTERS.keys())
    if any(context.get(k) is None or (isinstance(context.get(k), float) and np.isnan(context[k])) for k in required_indicators):
        return BINS[-1]
    for bin_name, condition in BIN_CONDITIONS:
        if condition(context):
            return bin_name
    return BINS[-1]

# =============================================================================
# COMBO LABEL
# =============================================================================

def combo_label(indicator_cfg: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in indicator_cfg.items())

# =============================================================================
# OHLCV LOADER
# =============================================================================

def load_ohlcv_raw(symbol: str, data_folder: str) -> pd.DataFrame:
    """Load raw OHLCV data for a symbol on REGIME_TIMEFRAME from the given data folder."""
    path = os.path.join(data_folder, f"{symbol}_{REGIME_TIMEFRAME}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if df.index.name and df.index.name.lower() in ("timestamp", "ts", "date", "time"):
        df.index.name = "ts"
        df = df.reset_index()
    if "volume_base" in df.columns and "volume_quote" in df.columns:
        df.drop(columns=["volume_base"], inplace=True)
    rename_map = {
        "timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts",
        "volume_quote": "volume", "vol": "volume",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["ts"] = df["ts"].dt.tz_localize("UTC") if df["ts"].dt.tz is None else df["ts"].dt.tz_convert("UTC")
    df.dropna(subset=["ts"], inplace=True)
    df.sort_values("ts", inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df

# =============================================================================
# TIME-SERIES PRECOMPUTATION
# =============================================================================

def precompute_indicators(df: pd.DataFrame, cfg: dict) -> dict:
    ts         = df["ts"].values
    indicators = {key: fn(df, cfg) for key, fn in INDICATOR_COMPUTERS.items()}
    valid      = np.ones(len(ts), dtype=bool)
    for arr in indicators.values():
        valid &= ~np.isnan(arr)
    return {
        "ts": np.array(ts[valid], dtype="datetime64[ns]"),
        **{key: arr[valid] for key, arr in indicators.items()},
    }

# =============================================================================
# LOOKUP
# =============================================================================

def lookup_indicator_batch(
    ts_arr:        np.ndarray,
    indicator_arr: np.ndarray,
    signal_ts_arr: np.ndarray,
    debug_n:       int = 0,
) -> np.ndarray:

    ts_lookup = signal_ts_arr.astype("datetime64[ns]")
    ts_fixed  = (ts_lookup.astype("datetime64[D]") - np.timedelta64(1, "D")).astype("datetime64[ns]")
    idxs      = np.searchsorted(ts_arr, ts_fixed, side="right") - 1

    valid = idxs >= 0
    out   = np.full(len(idxs), np.nan)
    out[valid] = indicator_arr[idxs[valid]]

    if debug_n > 0:
        print(f"\n  [LOOKUP DEBUG] first {min(debug_n, len(idxs))} signals — raw timestamps only")
        print(f"  {'SIGNAL_TS':<30} {'DAILY_CANDLE_TS (used for MA)'}")
        print(f"  {'─'*65}")
        for i in range(min(debug_n, len(idxs))):
            sig_ts    = str(signal_ts_arr[i])
            daily_idx = idxs[i]
            candle_ts = str(ts_arr[daily_idx]) if daily_idx >= 0 else "N/A"
            print(f"  {sig_ts:<30} {candle_ts}")

    return out
def apply_regime_filter(
    signals: np.ndarray,
    arr: dict,
    sym_cache: dict | None,
    cfg: dict,
    bins_to_filter: list[str],
) -> np.ndarray:
    """
    Zero out signals whose D-1 market regime is not in bins_to_filter.
    Returns the same array, mutated in place, for convenience.
    """
    if sym_cache is None:
        return signals

    signal_idxs = np.nonzero(signals)[0]
    if signal_idxs.size == 0:
        return signals

    signal_ts = arr["ts"][signal_idxs]
    lookups   = {
        key: lookup_indicator_batch(sym_cache["ts"], sym_cache[key], signal_ts)
        for key in sym_cache if key != "ts"
    }

    for i, idx in enumerate(signal_idxs):
        close_idx = idx - 1 if idx > 0 else idx
        context   = {"close": float(arr["close"][close_idx])}
        for key, values in lookups.items():
            context[key] = float(values[i]) if not np.isnan(values[i]) else None
        if classify_market_regime(context, cfg=cfg) not in bins_to_filter:
            signals[idx] = 0

    return signals


def classify_signal_regimes(
    signals: np.ndarray,
    arr: dict,
    sym_cache: dict | None,
    cfg: dict,
) -> dict[int, str]:
    """
    Classify the market regime for each non-zero signal index (D-1 lookup).
    Returns {signal_idx: regime_label}. Used when signals must be split per
    bin rather than filtered out (e.g. regime calibration).
    """
    if sym_cache is None:
        return {}

    signal_idxs = np.nonzero(signals)[0]
    if signal_idxs.size == 0:
        return {}

    signal_ts = arr["ts"][signal_idxs]
    lookups   = {
        key: lookup_indicator_batch(sym_cache["ts"], sym_cache[key], signal_ts)
        for key in sym_cache if key != "ts"
    }

    regimes = {}
    for i, idx in enumerate(signal_idxs):
        close_idx = idx - 1 if idx > 0 else idx
        context   = {"close": float(arr["close"][close_idx])}
        for key, values in lookups.items():
            context[key] = float(values[i]) if not np.isnan(values[i]) else None
        regimes[int(idx)] = classify_market_regime(context, cfg=cfg)

    return regimes


def compute_atr_pct(df: pd.DataFrame, period: int) -> np.ndarray:
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr  = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.full(len(tr), np.nan)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(close > 0, atr / close, np.nan)