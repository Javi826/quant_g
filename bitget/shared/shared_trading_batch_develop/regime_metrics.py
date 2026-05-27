#shared/shared_trading_batch_develop/regime_metrics.py
"""
Regime indicator calculations and lookup utilities.

Provides:
  - calc_atr_norm   : ATR (Wilder smoothing) normalised by close price
  - calc_er         : Efficiency Ratio
  - calc_hurst      : Hurst exponent (log-variance method)
  - calc_all_metrics: convenience wrapper returning all indicators
  - precompute_indicators : build time-series arrays for a full OHLCV DataFrame
  - lookup_indicators     : retrieve indicator values for a given signal timestamp
"""
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# =============================================================================
# SINGLE-VALUE CALCULATORS
# =============================================================================

def calc_atr_norm(
    high:   np.ndarray,
    low:    np.ndarray,
    close:  np.ndarray,
    window: int = 14,
) -> float:
    """
    ATR normalised by current close, using Wilder smoothing.
    Seed: SMA of first `window` true ranges.
    """
    needed = window + 1
    if len(close) < needed:
        return np.nan

    h  = high[-needed:]
    l  = low[-needed:]
    c  = close[-needed:]
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )

    atr = float(np.mean(tr[:window]))
    for i in range(window, len(tr)):
        atr = (atr * (window - 1) + tr[i]) / window

    current_price = float(c[-1])
    if current_price <= 0 or np.isnan(atr):
        return np.nan

    result = atr / current_price
    return float(result) if 0.0 <= result <= 1.0 else np.nan


def calc_er(close: np.ndarray, window: int = 14) -> float:
    """Efficiency Ratio: net directional change / sum of absolute changes."""
    if len(close) < window + 1:
        return np.nan

    series       = close[-(window + 1):]
    total_change = float(np.sum(np.abs(np.diff(series))))

    if total_change == 0.0:
        return 0.0

    er = abs(float(series[-1]) - float(series[0])) / total_change
    return float(np.clip(er, 0.0, 1.0))


def calc_hurst(close: np.ndarray, window: int = 30) -> float:
    """
    Hurst exponent via log-variance of aggregated log-returns.
    Returns value in [0, 1]:  <0.5 mean-reverting | ~0.5 random | >0.5 trending.
    """
    if len(close) < window:
        return np.nan

    log_returns = np.diff(np.log(close[-window:] + 1e-10))
    if len(log_returns) < 4:
        return np.nan

    log_lags, log_vars = [], []
    max_lag = max(3, len(log_returns) // 2)

    for lag in range(2, max_lag):
        agg = np.array([
            log_returns[i:i + lag].sum()
            for i in range(0, len(log_returns) - lag, lag)
        ])
        if len(agg) < 2:
            continue
        var = float(np.var(agg))
        if var <= 0.0:
            continue
        log_lags.append(np.log(lag))
        log_vars.append(np.log(var))

    if len(log_lags) < 2:
        return np.nan

    slope = float(np.polyfit(log_lags, log_vars, 1)[0])
    return float(np.clip(slope / 2.0, 0.0, 1.0))


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

_CALC_FN = {
    "atr_norm": lambda high, low, close, w: calc_atr_norm(high, low, close, w),
    "er":       lambda high, low, close, w: calc_er(close, w),
    "hurst":    lambda high, low, close, w: calc_hurst(close, w),
}


def calc_all_metrics(
    ohlc:       dict[str, np.ndarray],
    er_window:  int = 14,
    atr_window: int = 14,
    hurst_window: int = 30,
) -> dict[str, float]:
    """Return all indicator values for the latest candle."""
    close = ohlc["close"]
    high  = ohlc["high"]
    low   = ohlc["low"]
    return {
        "atr_norm": calc_atr_norm(high, low, close, atr_window),
        "er":       calc_er(close, er_window),
        "hurst":    calc_hurst(close, hurst_window),
    }


# =============================================================================
# TIME-SERIES PRECOMPUTATION
# =============================================================================

def precompute_indicators(
    df:      pd.DataFrame,
    windows: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute rolling indicator arrays for a full OHLCV DataFrame.

    Parameters
    ----------
    df      : DataFrame with columns ts, high, low, close
    windows : {indicator_key: window}  — only enabled indicators

    Returns
    -------
    ts_arr     : np.ndarray[datetime64[ns]]
    values_arr : {indicator_key: np.ndarray[float]}

    Only rows where ALL indicators produced a valid (non-NaN) value are kept.
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    ts    = df["ts"].values

    min_win = (
        max((w + 1 if k != "hurst" else w) for k, w in windows.items())
        if windows else 1
    )

    ts_list:     list            = []
    value_lists: dict[str, list] = {k: [] for k in windows}

    for i in range(min_win, len(close)):
        row_values: dict[str, float] = {}
        valid = True

        for key, w in windows.items():
            val = _CALC_FN[key](high[: i + 1], low[: i + 1], close[: i + 1], w)
            if np.isnan(val):
                valid = False
                break
            row_values[key] = val

        if not valid:
            continue

        ts_list.append(ts[i])
        for key in windows:
            value_lists[key].append(row_values[key])

    ts_arr     = np.array(ts_list, dtype="datetime64[ns]")
    values_arr = {k: np.array(v, dtype=float) for k, v in value_lists.items()}
    return ts_arr, values_arr


# =============================================================================
# LOOKUP
# =============================================================================

def lookup_indicators(
    ts_arr:     np.ndarray,
    values_arr: dict[str, np.ndarray],
    signal_ts,
) -> dict[str, float | None]:
    """
    Return {indicator_key: value} for the last available row before signal_ts.
    Applies normalize() - 1 day lookahead fix.
    """
    ts = pd.Timestamp(signal_ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    ts = ts.normalize() - pd.Timedelta(days=1)

    idx = np.searchsorted(ts_arr, np.datetime64(ts.value, "ns"), side="right") - 1
    if idx < 0:
        return {k: None for k in values_arr}

    return {k: float(arr[idx]) for k, arr in values_arr.items()}