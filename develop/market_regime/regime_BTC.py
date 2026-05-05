#!/usr/bin/env python3
"""
develop/market_regime/regime_predictive_analysis.py

Cross-lag correlation analysis between BTC weekly indicators and system performance.
For each indicator at week T, computes correlation with system profit/winrate at week T+1.
Identifies which market conditions have predictive power over strategy performance.

Usage:
    python regime_predictive_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_market_regime")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))
from regime_common import load_btc_for_timeframe, get_btc_macro_direction

# =============================================================================
# CONFIGURATION
# =============================================================================

TRADES_FOLDER = os.path.join(os.path.dirname(__file__), "..", "brief_trades")
SPLIT_MODE    = "expanding"
SPLIT_BASE    = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
BTC_FOLDER    = os.path.join(SPLIT_BASE, "IS", "crypto_full_IS")

PERIOD_LABELS = [
    ("IS",   "is_regime"),
    ("OOS1", "oos1_regime"),
    ("OOS2", "oos2_regime"),
    ("OOS3", "oos3_regime"),
]

# =============================================================================
# PERIOD_LABELS = [
#     ("IS",   "is_baseline"),
#     ("OOS1", "oos1_baseline"),
#     ("OOS2", "oos2_baseline"),
#     ("OOS3", "oos3_baseline"),
# ]
# =============================================================================

BTC_MA_PERIOD   = 4
INITIAL_CAPITAL = 800
MIN_TRADES_WEEK = 2       # minimum trades in a week to include it
MIN_PERIODS     = 8       # minimum weeks needed to compute correlation
PVALUE_TH       = 0.10    # significance threshold
GAP_THRESHOLD    = 3.0     # minimum pp gap between two groups to flag as predictive
MIN_CONSISTENT   = 3       # minimum periods where gap is in same direction
MIN_OOS_OK       = 3       # minimum OOS periods (of 3) to mark as OK (2 or 3)

def _calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    """Simplified ADX using smoothed directional movement."""
    n = len(close)
    if n < period + 2:
        return np.nan
    tr  = np.maximum(high[1:] - low[1:],
          np.maximum(np.abs(high[1:] - close[:-1]),
                     np.abs(low[1:]  - close[:-1])))
    dm_plus  = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                        np.maximum(high[1:] - high[:-1], 0), 0.0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0), 0.0)
    tr_s  = np.mean(tr[-period:])
    dmp_s = np.mean(dm_plus[-period:])
    dmm_s = np.mean(dm_minus[-period:])
    if tr_s == 0:
        return np.nan
    di_plus  = dmp_s / tr_s * 100
    di_minus = dmm_s / tr_s * 100
    dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 \
         if (di_plus + di_minus) > 0 else 0.0
    return float(dx)


def _calc_trix(close: np.ndarray, period: int) -> float:
    """TRIX: rate of change of triple EMA."""
    n = len(close)
    if n < period * 3:
        return np.nan
    def _ema(arr, p):
        k   = 2 / (p + 1)
        ema = arr[0]
        for x in arr[1:]:
            ema = x * k + ema * (1 - k)
        return ema
    ema1 = _ema(close[-(period * 3):], period)
    ema2 = _ema(close[-(period * 2):], period)
    ema3 = _ema(close[-period:], period)
    return float((ema3 - ema2) / ema2 * 100) if ema2 != 0 else np.nan


def _calc_slope(close: np.ndarray, period: int) -> float:
    """Linear regression slope of close over last `period` bars, normalized by price."""
    if len(close) < period:
        return np.nan
    y = close[-period:]
    x = np.arange(period, dtype=np.float64)
    slope = (np.cov(x, y)[0, 1]) / np.var(x)
    return float(slope / y[-1] * 100) if y[-1] != 0 else np.nan


def _calc_ichimoku_proxy(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
    """
    Simplified Ichimoku proxy: position of price relative to 9d and 26d midlines.
    Returns 1 if price > both, -1 if below both, 0 otherwise.
    """
    n = len(close)
    if n < 27:
        return np.nan
    mid_9  = (np.max(high[-9:])  + np.min(low[-9:]))  / 2
    mid_26 = (np.max(high[-26:]) + np.min(low[-26:])) / 2
    p = close[-1]
    if p > mid_9 and p > mid_26:   return 1.0
    if p < mid_9 and p < mid_26:   return -1.0
    return 0.0


def _calc_consecutive_days(close: np.ndarray) -> float:
    """Count consecutive up (+) or down (-) days ending at last bar."""
    if len(close) < 2:
        return np.nan
    diffs = np.diff(close)
    sign  = np.sign(diffs[-1])
    if sign == 0:
        return 0.0
    count = 0
    for d in reversed(diffs):
        if np.sign(d) == sign:
            count += 1
        else:
            break
    return float(count * sign)


def _calc_btc_eth_corr(btc_df: pd.DataFrame, eth_df: pd.DataFrame,
                        week_start: pd.Timestamp, window: int) -> float:
    """Pearson correlation of BTC vs ETH daily returns over last `window` bars."""
    if eth_df is None:
        return np.nan
    btc_c = btc_df[btc_df['ts'] < week_start]['close'].values[-window-1:]
    eth_c = eth_df[eth_df['ts'] < week_start]['close'].values[-window-1:]
    if len(btc_c) < window or len(eth_c) < window:
        return np.nan
    btc_r = np.diff(np.log(btc_c[-window:]))
    eth_r = np.diff(np.log(eth_c[-window:]))
    if len(btc_r) != len(eth_r) or np.std(btc_r) == 0 or np.std(eth_r) == 0:
        return np.nan
    return float(np.corrcoef(btc_r, eth_r)[0, 1])


def _calc_btc_eth_ret_ratio(btc_df: pd.DataFrame, eth_df: pd.DataFrame,
                             week_start: pd.Timestamp, window: int) -> float:
    """BTC return / ETH return over last `window` bars — >1 means BTC outperforming."""
    if eth_df is None:
        return np.nan
    btc_c = btc_df[btc_df['ts'] < week_start]['close'].values
    eth_c = eth_df[eth_df['ts'] < week_start]['close'].values
    if len(btc_c) < window + 1 or len(eth_c) < window + 1:
        return np.nan
    btc_ret = btc_c[-1] / btc_c[-window-1] - 1
    eth_ret = eth_c[-1] / eth_c[-window-1] - 1
    if abs(eth_ret) < 1e-9:
        return np.nan
    return float(btc_ret / eth_ret)


# =============================================================================
# BTC INDICATOR COMPUTATION
# =============================================================================

def compute_btc_weekly_indicators(btc_1d_df: pd.DataFrame, week_start: pd.Timestamp,
                                   eth_1d_df: pd.DataFrame = None) -> dict:
    """
    40 indicators focused on range, crosses, trend clarity and ranging conditions.
    Uses only closed bars strictly before week_start — no lookahead.
    """
    closed = btc_1d_df[btc_1d_df['ts'] < week_start].copy()
    if len(closed) < 30:
        return None

    close = closed['close'].values.astype(np.float64)
    high  = closed['high'].values.astype(np.float64)
    low   = closed['low'].values.astype(np.float64)
    n     = len(close)

    has_open = 'open' in closed.columns
    if has_open:
        open_ = closed['open'].values.astype(np.float64)

    def _sma(arr, p):
        return arr[-p:].mean() if len(arr) >= p else np.nan

    def _ema_val(arr, p):
        if len(arr) < p:
            return np.nan
        k, e = 2 / (p + 1), arr[-p:].mean()
        for x in arr[-p+1:]:
            e = x * k + e * (1 - k)
        return e

    # -------------------------------------------------------------------------
    # GROUP 1 — Range indicators (10)
    # -------------------------------------------------------------------------

    # 1. Weekly range ratio vs 4-week average
    range_7  = (np.max(high[-7:]) - np.min(low[-7:])) / close[-1] * 100 if n >= 7 else np.nan
    range_28 = (np.max(high[-28:]) - np.min(low[-28:])) / close[-1] * 100 if n >= 28 else np.nan
    range_ratio_4w = range_7 / (range_28 / 4) if n >= 28 and range_28 > 0 else np.nan

    # 2. Range compression (current 7d range / max 8-week range)
    ranges_8w = [(np.max(high[max(0,n-7-7*i):n-7*i]) - np.min(low[max(0,n-7-7*i):n-7*i]))
                 for i in range(8) if n >= 7*(i+1)]
    range_compression = range_7 / (max(ranges_8w) / close[-1] * 100) if ranges_8w and max(ranges_8w) > 0 and n >= 7 else np.nan

    # 3. Inside bars 7d (range within previous day's range)
    inside_bars_7d = float(np.sum(
        (high[-7:] <= high[-8:-1]) & (low[-7:] >= low[-8:-1])
    )) if n >= 8 else np.nan

    # 4. Range expansion (this week / last week)
    range_prev = (np.max(high[-14:-7]) - np.min(low[-14:-7])) / close[-8] * 100 if n >= 14 else np.nan
    range_expansion = range_7 / range_prev if range_prev and range_prev > 0 else np.nan

    # 5. 4-week range normalized by price
    range_4w_pct = range_28

    # 6. Number of inside bars 14d
    inside_bars_14d = float(np.sum(
        (high[-14:] <= high[-15:-1]) & (low[-14:] >= low[-15:-1])
    )) if n >= 15 else np.nan

    # 7. Average daily range 7d (% of price)
    avg_daily_range_7d = float(np.mean((high[-7:] - low[-7:]) / close[-7:] * 100)) if n >= 7 else np.nan

    # 8. Avg daily range 7d vs 30d ratio
    avg_daily_range_30d = float(np.mean((high[-30:] - low[-30:]) / close[-30:] * 100)) if n >= 30 else np.nan
    range_rel_7_30 = avg_daily_range_7d / avg_daily_range_30d if avg_daily_range_30d and avg_daily_range_30d > 0 else np.nan

    # 9. Wide range days 7d (range > 2x mean range 7d)
    mean_range_7  = np.mean(high[-7:] - low[-7:]) if n >= 7 else np.nan
    wide_range_7d = float(np.sum((high[-7:] - low[-7:]) > 2 * mean_range_7)) if n >= 7 else np.nan

    # 10. Narrow range days 7d (range < 0.5x mean range 7d)
    narrow_range_7d = float(np.sum((high[-7:] - low[-7:]) < 0.5 * mean_range_7)) if n >= 7 else np.nan

    # -------------------------------------------------------------------------
    # GROUP 2 — Crosses indicators (10)
    # -------------------------------------------------------------------------

    # 11. Number of SMA3 crosses in last 4 weeks (daily)
    sma3_arr = np.array([close[max(0,i-2):i+1].mean() if i >= 2 else np.nan for i in range(n)])
    crosses_sma3_4w = float(np.sum(
        np.diff(np.sign(close[-28:] - sma3_arr[-28:]))[~np.isnan(sma3_arr[-27:])] != 0
    )) if n >= 28 else np.nan

    # 12. Number of SMA8 crosses in last 8 weeks
    sma8_arr = np.array([close[max(0,i-7):i+1].mean() if i >= 7 else np.nan for i in range(n)])
    crosses_sma8_8w = float(np.sum(
        np.diff(np.sign(close[-56:] - sma8_arr[-56:]))[~np.isnan(sma8_arr[-55:])] != 0
    )) if n >= 56 else np.nan

    # 13. Days since last SMA3 cross
    signs3 = np.sign(close - sma3_arr)
    cross_mask3 = np.diff(signs3) != 0
    days_since_cross_sma3 = float(len(cross_mask3) - np.where(cross_mask3)[0][-1] - 1) \
                             if np.any(cross_mask3) else np.nan

    # 14. Days since last SMA8 cross
    signs8 = np.sign(close - sma8_arr)
    cross_mask8 = np.diff(signs8) != 0
    days_since_cross_sma8 = float(len(cross_mask8) - np.where(cross_mask8)[0][-1] - 1) \
                             if np.any(cross_mask8) else np.nan

    # 15. Number of SMA5 crosses in last 4 weeks
    sma5_arr = np.array([close[max(0,i-4):i+1].mean() if i >= 4 else np.nan for i in range(n)])
    crosses_sma5_4w = float(np.sum(
        np.diff(np.sign(close[-28:] - sma5_arr[-28:]))[~np.isnan(sma5_arr[-27:])] != 0
    )) if n >= 28 else np.nan

    # 16. Price above/below SMA3 streak (consecutive days)
    above3 = (close - sma3_arr > 0).astype(float)
    streak3 = 0.0
    for v in reversed(above3[-28:]):
        if np.isnan(v): break
        if v == above3[-1]: streak3 += 1
        else: break

    # 17. Cross frequency ratio SMA3 (crosses per week, last 4w vs last 8w)
    crosses_sma3_8w = float(np.sum(
        np.diff(np.sign(close[-56:] - sma3_arr[-56:]))[~np.isnan(sma3_arr[-55:])] != 0
    )) if n >= 56 else np.nan
    cross_freq_ratio = (crosses_sma3_4w / 4) / (crosses_sma3_8w / 8) \
                        if crosses_sma3_8w and crosses_sma3_8w > 0 and n >= 56 else np.nan

    # 18. SMA3 vs SMA8 cross (golden/death cross proxy: 1=SMA3>SMA8, -1=below)
    sma3_last = sma3_arr[-1] if not np.isnan(sma3_arr[-1]) else np.nan
    sma8_last = sma8_arr[-1] if not np.isnan(sma8_arr[-1]) else np.nan
    sma_cross_proxy = 1.0 if not np.isnan(sma3_last) and not np.isnan(sma8_last) and sma3_last > sma8_last \
                      else -1.0 if not np.isnan(sma3_last) and not np.isnan(sma8_last) else np.nan

    # 19. Number of high-low crosses of SMA3 in 14d (price oscillates around MA)
    hl_cross_sma3 = float(np.sum(
        (low[-14:] < sma3_arr[-14:]) & (high[-14:] > sma3_arr[-14:])
    )) if n >= 14 else np.nan

    # 20. Days since SMA3 crossed SMA8
    sma_diff = sma3_arr - sma8_arr
    valid_mask = ~np.isnan(sma_diff)
    if np.sum(valid_mask) >= 2:
        valid_diff = sma_diff[valid_mask]
        cross_mask_sma = np.diff(np.sign(valid_diff)) != 0
        days_since_sma_cross = float(len(cross_mask_sma) - np.where(cross_mask_sma)[0][-1] - 1) \
                                if np.any(cross_mask_sma) else np.nan
    else:
        days_since_sma_cross = np.nan

    # -------------------------------------------------------------------------
    # GROUP 3 — Trend clarity (10)
    # -------------------------------------------------------------------------

    # 21. ADX 7
    adx_7 = _calc_adx(high, low, close, 7)

    # 22. ADX 14
    adx_14 = _calc_adx(high, low, close, 14)

    # 23. Efficiency Ratio 7d
    er_7 = float(np.clip(abs(close[-1] - close[-8]) / (np.sum(np.abs(np.diff(close[-8:]))) + 1e-9), 0, 1)) if n >= 8 else np.nan

    # 24. Efficiency Ratio 14d
    er_14 = float(np.clip(abs(close[-1] - close[-15]) / (np.sum(np.abs(np.diff(close[-15:]))) + 1e-9), 0, 1)) if n >= 15 else np.nan

    # 25. Choppiness Index 14d (100*log10(sum_tr/range)/log10(14))
    if n >= 15:
        tr_arr  = np.maximum(high[1:] - low[1:],
                  np.maximum(np.abs(high[1:] - close[:-1]),
                             np.abs(low[1:]  - close[:-1])))
        sum_tr  = np.sum(tr_arr[-14:])
        rng_14  = np.max(high[-14:]) - np.min(low[-14:])
        chop    = 100 * np.log10(sum_tr / rng_14) / np.log10(14) if rng_14 > 0 else np.nan
    else:
        chop = np.nan

    # 26. Consecutive days without new high or low (consolidation)
    consol_days = 0
    last_high, last_low = high[-1], low[-1]
    for i in range(2, min(n, 30)):
        if high[-i] >= last_high or low[-i] <= last_low:
            break
        consol_days += 1

    # 27. % time price between SMA3 and SMA8 last 14d
    in_band = float(np.sum(
        (close[-14:] >= np.minimum(sma3_arr[-14:], sma8_arr[-14:])) &
        (close[-14:] <= np.maximum(sma3_arr[-14:], sma8_arr[-14:]))
    )) / 14 * 100 if n >= 14 else np.nan

    # 28. Slope of SMA5 (% change per day, last 7d)
    sma5_recent = sma5_arr[-7:]
    valid_sma5  = sma5_recent[~np.isnan(sma5_recent)]
    slope_sma5  = float((valid_sma5[-1] - valid_sma5[0]) / (len(valid_sma5) * valid_sma5[0]) * 100) \
                  if len(valid_sma5) >= 2 and valid_sma5[0] != 0 else np.nan

    # 29. Trend consistency 7d (% days moving in same direction as weekly trend)
    weekly_dir  = np.sign(close[-1] - close[-8]) if n >= 8 else 0
    daily_dirs  = np.sign(np.diff(close[-8:])) if n >= 8 else np.array([])
    trend_consistency = float(np.sum(daily_dirs == weekly_dir)) / 7 * 100 if len(daily_dirs) == 7 else np.nan

    # 30. Price position in 14d range (0=bottom, 100=top)
    h14, l14  = np.max(high[-14:]), np.min(low[-14:])
    price_pos_14d = float((close[-1] - l14) / (h14 - l14) * 100) if n >= 14 and (h14 - l14) > 0 else np.nan

    # -------------------------------------------------------------------------
    # GROUP 4 — Ranging / indecision (10)
    # -------------------------------------------------------------------------

    # 31. % doji days 7d (body < 30% of range)
    if n >= 7 and has_open:
        body   = np.abs(close[-7:] - open_[-7:])
        rng_7  = high[-7:] - low[-7:]
        doji_pct = float(np.sum(body < 0.3 * (rng_7 + 1e-9))) / 7 * 100
    else:
        doji_pct = np.nan

    # 32. Ratio up days vs down days last 14d
    diffs_14 = np.diff(close[-15:]) if n >= 15 else np.array([])
    up_days  = np.sum(diffs_14 > 0)
    dn_days  = np.sum(diffs_14 < 0)
    up_dn_ratio = float(up_days / dn_days) if dn_days > 0 else np.nan

    # 33. Volatility ratio rv_7d / rv_30d
    rv_7d  = np.std(np.diff(np.log(close[-8:])))  * np.sqrt(252) * 100 if n >= 8  else np.nan
    rv_30d = np.std(np.diff(np.log(close[-31:]))) * np.sqrt(252) * 100 if n >= 31 else np.nan
    vol_ratio = rv_7d / rv_30d if rv_30d and rv_30d > 0 else np.nan

    # 34. Upper shadow ratio 7d (upper shadow / total range)
    if n >= 7 and has_open:
        upper_sh = (high[-7:] - np.maximum(close[-7:], open_[-7:])) / (high[-7:] - low[-7:] + 1e-9)
        upper_shadow_7d = float(np.mean(upper_sh))
    else:
        upper_shadow_7d = np.nan

    # 35. Lower shadow ratio 7d
    if n >= 7 and has_open:
        lower_sh = (np.minimum(close[-7:], open_[-7:]) - low[-7:]) / (high[-7:] - low[-7:] + 1e-9)
        lower_shadow_7d = float(np.mean(lower_sh))
    else:
        lower_shadow_7d = np.nan

    # 36. Shadow ratio upper/lower (>1 = more upper pressure = bearish wick)
    shadow_ratio = upper_shadow_7d / lower_shadow_7d \
                   if lower_shadow_7d and lower_shadow_7d > 0 else np.nan

    # 37. % bearish candles 7d
    if n >= 7 and has_open:
        bearish_pct = float(np.sum(close[-7:] < open_[-7:])) / 7 * 100
    else:
        bearish_pct = np.nan

    # 38. Consecutive days in same direction
    consec = _calc_consecutive_days(close)

    # 39. Max consecutive days in same direction last 14d (ranging proxy — low = ranging)
    dirs_14 = np.sign(np.diff(close[-15:])) if n >= 15 else np.array([])
    max_consec_14d = 1
    cur = 1
    for i in range(1, len(dirs_14)):
        if dirs_14[i] == dirs_14[i-1] and dirs_14[i] != 0:
            cur += 1
            max_consec_14d = max(max_consec_14d, cur)
        else:
            cur = 1

    # 40. Bollinger Band width 14d (measure of compression)
    if n >= 14:
        sma14   = close[-14:].mean()
        std14   = close[-14:].std()
        bb_width = (std14 * 2) / sma14 * 100 if sma14 > 0 else np.nan
    else:
        bb_width = np.nan

    return {
        # Range
        'range_ratio_4w':       range_ratio_4w,
        'range_compression':    range_compression,
        'inside_bars_7d':       inside_bars_7d,
        'range_expansion':      range_expansion,
        'range_4w_pct':         range_4w_pct,
        'inside_bars_14d':      inside_bars_14d,
        'avg_daily_range_7d':   avg_daily_range_7d,
        'range_rel_7_30':       range_rel_7_30,
        'wide_range_7d':        wide_range_7d,
        'narrow_range_7d':      narrow_range_7d,
        # Crosses
        'crosses_sma3_4w':      crosses_sma3_4w,
        'crosses_sma8_8w':      crosses_sma8_8w,
        'days_since_cross_sma3':days_since_cross_sma3,
        'days_since_cross_sma8':days_since_cross_sma8,
        'crosses_sma5_4w':      crosses_sma5_4w,
        'streak_above_sma3':    streak3,
        'cross_freq_ratio':     cross_freq_ratio,
        'sma_cross_proxy':      sma_cross_proxy,
        'hl_cross_sma3':        hl_cross_sma3,
        'days_since_sma_cross': days_since_sma_cross,
        # Trend clarity
        'adx_7':                adx_7,
        'adx_14':               adx_14,
        'er_7':                 er_7,
        'er_14':                er_14,
        'choppiness_14':        chop,
        'consol_days':          float(consol_days),
        'pct_in_band_sma3_8':   in_band,
        'slope_sma5':           slope_sma5,
        'trend_consistency_7d': trend_consistency,
        'price_pos_14d':        price_pos_14d,
        # Ranging / indecision
        'doji_pct_7d':          doji_pct,
        'up_dn_ratio_14d':      up_dn_ratio,
        'vol_ratio_7_30':       vol_ratio,
        'upper_shadow_7d':      upper_shadow_7d,
        'lower_shadow_7d':      lower_shadow_7d,
        'shadow_ratio':         shadow_ratio,
        'bearish_pct_7d':       bearish_pct,
        'consec_days':          consec,
        'max_consec_14d':       float(max_consec_14d),
        'bb_width_14d':         bb_width,
        # Extra for binary analysis
        'weekly_return':        (close[-1] / close[-8] - 1) * 100 if n >= 8 else np.nan,
        'monthly_return':       (close[-1] / close[-31] - 1) * 100 if n >= 31 else np.nan,
        'ma_direction':         1.0 if n >= BTC_MA_PERIOD and close[-1] > close[-BTC_MA_PERIOD:].mean() else -1.0,
        'rsi_7':                float(100 - 100 / (1 + np.mean(np.where(np.diff(close[-8:]) > 0, np.diff(close[-8:]), 0)) /
                                (np.mean(np.where(np.diff(close[-8:]) < 0, -np.diff(close[-8:]), 0)) + 1e-9))) if n >= 8 else np.nan,
        'pct_pos_14d':          float(np.sum(np.diff(close[-15:]) > 0)) / 14 * 100 if n >= 15 else np.nan,
    }


# =============================================================================
# SYSTEM WEEKLY METRICS
# =============================================================================

def compute_system_weekly_metrics(trades_df: pd.DataFrame, week_start: pd.Timestamp,
                                   week_end: pd.Timestamp) -> dict:
    """Compute system profit and winrate for trades in [week_start, week_end)."""
    mask   = (trades_df['buy_time'] >= week_start) & (trades_df['buy_time'] < week_end)
    subset = trades_df[mask]
    n      = len(subset)
    if n < MIN_TRADES_WEEK:
        return None
    profit  = subset['profit'].sum()
    winrate = (subset['profit'] > 0).mean() * 100
    return {'profit': profit, 'winrate': winrate, 'n_trades': n}


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_predictive_analysis():
    print(f"\n{'='*100}")
    print(f"  PREDICTIVE CORRELATION ANALYSIS — BTC indicators[T] vs system performance[T+1]")
    print(f"{'='*100}\n")

    # Load BTC 1D
    btc_1d_path = Path(BTC_FOLDER) / "BTCUSDT_1Dutc.parquet"
    btc_1d_df   = pd.read_parquet(btc_1d_path)
    btc_1d_df.columns = btc_1d_df.columns.str.lower()
    btc_1d_df['ts'] = pd.to_datetime(btc_1d_df['timestamp'] if 'timestamp' in btc_1d_df.columns else btc_1d_df.index)
    btc_1d_df = btc_1d_df.sort_values('ts').reset_index(drop=True)

    # Load ETH 1D
    eth_1d_path = Path(BTC_FOLDER) / "ETHUSDT_1Dutc.parquet"
    if eth_1d_path.exists():
        eth_1d_df = pd.read_parquet(eth_1d_path)
        eth_1d_df.columns = eth_1d_df.columns.str.lower()
        eth_1d_df['ts'] = pd.to_datetime(eth_1d_df['timestamp'] if 'timestamp' in eth_1d_df.columns else eth_1d_df.index)
        eth_1d_df = eth_1d_df.sort_values('ts').reset_index(drop=True)
        print(f"  ETH data loaded: {len(eth_1d_df)} bars")
    else:
        eth_1d_df = None
        print(f"  ETH data not found — skipping ETH indicators")

    # Load all trades across all periods
    all_trades = []
    for period_label, trades_label in PERIOD_LABELS:
        files = sorted(glob(str(Path(TRADES_FOLDER) / f"trades_{trades_label}_*.csv")))
        for filepath in files:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower().str.strip()
            df['buy_time'] = pd.to_datetime(df['buy_time'])
            df['period']   = period_label
            all_trades.append(df)

    if not all_trades:
        print("  No trades found — aborting")
        return

    trades_df = pd.concat(all_trades, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
    print(f"  Total trades loaded: {len(trades_df)} across {len(all_trades)} files")

    # Build weekly time grid (Monday-based)
    t_min = trades_df['buy_time'].min().normalize()
    t_max = trades_df['buy_time'].max().normalize()
    weeks = pd.date_range(start=t_min - pd.Timedelta(days=t_min.weekday()), end=t_max, freq='W-MON')

    # Build weekly series
    rows = []
    for i in range(len(weeks) - 1):
        week_start = weeks[i]
        week_end   = weeks[i + 1]

        # Indicators at week T (use week_start as reference — closed bars before it)
        indicators = compute_btc_weekly_indicators(btc_1d_df, week_start, eth_1d_df)
        if indicators is None:
            continue

        # System performance at week T+1
        if i + 2 >= len(weeks):
            continue
        next_start = weeks[i + 1]
        next_end   = weeks[i + 2]
        perf = compute_system_weekly_metrics(trades_df, next_start, next_end)
        if perf is None:
            continue

        row = {'week': week_start, **indicators, **perf}
        rows.append(row)

    if len(rows) < MIN_PERIODS:
        print(f"  Insufficient weeks ({len(rows)}) for analysis — need at least {MIN_PERIODS}")
        return

    df = pd.DataFrame(rows)
    print(f"  Weekly periods with valid data: {len(df)}\n")

    df['period'] = df['week'].apply(lambda w: next(
        (p for p, _ in PERIOD_LABELS
         if w >= trades_df[trades_df['period'] == p]['buy_time'].min().normalize() - pd.Timedelta(days=7)
         and w <= trades_df[trades_df['period'] == p]['buy_time'].max().normalize()),
        'unknown'
    ))

    pct_pos_all  = (df['profit'] > 0).mean() * 100
    periods      = ['IS', 'OOS1', 'OOS2', 'OOS3']
    period_means = {p: (df[df['period'] == p]['profit'] > 0).mean() * 100 for p in periods}

    indicators = [
        # Trend / direction
        ('weekly_return',        'BTC up week',              0.0),
        ('monthly_return',       'BTC up month',             0.0),
        ('ma_direction',         'BTC above SMA4',           0.0),
        ('slope_sma5',           'SMA5 rising',              0.0),
        ('sma_cross_proxy',      'SMA3 above SMA8',          0.0),
        ('trend_consistency_7d', 'Trend consistent 7d',     57.0),
        ('price_pos_14d',        'Price top half 14d',       50.0),
        ('consec_days',          'BTC up streak',            0.0),
        ('streak_above_sma3',    'Above SMA3 streak',        0.0),
        # Momentum
        ('rsi_7',                'RSI7 overbought',         50.0),
        ('adx_14',               'Strong trend ADX14',      25.0),
        ('adx_7',                'Strong trend ADX7',       25.0),
        ('er_7',                 'High efficiency 7d',       0.5),
        ('er_14',                'High efficiency 14d',      0.5),
        ('cross_freq_ratio',     'Cross freq increasing',    1.0),
        # Bullish days
        ('bearish_pct_7d',       'Mostly bullish 7d',       50.0),
        ('pct_pos_14d',          'Mostly bullish 14d',      50.0),
        ('up_dn_ratio_14d',      'More up than down 14d',    1.0),
        # Volatility
        ('vol_ratio_7_30',       'Vol high vs history',      1.0),
        ('avg_daily_range_7d',   'High daily range 7d',      2.0),
        ('bb_width_14d',         'Wide Bollinger band',      3.0),
        ('range_rel_7_30',       'Range high vs history',    1.0),
        ('range_expansion',      'Range expanding',          1.0),
        ('wide_range_7d',        'Wide range days',          1.0),
        # Ranging / indecision
        ('choppiness_14',        'BTC ranging CHOP>61.8',   61.8),
        ('pct_in_band_sma3_8',   'Price in SMA band',       20.0),
        ('inside_bars_14d',      'Many inside bars 14d',     3.0),
        ('narrow_range_7d',      'Narrow range days',        2.0),
        ('doji_pct_7d',          'Many doji candles',       30.0),
        ('consol_days',          'Consolidation days',       3.0),
        # Crosses
        ('crosses_sma3_4w',      'Many SMA3 crosses 4w',     4.0),
        ('crosses_sma5_4w',      'Many SMA5 crosses 4w',     4.0),
        ('hl_cross_sma3',        'Price straddles SMA3',     3.0),
        # Candle structure
        ('upper_shadow_7d',      'High upper shadow 7d',     0.25),
        ('lower_shadow_7d',      'High lower shadow 7d',     0.25),
        ('shadow_ratio',         'More upper shadow',        1.0),
        ('max_consec_14d',       'Long directional run',     4.0),
        ('range_4w_pct',         'Wide 4w range',            5.0),
    ]

    print(f"  System average: {pct_pos_all:.1f}% positive weeks")
    print(f"  Green = above period average | Red = below period average\n")
    col_w = 8
    print(f"  {'INDICATOR':<20} {'CONDITION':<22} {'N':>5}" +
          "".join(f" {p:>{col_w}}" for p in periods) + f"  {'OK':>4}")
    print(f"  {'-'*92}")

    for col, label, threshold in indicators:
        if col not in df.columns:
            continue
        valid = df[[col, 'profit', 'period']].dropna()
        grp   = valid[valid[col] >= threshold]
        if len(grp) < 5:
            continue
        reset   = "\033[0m"
        row     = f"  {col:<20} {label:<22} {len(grp):>5}"
        n_green = 0
        n_red   = 0
        n_valid = 0
        for period in periods:
            sub = grp[grp['period'] == period]
            if len(sub) < 3:
                row += f" {'—':>{col_w}}"
            else:
                pct_p    = (sub['profit'] > 0).mean() * 100
                is_green = pct_p >= period_means[period]
                color_p  = "\033[92m" if is_green else "\033[91m"
                row     += f" {color_p}{pct_p:>{col_w-1}.1f}%{reset}"
                if period != 'IS':
                    n_valid += 1
                    if is_green:
                        n_green += 1
                    else:
                        n_red += 1
        n_oos = 3  # OOS1, OOS2, OOS3
        if n_valid >= MIN_OOS_OK and (n_green >= MIN_OOS_OK or n_red >= MIN_OOS_OK):
            ok = "\033[92m YES\033[0m"
        else:
            ok = "\033[91m  NO\033[0m"
        row += f"  {ok}"
        print(row)

    print(f"\n{'='*85}\n")

    # ==========================================================================
    # PEARSON CORRELATION — YES indicators vs profit per OOS period
    # ==========================================================================
    from scipy import stats as scipy_stats

    yes_indicators = [
        (col, label, threshold) for col, label, threshold in indicators
        if col in df.columns and len(df[[col, 'profit', 'period']].dropna()[
            df[[col, 'profit', 'period']].dropna()[col] >= threshold]) >= 5
    ]

    # Filter to only YES ones — recompute
    yes_cols = []
    for col, label, threshold in indicators:
        if col not in df.columns:
            continue
        valid = df[[col, 'profit', 'period']].dropna()
        grp   = valid[valid[col] >= threshold]
        if len(grp) < 5:
            continue
        n_green = n_red = n_valid = 0
        for period in ['OOS1', 'OOS2', 'OOS3']:
            sub = grp[grp['period'] == period]
            if len(sub) >= 3:
                pct_p = (sub['profit'] > 0).mean() * 100
                n_valid += 1
                if pct_p >= period_means[period]:
                    n_green += 1
                else:
                    n_red += 1
        if n_valid >= MIN_OOS_OK and (n_green >= MIN_OOS_OK or n_red >= MIN_OOS_OK):
            yes_cols.append((col, label, threshold))

    if yes_cols:
        print(f"\n{'='*90}")
        print(f"  PEARSON CORRELATION — YES indicators[T] vs profit[T+1] per OOS period")
        print(f"  (measures linear relationship strength)")
        print(f"{'='*90}")
        print(f"  {'INDICATOR':<25} {'CONDITION':<22}" +
              "".join(f"  {p+' corr':>10}  {p+' p':>8}" for p in ['OOS1', 'OOS2', 'OOS3']))
        print(f"  {'-'*100}")

        for col, label, threshold in yes_cols:
            valid = df[[col, 'profit', 'period']].dropna()
            row   = f"  {col:<25} {label:<22}"
            for period in ['OOS1', 'OOS2', 'OOS3']:
                sub = valid[valid['period'] == period]
                if len(sub) < 5:
                    row += f"  {'—':>10}  {'—':>8}"
                else:
                    corr, pval = scipy_stats.pearsonr(sub[col], sub['profit'])
                    sig    = "✅" if pval < 0.10 else "  "
                    color  = "\033[92m" if abs(corr) >= 0.15 and pval < 0.10 else ""
                    reset  = "\033[0m" if color else ""
                    row   += f"  {color}{corr:>+9.3f}{reset}  {pval:>6.3f}{sig}"
            print(row)

        print(f"\n{'='*90}\n")

    # ==========================================================================
    # MUTUAL INFORMATION — YES indicators vs profit per OOS period
    # ==========================================================================
    if yes_cols:
        from sklearn.feature_selection import mutual_info_regression

        N_PERM = 100
        rng    = np.random.default_rng(42)

        print(f"\n{'='*90}")
        print(f"  MUTUAL INFORMATION — YES indicators[T] vs profit[T+1] per OOS period")
        print(f"  (permutation p-value, {N_PERM} shuffles)")
        print(f"{'='*90}")
        print(f"  {'INDICATOR':<25} {'CONDITION':<22}" +
              "".join(f"  {p+' MI':>8}  {p+' p':>7}" for p in ['OOS1', 'OOS2', 'OOS3']))
        print(f"  {'-'*100}")

        for col, label, threshold in yes_cols:
            valid = df[[col, 'profit', 'period']].dropna()
            row   = f"  {col:<25} {label:<22}"
            for period in ['OOS1', 'OOS2', 'OOS3']:
                sub = valid[valid['period'] == period]
                if len(sub) < 5:
                    row += f"  {'—':>8}  {'—':>7}"
                else:
                    X      = sub[[col]].values
                    y      = sub['profit'].values
                    mi_obs = mutual_info_regression(X, y, random_state=42)[0]
                    mi_perm = np.array([
                        mutual_info_regression(X, rng.permutation(y), random_state=42)[0]
                        for _ in range(N_PERM)
                    ])
                    pval  = float(np.mean(mi_perm >= mi_obs))
                    sig   = "✅" if pval < 0.10 else "  "
                    color = "\033[92m" if pval < 0.10 else ""
                    reset = "\033[0m" if color else ""
                    row  += f"  {color}{mi_obs:>8.4f}{reset}  {pval:>6.3f}{sig}"
            print(row)

        print(f"\n{'='*90}\n")

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = run_predictive_analysis()