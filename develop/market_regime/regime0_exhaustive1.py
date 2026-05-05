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
    ("IS",   "is_baseline"),
    ("OOS1", "oos1_baseline"),
    ("OOS2", "oos2_baseline"),
    ("OOS3", "oos3_baseline"),
]

BTC_MA_PERIOD   = 4
INITIAL_CAPITAL = 800
MIN_TRADES_WEEK = 2       # minimum trades in a week to include it
MIN_PERIODS     = 8       # minimum weeks needed to compute correlation
PVALUE_TH       = 0.10    # significance threshold

# =============================================================================
# BTC INDICATOR COMPUTATION
# =============================================================================

def compute_btc_weekly_indicators(btc_1d_df: pd.DataFrame, week_start: pd.Timestamp) -> dict:
    """
    Compute 20 BTC indicators for the week ending at week_start (no lookahead).
    Uses only closed bars strictly before week_start.
    """
    closed = btc_1d_df[btc_1d_df['ts'] < week_start].copy()
    if len(closed) < 30:
        return None

    close  = closed['close'].values.astype(np.float64)
    high   = closed['high'].values.astype(np.float64)
    low    = closed['low'].values.astype(np.float64)
    volume = closed['volume'].values.astype(np.float64) if 'volume' in closed.columns else np.ones(len(closed))

    n = len(close)

    # --- 1. MA direction (1=uptrend, -1=dwtrend, 0=neutral) ---
    ma_val   = close[-BTC_MA_PERIOD:].mean() if n >= BTC_MA_PERIOD else np.nan
    if not np.isnan(ma_val):
        if close[-1] > ma_val:   direction = 1.0
        elif close[-1] < ma_val: direction = -1.0
        else:                    direction = 0.0
    else:
        direction = np.nan

    # --- 2. BTC weekly return % ---
    weekly_return = (close[-1] / close[-8] - 1) * 100 if n >= 8 else np.nan

    # --- 3. BTC monthly return % ---
    monthly_return = (close[-1] / close[-31] - 1) * 100 if n >= 31 else np.nan

    # --- 4. Efficiency Ratio (ER) 14d ---
    if n >= 15:
        net_change  = abs(close[-1] - close[-15])
        total_change = np.sum(np.abs(np.diff(close[-15:])))
        er = float(np.clip(net_change / total_change, 0, 1)) if total_change > 0 else 0.0
    else:
        er = np.nan

    # --- 5. Distance from MA % ---
    dist_ma = (close[-1] / ma_val - 1) * 100 if not np.isnan(ma_val) and ma_val != 0 else np.nan

    # --- 6. Higher highs structure (1=HH+HL, -1=LH+LL, 0=mixed) ---
    if n >= 3:
        hh = high[-1] > high[-2] > high[-3]
        hl = low[-1]  > low[-2]  > low[-3]
        lh = high[-1] < high[-2] < high[-3]
        ll = low[-1]  < low[-2]  < low[-3]
        if hh and hl:   structure = 1.0
        elif lh and ll: structure = -1.0
        else:           structure = 0.0
    else:
        structure = np.nan

    # --- 7. Rate of change 7d ---
    roc_7 = (close[-1] / close[-8] - 1) * 100 if n >= 8 else np.nan

    # --- 8. ATR % (14d Wilder) ---
    if n >= 15:
        tr = np.maximum(high[1:] - low[1:],
             np.maximum(np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:]  - close[:-1])))
        atr = np.mean(tr[-14:])
        for i in range(len(tr) - 14 + 1, len(tr)):
            atr = (atr * 13 + tr[i]) / 14
        atr_pct = (atr / close[-1]) * 100 if close[-1] != 0 else np.nan
    else:
        atr_pct = np.nan

    # --- 9. Realized volatility 7d ---
    if n >= 8:
        rets_7  = np.diff(np.log(close[-8:]))
        rv_7    = np.std(rets_7) * np.sqrt(252) * 100
    else:
        rv_7 = np.nan

    # --- 10. Realized volatility 30d ---
    if n >= 31:
        rets_30 = np.diff(np.log(close[-31:]))
        rv_30   = np.std(rets_30) * np.sqrt(252) * 100
    else:
        rv_30 = np.nan

    # --- 11. High-Low range % weekly ---
    hl_range = (np.max(high[-7:]) / np.min(low[-7:]) - 1) * 100 if n >= 7 else np.nan

    # --- 12. VIX proxy (std daily returns 7d) ---
    vix_proxy = np.std(np.diff(close[-8:]) / close[-8:-1]) * 100 if n >= 8 else np.nan

    # --- 13. RSI 14 ---
    if n >= 15:
        deltas = np.diff(close[-15:])
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rsi_14   = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100.0
    else:
        rsi_14 = np.nan

    # --- 14. RSI 7 ---
    if n >= 8:
        deltas = np.diff(close[-8:])
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rsi_7    = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100.0
    else:
        rsi_7 = np.nan

    # --- 15. MACD signal (12-26 EMA diff) ---
    def _ema(arr, period):
        k   = 2 / (period + 1)
        ema = arr[0]
        for x in arr[1:]:
            ema = x * k + ema * (1 - k)
        return ema

    if n >= 27:
        ema12  = _ema(close[-27:], 12)
        ema26  = _ema(close[-27:], 26)
        macd   = ema12 - ema26
        macd_pct = macd / close[-1] * 100 if close[-1] != 0 else np.nan
    else:
        macd_pct = np.nan

    # --- 16. Stochastic %K (14d) ---
    if n >= 14:
        highest_high = np.max(high[-14:])
        lowest_low   = np.min(low[-14:])
        stoch_k      = (close[-1] - lowest_low) / (highest_high - lowest_low) * 100 \
                       if (highest_high - lowest_low) > 0 else 50.0
    else:
        stoch_k = np.nan

    # --- 17. Hurst exponent (hardcoded — fast) ---
    hurst = 0.8  # matches regime_metrics.py

    # --- 18. Permutation Entropy (hardcoded — fast) ---
    pe = 0.8  # matches regime_metrics.py

    # --- 19. Trend strength (directional returns / ATR) ---
    if n >= 15 and not np.isnan(atr_pct) and atr_pct > 0:
        dir_returns = np.sum(np.abs(np.diff(close[-8:])))
        trend_str   = dir_returns / (atr_pct * close[-1] / 100 * 7) if atr_pct > 0 else np.nan
    else:
        trend_str = np.nan

    # --- 20. Volume change % vs 30d average ---
    if n >= 31 and np.mean(volume[-31:-1]) > 0:
        vol_change = (volume[-1] / np.mean(volume[-31:-1]) - 1) * 100
    else:
        vol_change = np.nan

    return {
        'ma_direction':    direction,
        'weekly_return':   weekly_return,
        'monthly_return':  monthly_return,
        'er_14':           er,
        'dist_from_ma':    dist_ma,
        'hh_structure':    structure,
        'roc_7d':          roc_7,
        'atr_pct':         atr_pct,
        'rv_7d':           rv_7,
        'rv_30d':          rv_30,
        'hl_range_weekly': hl_range,
        'vix_proxy':       vix_proxy,
        'rsi_14':          rsi_14,
        'rsi_7':           rsi_7,
        'macd_pct':        macd_pct,
        'stoch_k':         stoch_k,
        'hurst':           hurst,
        'perm_entropy':    pe,
        'trend_strength':  trend_str,
        'volume_change':   vol_change,
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
        indicators = compute_btc_weekly_indicators(btc_1d_df, week_start)
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

    indicator_cols = list(compute_btc_weekly_indicators(btc_1d_df, weeks[10]).keys())

    # ==========================================================================
    # CORRELATION TABLE
    # ==========================================================================
    for target in ['profit', 'winrate']:
        print(f"\n{'='*90}")
        print(f"  CORRELATIONS — BTC indicator[T] vs system {target.upper()}[T+1]")
        print(f"{'='*90}")
        print(f"  {'INDICATOR':<25} {'CORR':>8} {'P-VALUE':>10} {'N':>6} {'SIG':>6} {'INTERPRETATION'}")
        print(f"  {'-'*88}")

        results = []
        for col in indicator_cols:
            valid = df[[col, target]].dropna()
            n     = len(valid)
            if n < MIN_PERIODS:
                continue
            corr, pval = stats.pearsonr(valid[col], valid[target])
            sig  = "✅" if pval < PVALUE_TH else "  "
            results.append((col, corr, pval, n, sig))

        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x[1]), reverse=True)

        for col, corr, pval, n, sig in results:
            direction = "↑ higher → better" if corr > 0 else "↓ higher → worse"
            strength  = "strong" if abs(corr) > 0.4 else "moderate" if abs(corr) > 0.2 else "weak"
            interp    = f"{strength} {direction}" if pval < PVALUE_TH else "—"
            print(f"  {col:<25} {corr:>+8.3f} {pval:>10.4f} {n:>6} {sig:>6}  {interp}")

        print(f"  {'-'*88}")
        sig_count = sum(1 for _, _, pval, _, _ in results if pval < PVALUE_TH)
        print(f"  Significant indicators (p<{PVALUE_TH}): {sig_count}/{len(results)}")

    # ==========================================================================
    # SCATTER SUMMARY — top 5 significant predictors for profit
    # ==========================================================================
    print(f"\n{'='*90}")
    print(f"  TOP PREDICTORS SUMMARY")
    print(f"{'='*90}")
    print(f"  {'INDICATOR':<25} {'CORR_PROFIT':>12} {'CORR_WR':>10} {'SIG_PROFIT':>12} {'SIG_WR':>8}")
    print(f"  {'-'*75}")

    profit_corrs = {}
    wr_corrs     = {}
    for col in indicator_cols:
        valid_p = df[[col, 'profit']].dropna()
        valid_w = df[[col, 'winrate']].dropna()
        if len(valid_p) >= MIN_PERIODS:
            c, p = stats.pearsonr(valid_p[col], valid_p['profit'])
            profit_corrs[col] = (c, p)
        if len(valid_w) >= MIN_PERIODS:
            c, p = stats.pearsonr(valid_w[col], valid_w['winrate'])
            wr_corrs[col] = (c, p)

    # Sort by max significance
    all_cols = sorted(indicator_cols,
                      key=lambda c: max(abs(profit_corrs.get(c, (0, 1))[0]),
                                        abs(wr_corrs.get(c, (0, 1))[0])),
                      reverse=True)

    for col in all_cols:
        pc, pp = profit_corrs.get(col, (np.nan, np.nan))
        wc, wp = wr_corrs.get(col, (np.nan, np.nan))
        sig_p  = "✅" if not np.isnan(pp) and pp < PVALUE_TH else "  "
        sig_w  = "✅" if not np.isnan(wp) and wp < PVALUE_TH else "  "
        print(f"  {col:<25} {pc:>+11.3f}  {wc:>+9.3f}  {sig_p:>12} {sig_w:>8}")

    print(f"{'='*90}\n")

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = run_predictive_analysis()