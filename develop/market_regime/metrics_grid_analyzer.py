#!/usr/bin/env python3
"""
develop/market_regime/regime_threshold_optimizer.py

Grid search over efficiency_ratio and atr_pct thresholds to find the combination
that maximizes total filtered profit across all strategies.

Filtering logic: bins with profit < 0 and n >= MIN_TRADES are blocked.

Usage:
    python regime_threshold_optimizer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from itertools import product as iterproduct
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_market_regime")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))
from regime_common import extract_timeframe, load_btc_for_timeframe, calc_all_metrics_at_time, load_trades

# =============================================================================
# CONFIGURATION
# =============================================================================
TRADES_FOLDER = os.path.join(os.path.dirname(__file__), "..", "brief_trades")

SPLIT_MODE    = "expanding"
SPLIT_BASE    = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
BTC_FOLDER    = os.path.join(SPLIT_BASE, "IS", "crypto_full_IS")

FAMILY_SOURCE = 'strategy'   # 'strategy' | 'macro'

HURST_WINDOW  = 100
ER_WINDOW     = 14
ATR_WINDOW    = 14
PE_WINDOW     = 50
PE_ORDER      = 3
LOOKBACK_BARS = 100

MIN_TRADES    = 10
INITIAL_CAPITAL = 800

# Grid search ranges
ER_THRESHOLDS        = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ATR_THRESHOLDS       = [1.0, 1.5, 2.0, 2.5, 3.0]
VOL_RATIO_THRESHOLDS = [0.75, 1.0, 1.25, 1.5, 1.75]
AUTOCORR_THRESHOLDS  = [-0.3, -0.2, -0.1, 0.0, 0.1]

# Extra metrics params
VOL_RATIO_SHORT = 14
VOL_RATIO_LONG  = 50
AUTOCORR_WINDOW = 20

TOP_N_RESULTS  = 10

# =============================================================================
# CACHE
# =============================================================================
_btc_cache = {}


# =============================================================================
# HELPERS
# =============================================================================
def load_btc_for_family(timeframe: str) -> pd.DataFrame:
    if FAMILY_SOURCE == 'macro':
        filepath = Path(BTC_FOLDER) / "BTCUSDT_1Dutc.parquet"
        df = pd.read_parquet(filepath)
        df.columns = df.columns.str.lower()
        df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
        return df.sort_values('ts').reset_index(drop=True)
    return load_btc_for_timeframe(BTC_FOLDER, timeframe, _btc_cache)


def classify_family(metrics: dict, er_th: float, atr_th: float, vol_ratio_th: float, autocorr_th: float) -> str:
    """Classify trade into family using given thresholds."""
    if metrics is None:
        return 'unknown'
    er        = metrics.get('efficiency_ratio')
    atr       = metrics.get('atr_pct')
    vol_ratio = metrics.get('volatility_ratio')
    autocorr  = metrics.get('autocorr_lag1')

    if er is not None and not np.isnan(er) and er > er_th:
        return 'trending'
    volatile = (
        (atr       is not None and not np.isnan(atr)       and atr       > atr_th) or
        (vol_ratio is not None and not np.isnan(vol_ratio) and vol_ratio > vol_ratio_th)
    )
    if volatile:
        return 'volatile'
    if autocorr is not None and not np.isnan(autocorr) and autocorr < autocorr_th:
        return 'mean_reverting'
    return 'ranging'


def calc_filtered_profit(df: pd.DataFrame, er_th: float, atr_th: float, vol_ratio_th: float, autocorr_th: float) -> float:
    """
    Given a trades DataFrame with precomputed metrics, apply family classification
    and filter bins with profit < 0 and n >= MIN_TRADES. Return total filtered profit.
    """
    df = df.copy()
    df['family'] = df['metrics'].apply(lambda m: classify_family(m, er_th, atr_th, vol_ratio_th, autocorr_th))

    all_families   = ['trending', 'volatile', 'mean_reverting', 'ranging']
    all_directions = ['uptrend', 'dwtrend']

    bins_to_filter = set()
    for family in all_families:
        for direction in all_directions:
            subset = df[(df['family'] == family) & (df['direction'] == direction)]
            n      = len(subset)
            profit = subset['profit'].sum() if n > 0 else 0.0
            if n >= MIN_TRADES and profit < 0:
                bins_to_filter.add(f"{family}_{direction}")

    if not bins_to_filter:
        return df['profit'].sum()

    mask = df.apply(
        lambda r: f"{r['family']}_{r['direction']}" not in bins_to_filter, axis=1
    )
    return df.loc[mask, 'profit'].sum()


# =============================================================================
# PRECOMPUTE METRICS + DIRECTIONS
# =============================================================================
def precompute_trades(files: list) -> pd.DataFrame:
    """
    Load all trades, compute metrics and direction once.
    Returns a single DataFrame with columns: strategy, profit, direction, metrics.
    """
    from regime_common import get_btc_macro_direction

    BTC_MA_PERIOD = 5
    LONG_TH       = 1.00
    SHORT_TH      = 1.00

    btc_1d_path = Path(BTC_FOLDER) / "BTCUSDT_1Dutc.parquet"
    btc_1d_df   = pd.read_parquet(btc_1d_path)
    btc_1d_df.columns = btc_1d_df.columns.str.lower()
    btc_1d_df['ts'] = pd.to_datetime(btc_1d_df['timestamp'] if 'timestamp' in btc_1d_df.columns else btc_1d_df.index)
    btc_1d_df = btc_1d_df.sort_values('ts').reset_index(drop=True)

    all_rows = []
    for filepath in files:
        df       = load_trades(filepath)
        strategy = df['strategy'].iloc[0]
        tf       = extract_timeframe(df)
        btc_df   = load_btc_for_family(tf)

        print(f"  Precomputing {strategy} ({len(df)} trades)...")

        for _, trade in df.iterrows():
            metrics = calc_all_metrics_at_time(
                btc_df       = btc_df,
                buy_time     = trade['buy_time'],
                lookback     = LOOKBACK_BARS,
                hurst_window = HURST_WINDOW,
                er_window    = ER_WINDOW,
                atr_window   = ATR_WINDOW,
                pe_window    = PE_WINDOW,
                pe_order     = PE_ORDER,
            )
            direction = get_btc_macro_direction(
                btc_1d_df  = btc_1d_df,
                trade_time = trade['buy_time'],
                ma_period  = BTC_MA_PERIOD,
                long_th    = LONG_TH,
                short_th   = SHORT_TH,
            )
            if direction not in ('uptrend', 'dwtrend'):
                continue
            if metrics is None:
                continue

            closed = btc_df[btc_df['ts'] < trade['buy_time']]
            extra  = {'volatility_ratio': None, 'autocorr_lag1': None}
            if len(closed) >= VOL_RATIO_LONG:
                close = closed['close'].values.astype(np.float64)
                high  = closed['high'].values.astype(np.float64)
                low   = closed['low'].values.astype(np.float64)

                def _atr(h, l, c, n):
                    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
                    return tr[-n:].mean() if len(tr) >= n else None

                atr_s = _atr(high, low, close, VOL_RATIO_SHORT)
                atr_l = _atr(high, low, close, VOL_RATIO_LONG)
                extra['volatility_ratio'] = (atr_s / atr_l) if (atr_s and atr_l and atr_l > 0) else None

                rets = np.diff(close[-AUTOCORR_WINDOW - 1:]) / close[-AUTOCORR_WINDOW - 1:-1]
                extra['autocorr_lag1'] = float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if len(rets) >= 3 else None

            metrics.update(extra)
            all_rows.append({
                'strategy':  strategy,
                'profit':    trade['profit'],
                'direction': direction,
                'metrics':   metrics,
            })

    return pd.DataFrame(all_rows)


# =============================================================================
# GRID SEARCH
# =============================================================================
def run_grid_search(df_all: pd.DataFrame) -> pd.DataFrame:
    """Run grid search over ER, ATR, VolRatio and Autocorr thresholds."""
    results = []
    total   = len(ER_THRESHOLDS) * len(ATR_THRESHOLDS) * len(VOL_RATIO_THRESHOLDS) * len(AUTOCORR_THRESHOLDS)
    i       = 0

    for er_th, atr_th, vr_th, ac_th in iterproduct(ER_THRESHOLDS, ATR_THRESHOLDS, VOL_RATIO_THRESHOLDS, AUTOCORR_THRESHOLDS):
        i += 1
        profit = calc_filtered_profit(df_all, er_th, atr_th, vr_th, ac_th)

        df_tmp = df_all.copy()
        df_tmp['family'] = df_tmp['metrics'].apply(lambda m: classify_family(m, er_th, atr_th, vr_th, ac_th))
        n_trending      = (df_tmp['family'] == 'trending').sum()
        n_volatile      = (df_tmp['family'] == 'volatile').sum()
        n_mean_rev      = (df_tmp['family'] == 'mean_reverting').sum()
        n_ranging       = (df_tmp['family'] == 'ranging').sum()

        results.append({
            'er_th':    er_th,
            'atr_th':   atr_th,
            'vr_th':    vr_th,
            'ac_th':    ac_th,
            'profit':   round(profit, 2),
            'n_trend':  n_trending,
            'n_vol':    n_volatile,
            'n_mr':     n_mean_rev,
            'n_range':  n_ranging,
        })
        print(f"  [{i:>4}/{total}] ER>{er_th}  ATR>{atr_th}  VR>{vr_th}  AC<{ac_th}  →  profit={profit:.2f}")

    return pd.DataFrame(results).sort_values('profit', ascending=False).reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("REGIME THRESHOLD OPTIMIZER")
    print("=" * 80)
    print(f"\n  Trades folder  : {TRADES_FOLDER}")
    print(f"  BTC folder     : {BTC_FOLDER}")
    print(f"  Min trades     : {MIN_TRADES}")
    print(f"  ER thresholds       : {ER_THRESHOLDS}")
    print(f"  ATR thresholds      : {ATR_THRESHOLDS}")
    print(f"  VolRatio thresholds : {VOL_RATIO_THRESHOLDS}")
    print(f"  Autocorr thresholds : {AUTOCORR_THRESHOLDS}")
    print(f"  Combinations        : {len(ER_THRESHOLDS) * len(ATR_THRESHOLDS) * len(VOL_RATIO_THRESHOLDS) * len(AUTOCORR_THRESHOLDS)}")

    files = sorted(glob(str(Path(TRADES_FOLDER) / "*.csv")))
    if not files:
        print(f"\n  No CSV files found in {TRADES_FOLDER}")
        return
    print(f"\n  Found {len(files)} strategy files")

    print("\n  Precomputing metrics (once)...")
    df_all = precompute_trades(files)
    print(f"\n  Total valid trades: {len(df_all)}")

    baseline_profit = df_all['profit'].sum()
    print(f"  Baseline profit (no filter): {baseline_profit:.2f}")

    print(f"\n{'─'*80}")
    print("  Running grid search...")
    print(f"{'─'*80}\n")

    results_df = run_grid_search(df_all)

    print(f"\n{'='*80}")
    print(f"  TOP {TOP_N_RESULTS} COMBINATIONS")
    print(f"{'='*80}")
    print(f"\n  {'Rank':<6} {'ER_TH':>6} {'ATR_TH':>7} {'VR_TH':>7} {'AC_TH':>7} {'Profit':>10} {'Δ_Profit':>10} {'N_trend':>9} {'N_vol':>7} {'N_mr':>7} {'N_range':>9}")
    print(f"  {'─'*100}")
    for i, row in results_df.head(TOP_N_RESULTS).iterrows():
        delta = row['profit'] - baseline_profit
        print(f"  {i+1:<6} {row['er_th']:>6.2f} {row['atr_th']:>7.2f} {row['vr_th']:>7.2f} {row['ac_th']:>7.2f} "
              f"{row['profit']:>10.2f} {delta:>+10.2f} "
              f"{int(row['n_trend']):>9} {int(row['n_vol']):>7} {int(row['n_mr']):>7} {int(row['n_range']):>9}")
    print(f"  {'─'*80}")
    print(f"\n  Baseline (no filter): {baseline_profit:.2f}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()