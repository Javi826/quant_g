#!/usr/bin/env python3
"""
develop/market_regime/regime_metrics_distribution.py

Analyzes the distribution of regime metrics (hurst, efficiency_ratio, atr_pct,
permutation_entropy) across all trades in the brief_trades folder.

Useful for calibrating FAMILIES thresholds before running the full regime analysis.

Usage:
    python regime_metrics_distribution.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
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

METRICS_TO_ANALYZE = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy', 'volatility_ratio', 'autocorr_lag1']

# Volatility ratio params
VOL_RATIO_SHORT = 14   # short ATR window
VOL_RATIO_LONG  = 50   # long ATR window

# Autocorrelation params
AUTOCORR_WINDOW = 20   # lookback bars for autocorrelation
PERCENTILES        = [5, 10, 25, 50, 75, 90, 95]

# =============================================================================
# CACHE
# =============================================================================
_btc_cache = {}


# =============================================================================
# HELPERS
# =============================================================================
def load_btc_1d() -> pd.DataFrame:
    filepath = Path(BTC_FOLDER) / "BTCUSDT_1Dutc.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"BTC 1D file not found: {filepath}")
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
    return df.sort_values('ts').reset_index(drop=True)


def load_btc_for_family(timeframe: str) -> pd.DataFrame:
    if FAMILY_SOURCE == 'macro':
        return load_btc_1d()
    return load_btc_for_timeframe(BTC_FOLDER, timeframe, _btc_cache)


def _calc_extra_metrics(btc_df: pd.DataFrame, buy_time, vol_ratio_short: int, vol_ratio_long: int, autocorr_window: int) -> dict:
    """Compute volatility_ratio and autocorr_lag1 from closed candles."""
    closed = btc_df[btc_df['ts'] < buy_time]
    if len(closed) < vol_ratio_long:
        return {'volatility_ratio': None, 'autocorr_lag1': None}

    close  = closed['close'].values.astype(np.float64)
    high   = closed['high'].values.astype(np.float64)
    low    = closed['low'].values.astype(np.float64)

    def _atr(h, l, c, n):
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return tr[-n:].mean() if len(tr) >= n else None

    atr_short = _atr(high, low, close, vol_ratio_short)
    atr_long  = _atr(high, low, close, vol_ratio_long)
    vol_ratio = (atr_short / atr_long) if (atr_short and atr_long and atr_long > 0) else None

    returns   = np.diff(close[-autocorr_window - 1:]) / close[-autocorr_window - 1:-1]
    autocorr  = float(np.corrcoef(returns[:-1], returns[1:])[0, 1]) if len(returns) >= 3 else None

    return {'volatility_ratio': vol_ratio, 'autocorr_lag1': autocorr}


def print_metric_distribution(metric: str, values: list):
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        print(f"  {metric}: no data")
        return

    pcts = np.percentile(arr, PERCENTILES)

    print(f"\n  {'─'*80}")
    print(f"  {metric.upper()}")
    print(f"  {'─'*80}")
    print(f"  n={len(arr)}  min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}")
    pct_str = "  " + "  ".join(f"p{p}={v:.4f}" for p, v in zip(PERCENTILES, pcts))
    print(pct_str)




# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("REGIME METRICS DISTRIBUTION ANALYZER")
    print("=" * 80)
    print(f"\n  Trades folder : {TRADES_FOLDER}")
    print(f"  BTC folder    : {BTC_FOLDER}")
    print(f"  Family source : {FAMILY_SOURCE}")

    files = sorted(glob(str(Path(TRADES_FOLDER) / "*.csv")))
    if not files:
        print(f"\n  No CSV files found in {TRADES_FOLDER}")
        return
    print(f"\n  Found {len(files)} strategy files")

    all_metrics  = defaultdict(list)
    per_strategy = {m: {} for m in METRICS_TO_ANALYZE}

    for filepath in files:
        df       = load_trades(filepath)
        strategy = df['strategy'].iloc[0]
        tf       = extract_timeframe(df)
        btc_df   = load_btc_for_family(tf)

        print(f"\n  Processing {strategy} ({len(df)} trades)...")

        for m in METRICS_TO_ANALYZE:
            per_strategy[m][strategy] = []

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
            if metrics is None:
                continue
            extra = _calc_extra_metrics(btc_df, trade['buy_time'], VOL_RATIO_SHORT, VOL_RATIO_LONG, AUTOCORR_WINDOW)
            metrics.update(extra)
            for m in METRICS_TO_ANALYZE:
                val = metrics.get(m)
                all_metrics[m].append(val)
                per_strategy[m][strategy].append(val)

    print(f"\n\n{'='*80}")
    print("METRIC DISTRIBUTIONS — ALL STRATEGIES COMBINED")
    print(f"{'='*80}")

    for m in METRICS_TO_ANALYZE:
        print_metric_distribution(m, all_metrics[m])

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()