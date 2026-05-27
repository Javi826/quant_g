# develop/market_regime/analyze_autocor_returns.py
"""
Estimates the optimal block_size for MC price path generation (bootstrap of returns).

Analyzes the autocorrelation structure of positive/negative return streaks
for the top N symbols by volume, and recommends a block_size.

Usage: run directly, adjust CONFIGURATION section.
"""


import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

# =============================================================================
# CONFIGURATION
# =============================================================================
SPLIT_MODE  = "expanding"
SPLIT_BASE  = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER = os.path.join(SPLIT_BASE, "IS", "crypto_2024-01_2025-05_IS")
TIMEFRAME   = "1H"
TOP_N       = 20
VOLUME_COL  = "volume_quote"  # column used to rank symbols by volume


# =============================================================================
# DATA LOADING
# =============================================================================

def load_top_symbols(data_folder: str, timeframe: str, top_n: int) -> dict:
    """Load OHLCV for top N symbols by average volume."""
    pattern = str(Path(data_folder) / f"*_{timeframe}.parquet")
    files   = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_folder} for timeframe {timeframe}")

    vol_map = {}
    for f in files:
        symbol = Path(f).stem.replace(f"_{timeframe}", "")
        try:
            df = pd.read_parquet(f)
            if VOLUME_COL in df.columns:
                vol_map[symbol] = df[VOLUME_COL].mean()
            elif "volume" in df.columns:
                vol_map[symbol] = df["volume"].mean()
        except Exception:
            continue

    top_symbols = sorted(vol_map, key=vol_map.get, reverse=True)[:top_n]
    print(f"\n  Top {top_n} symbols by avg volume ({TIMEFRAME}):")
    for i, sym in enumerate(top_symbols, 1):
        print(f"    {i:>2}. {sym:<20} avg_vol={vol_map[sym]:,.0f}")

    ohlcv = {}
    for sym in top_symbols:
        f = str(Path(data_folder) / f"{sym}_{timeframe}.parquet")
        try:
            df = pd.read_parquet(f)
            df.columns = df.columns.str.lower()
            ohlcv[sym] = df
        except Exception:
            continue

    return ohlcv


# =============================================================================
# STREAK ANALYSIS
# =============================================================================

def compute_streaks(returns: np.ndarray) -> list:
    """Compute run lengths of consecutive positive/negative returns."""
    if len(returns) == 0:
        return []
    streaks     = []
    current_sign = np.sign(returns[0])
    current_len  = 1
    for r in returns[1:]:
        sign = np.sign(r)
        if sign == current_sign:
            current_len += 1
        else:
            streaks.append(current_len)
            current_sign = sign
            current_len  = 1
    streaks.append(current_len)
    return streaks


def analyze_symbol_streaks(df: pd.DataFrame) -> dict:
    """Compute streak stats for a single symbol."""
    returns = df["close"].pct_change().dropna().values
    streaks = compute_streaks(returns)
    if not streaks:
        return {}
    return {
        "mean":   np.mean(streaks),
        "median": np.median(streaks),
        "p75":    np.percentile(streaks, 75),
        "p90":    np.percentile(streaks, 90),
        "max":    max(streaks),
        "n_obs":  len(returns),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(ohlcv: dict) -> None:
    """Analyze return streaks for all symbols and print recommendation."""
    print(f"\n{'='*80}")
    print(f"  RETURN STREAK ANALYSIS — {DATA_FOLDER.split('/')[-1]}  |  {TIMEFRAME}")
    print(f"{'='*80}")
    print(f"\n  {'SYMBOL':<22} {'N_OBS':>8} {'MEAN':>8} {'MEDIAN':>8} {'P75':>8} {'P90':>8} {'MAX':>8}")
    print(f"  {'-'*75}")

    results = {}
    for sym, df in ohlcv.items():
        stats = analyze_symbol_streaks(df)
        if not stats:
            continue
        results[sym] = stats
        print(f"  {sym:<22} {stats['n_obs']:>8} {stats['mean']:>8.1f} {stats['median']:>8.1f} "
              f"{stats['p75']:>8.1f} {stats['p90']:>8.1f} {stats['max']:>8.0f}")

    if not results:
        print("  No data to analyze.")
        return

    # Aggregate across symbols
    means   = [s["mean"]   for s in results.values()]
    medians = [s["median"] for s in results.values()]
    p75s    = [s["p75"]    for s in results.values()]

    agg_mean   = np.mean(means)
    agg_median = np.median(medians)
    agg_p75    = np.mean(p75s)

    print(f"\n{'─'*80}")
    print(f"  AGGREGATE ACROSS {len(results)} SYMBOLS")
    print(f"{'─'*80}")
    print(f"  {'METRIC':<30} {'VALUE':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Mean of means':<30} {agg_mean:>10.1f}")
    print(f"  {'Median of medians':<30} {agg_median:>10.1f}")
    print(f"  {'Mean of P75s':<30} {agg_p75:>10.1f}")

    print(f"\n{'='*80}")
    print(f"  BLOCK_SIZE RECOMMENDATION (in {TIMEFRAME} bins)")
    print(f"{'='*80}")
    print(f"  Conservative (mean of means)  : block_size = {int(round(agg_mean))}")
    print(f"  Moderate     (median of P75s) : block_size = {int(round(agg_p75))}")
    print(f"  Aggressive   (2× mean)        : block_size = {int(round(agg_mean * 2))}")
    print(f"\n  → In hours (1 bin = 1 {TIMEFRAME}):")
    tf_hours = {"15m": 0.25, "30m": 0.5, "1H": 1, "4H": 4, "6Hutc": 6, "1Dutc": 24}
    h = tf_hours.get(TIMEFRAME, 1)
    print(f"    Conservative : {agg_mean * h:.1f}h")
    print(f"    Moderate     : {agg_p75 * h:.1f}h")
    print(f"{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Loading top {TOP_N} symbols from {DATA_FOLDER.split('/')[-1]} [{TIMEFRAME}]...")
    ohlcv = load_top_symbols(DATA_FOLDER, TIMEFRAME, TOP_N)
    run_analysis(ohlcv)