#!/usr/bin/env python3
"""
tools/data_comparison.py

Compares old vs new data folders to identify differences.
Checks: number of files, number of rows, date ranges, and value differences.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# =============================================================================
# CONFIGURATION
# =============================================================================
SPLIT_BASE = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "data", "04_split", "expanding")

COMPARISONS = [
    {
        "label": "IS",
        "old":   os.path.join(SPLIT_BASE, "IS", "crypto_2024-01_2025-04_IS"),
        "new":   os.path.join(SPLIT_BASE, "IS", "crypto_2024-01_2025-04_IS_new"),
    },
    {
        "label": "OOS1",
        "old":   os.path.join(SPLIT_BASE, "OOS", "crypto_2025-04_2026-04_OOS"),
        "new":   os.path.join(SPLIT_BASE, "OOS", "crypto_2025-04_2026-05_OOS_new"),
    },
]

TIMEFRAME = "1H"   # timeframe to compare
TOP_N_DIFF = 5     # show top N symbols with most differences

# =============================================================================
# ANALYSIS
# =============================================================================
def get_files(folder: str, timeframe: str) -> dict:
    """Returns {symbol: filepath} for all files matching timeframe."""
    files = sorted(glob(os.path.join(folder, f"*_{timeframe}.parquet")))
    return {os.path.basename(f).replace(f"_{timeframe}.parquet", ""): f for f in files}


def compare_symbol(symbol: str, path_old: str, path_new: str) -> dict:
    """Compare a single symbol between old and new."""
    df_old = pd.read_parquet(path_old)
    df_new = pd.read_parquet(path_new)

    # Normalize columns
    df_old.columns = df_old.columns.str.lower()
    df_new.columns = df_new.columns.str.lower()

    # Get timestamp column
    ts_col = 'timestamp' if 'timestamp' in df_old.columns else df_old.index.name or 'index'
    if ts_col in df_old.columns:
        df_old['ts'] = pd.to_datetime(df_old[ts_col])
        df_new['ts'] = pd.to_datetime(df_new[ts_col])
    else:
        df_old['ts'] = pd.to_datetime(df_old.index)
        df_new['ts'] = pd.to_datetime(df_new.index)

    n_old = len(df_old)
    n_new = len(df_new)

    # Common rows by timestamp
    merged = pd.merge(
        df_old[['ts', 'close']].rename(columns={'close': 'close_old'}),
        df_new[['ts', 'close']].rename(columns={'close': 'close_new'}),
        on='ts', how='inner'
    )
    n_common     = len(merged)
    n_diff_close = (merged['close_old'] != merged['close_new']).sum() if n_common > 0 else 0
    max_diff     = (merged['close_old'] - merged['close_new']).abs().max() if n_common > 0 else 0

    return {
        'symbol':       symbol,
        'n_old':        n_old,
        'n_new':        n_new,
        'n_diff_rows':  n_new - n_old,
        'date_start_old': df_old['ts'].min(),
        'date_end_old':   df_old['ts'].max(),
        'date_start_new': df_new['ts'].min(),
        'date_end_new':   df_new['ts'].max(),
        'n_common':     n_common,
        'n_diff_close': n_diff_close,
        'max_diff':     round(max_diff, 8),
    }


def analyze_comparison(label: str, folder_old: str, folder_new: str, timeframe: str):
    print(f"\n{'═'*120}")
    print(f"  COMPARISON: {label}  [{timeframe}]")
    print(f"  OLD: {folder_old}")
    print(f"  NEW: {folder_new}")
    print(f"{'═'*120}")

    if not os.path.exists(folder_old):
        print(f"  ❌ OLD folder not found: {folder_old}")
        return
    if not os.path.exists(folder_new):
        print(f"  ❌ NEW folder not found: {folder_new}")
        return

    files_old = get_files(folder_old, timeframe)
    files_new = get_files(folder_new, timeframe)

    print(f"\n  Files OLD: {len(files_old)}  |  Files NEW: {len(files_new)}")

    only_old = set(files_old) - set(files_new)
    only_new = set(files_new) - set(files_old)
    common   = set(files_old) & set(files_new)

    if only_old:
        print(f"  Only in OLD ({len(only_old)}): {sorted(only_old)}")
    if only_new:
        print(f"  Only in NEW ({len(only_new)}): {sorted(only_new)}")
    print(f"  Common symbols: {len(common)}")

    if not common:
        return

    # Compare common symbols
    results = []
    for symbol in sorted(common):
        r = compare_symbol(symbol, files_old[symbol], files_new[symbol])
        results.append(r)

    df_results = pd.DataFrame(results)

    # Summary
    n_row_diff    = (df_results['n_diff_rows'] != 0).sum()
    n_close_diff  = (df_results['n_diff_close'] > 0).sum()
    total_changed = (df_results['n_diff_close'] > 0).sum()

    print(f"\n  {'─'*80}")
    print(f"  Symbols with different row count : {n_row_diff}")
    print(f"  Symbols with different close values: {n_close_diff}")
    print(f"  {'─'*80}")

    # Header
    print(f"\n  {'Symbol':<20} {'N_old':>8} {'N_new':>8} {'Δ_rows':>8} {'N_common':>10} {'Δ_close':>10} {'Max_diff':>12}  Date_start_old → Date_start_new")
    print(f"  {'─'*115}")

    # Sort by n_diff_close descending
    df_sorted = df_results.sort_values('n_diff_close', ascending=False)
    for _, r in df_sorted.iterrows():
        date_diff = "✅" if r['date_start_old'] == r['date_start_new'] and r['date_end_old'] == r['date_end_new'] else "⚠️ "
        print(f"  {r['symbol']:<20} {r['n_old']:>8} {r['n_new']:>8} {r['n_diff_rows']:>+8} "
              f"{r['n_common']:>10} {r['n_diff_close']:>10} {r['max_diff']:>12}  "
              f"{date_diff} {str(r['date_start_old'])[:10]} → {str(r['date_start_new'])[:10]}  |  "
              f"{str(r['date_end_old'])[:10]} → {str(r['date_end_new'])[:10]}")

    print(f"  {'─'*115}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 120)
    print("DATA COMPARISON — old vs new downloads")
    print("=" * 120)
    print(f"\n  Timeframe: {TIMEFRAME}")

    for comp in COMPARISONS:
        analyze_comparison(comp['label'], comp['old'], comp['new'], TIMEFRAME)

    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()