#!/usr/bin/env python3
"""
tools/backtest_comparison.py

Compares OOS1 backtest results month by month between two runs.
Loads trades_oos1_baseline_*.csv from two folders and compares profit per month.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# =============================================================================
# CONFIGURATION
# =============================================================================
TRADES_FOLDER_OLD = os.path.join(os.path.dirname(__file__), "brief_trades")
TRADES_FOLDER_NEW = os.path.join(os.path.dirname(__file__), "brief_trades_new")

TRADES_LABEL = "oos1_baseline"
CUTOFF_DATE  = None  # e.g. '2026-04-20' to compare only up to this date

# =============================================================================
# DATA LOADING
# =============================================================================
def load_trades(folder: str, label: str) -> pd.DataFrame:
    files = sorted(glob(os.path.join(folder, f"trades_{label}_*.csv")))
    if not files:
        return pd.DataFrame()
    all_trades = []
    for filepath in files:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower().str.strip()
        df['buy_time'] = pd.to_datetime(df['buy_time'])
        df['month'] = df['buy_time'].dt.to_period('M')
        all_trades.append(df)
    return pd.concat(all_trades, ignore_index=True).sort_values('buy_time').reset_index(drop=True)


# =============================================================================
# ANALYSIS
# =============================================================================
def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly profit, n_trades, win_rate per strategy."""
    rows = []
    for (strategy, month), grp in df.groupby(['strategy', 'month']):
        rows.append({
            'strategy':  strategy,
            'month':     month,
            'n_trades':  len(grp),
            'profit':    round(grp['profit'].sum(), 2),
            'win_rate':  round((grp['profit'] > 0).mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def compare_monthly(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Merge old and new monthly stats and compute differences."""
    stats_old = monthly_stats(df_old).rename(columns={
        'n_trades': 'n_trades_old', 'profit': 'profit_old', 'win_rate': 'wr_old'
    })
    stats_new = monthly_stats(df_new).rename(columns={
        'n_trades': 'n_trades_new', 'profit': 'profit_new', 'win_rate': 'wr_new'
    })
    merged = pd.merge(stats_old, stats_new, on=['strategy', 'month'], how='outer').fillna(0)
    merged['Δ_profit']   = round(merged['profit_new'] - merged['profit_old'], 2)
    merged['Δ_n_trades'] = merged['n_trades_new'] - merged['n_trades_old']
    merged['Δ_wr']       = round(merged['wr_new'] - merged['wr_old'], 1)
    return merged.sort_values(['strategy', 'month'])


# =============================================================================
# PRINTING
# =============================================================================
def print_strategy_comparison(df_cmp: pd.DataFrame, strategy: str):
    df_s = df_cmp[df_cmp['strategy'] == strategy]
    if df_s.empty:
        return

    print(f"\n  {'─'*115}")
    print(f"  {strategy}")
    print(f"  {'─'*115}")
    print(f"  {'Month':<10} {'N_old':>7} {'N_new':>7} {'ΔN':>5} {'PF_old':>10} {'PF_new':>10} {'Δ_profit':>10} {'WR_old':>8} {'WR_new':>8} {'Δ_WR':>6}")
    print(f"  {'─'*115}")

    total_old = 0.0
    total_new = 0.0
    for _, row in df_s.iterrows():
        color  = "\033[92m" if row['Δ_profit'] > 0 else "\033[91m" if row['Δ_profit'] < 0 else ""
        reset  = "\033[0m" if color else ""
        only_old = row['n_trades_new'] == 0
        only_new = row['n_trades_old'] == 0
        flag   = " ← only OLD" if only_old else " ← only NEW" if only_new else ""
        print(f"  {str(row['month']):<10} {int(row['n_trades_old']):>7} {int(row['n_trades_new']):>7} "
              f"{int(row['Δ_n_trades']):>+5} {row['profit_old']:>10.2f} {row['profit_new']:>10.2f} "
              f"{color}{row['Δ_profit']:>+10.2f}{reset} "
              f"{row['wr_old']:>7.1f}% {row['wr_new']:>7.1f}% {row['Δ_wr']:>+5.1f}%{flag}")
        total_old += row['profit_old']
        total_new += row['profit_new']

    delta_total = total_new - total_old
    color  = "\033[92m" if delta_total > 0 else "\033[91m" if delta_total < 0 else ""
    reset  = "\033[0m" if color else ""
    print(f"  {'─'*115}")
    print(f"  {'TOTAL':<10} {'':>7} {'':>7} {'':>5} {total_old:>10.2f} {total_new:>10.2f} "
          f"{color}{delta_total:>+10.2f}{reset}")


def print_system_summary(df_cmp: pd.DataFrame):
    """Print aggregated monthly comparison across all strategies."""
    df_agg = df_cmp.groupby('month').agg(
        n_trades_old=('n_trades_old', 'sum'),
        n_trades_new=('n_trades_new', 'sum'),
        profit_old=('profit_old', 'sum'),
        profit_new=('profit_new', 'sum'),
    ).reset_index()
    df_agg['Δ_profit']   = round(df_agg['profit_new'] - df_agg['profit_old'], 2)
    df_agg['Δ_n_trades'] = df_agg['n_trades_new'] - df_agg['n_trades_old']

    print(f"\n{'═'*115}")
    print(f"  SYSTEM TOTAL — ALL STRATEGIES COMBINED")
    print(f"{'═'*115}")
    print(f"  {'Month':<10} {'N_old':>8} {'N_new':>8} {'ΔN':>6} {'PF_old':>12} {'PF_new':>12} {'Δ_profit':>12}")
    print(f"  {'─'*80}")

    sys_old = 0.0
    sys_new = 0.0
    for _, row in df_agg.sort_values('month').iterrows():
        color = "\033[92m" if row['Δ_profit'] > 0 else "\033[91m" if row['Δ_profit'] < 0 else ""
        reset = "\033[0m" if color else ""
        print(f"  {str(row['month']):<10} {int(row['n_trades_old']):>8} {int(row['n_trades_new']):>8} "
              f"{int(row['Δ_n_trades']):>+6} {row['profit_old']:>12.2f} {row['profit_new']:>12.2f} "
              f"{color}{row['Δ_profit']:>+12.2f}{reset}")
        sys_old += row['profit_old']
        sys_new += row['profit_new']

    delta = sys_new - sys_old
    color = "\033[92m" if delta > 0 else "\033[91m" if delta < 0 else ""
    reset = "\033[0m" if color else ""
    print(f"  {'─'*80}")
    print(f"  {'TOTAL':<10} {'':>8} {'':>8} {'':>6} {sys_old:>12.2f} {sys_new:>12.2f} "
          f"{color}{delta:>+12.2f}{reset}")
    print(f"  {'═'*115}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 115)
    print(f"BACKTEST COMPARISON — {TRADES_LABEL}")
    print("=" * 115)
    print(f"\n  OLD folder: {TRADES_FOLDER_OLD}")
    print(f"  NEW folder: {TRADES_FOLDER_NEW}")

    df_old = load_trades(TRADES_FOLDER_OLD, TRADES_LABEL)
    df_new = load_trades(TRADES_FOLDER_NEW, TRADES_LABEL)

    if df_old.empty:
        print(f"\n  ❌ No trades found in OLD folder")
        return
    if df_new.empty:
        print(f"\n  ❌ No trades found in NEW folder")
        return

    if CUTOFF_DATE:
        df_old = df_old[df_old['buy_time'] <= CUTOFF_DATE]
        df_new = df_new[df_new['buy_time'] <= CUTOFF_DATE]
        print(f"\n  Cutoff date: {CUTOFF_DATE}")

    print(f"\n  OLD: {len(df_old)} trades | {df_old['strategy'].nunique()} strategies")
    print(f"  NEW: {len(df_new)} trades | {df_new['strategy'].nunique()} strategies")

    df_cmp = compare_monthly(df_old, df_new)

    # Per strategy
    print(f"\n{'═'*115}")
    print(f"  PER STRATEGY — monthly comparison")
    print(f"{'═'*115}")
    for strategy in sorted(df_cmp['strategy'].unique()):
        print_strategy_comparison(df_cmp, strategy)

    # System summary
    print_system_summary(df_cmp)

    print(f"\n{'='*115}\n")


if __name__ == "__main__":
    main()