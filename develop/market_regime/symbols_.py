#!/usr/bin/env python3
"""
develop/market_regime/symbol_analysis.py

Analyzes performance per symbol across OOS periods.
Shows per-symbol metrics for baseline or regime trades.

Usage:
    python symbol_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

# =============================================================================
# CONFIGURATION
# =============================================================================
TRADES_FOLDER  = os.path.join(os.path.dirname(__file__), "..", "brief_trades")
ANALYSIS_TYPE  = "regime"  # "baseline" | "regime"

PERIOD_LABELS = [
    ("OOS1", f"oos1_{ANALYSIS_TYPE}"),
    ("OOS2", f"oos2_{ANALYSIS_TYPE}"),
    ("OOS3", f"oos3_{ANALYSIS_TYPE}"),
]

# =============================================================================
# DATA LOADING
# =============================================================================
def load_trades_for_label(label: str) -> pd.DataFrame:
    files = sorted(glob(str(Path(TRADES_FOLDER) / f"trades_{label}_*.csv")))
    if not files:
        return pd.DataFrame()
    all_trades = []
    for filepath in files:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower().str.strip()
        df['buy_time'] = pd.to_datetime(df['buy_time'])
        all_trades.append(df)
    return pd.concat(all_trades, ignore_index=True).sort_values('buy_time').reset_index(drop=True)


# =============================================================================
# ANALYSIS
# =============================================================================
def analyze_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol metrics for a given trades DataFrame."""
    if df.empty:
        return pd.DataFrame()

    n_trades_period = len(df)
    wr_period       = round((df['profit'] > 0).mean() * 100, 2)
    profit_period   = round(df['profit'].sum(), 2)

    rows = []
    for symbol, df_sym in df.groupby('symbol'):
        n_sym      = len(df_sym)
        wr_sym     = round((df_sym['profit'] > 0).mean() * 100, 2)
        profit_sym = round(df_sym['profit'].sum(), 2)
        pct_trades = round(n_sym / n_trades_period * 100, 2)
        pct_profit = round(profit_sym / profit_period * 100, 2) if profit_period != 0 else 0.0

        rows.append({
            'Symbol':          symbol,
            'N_trades_period': n_trades_period,
            'N_trades_symbol': n_sym,
            '%Trades':         pct_trades,
            'WR_period':       wr_period,
            'WR_symbol':       wr_sym,
            'Profit_period':   profit_period,
            'Profit_symbol':   profit_sym,
            '%Profit':         pct_profit,
        })

    df_out = pd.DataFrame(rows).sort_values('Profit_symbol', ascending=False).reset_index(drop=True)
    return df_out


# =============================================================================
# PRINTING
# =============================================================================
def print_symbol_table(df_out: pd.DataFrame, title: str) -> None:
    if df_out.empty:
        print(f"\n  No data for {title}")
        return

    print(f"\n{'═'*120}")
    print(f"  {title}")
    print(f"{'═'*120}")
    print(f"  {'Symbol':<20} {'N_period':>10} {'N_symbol':>10} {'%Trades':>9} {'WR_period':>11} {'WR_symbol':>11} {'PF_period':>12} {'PF_symbol':>12} {'%Profit':>9}")
    print(f"  {'─'*115}")

    for _, row in df_out.iterrows():
        wr_color = "\033[92m" if row['WR_symbol'] > row['WR_period'] else "\033[91m"
        reset    = "\033[0m"
        pf_color = "\033[92m" if row['Profit_symbol'] > 0 else "\033[91m"
        print(f"  {row['Symbol']:<20} {row['N_trades_period']:>10} {row['N_trades_symbol']:>10} "
              f"{row['%Trades']:>8.2f}% "
              f"{row['WR_period']:>10.2f}% "
              f"{wr_color}{row['WR_symbol']:>10.2f}%{reset} "
              f"{row['Profit_period']:>12.2f} "
              f"{pf_color}{row['Profit_symbol']:>12.2f}{reset} "
              f"{row['%Profit']:>8.2f}%")

    print(f"  {'─'*115}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print(f"SYMBOL ANALYSIS — {ANALYSIS_TYPE.upper()}")
    print("=" * 80)
    print(f"\n  Trades folder  : {TRADES_FOLDER}")
    print(f"  Analysis type  : {ANALYSIS_TYPE}")

    all_dfs  = []
    for period_name, label in PERIOD_LABELS:
        df = load_trades_for_label(label)
        if df.empty:
            print(f"\n  No files for {label} — skipping.")
            continue
        print(f"\n  [{period_name}] {len(df)} trades loaded")
        df_out = analyze_symbols(df)
        print_symbol_table(df_out, f"{period_name} — {ANALYSIS_TYPE}")
        all_dfs.append(df)

    # Aggregated table
    if len(all_dfs) > 1:
        df_all = pd.concat(all_dfs, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
        df_agg = analyze_symbols(df_all)
        print_symbol_table(df_agg, f"ALL OOS COMBINED — {ANALYSIS_TYPE}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()