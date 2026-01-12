"""
market_regime/position_sizer.py

Applies position sizing based on regime family.
Processes all enriched files and shows summary.

Usage:
    python position_sizer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.config import (
    OUTPUT_FOLDER, FAMILIES, FAMILY_SIZING, INITIAL_CAPITAL
)


def classify_trade(row: pd.Series, families: dict) -> str:
    """Classifies a trade into a family based on its metrics."""
    for family_name, rules in families.items():
        if not rules:
            continue
        match = True
        for metric, (op, val) in rules.items():
            if metric not in row or pd.isna(row[metric]):
                match = False
                break
            if op == '>' and not (row[metric] > val):
                match = False
                break
            elif op == '<' and not (row[metric] < val):
                match = False
                break
        if match:
            return family_name
    for family_name, rules in families.items():
        if not rules:
            return family_name
    return 'unknown'


def load_enriched_trades(filepath: str) -> pd.DataFrame:
    """Loads enriched trades from Excel file."""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    return df


def process_single_file(filepath: str, families: dict, sizing: dict, initial_capital: float) -> dict:
    """Processes a single enriched file and returns results."""
    strategy = Path(filepath).stem.replace('trades_enriched_', '')
    df = load_enriched_trades(filepath)
    
    # Classify and apply sizing
    df['family'] = df.apply(lambda row: classify_trade(row, families), axis=1)
    df['sizing_mult'] = df['family'].map(sizing).fillna(1.0)
    df['profit_sized'] = df['profit'] * df['sizing_mult']
    
    # Sort by time
    df = df.sort_values('buy_time').reset_index(drop=True)
    
    # Calculate cumulative and drawdowns
    df['cum_base'] = df['profit'].cumsum()
    df['cum_sized'] = df['profit_sized'].cumsum()
    df['dd_base'] = df['cum_base'].cummax() - df['cum_base']
    df['dd_sized'] = df['cum_sized'].cummax() - df['cum_sized']
    
    # Metrics
    num_trades = len(df)
    profit_base = df['profit'].sum()
    profit_sized = df['profit_sized'].sum()
    delta_pct = ((profit_sized - profit_base) / abs(profit_base) * 100) if profit_base != 0 else 0
    win_rate = (df['profit'] > 0).mean() * 100
    dd_base_pct = (df['dd_base'].max() / initial_capital) * 100
    dd_sized_pct = (df['dd_sized'].max() / initial_capital) * 100

    # Delta de drawdown respecto al base (en puntos porcentuales)
    dd_delta_pct = ((df['dd_sized'].max() - df['dd_base'].max()) / initial_capital) * 100
    
    # Family breakdown
    family_stats = []
    for fam in df['family'].unique():
        fam_df = df[df['family'] == fam]
        fam_profit_b = fam_df['profit'].sum()
        fam_profit_s = fam_df['profit_sized'].sum()
        fam_delta = ((fam_profit_s - fam_profit_b) / abs(fam_profit_b) * 100) if fam_profit_b != 0 else 0
        family_stats.append({
            'family': fam,
            'trades': len(fam_df),
            'profit_b': fam_profit_b,
            'profit_s': fam_profit_s,
            'delta_pct': fam_delta
        })
    
    return {
        'strategy': strategy,
        'num_trades': num_trades,
        'profit_base': profit_base,
        'profit_sized': profit_sized,
        'delta_pct': delta_pct,
        'win_rate': win_rate,
        'dd_base_pct': dd_base_pct,
        'dd_sized_pct': dd_sized_pct,
        'dd_delta_pct': dd_delta_pct,
        'family_stats': family_stats
    }


def print_file_results(r: dict):
    """Prints results for a single file."""
    print(f"\n{'='*70}")
    print(f"STRATEGY: {r['strategy']}")
    print(f"{'='*70}")
    
    # Main metrics
    profit_ok = "✅" if r['profit_sized'] > r['profit_base'] else "❌"
    dd_ok = "✅" if r['dd_sized_pct'] < r['dd_base_pct'] else "❌"
    
    print(f"Trades: {r['num_trades']}  |  Win%: {r['win_rate']:.1f}%")
    print(f"Profit:  {r['profit_base']:>8.2f} → {r['profit_sized']:>8.2f}  ({r['delta_pct']:+.1f}%) {profit_ok}")
    print(f"Max DD%: {r['dd_base_pct']:>7.2f}% → {r['dd_sized_pct']:>7.2f}% {dd_ok}")
    
    # Family breakdown
    print(f"\n{'FAMILY':<12} {'TRADES':>7} {'PROFIT_B':>10} {'PROFIT_S':>10} {'Δ%':>8}")
    print("-" * 50)
    for fs in r['family_stats']:
        print(f"{fs['family']:<12} {fs['trades']:>7} {fs['profit_b']:>10.2f} {fs['profit_s']:>10.2f} {fs['delta_pct']:>+7.1f}%")
    print("-" * 50)


def apply_sizing(
    output_folder: str = None,
    families: dict = None,
    sizing: dict = None,
    initial_capital: float = None
) -> list:
    """Applies position sizing to all enriched files."""
    output_folder = output_folder or OUTPUT_FOLDER
    families = families or FAMILIES
    sizing = sizing or FAMILY_SIZING
    initial_capital = initial_capital or INITIAL_CAPITAL
    
    print("=" * 70)
    print("POSITION SIZER - Regime-based position sizing")
    print("=" * 70)
    
    # Show config
    print("\nSizing configuration:")
    for fam, mult in sizing.items():
        rules = families.get(fam, {})
        rules_str = ' & '.join([f"{m}{op}{v}" for m, (op, v) in rules.items()]) if rules else "(default)"
        print(f"  {fam:<12}: x{mult:.1f}  [{rules_str}]")
    
    # Find files
    pattern = os.path.join(output_folder, "trades_enriched_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {output_folder}")
        return []
    
    print(f"\nFiles found: {len(files)}")
    
    # Process each file
    results = []
    for f in files:
        r = process_single_file(f, families, sizing, initial_capital)
        results.append(r)
        print_file_results(r)
           # =========================
        # Equity curve plot
        # =========================
        try:
            import matplotlib.pyplot as plt

            df_plot = load_enriched_trades(f)
            df_plot['family'] = df_plot.apply(lambda row: classify_trade(row, families), axis=1)
            df_plot['sizing_mult'] = df_plot['family'].map(sizing).fillna(1.0)
            df_plot['profit_sized'] = df_plot['profit'] * df_plot['sizing_mult']

            df_plot = df_plot.sort_values('buy_time').reset_index(drop=True)

            df_plot['cum_base'] = df_plot['profit'].cumsum()
            df_plot['cum_sized'] = df_plot['profit_sized'].cumsum()

            plt.figure(figsize=(10, 5))
            plt.plot(df_plot['buy_time'], df_plot['cum_base'], label='Equity Base')
            plt.plot(df_plot['buy_time'], df_plot['cum_sized'], label='Equity Regime Sizing')

            plt.title(f"Equity Curve – {r['strategy']}")
            plt.xlabel("Time")
            plt.ylabel("Cumulative Profit")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️ Plot skipped for {r['strategy']} ({e})")
    
    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY - ALL STRATEGIES")
    print(f"{'='*70}")
    
    # Header now incluye ΔDD%
    print(f"\n{'STRATEGY':<30} {'TRADES':>7} {'PROFIT_BASE':>12} {'PROFIT_SIZING':>14} {'Δ%':>7} {'DD%_BASE':>9} {'DD%_SIZING':>11} {'ΔDD%':>8}")
    print("-" * 100)
    
    for r in results:
        profit_ok = "✅" if r['profit_sized'] > r['profit_base'] else "❌"
        dd_ok = "✅" if r['dd_sized_pct'] < r['dd_base_pct'] else "❌"
        print(f"{r['strategy']:<30} {r['num_trades']:>7} {r['profit_base']:>12.2f} {r['profit_sized']:>11.2f} {profit_ok} {r['delta_pct']:>+6.1f}% {r['dd_base_pct']:>9.1f}% {r['dd_sized_pct']:>10.1f}% {r['dd_delta_pct']:>+7.1f}% {dd_ok}")
    
    print("-" * 100)
    
    # Totals
    n = len(results)
    if n > 0:
        total_trades = sum(r['num_trades'] for r in results)
        total_profit_b = sum(r['profit_base'] for r in results)
        total_profit_s = sum(r['profit_sized'] for r in results)
        total_delta = ((total_profit_s - total_profit_b) / abs(total_profit_b) * 100) if total_profit_b != 0 else 0
        avg_dd_b = sum(r['dd_base_pct'] for r in results) / n
        avg_dd_s = sum(r['dd_sized_pct'] for r in results) / n
        # Delta medio de DD (sizing - base)
        total_dd_delta = avg_dd_s - avg_dd_b
        
        profit_ok = "✅" if total_profit_s > total_profit_b else "❌"
        dd_ok = "✅" if avg_dd_s < avg_dd_b else "❌"
        print(f"{'TOTAL':<30} {total_trades:>7} {total_profit_b:>12.2f} {total_profit_s:>11.2f} {profit_ok} {total_delta:>+6.1f}% {avg_dd_b:>9.1f}% {avg_dd_s:>10.1f}% {total_dd_delta:>+7.1f}% {dd_ok}")
    
    print(f"\n{'='*70}")
    
    return results


if __name__ == "__main__":
    apply_sizing()

