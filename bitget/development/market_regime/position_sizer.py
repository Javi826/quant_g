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


# Define generator display order
GENERATOR_ORDER = ['parity', 'reversal', 'ranging', 'double_top', 'orderblocks', 'unknown']


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


def extract_generator(strategy_name: str) -> str:
    """Extracts generator type from strategy name."""
    name_lower = strategy_name.lower()
    if 'parity' in name_lower:
        return 'parity'
    elif 'reversal' in name_lower:
        return 'reversal'
    elif 'ranging' in name_lower:
        return 'ranging'
    elif 'double_top' in name_lower or 'doubletop' in name_lower:
        return 'double_top'
    elif 'orderblock' in name_lower:
        return 'orderblocks'
    else:
        return 'unknown'


def load_enriched_trades(filepath: str) -> pd.DataFrame:
    """Loads enriched trades from Excel file."""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    return df


def calculate_max_dd_pct(equity_curve: pd.Series) -> float:
    """
    Calculates Maximum Drawdown % correctly.
    DD% = max((peak - valley) / peak * 100)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    cummax = equity_curve.cummax()
    
    # Avoid division by zero
    drawdown_pct = np.where(
        cummax > 0,
        ((cummax - equity_curve) / cummax) * 100,
        0.0
    )
    
    return float(np.max(drawdown_pct))


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
    
    # Calculate cumulative equity (starting from initial_capital)
    df['equity_base'] = initial_capital + df['profit'].cumsum()
    df['equity_sized'] = initial_capital + df['profit_sized'].cumsum()
    
    # Metrics
    num_trades = len(df)
    trades_sizing = (df['sizing_mult'] > 0).sum()
    profit_base = df['profit'].sum()
    profit_sized = df['profit_sized'].sum()
    delta_pct = ((profit_sized - profit_base) / abs(profit_base) * 100) if profit_base != 0 else 0
    win_rate = (df['profit'] > 0).mean() * 100
    
    # Calculate Max DD% correctly (from peak)
    dd_base_pct = calculate_max_dd_pct(df['equity_base'])
    dd_sized_pct = calculate_max_dd_pct(df['equity_sized'])
    dd_delta_pct = dd_sized_pct - dd_base_pct
    
    # Family breakdown
    family_stats = []
    for fam in df['family'].unique():
        fam_df = df[df['family'] == fam]
        fam_trades_base = len(fam_df)
        fam_trades_sizing = (fam_df['sizing_mult'] > 0).sum()
        fam_profit_b = fam_df['profit'].sum()
        fam_profit_s = fam_df['profit_sized'].sum()
        fam_delta = ((fam_profit_s - fam_profit_b) / abs(fam_profit_b) * 100) if fam_profit_b != 0 else 0
        family_stats.append({
            'family': fam,
            'trades_base': fam_trades_base,
            'trades_sizing': fam_trades_sizing,
            'profit_b': fam_profit_b,
            'profit_s': fam_profit_s,
            'delta_pct': fam_delta
        })
    
    return {
        'strategy': strategy,
        'generator': extract_generator(strategy),
        'num_trades': num_trades,
        'trades_sizing': trades_sizing,
        'profit_base': profit_base,
        'profit_sized': profit_sized,
        'delta_pct': delta_pct,
        'win_rate': win_rate,
        'dd_base_pct': dd_base_pct,
        'dd_sized_pct': dd_sized_pct,
        'dd_delta_pct': dd_delta_pct,
        'family_stats': family_stats,
        'df': df
    }


def print_file_results(r: dict):
    """Prints results for a single file."""
    print(f"\n{'='*70}")
    print(f"STRATEGY: {r['strategy']}")
    print(f"{'='*70}")
    
    profit_ok = "✅" if r['profit_sized'] > r['profit_base'] else "❌"
    dd_ok = "✅" if r['dd_sized_pct'] < r['dd_base_pct'] else "❌"
    
    print(f"Trades: {r['num_trades']} (base) → {r['trades_sizing']} (sizing)  |  Win%: {r['win_rate']:.1f}%")
    print(f"Profit:  {r['profit_base']:>8.2f} → {r['profit_sized']:>8.2f}  ({r['delta_pct']:+.1f}%) {profit_ok}")
    print(f"Max DD%: {r['dd_base_pct']:>7.2f}% → {r['dd_sized_pct']:>7.2f}% {dd_ok}")
    
    print(f"\n{'FAMILY':<12} {'TRADES_B':>9} {'TRADES_S':>9} {'PROFIT_B':>10} {'PROFIT_S':>10} {'Δ%':>8}")
    print("-" * 62)
    for fs in r['family_stats']:
        print(f"{fs['family']:<12} {fs['trades_base']:>9} {fs['trades_sizing']:>9} {fs['profit_b']:>10.2f} {fs['profit_s']:>10.2f} {fs['delta_pct']:>+7.1f}%")
    print("-" * 62)


def apply_sizing(
    output_folder: str = None,
    families: dict = None,
    sizing: dict = None,
    initial_capital: float = None,
    show_plots: bool = True
) -> list:
    """Applies position sizing to all enriched files."""
    output_folder = output_folder or OUTPUT_FOLDER
    families = families or FAMILIES
    sizing = sizing or FAMILY_SIZING
    initial_capital = initial_capital or INITIAL_CAPITAL
    
    print("=" * 70)
    print("POSITION SIZER - Regime-based position sizing")
    print("=" * 70)
    
    print("\nSizing configuration:")
    for fam, mult in sizing.items():
        rules = families.get(fam, {})
        rules_str = ' & '.join([f"{m}{op}{v}" for m, (op, v) in rules.items()]) if rules else "(default)"
        print(f"  {fam:<12}: x{mult:.1f}  [{rules_str}]")
    
    pattern = os.path.join(output_folder, "trades_enriched_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {output_folder}")
        return []
    
    print(f"\nFiles found: {len(files)}")
    
    results = []
    for f in files:
        r = process_single_file(f, families, sizing, initial_capital)
        results.append(r)
        print_file_results(r)
        
        if show_plots:
            try:
                import matplotlib.pyplot as plt
                df_plot = r['df']
                plt.figure(figsize=(10, 5))
                plt.plot(df_plot['buy_time'], df_plot['equity_base'], label='Equity Base')
                plt.plot(df_plot['buy_time'], df_plot['equity_sized'], label='Equity Regime Sizing')
                plt.title(f"Equity Curve – {r['strategy']}")
                plt.xlabel("Time")
                plt.ylabel("Equity")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"⚠️ Plot skipped for {r['strategy']} ({e})")
    
    # Sort results by generator order
    results.sort(key=lambda x: GENERATOR_ORDER.index(x['generator']) if x['generator'] in GENERATOR_ORDER else 999)
    
    # =================================================================
    # CREATE COMBINED DATAFRAME (for portfolio-level calculations)
    # =================================================================
    all_dfs = []
    for r in results:
        df_copy = r['df'].copy()
        df_copy['strategy'] = r['strategy']
        all_dfs.append(df_copy)
    
    combined_df = pd.concat(all_dfs, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
    
    # Calculate PORTFOLIO-LEVEL equity and DD
    combined_df['equity_base_portfolio'] = initial_capital + combined_df['profit'].cumsum()
    combined_df['equity_sized_portfolio'] = initial_capital + combined_df['profit_sized'].cumsum()
    
    portfolio_dd_base_pct = calculate_max_dd_pct(combined_df['equity_base_portfolio'])
    portfolio_dd_sized_pct = calculate_max_dd_pct(combined_df['equity_sized_portfolio'])
    portfolio_dd_delta_pct = portfolio_dd_sized_pct - portfolio_dd_base_pct
    
    # =================================================================
    # RESUMEN POR FAMILIA
    # =================================================================
    print(f"\n{'='*150}")
    print("RESUMEN POR FAMILIA (todas las estrategias agregadas)")
    print(f"{'='*150}")
    
    family_aggregates = {}
    for fam in combined_df['family'].unique():
        fam_df = combined_df[combined_df['family'] == fam].copy()
        fam_df = fam_df.sort_values('buy_time').reset_index(drop=True)
        
        # Calculate equity curves for this family
        fam_df['equity_base'] = initial_capital + fam_df['profit'].cumsum()
        fam_df['equity_sized'] = initial_capital + fam_df['profit_sized'].cumsum()
        
        family_aggregates[fam] = {
            'trades_base': len(fam_df),
            'trades_sizing': (fam_df['sizing_mult'] > 0).sum(),
            'profit_b': fam_df['profit'].sum(),
            'profit_s': fam_df['profit_sized'].sum(),
            'dd_base_pct': calculate_max_dd_pct(fam_df['equity_base']),
            'dd_sized_pct': calculate_max_dd_pct(fam_df['equity_sized'])
        }
    
    # Calculate total trades for percentage calculation
    total_trades_base_all = sum(agg['trades_base'] for agg in family_aggregates.values())
    
    print(f"\n{'FAMILY':<15} {'TRADES_BASE':>12} {'TRADES_%':>10} {'TRADES_SIZING':>14} {'PROFIT_BASE':>13} {'PROFIT_SIZING':>15} {'Δ%':>8} {'DD%_BASE':>10} {'DD%_SIZING':>12} {'ΔDD%':>8}")
    print("-" * 150)
    
    total_trades_base = 0
    total_trades_sizing = 0
    total_profit_base = 0
    total_profit_sizing = 0
    
    for fam in sorted(family_aggregates.keys()):
        agg = family_aggregates[fam]
        trades_pct = (agg['trades_base'] / total_trades_base_all * 100) if total_trades_base_all > 0 else 0
        delta_pct = ((agg['profit_s'] - agg['profit_b']) / abs(agg['profit_b']) * 100) if agg['profit_b'] != 0 else 0
        dd_delta_pct = agg['dd_sized_pct'] - agg['dd_base_pct']
        profit_ok = "✅" if agg['profit_s'] > agg['profit_b'] else "❌"
        dd_ok = "✅" if agg['dd_sized_pct'] < agg['dd_base_pct'] else "❌"
        
        print(f"{fam:<15} {agg['trades_base']:>12} {trades_pct:>9.1f}% {agg['trades_sizing']:>14} {agg['profit_b']:>13.2f} {agg['profit_s']:>15.2f} {profit_ok} {delta_pct:>+6.1f}% {agg['dd_base_pct']:>9.1f}% {agg['dd_sized_pct']:>11.1f}% {dd_ok} {dd_delta_pct:>+6.1f}%")
        
        total_trades_base += agg['trades_base']
        total_trades_sizing += agg['trades_sizing']
        total_profit_base += agg['profit_b']
        total_profit_sizing += agg['profit_s']
    
    print("-" * 150)
    
    # TOTAL: Use portfolio-level DD
    total_trades_pct = 100.0
    total_delta_pct = ((total_profit_sizing - total_profit_base) / abs(total_profit_base) * 100) if total_profit_base != 0 else 0
    total_profit_ok = "✅" if total_profit_sizing > total_profit_base else "❌"
    total_dd_ok = "✅" if portfolio_dd_sized_pct < portfolio_dd_base_pct else "❌"
    print(f"{'TOTAL':<15} {total_trades_base:>12} {total_trades_pct:>9.1f}% {total_trades_sizing:>14} {total_profit_base:>13.2f} {total_profit_sizing:>15.2f} {total_profit_ok} {total_delta_pct:>+6.1f}% {portfolio_dd_base_pct:>9.1f}% {portfolio_dd_sized_pct:>11.1f}% {total_dd_ok} {portfolio_dd_delta_pct:>+6.1f}%")
    
    # =================================================================
    # RESUMEN POR GENERADOR
    # =================================================================
    print(f"\n{'='*165}")
    print("RESUMEN POR GENERADOR (todas las estrategias agregadas)")
    print(f"{'='*165}")
    
    generator_aggregates = {}
    for gen in combined_df['strategy'].apply(lambda x: extract_generator(x)).unique():
        gen_strategies = [r for r in results if r['generator'] == gen]
        gen_dfs = [r['df'] for r in gen_strategies]
        gen_combined = pd.concat(gen_dfs, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
        
        # Calculate equity curves
        gen_combined['equity_base'] = initial_capital + gen_combined['profit'].cumsum()
        gen_combined['equity_sized'] = initial_capital + gen_combined['profit_sized'].cumsum()
        
        generator_aggregates[gen] = {
            'trades_base': len(gen_combined),
            'trades_sizing': (gen_combined['sizing_mult'] > 0).sum(),
            'profit_b': gen_combined['profit'].sum(),
            'profit_s': gen_combined['profit_sized'].sum(),
            'dd_base_pct': calculate_max_dd_pct(gen_combined['equity_base']),
            'dd_sized_pct': calculate_max_dd_pct(gen_combined['equity_sized'])
        }
    
    # Calculate total trades for percentage calculation
    gen_total_trades_base_all = sum(agg['trades_base'] for agg in generator_aggregates.values())
    
    print(f"\n{'GENERATOR':<15} {'TRADES_BASE':>12} {'TRADES_%':>10} {'TRADES_SIZING':>14} {'TRADES_ACTIVE%':>15} {'PROFIT_BASE':>13} {'PROFIT_SIZING':>15} {'Δ%':>8} {'DD%_BASE':>10} {'DD%_SIZING':>12} {'ΔDD%':>8}")
    print("-" * 165)
    
    gen_total_trades_base = 0
    gen_total_trades_sizing = 0
    gen_total_profit_base = 0
    gen_total_profit_sizing = 0
    
    # Sort generators by custom order
    sorted_gens = sorted(generator_aggregates.keys(), key=lambda x: GENERATOR_ORDER.index(x) if x in GENERATOR_ORDER else 999)
    
    for gen in sorted_gens:
        agg = generator_aggregates[gen]
        trades_pct = (agg['trades_base'] / gen_total_trades_base_all * 100) if gen_total_trades_base_all > 0 else 0
        trades_active_pct = (agg['trades_sizing'] / agg['trades_base'] * 100) if agg['trades_base'] > 0 else 0
        delta_pct = ((agg['profit_s'] - agg['profit_b']) / abs(agg['profit_b']) * 100) if agg['profit_b'] != 0 else 0
        dd_delta_pct = agg['dd_sized_pct'] - agg['dd_base_pct']
        profit_ok = "✅" if agg['profit_s'] > agg['profit_b'] else "❌"
        dd_ok = "✅" if agg['dd_sized_pct'] < agg['dd_base_pct'] else "❌"
        
        print(f"{gen:<15} {agg['trades_base']:>12} {trades_pct:>9.1f}% {agg['trades_sizing']:>14} {trades_active_pct:>14.1f}% {agg['profit_b']:>13.2f} {agg['profit_s']:>15.2f} {profit_ok} {delta_pct:>+6.1f}% {agg['dd_base_pct']:>9.1f}% {agg['dd_sized_pct']:>11.1f}% {dd_ok} {dd_delta_pct:>+6.1f}%")
        
        gen_total_trades_base += agg['trades_base']
        gen_total_trades_sizing += agg['trades_sizing']
        gen_total_profit_base += agg['profit_b']
        gen_total_profit_sizing += agg['profit_s']
    
    print("-" * 165)
    
    # TOTAL: Use portfolio-level DD
    gen_total_trades_pct = 100.0
    gen_total_trades_active_pct = (gen_total_trades_sizing / gen_total_trades_base * 100) if gen_total_trades_base > 0 else 0
    gen_total_delta_pct = ((gen_total_profit_sizing - gen_total_profit_base) / abs(gen_total_profit_base) * 100) if gen_total_profit_base != 0 else 0
    gen_total_profit_ok = "✅" if gen_total_profit_sizing > gen_total_profit_base else "❌"
    gen_total_dd_ok = "✅" if portfolio_dd_sized_pct < portfolio_dd_base_pct else "❌"
    print(f"{'TOTAL':<15} {gen_total_trades_base:>12} {gen_total_trades_pct:>9.1f}% {gen_total_trades_sizing:>14} {gen_total_trades_active_pct:>14.1f}% {gen_total_profit_base:>13.2f} {gen_total_profit_sizing:>15.2f} {gen_total_profit_ok} {gen_total_delta_pct:>+6.1f}% {portfolio_dd_base_pct:>9.1f}% {portfolio_dd_sized_pct:>11.1f}% {gen_total_dd_ok} {portfolio_dd_delta_pct:>+6.1f}%")
    
    # =================================================================
    # SUMMARY - ALL STRATEGIES
    # =================================================================
    print(f"\n{'='*120}")
    print("SUMMARY - ALL STRATEGIES")
    print(f"{'='*120}")
    
    print(f"\n{'STRATEGY':<30} {'TRADES_BASE':>12} {'TRADES_SIZING':>14} {'PROFIT_BASE':>13} {'PROFIT_SIZING':>15} {'Δ%':>7} {'DD%_BASE':>9} {'DD%_SIZING':>11} {'ΔDD%':>8}")
    print("-" * 120)
    
    for r in results:
        profit_ok = "✅" if r['profit_sized'] > r['profit_base'] else "❌"
        dd_ok = "✅" if r['dd_sized_pct'] < r['dd_base_pct'] else "❌"
        print(f"{r['strategy']:<30} {r['num_trades']:>12} {r['trades_sizing']:>14} {r['profit_base']:>13.2f} {r['profit_sized']:>12.2f} {profit_ok} {r['delta_pct']:>+6.1f}% {r['dd_base_pct']:>9.1f}% {r['dd_sized_pct']:>10.1f}% {r['dd_delta_pct']:>+7.1f}% {dd_ok}")
    
    print("-" * 120)
    
    n = len(results)
    if n > 0:
        total_trades_base = sum(r['num_trades'] for r in results)
        total_trades_sizing = sum(r['trades_sizing'] for r in results)
        total_profit_b = sum(r['profit_base'] for r in results)
        total_profit_s = sum(r['profit_sized'] for r in results)
        total_delta = ((total_profit_s - total_profit_b) / abs(total_profit_b) * 100) if total_profit_b != 0 else 0
        
        # TOTAL: Use portfolio-level DD
        profit_ok = "✅" if total_profit_s > total_profit_b else "❌"
        dd_ok = "✅" if portfolio_dd_sized_pct < portfolio_dd_base_pct else "❌"
        print(f"{'TOTAL':<30} {total_trades_base:>12} {total_trades_sizing:>14} {total_profit_b:>13.2f} {total_profit_s:>12.2f} {profit_ok} {total_delta:>+6.1f}% {portfolio_dd_base_pct:>9.1f}% {portfolio_dd_sized_pct:>10.1f}% {portfolio_dd_delta_pct:>+7.1f}% {dd_ok}")
    
    print(f"\n{'='*120}")
    
    return results


if __name__ == "__main__":
    apply_sizing()