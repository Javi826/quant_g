"""
market_regime/optimize_thresholds.py

Optimizes volatile family thresholds to improve portfolio drawdown.
Tests multiple configurations and shows detailed comparison.

Usage:
    python optimize_thresholds.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.config import OUTPUT_FOLDER, FAMILIES, FAMILY_SIZING, INITIAL_CAPITAL


# =============================================================================
# OPTIMIZATION GRID
# =============================================================================

# Define threshold configurations to test
THRESHOLD_CONFIGS = [
    # ATR only
    {'atr_pct': 1.3, 'pe': None},
    {'atr_pct': 1.5, 'pe': None},
    {'atr_pct': 2.0, 'pe': None},
    {'atr_pct': 2.5, 'pe': None},
    {'atr_pct': 3.0, 'pe': None},
    
    # ATR + PE combinations
    {'atr_pct': 1.5, 'pe': 0.80},
    {'atr_pct': 1.5, 'pe': 0.85},
    {'atr_pct': 2.0, 'pe': 0.80},
    {'atr_pct': 2.0, 'pe': 0.85},
    {'atr_pct': 2.0, 'pe': 0.90},
    {'atr_pct': 2.5, 'pe': 0.85},
    {'atr_pct': 2.5, 'pe': 0.90},
]

# Define generator display order
GENERATOR_ORDER = ['parity', 'reversal', 'ranging', 'double_top', 'orderblocks', 'unknown']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_families(atr_threshold: float, pe_threshold: float = None) -> dict:
    """Builds FAMILIES dict with given thresholds."""
    volatile_rules = {'atr_pct': ('>', atr_threshold)}
    
    if pe_threshold is not None:
        volatile_rules['permutation_entropy'] = ('>', pe_threshold)
    
    return {
        'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
        'volatile': volatile_rules,
        'ranging': {},
    }


def extract_metrics_from_results(results: list, initial_capital: float) -> dict:
    """Extracts key metrics from position_sizer results."""
    
    # Import here to avoid circular import
    from market_regime.position_sizer import calculate_max_dd_pct, extract_generator
    
    # Combine all dataframes
    all_dfs = []
    for r in results:
        df_copy = r['df'].copy()
        df_copy['strategy'] = r['strategy']
        all_dfs.append(df_copy)
    
    combined_df = pd.concat(all_dfs, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
    
    # Portfolio-level metrics
    combined_df['equity_base'] = initial_capital + combined_df['profit'].cumsum()
    combined_df['equity_sized'] = initial_capital + combined_df['profit_sized'].cumsum()
    
    portfolio_dd_base = calculate_max_dd_pct(combined_df['equity_base'])
    portfolio_dd_sized = calculate_max_dd_pct(combined_df['equity_sized'])
    
    total_profit_base = combined_df['profit'].sum()
    total_profit_sized = combined_df['profit_sized'].sum()
    
    total_trades_base = len(combined_df)
    total_trades_sized = (combined_df['sizing_mult'] > 0).sum()
    trades_filtered = total_trades_base - total_trades_sized
    
    # Volatile family metrics
    volatile_df = combined_df[combined_df['family'] == 'volatile']
    volatile_trades = len(volatile_df)
    volatile_profit_base = volatile_df['profit'].sum() if len(volatile_df) > 0 else 0.0
    
    # Generator-level metrics
    generator_metrics = {}
    for gen in combined_df['strategy'].apply(extract_generator).unique():
        gen_df = combined_df[combined_df['strategy'].apply(extract_generator) == gen].copy()
        gen_df = gen_df.sort_values('buy_time').reset_index(drop=True)
        
        gen_df['equity_base'] = initial_capital + gen_df['profit'].cumsum()
        gen_df['equity_sized'] = initial_capital + gen_df['profit_sized'].cumsum()
        
        dd_base = calculate_max_dd_pct(gen_df['equity_base'])
        dd_sized = calculate_max_dd_pct(gen_df['equity_sized'])
        
        generator_metrics[gen] = {
            'trades_base': len(gen_df),
            'trades_sizing': (gen_df['sizing_mult'] > 0).sum(),
            'profit_base': gen_df['profit'].sum(),
            'profit_sized': gen_df['profit_sized'].sum(),
            'profit_delta_pct': ((gen_df['profit_sized'].sum() - gen_df['profit'].sum()) / abs(gen_df['profit'].sum()) * 100) if gen_df['profit'].sum() != 0 else 0,
            'dd_base': dd_base,
            'dd_sized': dd_sized,
            'dd_delta': dd_sized - dd_base,
            'trades_filtered': len(gen_df) - (gen_df['sizing_mult'] > 0).sum()
        }
    
    return {
        'total_trades_base': total_trades_base,
        'total_trades_sized': total_trades_sized,
        'trades_filtered': trades_filtered,
        'total_profit_base': total_profit_base,
        'total_profit_sized': total_profit_sized,
        'profit_delta_pct': ((total_profit_sized - total_profit_base) / abs(total_profit_base) * 100) if total_profit_base != 0 else 0,
        'portfolio_dd_base': portfolio_dd_base,
        'portfolio_dd_sized': portfolio_dd_sized,
        'portfolio_dd_delta': portfolio_dd_sized - portfolio_dd_base,
        'volatile_trades': volatile_trades,
        'volatile_profit_base': volatile_profit_base,
        'generator_metrics': generator_metrics
    }


def print_config_header(config_num: int, total_configs: int, atr: float, pe: float = None):
    """Prints configuration header."""
    print(f"\n{'='*160}")
    print(f"CONFIG {config_num}/{total_configs}: atr_pct > {atr}" + (f", permutation_entropy > {pe}" if pe else ""))
    print(f"{'='*160}")


def print_generator_detail(gen_metrics: dict):
    """Prints generator-level detail table."""
    print(f"\n{'GENERATOR':<15} {'TRADES_BASE':>12} {'TRADES_SIZING':>14} {'PROFIT_BASE':>13} {'PROFIT_SIZED':>15} {'ΔProfit%':>11} {'DD_BASE':>10} {'DD_SIZED':>12} {'ΔDD%':>10}")
    print("-" * 160)
    
    # Sort generators by custom order
    sorted_gens = sorted(gen_metrics.keys(), key=lambda x: GENERATOR_ORDER.index(x) if x in GENERATOR_ORDER else 999)
    
    for gen in sorted_gens:
        m = gen_metrics[gen]
        profit_ok = "✅" if m['profit_sized'] > m['profit_base'] else "❌"
        dd_ok = "✅" if m['dd_delta'] < 0 else "❌"
        print(f"{gen:<15} {m['trades_base']:>12} {m['trades_sizing']:>14} {m['profit_base']:>13.2f} {m['profit_sized']:>15.2f} {profit_ok} {m['profit_delta_pct']:>+9.1f}% {m['dd_base']:>9.2f}% {m['dd_sized']:>11.2f}% {dd_ok} {m['dd_delta']:>+8.2f}%")


def print_portfolio_summary(metrics: dict):
    """Prints portfolio-level summary."""
    print(f"\n{'PORTFOLIO TOTAL':<15}")
    print("-" * 160)
    
    profit_ok = "✅" if metrics['total_profit_sized'] > metrics['total_profit_base'] else "❌"
    dd_ok = "✅" if metrics['portfolio_dd_delta'] < 0 else "❌"
    
    print(f"  Trades: {metrics['total_trades_base']} → {metrics['total_trades_sized']} (filtered: {metrics['trades_filtered']})")
    print(f"  Profit: {metrics['total_profit_base']:>7.2f} → {metrics['total_profit_sized']:>7.2f} {profit_ok} ({metrics['profit_delta_pct']:>+6.1f}%)")
    print(f"  DD:     {metrics['portfolio_dd_base']:>6.2f}% → {metrics['portfolio_dd_sized']:>6.2f}% {dd_ok} ({metrics['portfolio_dd_delta']:>+6.2f}%)")
    print(f"  Volatile trades filtered: {metrics['volatile_trades']} (profit: {metrics['volatile_profit_base']:>+7.2f})")


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def run_optimization():
    """Runs threshold optimization and prints results."""
    
    # Import here to avoid circular import
    from market_regime.position_sizer import apply_sizing
    
    print("=" * 160)
    print("THRESHOLD OPTIMIZATION - Finding best volatile thresholds")
    print("=" * 160)
    print(f"\nTesting {len(THRESHOLD_CONFIGS)} configurations...")
    print(f"Sizing strategy: {FAMILY_SIZING}")
    print(f"Goal: Maximize drawdown reduction (negative ΔDD% = improvement)\n")
    
    all_results = []
    
    # Test each configuration
    for i, config in enumerate(THRESHOLD_CONFIGS, 1):
        atr = config['atr_pct']
        pe = config.get('pe')
        
        print_config_header(i, len(THRESHOLD_CONFIGS), atr, pe)
        
        # Build families with this threshold
        families = build_families(atr, pe)
        
        # Run position sizer (suppress individual strategy output)
        print("\n  Running backtest...")
        import io
        import contextlib
        
        # Capture stdout to suppress detailed output
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            results = apply_sizing(
                output_folder=OUTPUT_FOLDER,
                families=families,
                sizing=FAMILY_SIZING,
                initial_capital=INITIAL_CAPITAL
            )
        
        # Extract metrics
        metrics = extract_metrics_from_results(results, INITIAL_CAPITAL)
        
        # Print this config's results
        print_generator_detail(metrics['generator_metrics'])
        print_portfolio_summary(metrics)
        
        # Store for final comparison
        all_results.append({
            'config_num': i,
            'atr_threshold': atr,
            'pe_threshold': pe if pe else 'None',
            **metrics
        })
    
    # =================================================================
    # FINAL COMPARISON TABLE
    # =================================================================
    print(f"\n\n{'='*160}")
    print("FINAL COMPARISON - All Configurations Ranked by DD Improvement")
    print(f"{'='*160}\n")
    
    # Sort by DD delta (most negative = best improvement)
    all_results.sort(key=lambda x: x['portfolio_dd_delta'])
    
    print(f"{'#':<4} {'ATR>':<8} {'PE>':<8} {'TRADES_BASE':>12} {'TRADES_SIZED':>13} {'PROFIT_BASE':>13} {'PROFIT_SIZED':>15} {'ΔProfit%':>11} {'VOL_PROFIT':>12} {'DD_BASE':>10} {'DD_SIZED':>12} {'ΔDD%':>10}")
    print("-" * 160)
    
    for r in all_results:
        rank_icon = "🏆" if r == all_results[0] else "✅" if r['portfolio_dd_delta'] < 0 else "❌"
        pe_str = f"{r['pe_threshold']:.2f}" if r['pe_threshold'] != 'None' else "None"
        profit_ok = "✅" if r['total_profit_sized'] > r['total_profit_base'] else "❌"
        
        print(f"{r['config_num']:<4} {r['atr_threshold']:<8.1f} {pe_str:<8} {r['total_trades_base']:>12} {r['total_trades_sized']:>13} {r['total_profit_base']:>13.2f} {r['total_profit_sized']:>15.2f} {profit_ok} {r['profit_delta_pct']:>+9.1f}% {r['volatile_profit_base']:>+11.2f} "
              f"{r['portfolio_dd_base']:>9.2f}% {r['portfolio_dd_sized']:>11.2f}% {rank_icon} {r['portfolio_dd_delta']:>+8.2f}%")
    
    # Best configuration summary
    best = all_results[0]
    print(f"\n{'='*160}")
    print("🏆 BEST CONFIGURATION:")
    print(f"{'='*160}")
    print(f"  Thresholds: atr_pct > {best['atr_threshold']}" + (f", permutation_entropy > {best['pe_threshold']}" if best['pe_threshold'] != 'None' else ""))
    print(f"  Trades: {best['total_trades_base']} → {best['total_trades_sized']} (filtered: {best['trades_filtered']})")
    print(f"  Profit: {best['total_profit_base']:.2f} → {best['total_profit_sized']:.2f} ({best['profit_delta_pct']:+.1f}%)")
    print(f"  DD: {best['portfolio_dd_base']:.2f}% → {best['portfolio_dd_sized']:.2f}% ({best['portfolio_dd_delta']:+.2f}%)")
    print(f"  Volatile trades filtered: {best['volatile_trades']} (profit: {best['volatile_profit_base']:+.2f})")
    
    print(f"\n  Generator breakdown:")
    print(f"  {'GENERATOR':<15} {'TRADES_BASE':>12} {'TRADES_SIZING':>14} {'PROFIT_BASE':>13} {'PROFIT_SIZED':>15} {'ΔProfit%':>11} {'DD_BASE':>10} {'DD_SIZED':>12} {'ΔDD%':>10}")
    print(f"  {'-' * 145}")
    
    # Sort generators by custom order in breakdown
    sorted_gens = sorted(best['generator_metrics'].keys(), key=lambda x: GENERATOR_ORDER.index(x) if x in GENERATOR_ORDER else 999)
    
    for gen in sorted_gens:
        m = best['generator_metrics'][gen]
        profit_ok = "✅" if m['profit_sized'] > m['profit_base'] else "❌"
        dd_ok = "✅" if m['dd_delta'] < 0 else "❌"
        print(f"  {gen:<15} {m['trades_base']:>12} {m['trades_sizing']:>14} {m['profit_base']:>13.2f} {m['profit_sized']:>15.2f} {profit_ok} {m['profit_delta_pct']:>+9.1f}% {m['dd_base']:>9.2f}% {m['dd_sized']:>11.2f}% {dd_ok} {m['dd_delta']:>+8.2f}%")
    
    print(f"\n{'='*160}\n")


if __name__ == "__main__":
    run_optimization()