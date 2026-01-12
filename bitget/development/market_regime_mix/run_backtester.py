"""
market_regime/reg_backtester.py

Analyzes strategy performance with regime filters.

MODES:
- 'single': Analyzes an individual strategy, tests all filter combinations
- 'confluence': Analyzes ALL strategies by generator OR direction, looks for robust rules
- 'families': Tests predefined regime families on ALL enriched files

FILTER_BY:
- 'generator': Groups by generator (e.g., all parity_*)
- 'direction': Groups by direction (e.g., all *_long_*)

Usage:
    %runfile reg_backtester.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from itertools import combinations

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CONFIGURATION - EDIT HERE
# =============================================================================
MODE = 'sizing'               # 'single', 'confluence', 'families', or 'sizing'
FILTER_BY = 'generator'       # 'generator' or 'direction' (for confluence mode)
GENERATOR = 'parity'          # For FILTER_BY='generator': searches all {GENERATOR}_*
DIRECTION = 'long'            # For FILTER_BY='direction': searches all *_{DIRECTION}_*
STRATEGY = 'parity_long_4H_IS'   # For single mode: specific strategy
INITIAL_CAPITAL = 800         # Initial capital for profit % calculation
SHOW_PLOTS = False            # Show equity curve plots

# =============================================================================
# PREDEFINED FAMILIES - Edit thresholds as needed
# =============================================================================
FAMILIES = {
    'trending':     {'hurst': ('>', 0.55)},
    'volatile':     {'atr_pct': ('>', 2.0)},
    'ranging':      {},  # Default: everything else
}

# =============================================================================
# FAMILY SIZING - Multipliers for position sizing by family
# =============================================================================
FAMILY_SIZING = {
    'trending':  1.5,   # More size in trending
    'volatile':  1.0,   # Less in volatile
    'ranging':   1.0,   # Less in ranging
}
# =============================================================================


def get_output_folder() -> str:
    """Gets the output folder path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'market_regime', 'output')


def load_enriched_trades(strategy_name: str, output_folder: str = None) -> pd.DataFrame:
    """Loads the enriched trades file."""
    if output_folder is None:
        output_folder = get_output_folder()
    
    filepath = os.path.join(output_folder, f'trades_enriched_{strategy_name}.xlsx')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return pd.read_excel(filepath)


def load_profile(strategy_name: str, output_folder: str = None) -> pd.DataFrame:
    """Loads the strategy profile."""
    if output_folder is None:
        output_folder = get_output_folder()
    
    filepath = os.path.join(output_folder, f'profile_{strategy_name}.xlsx')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return pd.read_excel(filepath)


def get_activation_rules(profile_df: pd.DataFrame) -> Dict[str, Tuple[str, float]]:
    """Extracts activation rules from profile."""
    rules = {}
    for _, row in profile_df.iterrows():
        metric = row['metric']
        op = row['threshold_op']
        val = row['threshold_val']
        if pd.notna(op) and pd.notna(val):
            rules[metric] = (op, float(val))
    return rules


def apply_filter(df: pd.DataFrame, rules: Dict[str, Tuple[str, float]]) -> pd.DataFrame:
    """Applies filter rules to DataFrame."""
    mask = pd.Series([True] * len(df), index=df.index)
    
    for metric, (op, val) in rules.items():
        if metric not in df.columns:
            continue
        if op == '>':
            mask &= df[metric] > val
        elif op == '<':
            mask &= df[metric] < val
        elif op == '>=':
            mask &= df[metric] >= val
        elif op == '<=':
            mask &= df[metric] <= val
    
    return df[mask]


def calculate_metrics(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> dict:
    """Calculates performance metrics."""
    if len(df) == 0:
        return {
            'num_trades': 0,
            'profit_total': 0,
            'profit_pct': 0,
            'win_rate': 0,
            'max_dd': 0,
            'max_dd_pct': 0,
            'profit_factor': 0,
        }
    
    profits = df['profit']
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    profit_total = profits.sum()
    profit_pct = (profit_total / initial_capital) * 100
    
    # Calculate max drawdown from cumulative profit
    cumsum = profits.cumsum()
    running_max = cumsum.cummax()
    drawdown = running_max - cumsum
    max_dd = drawdown.max() if len(drawdown) > 0 else 0
    max_dd_pct = (max_dd / initial_capital) * 100
    
    return {
        'num_trades': len(df),
        'profit_total': profit_total,
        'profit_pct': profit_pct,
        'win_rate': len(wins) / len(df) * 100,
        'max_dd': max_dd,
        'max_dd_pct': max_dd_pct,
        'profit_factor': profit_factor,
    }


def plot_equity_comparison(
    df_base: pd.DataFrame,
    df_filtered: pd.DataFrame,
    title: str,
    initial_capital: float = INITIAL_CAPITAL
):
    """
    Plots equity curves comparing base (no filter) vs filtered trades.
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not available, skipping plot")
        return
    
    # Sort by buy_time
    df_base = df_base.sort_values('buy_time').copy()
    df_filtered = df_filtered.sort_values('buy_time').copy() if len(df_filtered) > 0 else df_filtered
    
    # Calculate cumulative profit
    df_base['cum_profit'] = df_base['profit'].cumsum()
    df_base['equity_pct'] = (df_base['cum_profit'] / initial_capital) * 100
    
    if len(df_filtered) > 0:
        df_filtered['cum_profit'] = df_filtered['profit'].cumsum()
        df_filtered['equity_pct'] = (df_filtered['cum_profit'] / initial_capital) * 100
    
    # Calculate drawdown for base
    df_base['running_max'] = df_base['cum_profit'].cummax()
    df_base['dd'] = df_base['running_max'] - df_base['cum_profit']
    df_base['dd_pct'] = (df_base['dd'] / initial_capital) * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curves
    ax1.plot(df_base['buy_time'], df_base['equity_pct'], 
             color='blue', linewidth=1.5, label=f'No Filter ({len(df_base)} trades)', alpha=0.7)
    
    if len(df_filtered) > 0:
        ax1.plot(df_filtered['buy_time'], df_filtered['equity_pct'], 
                 color='green', linewidth=1.5, label=f'With Filter ({len(df_filtered)} trades)', alpha=0.9)
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Equity %')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Add stats box
    base_final = df_base['equity_pct'].iloc[-1] if len(df_base) > 0 else 0
    filt_final = df_filtered['equity_pct'].iloc[-1] if len(df_filtered) > 0 else 0
    base_dd = df_base['dd_pct'].max() if len(df_base) > 0 else 0
    
    stats_text = f'No Filter: {base_final:.1f}%\nWith Filter: {filt_final:.1f}%\nMax DD (base): {base_dd:.1f}%'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot drawdown
    ax2.fill_between(df_base['buy_time'], 0, -df_base['dd_pct'], 
                     color='red', alpha=0.3, label='Drawdown (No Filter)')
    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Time')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# SINGLE MODE - Analyzes one strategy
# =============================================================================

def test_all_combinations(strategy_name: str, output_folder: str = None) -> pd.DataFrame:
    """Tests all rule combinations for a strategy."""
    df = load_enriched_trades(strategy_name, output_folder)
    profile_df = load_profile(strategy_name, output_folder)
    all_rules = get_activation_rules(profile_df)
    
    metrics_list = list(all_rules.keys())
    results = []
    
    # Baseline without filter
    baseline = calculate_metrics(df)
    baseline['config'] = 'NO FILTER'
    baseline['rules'] = '-'
    baseline['pct_trades'] = 100.0
    results.append(baseline)
    
    # All combinations
    for n in range(1, len(metrics_list) + 1):
        for combo in combinations(metrics_list, n):
            rules_subset = {m: all_rules[m] for m in combo}
            df_filtered = apply_filter(df, rules_subset)
            
            metrics = calculate_metrics(df_filtered)
            rules_str = ' & '.join([f"{m}{rules_subset[m][0]}{rules_subset[m][1]:.2f}" for m in combo])
            
            metrics['config'] = f"{n} rule(s)"
            metrics['rules'] = rules_str
            metrics['pct_trades'] = (metrics['num_trades'] / baseline['num_trades'] * 100) if baseline['num_trades'] > 0 else 0
            results.append(metrics)
    
    return pd.DataFrame(results)


def run_single_mode(strategy_name: str):
    """Runs analysis for a single strategy."""
    output_folder = get_output_folder()
    
    print("=" * 140)
    print(f"📊 SINGLE ANALYSIS: {strategy_name}")
    print("=" * 140)
    
    df_results = test_all_combinations(strategy_name, output_folder)
    df_sorted = df_results.sort_values('profit_total', ascending=False)
    
    print(f"\n{'CONFIG':<12} {'TRADES':>8} {'%TRADES':>8} {'PROFIT':>10} {'PROFIT%':>10} {'WIN%':>8} {'DD%':>8} {'PF':>8}  RULES")
    print("-" * 140)
    
    baseline_profit = df_results[df_results['config'] == 'NO FILTER']['profit_total'].values[0]
    
    for _, row in df_sorted.iterrows():
        profit = row['profit_total']
        pf = row['profit_factor']
        pf_str = f"{pf:.2f}" if pf != np.inf else "∞"
        marker = "✅" if profit > baseline_profit and row['config'] != 'NO FILTER' else "  "
        print(f"{row['config']:<12} {row['num_trades']:>8.0f} {row['pct_trades']:>7.1f}% {profit:>10.2f} {row['profit_pct']:>9.2f}% {row['win_rate']:>7.1f}% {row['max_dd_pct']:>7.2f}% {pf_str:>8} {marker} {row['rules']}")
    
    print("-" * 140)
    
    # Best configuration (by profit_total)
    best = df_sorted[df_sorted['config'] != 'NO FILTER'].iloc[0]
    baseline = df_results[df_results['config'] == 'NO FILTER'].iloc[0]
    
    print(f"\n🏆 BEST CONFIGURATION (by profit_total):")
    print(f"   Rules: {best['rules']}")
    print(f"   Profit: {best['profit_total']:.2f} vs {baseline['profit_total']:.2f} (baseline)")
    
    if best['profit_total'] > baseline['profit_total']:
        improvement = best['profit_total'] - baseline['profit_total']
        print(f"   Improvement: +{improvement:.2f}")
    else:
        print(f"   ⚠️  No combination improves baseline")
    
    # =================================================================
    # IS PLOT - Equity curve with and without filter
    # =================================================================
    
    # Get best rules from the combinations table (not the full profile)
    profile_df = load_profile(strategy_name, output_folder)
    all_rules = get_activation_rules(profile_df)
    
    # Find the best combination (highest profit_total)
    df_results_sorted = df_results.sort_values('profit_total', ascending=False)
    best_row = df_results_sorted[df_results_sorted['config'] != 'NO FILTER'].iloc[0]
    best_rules_str = best_row['rules']
    
    # Parse the best rules string back to dict
    # Format: "hurst>0.62" or "hurst>0.62 & efficiency_ratio>0.61"
    best_rules = {}
    if best_rules_str != '-':
        rule_parts = best_rules_str.split(' & ')
        for part in rule_parts:
            for metric in all_rules.keys():
                if part.startswith(metric):
                    op = '>' if '>' in part else '<'
                    val = float(part.split(op)[1])
                    best_rules[metric] = (op, val)
                    break
    
    if SHOW_PLOTS:
        # Load IS trades
        is_df = load_enriched_trades(strategy_name, output_folder)
        is_filtered_df = apply_filter(is_df, best_rules)
        
        plot_equity_comparison(
            df_base=is_df,
            df_filtered=is_filtered_df,
            title=f"IN-SAMPLE: {strategy_name} - Equity Curve (No Filter vs Best: {best_rules_str})",
            initial_capital=INITIAL_CAPITAL
        )
    
    # =================================================================
    # OOS VALIDATION - If this is an IS strategy, look for OOS pair
    # =================================================================
    
    if strategy_name.endswith('_IS'):
        oos_name = strategy_name.replace('_IS', '_OOS')
        oos_file = os.path.join(output_folder, f'trades_enriched_{oos_name}.xlsx')
        
        if os.path.exists(oos_file):
            print("\n" + "=" * 140)
            print(f"🧪 OUT-OF-SAMPLE VALIDATION: {oos_name}")
            print("=" * 140)
            
            print(f"\n📐 Applying BEST IS rule to OOS:")
            print(f"   {best_rules_str}")
            for metric, (op, val) in best_rules.items():
                print(f"   {metric} {op} {val:.4f}")
            
            # Load OOS trades
            oos_df = load_enriched_trades(oos_name, output_folder)
            
            # Baseline OOS (no filter)
            oos_base = calculate_metrics(oos_df)
            
            # Filtered OOS (with best IS rules)
            oos_filtered_df = apply_filter(oos_df, best_rules)
            oos_filt = calculate_metrics(oos_filtered_df)
            
            pct_filt = (oos_filt['num_trades'] / oos_base['num_trades'] * 100) if oos_base['num_trades'] > 0 else 0
            
            # Display
            print(f"\n{'OOS DATASET':<25} {'TR_BASE':>8} {'TR_FILT':>8} {'%_FILT':>7} │ {'PROFIT_B':>9} {'PROFIT_F':>9} {'Δ_PROFIT':>9} {'':>3} │ {'PROF%_B':>8} {'PROF%_F':>8} {'':>3} │ {'WIN%_B':>7} {'WIN%_F':>7} {'':>3} │ {'DD%_B':>7} {'DD%_F':>7} {'':>3}")
            print("-" * 140)
            
            # OOS row - comparing base (no filter) vs filtered (with best IS rules)
            oos_delta = oos_filt['profit_total'] - oos_base['profit_total']
            oos_profit_ok = "✅" if oos_filt['profit_total'] > oos_base['profit_total'] else "❌"
            oos_prof_pct_ok = "✅" if oos_filt['profit_pct'] > oos_base['profit_pct'] else "❌"
            oos_win_ok = "✅" if oos_filt['win_rate'] >= oos_base['win_rate'] else "❌"
            oos_dd_ok = "✅" if oos_filt.get('max_dd_pct', 0) <= oos_base.get('max_dd_pct', 0) else "❌"
            
            print(f"{oos_name:<25} {oos_base['num_trades']:>8.0f} {oos_filt['num_trades']:>8.0f} {pct_filt:>6.1f}% │ {oos_base['profit_total']:>9.2f} {oos_filt['profit_total']:>9.2f} {oos_delta:>+9.2f} {oos_profit_ok:>3} │ {oos_base['profit_pct']:>7.2f}% {oos_filt['profit_pct']:>7.2f}% {oos_prof_pct_ok:>3} │ {oos_base['win_rate']:>6.1f}% {oos_filt['win_rate']:>6.1f}% {oos_win_ok:>3} │ {oos_base.get('max_dd_pct', 0):>6.2f}% {oos_filt.get('max_dd_pct', 0):>6.2f}% {oos_dd_ok:>3}")
            
            print("-" * 140)
            
            # Verdict
            oos_improved = oos_filt['profit_total'] > oos_base['profit_total']
            oos_dd_better = oos_filt.get('max_dd_pct', 0) <= oos_base.get('max_dd_pct', 0)
            
            if oos_filt['num_trades'] == 0:
                print(f"\n   ❌ Rule FAILED OOS validation - 0 trades pass the filter")
                print(f"   → IS rules are too restrictive, likely overfitting")
            elif oos_improved and oos_dd_better:
                print(f"\n   ✅ Rule VALIDATED in OOS - profit improved and DD reduced")
                print(f"   → Safe to use in production")
            elif oos_dd_better and oos_filt['win_rate'] > oos_base['win_rate']:
                print(f"\n   ⚠️  Rule partially validated - better Win% and DD, but less profit")
                print(f"   → Consider using if you prefer quality over quantity")
            else:
                print(f"\n   ❌ Rule FAILED OOS validation")
                print(f"   → IS results were likely overfitting")
            
            # =================================================================
            # OOS PLOT - Equity curve with and without filter
            # =================================================================
            
            if SHOW_PLOTS and oos_filt['num_trades'] > 0:
                plot_equity_comparison(
                    df_base=oos_df,
                    df_filtered=oos_filtered_df,
                    title=f"OUT-OF-SAMPLE: {oos_name} - Equity Curve (No Filter vs Best: {best_rules_str})",
                    initial_capital=INITIAL_CAPITAL
                )
            elif SHOW_PLOTS:
                print(f"\n   ⚠️  Skipping OOS plot - no trades pass the filter")
        else:
            print(f"\n📁 No OOS file found: {oos_file}")
            print(f"   Run run_analysis.py for {oos_name} to enable OOS validation")
    
    print("\n" + "=" * 140)


# =============================================================================
# CONFLUENCE MODE - Analyzes all strategies by generator or direction
# =============================================================================

def find_strategies(filter_by: str, filter_value: str, output_folder: str = None, is_only: bool = True) -> List[str]:
    """Finds all strategies matching the filter criteria.
    
    Args:
        filter_by: 'generator' or 'direction'
        filter_value: The value to filter by (e.g., 'parity' or 'long')
        output_folder: Output folder path
        is_only: If True, only return IS strategies when IS/OOS pairs exist
    
    Returns:
        List of matching strategy names
    """
    if output_folder is None:
        output_folder = get_output_folder()
    
    path = Path(output_folder)
    
    # Get all profiles
    all_profiles = list(path.glob('profile_*.xlsx'))
    
    strategies = []
    for p in all_profiles:
        name = p.stem.replace('profile_', '')
        
        # Parse strategy name to check filter
        # Format: generator_direction_timeframe or generator_direction_timeframe_IS/OOS
        parts = name.split('_')
        
        # Remove IS/OOS suffix for parsing
        if parts[-1].upper() in ['IS', 'OOS']:
            data_type = parts[-1].upper()
            parts_clean = parts[:-1]
        else:
            data_type = None
            parts_clean = parts
        
        if len(parts_clean) >= 2:
            # Assume format: generator_direction_timeframe
            generator = parts_clean[0]
            direction = parts_clean[1] if len(parts_clean) > 1 else None
        else:
            generator = parts_clean[0] if parts_clean else None
            direction = None
        
        # Apply filter
        if filter_by == 'generator':
            if generator == filter_value:
                strategies.append(name)
        elif filter_by == 'direction':
            if direction == filter_value:
                strategies.append(name)
    
    strategies = sorted(strategies)
    
    # If is_only, filter to only IS strategies when IS/OOS pairs exist
    if is_only:
        has_is = any(s.endswith('_IS') for s in strategies)
        has_oos = any(s.endswith('_OOS') for s in strategies)
        
        if has_is and has_oos:
            strategies = [s for s in strategies if s.endswith('_IS')]
        elif has_is:
            strategies = [s for s in strategies if s.endswith('_IS')]
    
    return strategies


def run_confluence_mode(filter_by: str, filter_value: str):
    """Runs confluence analysis for strategies matching the filter."""
    output_folder = get_output_folder()
    
    filter_label = f"{filter_by.upper()}: {filter_value.upper()}"
    
    print("=" * 140)
    print(f"📊 CONFLUENCE ANALYSIS BY {filter_label}")
    print("=" * 140)
    
    # Find strategies
    strategies = find_strategies(filter_by, filter_value, output_folder)
    
    if not strategies:
        print(f"\n❌ No strategies found for {filter_label}")
        print(f"   Searched in: {output_folder}")
        if filter_by == 'generator':
            print(f"   Looking for: profile_{filter_value}_*.xlsx")
        else:
            print(f"   Looking for: profile_*_{filter_value}_*.xlsx")
        print(f"\n   Run run_analysis.py first for each strategy")
        return
    
    print(f"\n📁 Strategies found: {len(strategies)}")
    for s in strategies:
        print(f"   • {s}")
    
    # Load rules for each strategy
    all_strategy_rules = {}
    all_strategy_stats = {}
    
    print("\n" + "-" * 140)
    print("📐 RULES BY STRATEGY")
    print("-" * 140)
    
    for strategy in strategies:
        try:
            profile_df = load_profile(strategy, output_folder)
            trades_df = load_enriched_trades(strategy, output_folder)
            
            rules = get_activation_rules(profile_df)
            stats = calculate_metrics(trades_df)
            
            all_strategy_rules[strategy] = rules
            all_strategy_stats[strategy] = stats
            
            print(f"\n▸ {strategy}")
            print(f"  Trades: {stats['num_trades']} | Profit: {stats['profit_total']:.2f} | Win%: {stats['win_rate']:.1f}%")
            print(f"  Rules:")
            for metric, (op, val) in rules.items():
                print(f"    {metric:25s} {op} {val:.4f}")
                
        except Exception as e:
            print(f"\n▸ {strategy}")
            print(f"  ❌ Error: {e}")
    
    if len(all_strategy_rules) < 2:
        print("\n⚠️  Need at least 2 strategies for confluence analysis")
        return
    
    # Analyze confluence
    print("\n" + "=" * 140)
    print("🔍 CONFLUENCE ANALYSIS")
    print("=" * 140)
    
    metrics = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
    confluence_results = {}
    
    for metric in metrics:
        directions = []
        values = []
        
        for strategy, rules in all_strategy_rules.items():
            if metric in rules:
                op, val = rules[metric]
                directions.append(op)
                values.append(val)
        
        if not directions:
            confluence_results[metric] = {
                'status': 'NO_DATA',
                'agreement': 0,
                'direction': None,
                'values': []
            }
            continue
        
        # Calculate confluence
        count_gt = directions.count('>')
        count_lt = directions.count('<')
        total = len(directions)
        
        DISPERSION_THRESHOLD = 0.30  # 30% - if std/mean > this, weak confluence
        
        if count_gt == total:
            mean_val = np.mean(values)
            std_val = np.std(values)
            dispersion = std_val / abs(mean_val) if mean_val != 0 else 0
            
            if dispersion > DISPERSION_THRESHOLD:
                status = 'CONFLUENCE_WEAK'
            else:
                status = 'CONFLUENCE_STRONG'
            
            confluence_results[metric] = {
                'status': status,
                'agreement': 100,
                'direction': '>',
                'values': values,
                'mean_val': mean_val,
                'std_val': std_val,
                'dispersion': dispersion
            }
        elif count_lt == total:
            mean_val = np.mean(values)
            std_val = np.std(values)
            dispersion = std_val / abs(mean_val) if mean_val != 0 else 0
            
            if dispersion > DISPERSION_THRESHOLD:
                status = 'CONFLUENCE_WEAK'
            else:
                status = 'CONFLUENCE_STRONG'
            
            confluence_results[metric] = {
                'status': status,
                'agreement': 100,
                'direction': '<',
                'values': values,
                'mean_val': mean_val,
                'std_val': std_val,
                'dispersion': dispersion
            }
        else:
            majority_dir = '>' if count_gt > count_lt else '<'
            agreement = max(count_gt, count_lt) / total * 100
            confluence_results[metric] = {
                'status': 'DIVERGENCE',
                'agreement': agreement,
                'direction': majority_dir,
                'count_gt': count_gt,
                'count_lt': count_lt,
                'values': values
            }
    
    # Show confluence results
    print(f"\n{'METRIC':<25} {'STATUS':<20} {'AGREEMENT':>10} {'DIRECTION':>10} {'DISPERSION':>12} {'VALUES'}")
    print("-" * 140)
    
    for metric, result in confluence_results.items():
        status = result['status']
        agreement = result['agreement']
        direction = result['direction'] or '-'
        
        if status == 'CONFLUENCE_STRONG':
            status_icon = "✅"
            disp_str = f"{result['dispersion']*100:.1f}%"
            values_str = f"mean={result['mean_val']:.4f} ± {result['std_val']:.4f}"
        elif status == 'CONFLUENCE_WEAK':
            status_icon = "⚠️"
            disp_str = f"{result['dispersion']*100:.1f}%"
            values_str = f"mean={result['mean_val']:.4f} ± {result['std_val']:.4f} (HIGH DISPERSION)"
        elif status == 'DIVERGENCE':
            status_icon = "❌"
            disp_str = "-"
            values_str = f">{result['count_gt']} vs <{result['count_lt']}"
        else:
            status_icon = "❓"
            disp_str = "-"
            values_str = "-"
        
        print(f"{metric:<25} {status_icon} {status:<17} {agreement:>9.0f}% {direction:>10} {disp_str:>12} {values_str}")
    
    # Consolidated rule - get all STRONG metrics
    print("\n" + "=" * 140)
    print("🎛️  AVAILABLE STRONG METRICS")
    print("=" * 140)
    
    strong_rules = {}
    for metric, result in confluence_results.items():
        if result['status'] == 'CONFLUENCE_STRONG':
            strong_rules[metric] = (result['direction'], round(result['mean_val'], 4))
    
    if strong_rules:
        print(f"\n```python")
        print(f"STRONG_METRICS = {{")
        for metric, (op, val) in strong_rules.items():
            print(f"    '{metric}': ('{op}', {val}),")
        print(f"}}")
        print(f"```")
    else:
        print(f"\n⚠️  No metrics with STRONG confluence")
        print(f"   Strategies don't share a clear regime pattern")
        print(f"   (All confluences are either WEAK or DIVERGENT)")
        print("\n" + "=" * 140)
        return
    
    # Test all combinations of STRONG rules and find best by profit_total
    print("\n" + "-" * 140)
    print("🔬 TESTING ALL COMBINATIONS OF STRONG RULES")
    print("-" * 140)
    
    metrics_list = list(strong_rules.keys())
    combination_results = []
    
    # Test each combination
    for n in range(1, len(metrics_list) + 1):
        for combo in combinations(metrics_list, n):
            rules_subset = {m: strong_rules[m] for m in combo}
            rules_str = ' & '.join([f"{m}{rules_subset[m][0]}{rules_subset[m][1]:.2f}" for m in combo])
            
            # Calculate total profit across all IS strategies
            total_profit = 0
            total_trades_base = 0
            total_trades_filt = 0
            
            for strategy in strategies:
                try:
                    trades_df = load_enriched_trades(strategy, output_folder)
                    filtered_df = apply_filter(trades_df, rules_subset)
                    
                    total_profit += filtered_df['profit'].sum() if len(filtered_df) > 0 else 0
                    total_trades_base += len(trades_df)
                    total_trades_filt += len(filtered_df)
                except:
                    pass
            
            pct_trades = (total_trades_filt / total_trades_base * 100) if total_trades_base > 0 else 0
            
            combination_results.append({
                'n_rules': n,
                'rules': rules_subset,
                'rules_str': rules_str,
                'total_profit': total_profit,
                'total_trades_filt': total_trades_filt,
                'pct_trades': pct_trades
            })
    
    # Sort by total_profit descending
    combination_results.sort(key=lambda x: x['total_profit'], reverse=True)
    
    # Show results
    print(f"\n{'CONFIG':<10} {'TRADES':>8} {'%TRADES':>8} {'TOTAL_PROFIT':>14}  RULES")
    print("-" * 140)
    
    # Calculate baseline (no filter)
    baseline_profit = 0
    baseline_trades = 0
    for strategy in strategies:
        try:
            trades_df = load_enriched_trades(strategy, output_folder)
            baseline_profit += trades_df['profit'].sum()
            baseline_trades += len(trades_df)
        except:
            pass
    
    print(f"{'NO FILTER':<10} {baseline_trades:>8} {'100.0':>7}% {baseline_profit:>14.2f}  -")
    
    for result in combination_results:
        marker = "✅" if result['total_profit'] > baseline_profit else "  "
        print(f"{result['n_rules']} rule(s)   {result['total_trades_filt']:>8} {result['pct_trades']:>7.1f}% {result['total_profit']:>14.2f} {marker} {result['rules_str']}")
    
    print("-" * 140)
    
    # Best combination
    best_combo = combination_results[0]
    consolidated_rules = best_combo['rules']
    best_rules_str = best_combo['rules_str']
    
    print(f"\n🏆 BEST COMBINATION (by total profit across all IS strategies):")
    print(f"   Rules: {best_rules_str}")
    print(f"   Total Profit: {best_combo['total_profit']:.2f} vs {baseline_profit:.2f} (baseline)")
    
    if best_combo['total_profit'] > baseline_profit:
        improvement = best_combo['total_profit'] - baseline_profit
        print(f"   Improvement: +{improvement:.2f}")
    else:
        print(f"   ⚠️  No combination improves baseline")
    
    # Validate best rule on each strategy
    print("\n" + "-" * 140)
    print("📈 BEST RULE VALIDATION BY STRATEGY")
    print("-" * 140)
    
    # Header
    print(f"\n{'STRATEGY':<25} {'TR_BASE':>8} {'TR_FILT':>8} {'%_FILT':>7} │ {'PROFIT_B':>9} {'PROFIT_F':>9} {'Δ_PROFIT':>9} {'':>3} │ {'PROF%_B':>8} {'PROF%_F':>8} {'':>3} │ {'WIN%_B':>7} {'WIN%_F':>7} {'':>3} │ {'DD%_B':>7} {'DD%_F':>7} {'':>3}")
    print("-" * 140)
    
    total_strategies = len(strategies)
    profit_improved = 0
    profit_pct_improved = 0
    win_improved = 0
    dd_improved = 0
    
    for strategy in strategies:
        try:
            trades_df = load_enriched_trades(strategy, output_folder)
            
            # Baseline
            base = calculate_metrics(trades_df)
            
            # Filtered
            filtered_df = apply_filter(trades_df, consolidated_rules)
            filt = calculate_metrics(filtered_df)
            
            pct_filt = (filt['num_trades'] / base['num_trades'] * 100) if base['num_trades'] > 0 else 0
            
            # Deltas and indicators
            delta_profit = filt['profit_total'] - base['profit_total']
            
            profit_ok = "✅" if filt['profit_total'] > base['profit_total'] else "❌"
            prof_pct_ok = "✅" if filt['profit_pct'] > base['profit_pct'] else "❌"
            win_ok = "✅" if filt['win_rate'] >= base['win_rate'] else "❌"
            dd_ok = "✅" if filt['max_dd_pct'] <= base['max_dd_pct'] else "❌"
            
            if filt['profit_total'] > base['profit_total']:
                profit_improved += 1
            if filt['profit_pct'] > base['profit_pct']:
                profit_pct_improved += 1
            if filt['win_rate'] >= base['win_rate']:
                win_improved += 1
            if filt['max_dd_pct'] <= base['max_dd_pct']:
                dd_improved += 1
            
            print(f"{strategy:<25} {base['num_trades']:>8.0f} {filt['num_trades']:>8.0f} {pct_filt:>6.1f}% │ {base['profit_total']:>9.2f} {filt['profit_total']:>9.2f} {delta_profit:>+9.2f} {profit_ok:>3} │ {base['profit_pct']:>7.2f}% {filt['profit_pct']:>7.2f}% {prof_pct_ok:>3} │ {base['win_rate']:>6.1f}% {filt['win_rate']:>6.1f}% {win_ok:>3} │ {base['max_dd_pct']:>6.2f}% {filt['max_dd_pct']:>6.2f}% {dd_ok:>3}")
            
            # Store for plots
            if SHOW_PLOTS:
                if not hasattr(run_confluence_mode, 'is_plot_data'):
                    run_confluence_mode.is_plot_data = []
                run_confluence_mode.is_plot_data.append({
                    'strategy': strategy,
                    'df_base': trades_df.copy(),
                    'df_filtered': filtered_df.copy()
                })
            
        except Exception as e:
            print(f"{strategy:<25} Error: {e}")
    
    print("-" * 140)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Profit improved:   {profit_improved}/{total_strategies} strategies")
    print(f"   Profit% improved:  {profit_pct_improved}/{total_strategies} strategies")
    print(f"   Win% improved:     {win_improved}/{total_strategies} strategies")
    print(f"   DD% improved:      {dd_improved}/{total_strategies} strategies")
    
    # Final verdict
    all_improved = profit_improved == total_strategies and dd_improved == total_strategies
    majority_improved = profit_improved > total_strategies / 2
    
    if all_improved:
        print(f"\n   ✅ Best rule IMPROVES all strategies")
        print(f"   → ROBUST RULE, can be used in production")
    elif majority_improved:
        print(f"\n   ⚠️  Rule improves majority but not all")
        print(f"   → Consider adjusting thresholds or use with caution")
    else:
        print(f"\n   ❌ Rule does NOT improve consistently")
        print(f"   → DO NOT use this rule, it's overfitting")
    
    # =================================================================
    # OOS VALIDATION - Apply IS rule to OOS data
    # =================================================================
    
    # Find OOS pairs for IS strategies
    oos_strategies = []
    is_to_oos_map = {}
    
    for strategy in strategies:
        if strategy.endswith('_IS'):
            oos_name = strategy.replace('_IS', '_OOS')
            oos_file = os.path.join(output_folder, f'trades_enriched_{oos_name}.xlsx')
            if os.path.exists(oos_file):
                oos_strategies.append(oos_name)
                is_to_oos_map[strategy] = oos_name
    
    if oos_strategies:
        print("\n" + "=" * 140)
        print("🧪 OUT-OF-SAMPLE VALIDATION (applying best IS rule to OOS data)")
        print("=" * 140)
        
        print(f"\n📐 Best rule: {best_rules_str}")
        
        print(f"\n📁 OOS strategies found: {len(oos_strategies)}")
        for s in oos_strategies:
            print(f"   • {s}")
        
        # Header
        print(f"\n{'STRATEGY':<25} {'TR_BASE':>8} {'TR_FILT':>8} {'%_FILT':>7} │ {'PROFIT_B':>9} {'PROFIT_F':>9} {'Δ_PROFIT':>9} {'':>3} │ {'PROF%_B':>8} {'PROF%_F':>8} {'':>3} │ {'WIN%_B':>7} {'WIN%_F':>7} {'':>3} │ {'DD%_B':>7} {'DD%_F':>7} {'':>3}")
        print("-" * 140)
        
        oos_total = len(oos_strategies)
        oos_profit_improved = 0
        oos_profit_pct_improved = 0
        oos_win_improved = 0
        oos_dd_improved = 0
        
        for oos_strategy in oos_strategies:
            try:
                trades_df = load_enriched_trades(oos_strategy, output_folder)
                
                # Baseline
                base = calculate_metrics(trades_df)
                
                # Filtered with best IS rule
                filtered_df = apply_filter(trades_df, consolidated_rules)
                filt = calculate_metrics(filtered_df)
                
                pct_filt = (filt['num_trades'] / base['num_trades'] * 100) if base['num_trades'] > 0 else 0
                
                # Deltas and indicators
                delta_profit = filt['profit_total'] - base['profit_total']
                
                profit_ok = "✅" if filt['profit_total'] > base['profit_total'] else "❌"
                prof_pct_ok = "✅" if filt['profit_pct'] > base['profit_pct'] else "❌"
                win_ok = "✅" if filt['win_rate'] >= base['win_rate'] else "❌"
                dd_ok = "✅" if filt['max_dd_pct'] <= base['max_dd_pct'] else "❌"
                
                if filt['profit_total'] > base['profit_total']:
                    oos_profit_improved += 1
                if filt['profit_pct'] > base['profit_pct']:
                    oos_profit_pct_improved += 1
                if filt['win_rate'] >= base['win_rate']:
                    oos_win_improved += 1
                if filt['max_dd_pct'] <= base['max_dd_pct']:
                    oos_dd_improved += 1
                
                print(f"{oos_strategy:<25} {base['num_trades']:>8.0f} {filt['num_trades']:>8.0f} {pct_filt:>6.1f}% │ {base['profit_total']:>9.2f} {filt['profit_total']:>9.2f} {delta_profit:>+9.2f} {profit_ok:>3} │ {base['profit_pct']:>7.2f}% {filt['profit_pct']:>7.2f}% {prof_pct_ok:>3} │ {base['win_rate']:>6.1f}% {filt['win_rate']:>6.1f}% {win_ok:>3} │ {base['max_dd_pct']:>6.2f}% {filt['max_dd_pct']:>6.2f}% {dd_ok:>3}")
                
                # Store for plots
                if SHOW_PLOTS:
                    if not hasattr(run_confluence_mode, 'oos_plot_data'):
                        run_confluence_mode.oos_plot_data = []
                    run_confluence_mode.oos_plot_data.append({
                        'strategy': oos_strategy,
                        'df_base': trades_df.copy(),
                        'df_filtered': filtered_df.copy()
                    })
                
            except Exception as e:
                print(f"{oos_strategy:<25} Error: {e}")
        
        print("-" * 140)
        
        # OOS Summary
        print(f"\n📊 OOS SUMMARY:")
        print(f"   Profit improved:   {oos_profit_improved}/{oos_total} strategies")
        print(f"   Profit% improved:  {oos_profit_pct_improved}/{oos_total} strategies")
        print(f"   Win% improved:     {oos_win_improved}/{oos_total} strategies")
        print(f"   DD% improved:      {oos_dd_improved}/{oos_total} strategies")
        
        # Final OOS verdict
        oos_all_improved = oos_profit_improved == oos_total and oos_dd_improved == oos_total
        oos_majority_improved = oos_profit_improved > oos_total / 2
        
        if oos_all_improved:
            print(f"\n   ✅ Rule VALIDATED in OOS - ROBUST RULE")
            print(f"   → Safe to use in production")
        elif oos_majority_improved:
            print(f"\n   ⚠️  Rule partially validated in OOS")
            print(f"   → Use with caution")
        else:
            print(f"\n   ❌ Rule FAILED OOS validation")
            print(f"   → DO NOT use, IS results were likely overfitting")
    
    # =================================================================
    # PLOTS - Generate all plots at the end
    # =================================================================
    
    if SHOW_PLOTS and consolidated_rules:
        print("\n" + "=" * 140)
        print("📊 GENERATING PLOTS")
        print("=" * 140)
        
        # IS plots
        if hasattr(run_confluence_mode, 'is_plot_data'):
            for plot_data in run_confluence_mode.is_plot_data:
                if len(plot_data['df_filtered']) > 0:
                    plot_equity_comparison(
                        df_base=plot_data['df_base'],
                        df_filtered=plot_data['df_filtered'],
                        title=f"IN-SAMPLE: {plot_data['strategy']} - Equity Curve (No Filter vs Best Rule)",
                        initial_capital=INITIAL_CAPITAL
                    )
                else:
                    print(f"   ⚠️  Skipping IS plot for {plot_data['strategy']} - no trades pass filter")
            # Clean up
            del run_confluence_mode.is_plot_data
        
        # OOS plots
        if hasattr(run_confluence_mode, 'oos_plot_data'):
            for plot_data in run_confluence_mode.oos_plot_data:
                if len(plot_data['df_filtered']) > 0:
                    plot_equity_comparison(
                        df_base=plot_data['df_base'],
                        df_filtered=plot_data['df_filtered'],
                        title=f"OUT-OF-SAMPLE: {plot_data['strategy']} - Equity Curve (No Filter vs Best Rule)",
                        initial_capital=INITIAL_CAPITAL
                    )
                else:
                    print(f"   ⚠️  Skipping OOS plot for {plot_data['strategy']} - no trades pass filter")
            # Clean up
            del run_confluence_mode.oos_plot_data
    
    print("\n" + "=" * 140)


# =============================================================================
# FAMILIES MODE - Tests predefined families on all enriched files
# =============================================================================

def find_all_enriched_files(output_folder: str = None) -> List[str]:
    """Finds all trades_enriched_*.xlsx files in output folder."""
    if output_folder is None:
        output_folder = get_output_folder()
    
    path = Path(output_folder)
    files = list(path.glob('trades_enriched_*.xlsx'))
    
    # Extract strategy names
    strategies = []
    for f in files:
        name = f.stem.replace('trades_enriched_', '')
        strategies.append(name)
    
    return sorted(strategies)


def run_families_mode():
    """Tests predefined families on all enriched files."""
    output_folder = get_output_folder()
    
    print("=" * 140)
    print("📊 FAMILIES MODE - Testing predefined regime families")
    print("=" * 140)
    
    # Show families being tested
    print(f"\n📋 Predefined families:")
    for family_name, rules in FAMILIES.items():
        rules_str = ' & '.join([f"{m}{op}{v}" for m, (op, v) in rules.items()])
        print(f"   • {family_name:20s}: {rules_str}")
    
    # Find all enriched files
    strategies = find_all_enriched_files(output_folder)
    
    if not strategies:
        print(f"\n❌ No enriched files found in: {output_folder}")
        print(f"   Run run_analysis.py first to generate trades_enriched_*.xlsx files")
        return
    
    print(f"\n📁 Enriched files found: {len(strategies)}")
    for s in strategies:
        print(f"   • {s}")
    
    # Store results for final summary
    summary_results = []
    
    # Process each strategy
    for strategy in strategies:
        print("\n" + "=" * 140)
        print(f"📊 FAMILY ANALYSIS: {strategy}")
        print("=" * 140)
        
        try:
            df = load_enriched_trades(strategy, output_folder)
            
            # Baseline (no filter)
            base = calculate_metrics(df)
            
            # Table header
            print(f"\n{'FAMILY':<20} {'TR_BASE':>8} {'TR_ACTV':>8} {'%_ACTV':>7} │ {'PROFIT':>10} {'PROFIT%':>9} │ {'WIN%':>7} │ {'DD%':>7}")
            print("-" * 100)
            
            # Baseline row
            print(f"{'NO_FILTER':<20} {base['num_trades']:>8} {base['num_trades']:>8} {'100.0':>6}% │ {base['profit_total']:>10.2f} {base['profit_pct']:>8.2f}% │ {base['win_rate']:>6.1f}% │ {base['max_dd_pct']:>6.2f}%")
            
            # Track best family (by profit, among real families only)
            best_family = None
            best_profit = float('-inf')
            best_metrics = None
            best_filtered_df = None
            
            # Test each family
            family_results = []
            for family_name, rules in FAMILIES.items():
                filtered_df = apply_filter(df, rules)
                filt = calculate_metrics(filtered_df)
                
                pct_active = (filt['num_trades'] / base['num_trades'] * 100) if base['num_trades'] > 0 else 0
                
                # Track best family (must have at least 1 trade)
                if filt['num_trades'] > 0 and filt['profit_total'] > best_profit:
                    best_profit = filt['profit_total']
                    best_family = family_name
                    best_metrics = filt
                    best_filtered_df = filtered_df.copy()
                
                marker = ""
                if filt['num_trades'] > 0:
                    if filt['profit_total'] > base['profit_total']:
                        marker = "✅"
                    elif filt['profit_total'] < 0:
                        marker = "❌"
                
                print(f"{family_name:<20} {base['num_trades']:>8} {filt['num_trades']:>8} {pct_active:>6.1f}% │ {filt['profit_total']:>10.2f} {filt['profit_pct']:>8.2f}% │ {filt['win_rate']:>6.1f}% │ {filt['max_dd_pct']:>6.2f}% {marker}")
                
                family_results.append({
                    'family': family_name,
                    'metrics': filt,
                    'pct_active': pct_active
                })
            
            print("-" * 100)
            
            # Best family for this strategy
            if best_family and best_metrics:
                if best_profit > base['profit_total']:
                    print(f"\n🏆 Best family: {best_family} (profit: {best_profit:.2f} vs baseline: {base['profit_total']:.2f})")
                else:
                    print(f"\n⚠️  Best family: {best_family} (profit: {best_profit:.2f}, but baseline is better: {base['profit_total']:.2f})")
            else:
                print(f"\n❌ No family has any trades")
                best_family = list(FAMILIES.keys())[0]  # Default to first family
                best_metrics = {'num_trades': 0, 'profit_total': 0, 'win_rate': 0, 'max_dd_pct': 0}
                best_filtered_df = pd.DataFrame()
            
            # Store for summary and plots
            pct_active = (best_metrics['num_trades'] / base['num_trades'] * 100) if base['num_trades'] > 0 and best_metrics['num_trades'] > 0 else 0.0
            summary_results.append({
                'strategy': strategy,
                'best_family': best_family,
                'tr_base': base['num_trades'],
                'tr_active': best_metrics['num_trades'],
                'pct_active': pct_active,
                'profit_base': base['profit_total'],
                'profit_family': best_metrics['profit_total'],
                'win_base': base['win_rate'],
                'win_family': best_metrics['win_rate'],
                'dd_base': base['max_dd_pct'],
                'dd_family': best_metrics['max_dd_pct'],
                'df_base': df.copy(),
                'df_filtered': best_filtered_df,
            })
            
        except Exception as e:
            print(f"\n❌ Error processing {strategy}: {e}")
    
    # Final summary
    print("\n" + "=" * 140)
    print("📊 SUMMARY - BEST FAMILY PER STRATEGY")
    print("=" * 140)
    
    # Header
    print(f"\n{'STRATEGY':<30} {'FAMILY':<18} {'TR_BASE':>8} {'TR_ACTV':>8} {'%_ACTV':>7} │ {'PROFIT_B':>9} {'PROFIT_F':>9} {'':>2} │ {'WIN%_B':>7} {'WIN%_F':>7} {'':>2} │ {'DD%_B':>7} {'DD%_F':>7} {'':>2}")
    print("-" * 145)
    
    for r in summary_results:
        profit_ok = "✅" if r['profit_family'] > r['profit_base'] else "❌"
        win_ok = "✅" if r['win_family'] > r['win_base'] else "❌"
        dd_ok = "✅" if r['dd_family'] < r['dd_base'] else "❌"
        
        print(f"{r['strategy']:<30} {r['best_family']:<18} {r['tr_base']:>8} {r['tr_active']:>8} {r['pct_active']:>6.1f}% │ {r['profit_base']:>9.2f} {r['profit_family']:>9.2f} {profit_ok:>2} │ {r['win_base']:>6.1f}% {r['win_family']:>6.1f}% {win_ok:>2} │ {r['dd_base']:>6.2f}% {r['dd_family']:>6.2f}% {dd_ok:>2}")
    
    print("-" * 145)
    
    # Calculate averages
    n = len(summary_results)
    if n > 0:
        avg_tr_base = sum(r['tr_base'] for r in summary_results) / n
        avg_tr_active = sum(r['tr_active'] for r in summary_results) / n
        avg_pct_active = sum(r['pct_active'] for r in summary_results) / n
        avg_profit_base = sum(r['profit_base'] for r in summary_results) / n
        avg_profit_family = sum(r['profit_family'] for r in summary_results) / n
        avg_win_base = sum(r['win_base'] for r in summary_results) / n
        avg_win_family = sum(r['win_family'] for r in summary_results) / n
        avg_dd_base = sum(r['dd_base'] for r in summary_results) / n
        avg_dd_family = sum(r['dd_family'] for r in summary_results) / n
        
        avg_profit_ok = "✅" if avg_profit_family > avg_profit_base else "❌"
        avg_win_ok = "✅" if avg_win_family > avg_win_base else "❌"
        avg_dd_ok = "✅" if avg_dd_family < avg_dd_base else "❌"
        
        print(f"{'AVERAGE':<30} {'-':<18} {avg_tr_base:>8.0f} {avg_tr_active:>8.0f} {avg_pct_active:>6.1f}% │ {avg_profit_base:>9.2f} {avg_profit_family:>9.2f} {avg_profit_ok:>2} │ {avg_win_base:>6.1f}% {avg_win_family:>6.1f}% {avg_win_ok:>2} │ {avg_dd_base:>6.2f}% {avg_dd_family:>6.2f}% {avg_dd_ok:>2}")
    
    print("-" * 145)
    
    # Count families used
    family_counts = {}
    for r in summary_results:
        fam = r['best_family']
        family_counts[fam] = family_counts.get(fam, 0) + 1
    
    print(f"\n📊 Family usage:")
    for fam, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"   • {fam:<20s}: {count} strategies")
    
    # =================================================================
    # PLOTS - Generate all plots at the end
    # =================================================================
    
    if SHOW_PLOTS:
        print("\n" + "=" * 140)
        print("📊 GENERATING PLOTS")
        print("=" * 140)
        
        for r in summary_results:
            if r['tr_active'] > 0 and r['df_filtered'] is not None and len(r['df_filtered']) > 0:
                plot_equity_comparison(
                    df_base=r['df_base'],
                    df_filtered=r['df_filtered'],
                    title=f"{r['strategy']} - No Filter vs {r['best_family']}",
                    initial_capital=INITIAL_CAPITAL
                )
            else:
                print(f"   ⚠️  Skipping plot for {r['strategy']} - no trades in best family")
    
    print("\n" + "=" * 140)


# =============================================================================
# SIZING MODE - Tests position sizing by family on all enriched files
# =============================================================================

def classify_trade_family(row: pd.Series) -> str:
    """Classifies a single trade into a family based on its metrics.
    
    Checks families in order (first match wins). 'ranging' should be last as default.
    """
    for family_name, rules in FAMILIES.items():
        if not rules:  # Empty rules = default family (ranging)
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
            elif op == '>=' and not (row[metric] >= val):
                match = False
                break
            elif op == '<=' and not (row[metric] <= val):
                match = False
                break
        
        if match:
            return family_name
    
    # Default to the family with empty rules (ranging)
    for family_name, rules in FAMILIES.items():
        if not rules:
            return family_name
    
    return 'unknown'


def run_sizing_mode():
    """Tests position sizing by family on all enriched files."""
    output_folder = get_output_folder()
    
    print("=" * 140)
    print("📊 SIZING MODE - Position sizing by regime family")
    print("=" * 140)
    
    # Show families and sizing
    print(f"\n📋 Family sizing multipliers:")
    for family_name, multiplier in FAMILY_SIZING.items():
        rules = FAMILIES.get(family_name, {})
        rules_str = ' & '.join([f"{m}{op}{v}" for m, (op, v) in rules.items()]) if rules else "(default)"
        print(f"   • {family_name:15s}: x{multiplier:.1f}  [{rules_str}]")
    
    # Find all enriched files
    strategies = find_all_enriched_files(output_folder)
    
    if not strategies:
        print(f"\n❌ No enriched files found in: {output_folder}")
        print(f"   Run run_analysis.py first to generate trades_enriched_*.xlsx files")
        return
    
    print(f"\n📁 Enriched files found: {len(strategies)}")
    for s in strategies:
        print(f"   • {s}")
    
    # Store results for final summary
    summary_results = []
    
    # Process each strategy
    for strategy in strategies:
        print("\n" + "=" * 140)
        print(f"📊 SIZING ANALYSIS: {strategy}")
        print("=" * 140)
        
        try:
            df = load_enriched_trades(strategy, output_folder)
            
            # Classify each trade into a family
            df['family'] = df.apply(classify_trade_family, axis=1)
            
            # Apply sizing multiplier
            df['sizing_mult'] = df['family'].map(FAMILY_SIZING).fillna(1.0)
            df['profit_sized'] = df['profit'] * df['sizing_mult']
            
            # Show family distribution
            family_counts = df['family'].value_counts()
            print(f"\n📋 Trade distribution by family:")
            for fam, count in family_counts.items():
                pct = count / len(df) * 100
                mult = FAMILY_SIZING.get(fam, 1.0)
                print(f"   • {fam:15s}: {count:5d} trades ({pct:5.1f}%) x{mult:.1f}")
            
            # Calculate metrics
            profit_base = df['profit'].sum()
            profit_sized = df['profit_sized'].sum()
            delta_pct = ((profit_sized - profit_base) / abs(profit_base) * 100) if profit_base != 0 else 0
            
            # Calculate drawdown for both
            df_sorted = df.sort_values('buy_time').copy()
            
            # Base DD
            df_sorted['cum_profit_base'] = df_sorted['profit'].cumsum()
            df_sorted['running_max_base'] = df_sorted['cum_profit_base'].cummax()
            df_sorted['dd_base'] = df_sorted['running_max_base'] - df_sorted['cum_profit_base']
            max_dd_base = df_sorted['dd_base'].max()
            max_dd_base_pct = (max_dd_base / INITIAL_CAPITAL) * 100
            
            # Sized DD
            df_sorted['cum_profit_sized'] = df_sorted['profit_sized'].cumsum()
            df_sorted['running_max_sized'] = df_sorted['cum_profit_sized'].cummax()
            df_sorted['dd_sized'] = df_sorted['running_max_sized'] - df_sorted['cum_profit_sized']
            max_dd_sized = df_sorted['dd_sized'].max()
            max_dd_sized_pct = (max_dd_sized / INITIAL_CAPITAL) * 100
            
            # Win rate (same for both)
            win_rate = (df['profit'] > 0).mean() * 100
            
            # Show results
            print(f"\n{'METRIC':<20} {'BASE':>12} {'SIZED':>12} {'Δ':>10}")
            print("-" * 60)
            print(f"{'Profit':<20} {profit_base:>12.2f} {profit_sized:>12.2f} {delta_pct:>+9.1f}%")
            print(f"{'Max DD%':<20} {max_dd_base_pct:>11.2f}% {max_dd_sized_pct:>11.2f}%")
            print(f"{'Win%':<20} {win_rate:>11.1f}% {win_rate:>11.1f}%")
            print("-" * 60)
            
            profit_ok = "✅" if profit_sized > profit_base else "❌"
            print(f"\n{'Result:':<20} {profit_ok} Profit {'improved' if profit_sized > profit_base else 'decreased'} by {abs(delta_pct):.1f}%")
            
            # Store for summary
            summary_results.append({
                'strategy': strategy,
                'num_trades': len(df),
                'profit_base': profit_base,
                'profit_sized': profit_sized,
                'delta_pct': delta_pct,
                'win_rate': win_rate,
                'dd_base': max_dd_base_pct,
                'dd_sized': max_dd_sized_pct,
                'df_sorted': df_sorted,
            })
            
        except Exception as e:
            print(f"\n❌ Error processing {strategy}: {e}")
    
    # Final summary
    print("\n" + "=" * 140)
    print("📊 SUMMARY - POSITION SIZING RESULTS")
    print("=" * 140)
    
    # Header
    print(f"\n{'STRATEGY':<30} {'TRADES':>8} │ {'PROFIT_B':>10} {'PROFIT_S':>10} {'Δ%':>8} {'':>2} │ {'WIN%':>7} │ {'DD%_B':>8} {'DD%_S':>8} {'':>2}")
    print("-" * 120)
    
    for r in summary_results:
        profit_ok = "✅" if r['profit_sized'] > r['profit_base'] else "❌"
        dd_ok = "✅" if r['dd_sized'] < r['dd_base'] else "❌"
        
        print(f"{r['strategy']:<30} {r['num_trades']:>8} │ {r['profit_base']:>10.2f} {r['profit_sized']:>10.2f} {r['delta_pct']:>+7.1f}% {profit_ok:>2} │ {r['win_rate']:>6.1f}% │ {r['dd_base']:>7.2f}% {r['dd_sized']:>7.2f}% {dd_ok:>2}")
    
    print("-" * 120)
    
    # Calculate averages
    n = len(summary_results)
    if n > 0:
        avg_trades = sum(r['num_trades'] for r in summary_results) / n
        avg_profit_base = sum(r['profit_base'] for r in summary_results) / n
        avg_profit_sized = sum(r['profit_sized'] for r in summary_results) / n
        avg_delta_pct = ((avg_profit_sized - avg_profit_base) / abs(avg_profit_base) * 100) if avg_profit_base != 0 else 0
        avg_win_rate = sum(r['win_rate'] for r in summary_results) / n
        avg_dd_base = sum(r['dd_base'] for r in summary_results) / n
        avg_dd_sized = sum(r['dd_sized'] for r in summary_results) / n
        
        avg_profit_ok = "✅" if avg_profit_sized > avg_profit_base else "❌"
        avg_dd_ok = "✅" if avg_dd_sized < avg_dd_base else "❌"
        
        print(f"{'AVERAGE':<30} {avg_trades:>8.0f} │ {avg_profit_base:>10.2f} {avg_profit_sized:>10.2f} {avg_delta_pct:>+7.1f}% {avg_profit_ok:>2} │ {avg_win_rate:>6.1f}% │ {avg_dd_base:>7.2f}% {avg_dd_sized:>7.2f}% {avg_dd_ok:>2}")
    
    print("-" * 120)
    
    # =================================================================
    # PLOTS - Generate all plots at the end
    # =================================================================
    
    if SHOW_PLOTS:
        print("\n" + "=" * 140)
        print("📊 GENERATING PLOTS")
        print("=" * 140)
        
        for r in summary_results:
            df_sorted = r['df_sorted']
            
            if not HAS_MATPLOTLIB:
                print("⚠️  matplotlib not available, skipping plots")
                break
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot both equity curves
            ax.plot(df_sorted['buy_time'], df_sorted['cum_profit_base'], 
                    color='blue', linewidth=1.5, label=f'Base (profit: {r["profit_base"]:.2f})', alpha=0.7)
            ax.plot(df_sorted['buy_time'], df_sorted['cum_profit_sized'], 
                    color='green', linewidth=1.5, label=f'Sized (profit: {r["profit_sized"]:.2f})', alpha=0.9)
            
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_ylabel('Cumulative Profit')
            ax.set_xlabel('Time')
            ax.set_title(f"{r['strategy']} - Base vs Position Sizing (Δ {r['delta_pct']:+.1f}%)")
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    print("\n" + "=" * 140)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"📁 Output folder: {get_output_folder()}")
    print(f"🔧 Mode: {MODE}")
    if MODE == 'confluence':
        print(f"🔍 Filter by: {FILTER_BY}")
        if FILTER_BY == 'generator':
            print(f"   Generator: {GENERATOR}")
        else:
            print(f"   Direction: {DIRECTION}")
    elif MODE == 'families':
        print(f"📋 Families: {len(FAMILIES)}")
    elif MODE == 'sizing':
        print(f"📋 Sizing families: {len(FAMILY_SIZING)}")
    print(f"💰 Initial capital: {INITIAL_CAPITAL}")
    print(f"📊 Show plots: {SHOW_PLOTS}")
    print()
    
    if MODE == 'single':
        run_single_mode(STRATEGY)
    elif MODE == 'confluence':
        if FILTER_BY == 'generator':
            run_confluence_mode('generator', GENERATOR)
        elif FILTER_BY == 'direction':
            run_confluence_mode('direction', DIRECTION)
        else:
            print(f"❌ Unknown FILTER_BY: {FILTER_BY}")
            print(f"   Use 'generator' or 'direction'")
    elif MODE == 'families':
        run_families_mode()
    elif MODE == 'sizing':
        run_sizing_mode()
    else:
        print(f"❌ Unknown mode: {MODE}")
        print(f"   Use 'single', 'confluence', 'families', or 'sizing'")