"""
market_regime/regime_analyzer.py

Analyzes strategy performance across 3 dimensions:
1. By FAMILY (trending/volatile/ranging) - ignoring BTC direction
2. By DIRECTION (uptrend/downtrend) - ignoring family
3. By REGIME (6 combined) - full granularity

Usage:
    python regime_analyzer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.config import (
    OUTPUT_FOLDER, FAMILIES, 
    DIRECTION_METHOD, DIRECTION_MA_PERIOD,
    DIRECTION_MA_FAST, DIRECTION_MA_SLOW,
    INITIAL_CAPITAL, DATE_RANGE_FILTER
)


# Minimum trades for confidence
MIN_TRADES_CONFIDENCE = 50


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


def calculate_max_dd_pct(equity_curve: pd.Series) -> float:
    """Calculates Maximum Drawdown % correctly."""
    if len(equity_curve) == 0:
        return 0.0
    
    cummax = equity_curve.cummax()
    drawdown_pct = np.where(
        cummax > 0,
        ((cummax - equity_curve) / cummax) * 100,
        0.0
    )
    return float(np.max(drawdown_pct))


def bootstrap_confidence_interval(profits: list, n_bootstrap: int = 1000, confidence: float = 0.95) -> tuple:
    """
    Simple bootstrap to estimate confidence interval for mean profit per trade.
    Returns (lower_bound, upper_bound)
    """
    if len(profits) < 10:
        return (np.nan, np.nan)
    
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(profits, size=len(profits), replace=True)
        means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    
    return (lower, upper)


def permutation_test(profits1: list, profits2: list, n_permutations: int = 1000) -> float:
    """
    Permutation test to check if two profit distributions are significantly different.
    Returns p-value.
    """
    if len(profits1) < 10 or len(profits2) < 10:
        return 1.0  # Not enough data
    
    observed_diff = np.mean(profits1) - np.mean(profits2)
    combined = profits1 + profits2
    n1 = len(profits1)
    
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    
    p_value = count_extreme / n_permutations
    return p_value


def load_enriched_trades(filepath: str) -> pd.DataFrame:
    """Loads enriched trades from Excel file."""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    return df


def analyze_by_dimension(df: pd.DataFrame, dimension: str, initial_capital: float) -> dict:
    """
    Analyzes performance by a single dimension.
    
    dimension: 'family', 'trend', or 'regime'
    """
    stats = {}
    
    for category in df[dimension].unique():
        cat_df = df[df[dimension] == category].copy()
        cat_df = cat_df.sort_values('buy_time').reset_index(drop=True)
        cat_df['equity'] = initial_capital + cat_df['profit'].cumsum()
        
        num_trades = len(cat_df)
        profit = cat_df['profit'].sum()
        profits_list = cat_df['profit'].tolist()
        
        # Confidence indicator: tick if >=50, X otherwise
        if num_trades >= MIN_TRADES_CONFIDENCE:
            confidence = "✓"  # Normal tick
        else:
            confidence = "✗"  # Normal X
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_confidence_interval(profits_list)
        
        stats[category] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(cat_df['equity']),
            'win_rate': (cat_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'profits_list': profits_list,
            'confidence': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return stats


def analyze_strategy_all_dimensions(filepath: str, families: dict, initial_capital: float, 
                                     direction_method: str = None, date_range: tuple = None) -> dict:
    """Analyzes a single strategy across all 3 dimensions."""
    strategy = Path(filepath).stem.replace('trades_enriched_', '')
    df = load_enriched_trades(filepath)
    
    # Apply date range filter if specified
    if date_range is not None:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['buy_time'] >= start_date) & (df['buy_time'] <= end_date)].copy()
    
    # Classify family
    df['family'] = df.apply(lambda row: classify_trade(row, families), axis=1)
    
    # ==========================================
    # NEW: Determine trend based on configured method
    # ==========================================
    direction_method = direction_method or DIRECTION_METHOD
    
    if direction_method == 'price_vs_ma':
        # Method 1: Price vs single MA
        price_vs_ma_col = f'price_vs_ma_{DIRECTION_MA_PERIOD}'
        df['trend'] = df.apply(
            lambda r: 'uptrend' if (not pd.isna(r.get(price_vs_ma_col)) and r[price_vs_ma_col] > 1.0) else 
                      'downtrend' if (not pd.isna(r.get(price_vs_ma_col))) else 
                      'unknown',
            axis=1
        )
    
    elif direction_method == 'ma_cross':
        # Method 2: MA cross (e.g., MA50 vs MA200)
        ma_cross_col = f'ma_{DIRECTION_MA_FAST}_vs_ma_{DIRECTION_MA_SLOW}'
        df['trend'] = df.apply(
            lambda r: 'uptrend' if (not pd.isna(r.get(ma_cross_col)) and r[ma_cross_col] > 1.0) else 
                      'downtrend' if (not pd.isna(r.get(ma_cross_col))) else 
                      'unknown',
            axis=1
        )
    
    else:
        # Fallback: default to price_vs_ma_50
        print(f"    ⚠️  Unknown DIRECTION_METHOD '{direction_method}', defaulting to price_vs_ma_50")
        df['trend'] = df.apply(
            lambda r: 'uptrend' if (not pd.isna(r.get('price_vs_ma_50')) and r['price_vs_ma_50'] > 1.0) else 
                      'downtrend' if (not pd.isna(r.get('price_vs_ma_50'))) else 
                      'unknown',
            axis=1
        )
    
    # Create regime column (family_trend)
    df['regime'] = df['family'] + '_' + df['trend']
    
    # Sort by time
    df = df.sort_values('buy_time').reset_index(drop=True)
    
    # Analyze by each dimension
    family_stats = analyze_by_dimension(df, 'family', initial_capital)
    trend_stats = analyze_by_dimension(df, 'trend', initial_capital)
    regime_stats = analyze_by_dimension(df, 'regime', initial_capital)
    
    # Calculate total equity and DD for entire strategy
    df_sorted = df.sort_values('buy_time').reset_index(drop=True)
    df_sorted['equity_total'] = initial_capital + df_sorted['profit'].cumsum()
    total_dd_pct = calculate_max_dd_pct(df_sorted['equity_total'])
    total_win_rate = (df_sorted['profit'] > 0).mean() * 100 if len(df_sorted) > 0 else 0.0
    
    return {
        'strategy': strategy,
        'total_trades': len(df),
        'total_profit': df['profit'].sum(),
        'total_dd_pct': total_dd_pct,
        'total_win_rate': total_win_rate,
        'family_stats': family_stats,
        'trend_stats': trend_stats,
        'regime_stats': regime_stats
    }


def format_significance(p_value: float) -> str:
    """Formats significance with green tick or red X."""
    if p_value < 0.1:
        return f"✅ (p={p_value:.3f})"
    else:
        return f"❌ (p={p_value:.2f})"


def print_single_strategy_all_dimensions(r: dict):
    """Prints all 3 dimension tables for a single strategy."""
    print(f"\n\033[93m{'='*200}\033[0m")
    print(f"\033[93mSTRATEGY: {r['strategy']} (Total: {r['total_trades']} trades, Profit: ${r['total_profit']:.2f}, DD: {r['total_dd_pct']:.2f}%, WR: {r['total_win_rate']:.1f}%)\033[0m")
    print(f"\033[93m{'='*200}\033[0m")
    
    # TABLE 1: BY FAMILY
    print(f"\n{'─'*120}")
    print(f"BY FAMILY (trending/volatile/ranging)")
    print(f"{'─'*120}")
    print(f"{'FAMILY':<20} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    family_stats = r['family_stats']
    sorted_family = sorted(family_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (category, stats) in enumerate(sorted_family):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        
        # Calculate p-value
        if len(sorted_family) < 2:
            p_str = "N/A"
        elif idx == 0:
            # Best vs 2nd best
            p_value = permutation_test(sorted_family[0][1]['profits_list'], sorted_family[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            # Others vs best
            p_value = permutation_test(stats['profits_list'], sorted_family[0][1]['profits_list'])
            p_str = format_significance(p_value)
        
        print(f"{category:<20} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)
    print(f"{'TOTAL':<20} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    # Best family comparison
    if len(sorted_family) >= 2:
        best_fam, best_stats = sorted_family[0]
        second_fam, second_stats = sorted_family[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_fam} (${best_stats['profit']:.2f}) vs 2ND: {second_fam} (${second_stats['profit']:.2f}) | {sig_str}")
    
    # TABLE 2: BY DIRECTION
    print(f"\n{'─'*120}")
    print(f"BY DIRECTION (uptrend/downtrend)")
    print(f"{'─'*120}")
    print(f"{'DIRECTION':<20} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    trend_stats = r['trend_stats']
    sorted_trend = sorted(trend_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (category, stats) in enumerate(sorted_trend):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        
        # Calculate p-value
        if len(sorted_trend) < 2:
            p_str = "N/A"
        elif idx == 0:
            # Best vs 2nd best
            p_value = permutation_test(sorted_trend[0][1]['profits_list'], sorted_trend[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            # Others vs best
            p_value = permutation_test(stats['profits_list'], sorted_trend[0][1]['profits_list'])
            p_str = format_significance(p_value)
        
        print(f"{category:<20} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)
    print(f"{'TOTAL':<20} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    # Best direction comparison
    if len(sorted_trend) >= 2:
        best_dir, best_stats = sorted_trend[0]
        second_dir, second_stats = sorted_trend[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_dir} (${best_stats['profit']:.2f}) vs 2ND: {second_dir} (${second_stats['profit']:.2f}) | {sig_str}")
    
    # TABLE 3: BY REGIME
    print(f"\n{'─'*120}")
    print(f"BY REGIME (6 combined categories)")
    print(f"{'─'*120}")
    print(f"{'REGIME':<20} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    regime_stats = r['regime_stats']
    sorted_regime = sorted(regime_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (category, stats) in enumerate(sorted_regime):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        
        # Calculate p-value
        if len(sorted_regime) < 2:
            p_str = "N/A"
        elif idx == 0:
            # Best vs 2nd best
            p_value = permutation_test(sorted_regime[0][1]['profits_list'], sorted_regime[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            # Others vs best
            p_value = permutation_test(stats['profits_list'], sorted_regime[0][1]['profits_list'])
            p_str = format_significance(p_value)
        
        print(f"{category:<20} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)
    print(f"{'TOTAL':<20} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    # Best regime comparison
    if len(sorted_regime) >= 2:
        best_reg, best_stats = sorted_regime[0]
        second_reg, second_stats = sorted_regime[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_reg} (${best_stats['profit']:.2f}) vs 2ND: {second_reg} (${second_stats['profit']:.2f}) | {sig_str}")

def print_summary_tables(results: list):
    """Prints final summary tables across all strategies."""
    print(f"\n{'='*200}")
    print(f"{'='*200}")
    print(f"SUMMARY - ALL STRATEGIES")
    print(f"{'='*200}")
    print(f"{'='*200}")
    
    # Summary 1: Best Family
    print(f"\n{'─'*145}")
    print("BEST FAMILY PER STRATEGY")
    print(f"{'─'*145}")
    print(f"{'STRATEGY':<30} {'BEST_FAMILY':<20} {'CONF':>5} {'TRADES':>8} {'PROFIT':>10} {'2ND_BEST':<20} {'TRADES':>8} {'PROFIT':>10} {'SIGNIFICANT?':>15}")
    print("-" * 145)
    
    for r in results:
        family_stats = r['family_stats']
        if family_stats and len(family_stats) >= 2:
            sorted_fam = sorted(family_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
            best_fam, best_stats = sorted_fam[0]
            second_fam, second_stats = sorted_fam[1]
            p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
            
            sig_str = format_significance(p_value)
            
            print(f"{r['strategy']:<30} {best_fam:<20} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {second_fam:<20} {second_stats['num_trades']:>8} {second_stats['profit']:>10.2f} {sig_str:>15}")
        elif family_stats and len(family_stats) == 1:
            best_fam, best_stats = list(family_stats.items())[0]
            print(f"{r['strategy']:<30} {best_fam:<20} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {'(only one)':<20} {0:>8} {0.0:>10.2f} {'N/A':>15}")
    
    print("-" * 145)
    
    # Summary 2: Best Direction
    print(f"\n{'─'*145}")
    print("BEST DIRECTION PER STRATEGY")
    print(f"{'─'*145}")
    print(f"{'STRATEGY':<30} {'BEST_DIRECTION':<20} {'CONF':>5} {'TRADES':>8} {'PROFIT':>10} {'2ND_BEST':<20} {'TRADES':>8} {'PROFIT':>10} {'SIGNIFICANT?':>15}")
    print("-" * 145)
    
    for r in results:
        trend_stats = r['trend_stats']
        if trend_stats and len(trend_stats) >= 2:
            sorted_trend = sorted(trend_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
            best_dir, best_stats = sorted_trend[0]
            second_dir, second_stats = sorted_trend[1]
            p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
            
            sig_str = format_significance(p_value)
            
            print(f"{r['strategy']:<30} {best_dir:<20} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {second_dir:<20} {second_stats['num_trades']:>8} {second_stats['profit']:>10.2f} {sig_str:>15}")
        elif trend_stats and len(trend_stats) == 1:
            best_dir, best_stats = list(trend_stats.items())[0]
            print(f"{r['strategy']:<30} {best_dir:<20} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {'(only one)':<20} {0:>8} {0.0:>10.2f} {'N/A':>15}")
    
    print("-" * 145)
    
    # Summary 3: Best Regime
    print(f"\n{'─'*145}")
    print("BEST REGIME PER STRATEGY")
    print(f"{'─'*145}")
    print(f"{'STRATEGY':<30} {'BEST_REGIME':<20} {'CONF':>5} {'TRADES':>8} {'PROFIT':>10} {'2ND_BEST':<20} {'TRADES':>8} {'PROFIT':>10} {'SIGNIFICANT?':>15}")
    print("-" * 145)
    
    for r in results:
        regime_stats = r['regime_stats']
        if regime_stats and len(regime_stats) >= 2:
            sorted_reg = sorted(regime_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
            best_reg, best_stats = sorted_reg[0]
            second_reg, second_stats = sorted_reg[1]
            p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
            
            sig_str = format_significance(p_value)
            
            print(f"{r['strategy']:<30} {best_reg:<20} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {second_reg:<20} {second_stats['num_trades']:>8} {second_stats['profit']:>10.2f} {sig_str:>15}")
        elif regime_stats and len(regime_stats) == 1:
            best_reg, best_stats = list(regime_stats.items())[0]
            print(f"{r['strategy']:<30} {best_reg:<20} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {'(only one)':<20} {0:>8} {0.0:>10.2f} {'N/A':>15}")
    
    print("-" * 145)


def analyze_all_strategies(output_folder: str = None, families: dict = None, 
                           initial_capital: float = None, direction_method: str = None,
                           date_range: tuple = None) -> list:
    """Analyzes all strategies across all 3 dimensions."""
    output_folder = output_folder or OUTPUT_FOLDER
    families = families or FAMILIES
    initial_capital = initial_capital or INITIAL_CAPITAL
    direction_method = direction_method or DIRECTION_METHOD
    
    print("=" * 70)
    print("REGIME ANALYZER - Performance across 3 dimensions")
    print("=" * 70)
    
    if date_range:
        print(f"\n⚠️  DATE RANGE FILTER ACTIVE: {date_range[0]} → {date_range[1]}")
    
    # Display direction detection method
    print(f"\nDirection detection method: {direction_method}")
    if direction_method == 'price_vs_ma':
        print(f"  Using: Price vs MA{DIRECTION_MA_PERIOD}")
    elif direction_method == 'ma_cross':
        print(f"  Using: MA{DIRECTION_MA_FAST} vs MA{DIRECTION_MA_SLOW}")
    
    print(f"Initial capital: ${initial_capital}")
    
    print("\nDimensions analyzed:")
    print("  1. FAMILY: trending/volatile/ranging (ignoring BTC direction)")
    print("  2. DIRECTION: uptrend/downtrend (ignoring family)")
    print("  3. REGIME: 6 combined categories (full granularity)")
    
    print("\nConfidence indicator (CONF):")
    print(f"  ✓ = ≥{MIN_TRADES_CONFIDENCE} trades (reliable)")
    print(f"  ✗ = <{MIN_TRADES_CONFIDENCE} trades (unreliable)")
    
    print("\nSignificance indicator (SIGNIFICANT?):")
    print(f"  ✅ = p<0.10 (statistically significant difference)")
    print(f"  ❌ = p≥0.10 (no significant difference)")
    
    pattern = os.path.join(output_folder, "trades_enriched_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {output_folder}")
        return []
    
    print(f"\nFiles found: {len(files)}\n")
    
    results = []
    for f in files:
        r = analyze_strategy_all_dimensions(f, families, initial_capital, direction_method, date_range=date_range)
        results.append(r)
    
    # Print each strategy with all 3 dimensions
    for r in results:
        print_single_strategy_all_dimensions(r)
    
    # Print final summary tables
    print_summary_tables(results)
    
    # Interpretation guide
    print(f"\n{'='*200}")
    print("INTERPRETATION GUIDE:")
    print("\n  CONF (Confidence):")
    print("    ✓ = Reliable sample (≥50 trades) - trust these results")
    print("    ✗ = Unreliable sample (<50 trades) - don't trust these results")
    print("\n  SIGNIFICANT? (Statistical test):")
    print("    ✅ (p<0.10) = Difference is real, not random")
    print("    ❌ (p≥0.10) = Difference could be random chance")
    print("\n  FILTERING DECISION:")
    print("    - Only filter if BOTH: ✓ (reliable) AND ✅ (significant)")
    print("    - If FAMILY is ✓✅: filter by family only")
    print("    - If DIRECTION is ✓✅: filter by direction only")
    print("    - If REGIME is ✓✅: filter by specific regime")
    print("    - Otherwise: don't filter, operate in all conditions")
    print(f"{'='*200}")
    
    return results


if __name__ == "__main__":
    analyze_all_strategies(date_range=DATE_RANGE_FILTER)