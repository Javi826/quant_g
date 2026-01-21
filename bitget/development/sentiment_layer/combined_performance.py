"""
sentiment_layer/combined_analyzer.py

Analyzes strategy performance by COMBINED STATES:
- regime (family_trend) + sentiment_state
- Example: trending_uptrend_greed, volatile_downtrend_fear, etc.

This reveals which specific regime+sentiment combinations work best.

Usage:
    python combined_analyzer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_layer.config import INITIAL_CAPITAL, DATE_RANGE_FILTER

# Output folder for combined enriched trades
OUTPUT_FOLDER_COMBINED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'sentiment_layer',
    'output_combined'
)


# Minimum trades for confidence
MIN_TRADES_CONFIDENCE = 50


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


def analyze_by_combined_state(df: pd.DataFrame, initial_capital: float) -> dict:
    """
    Analyzes performance by combined state (regime_sentiment).
    """
    stats = {}
    
    for combined_state in df['combined_state'].unique():
        state_df = df[df['combined_state'] == combined_state].copy()
        state_df = state_df.sort_values('buy_time').reset_index(drop=True)
        state_df['equity'] = initial_capital + state_df['profit'].cumsum()
        
        num_trades = len(state_df)
        profit = state_df['profit'].sum()
        profits_list = state_df['profit'].tolist()
        
        # Confidence indicator: tick if >=50, X otherwise
        if num_trades >= MIN_TRADES_CONFIDENCE:
            confidence = "✓"
        else:
            confidence = "✗"
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_confidence_interval(profits_list)
        
        stats[combined_state] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(state_df['equity']),
            'win_rate': (state_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'profits_list': profits_list,
            'confidence': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return stats


def analyze_by_regime(df: pd.DataFrame, initial_capital: float) -> dict:
    """Analyzes performance by regime only (ignoring sentiment)."""
    stats = {}
    
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime].copy()
        regime_df = regime_df.sort_values('buy_time').reset_index(drop=True)
        regime_df['equity'] = initial_capital + regime_df['profit'].cumsum()
        
        num_trades = len(regime_df)
        profit = regime_df['profit'].sum()
        profits_list = regime_df['profit'].tolist()
        
        if num_trades >= MIN_TRADES_CONFIDENCE:
            confidence = "✓"
        else:
            confidence = "✗"
        
        ci_lower, ci_upper = bootstrap_confidence_interval(profits_list)
        
        stats[regime] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(regime_df['equity']),
            'win_rate': (regime_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'profits_list': profits_list,
            'confidence': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return stats


def analyze_by_sentiment(df: pd.DataFrame, initial_capital: float) -> dict:
    """Analyzes performance by sentiment only (ignoring regime)."""
    stats = {}
    
    for sentiment in df['sentiment_state'].unique():
        sent_df = df[df['sentiment_state'] == sentiment].copy()
        sent_df = sent_df.sort_values('buy_time').reset_index(drop=True)
        sent_df['equity'] = initial_capital + sent_df['profit'].cumsum()
        
        num_trades = len(sent_df)
        profit = sent_df['profit'].sum()
        profits_list = sent_df['profit'].tolist()
        
        if num_trades >= MIN_TRADES_CONFIDENCE:
            confidence = "✓"
        else:
            confidence = "✗"
        
        ci_lower, ci_upper = bootstrap_confidence_interval(profits_list)
        
        stats[sentiment] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(sent_df['equity']),
            'win_rate': (sent_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'profits_list': profits_list,
            'confidence': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return stats


def analyze_strategy_combined(filepath: str, initial_capital: float, date_range: tuple = None) -> dict:
    """Analyzes a single strategy by combined states and individual dimensions."""
    strategy = Path(filepath).stem.replace('trades_combined_', '')
    df = load_enriched_trades(filepath)
    
    # Apply date range filter if specified
    if date_range is not None:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['buy_time'] >= start_date) & (df['buy_time'] <= end_date)].copy()
    
    # Validate required columns
    required_cols = ['combined_state', 'regime', 'sentiment_state']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"File {filepath} missing '{col}' column")
    
    # Sort by time
    df = df.sort_values('buy_time').reset_index(drop=True)
    
    # Analyze by each dimension
    combined_stats = analyze_by_combined_state(df, initial_capital)
    regime_stats = analyze_by_regime(df, initial_capital)
    sentiment_stats = analyze_by_sentiment(df, initial_capital)
    
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
        'combined_stats': combined_stats,
        'regime_stats': regime_stats,
        'sentiment_stats': sentiment_stats
    }


def format_significance(p_value: float) -> str:
    """Formats significance with green tick or red X."""
    if p_value < 0.1:
        return f"✅ (p={p_value:.3f})"
    else:
        return f"❌ (p={p_value:.2f})"


def print_single_strategy_combined(r: dict):
    """Prints analysis tables for a single strategy."""
    print(f"\n{'='*200}")
    print(f"STRATEGY: {r['strategy']} (Total: {r['total_trades']} trades, Profit: ${r['total_profit']:.2f}, DD: {r['total_dd_pct']:.2f}%, WR: {r['total_win_rate']:.1f}%)")
    print(f"{'='*200}")
    
    # TABLE 1: BY COMBINED STATE (regime + sentiment)
    print(f"\n{'─'*150}")
    print(f"BY COMBINED STATE (regime + sentiment)")
    print(f"{'─'*150}")
    print(f"{'COMBINED_STATE':<40} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 150)
    
    combined_stats = r['combined_stats']
    sorted_combined = sorted(combined_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (state, stats) in enumerate(sorted_combined):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        
        # Calculate p-value
        if len(sorted_combined) < 2:
            p_str = "N/A"
        elif idx == 0:
            # Best vs 2nd best
            p_value = permutation_test(sorted_combined[0][1]['profits_list'], sorted_combined[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            # Others vs best
            p_value = permutation_test(stats['profits_list'], sorted_combined[0][1]['profits_list'])
            p_str = format_significance(p_value)
        
        print(f"{state:<40} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 150)
    print(f"{'TOTAL':<40} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    # Best combined state comparison
    if len(sorted_combined) >= 2:
        best_state, best_stats = sorted_combined[0]
        second_state, second_stats = sorted_combined[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_state} (${best_stats['profit']:.2f}) vs 2ND: {second_state} (${second_stats['profit']:.2f}) | {sig_str}")
    
    # TABLE 2: BY REGIME ONLY (for comparison)
    print(f"\n{'─'*120}")
    print(f"BY REGIME ONLY (ignoring sentiment)")
    print(f"{'─'*120}")
    print(f"{'REGIME':<30} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    regime_stats = r['regime_stats']
    sorted_regime = sorted(regime_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (regime, stats) in enumerate(sorted_regime):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        
        if len(sorted_regime) < 2:
            p_str = "N/A"
        elif idx == 0:
            p_value = permutation_test(sorted_regime[0][1]['profits_list'], sorted_regime[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            p_value = permutation_test(stats['profits_list'], sorted_regime[0][1]['profits_list'])
            p_str = format_significance(p_value)
        
        print(f"{regime:<30} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)
    
    # TABLE 3: BY SENTIMENT ONLY (for comparison)
    print(f"\n{'─'*120}")
    print(f"BY SENTIMENT ONLY (ignoring regime)")
    print(f"{'─'*120}")
    print(f"{'SENTIMENT':<30} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    sentiment_stats = r['sentiment_stats']
    sorted_sentiment = sorted(sentiment_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (sentiment, stats) in enumerate(sorted_sentiment):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        
        if len(sorted_sentiment) < 2:
            p_str = "N/A"
        elif idx == 0:
            p_value = permutation_test(sorted_sentiment[0][1]['profits_list'], sorted_sentiment[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            p_value = permutation_test(stats['profits_list'], sorted_sentiment[0][1]['profits_list'])
            p_str = format_significance(p_value)
        
        print(f"{sentiment:<30} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)


def print_summary_table(results: list):
    """Prints final summary table across all strategies."""
    print(f"\n{'='*200}")
    print(f"{'='*200}")
    print(f"SUMMARY - ALL STRATEGIES")
    print(f"{'='*200}")
    print(f"{'='*200}")
    
    # Summary: Best Combined State
    print(f"\n{'─'*170}")
    print("BEST COMBINED STATE PER STRATEGY")
    print(f"{'─'*170}")
    print(f"{'STRATEGY':<30} {'BEST_COMBINED':<35} {'CONF':>5} {'TRADES':>8} {'PROFIT':>10} {'2ND_BEST':<35} {'TRADES':>8} {'PROFIT':>10} {'SIG?':>10}")
    print("-" * 170)
    
    for r in results:
        combined_stats = r['combined_stats']
        if combined_stats and len(combined_stats) >= 2:
            sorted_comb = sorted(combined_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
            best_comb, best_stats = sorted_comb[0]
            second_comb, second_stats = sorted_comb[1]
            p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
            
            sig_str = format_significance(p_value)
            
            print(f"{r['strategy']:<30} {best_comb:<35} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {second_comb:<35} {second_stats['num_trades']:>8} {second_stats['profit']:>10.2f} {sig_str:>10}")
        elif combined_stats and len(combined_stats) == 1:
            best_comb, best_stats = list(combined_stats.items())[0]
            print(f"{r['strategy']:<30} {best_comb:<35} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {'(only one)':<35} {0:>8} {0.0:>10.2f} {'N/A':>10}")
    
    print("-" * 170)


def analyze_all_strategies(output_folder: str = None, initial_capital: float = None,
                           date_range: tuple = None) -> list:
    """Analyzes all strategies by combined states."""
    output_folder = output_folder or OUTPUT_FOLDER_COMBINED
    initial_capital = initial_capital or INITIAL_CAPITAL
    
    print("=" * 70)
    print("COMBINED ANALYZER - Regime + Sentiment")
    print("=" * 70)
    
    if date_range:
        print(f"\n⚠️  DATE RANGE FILTER ACTIVE: {date_range[0]} → {date_range[1]}")
    
    print(f"\nInitial capital: ${initial_capital}")
    
    print("\nAnalysis dimensions:")
    print("  1. COMBINED STATE: regime + sentiment (full granularity)")
    print("  2. REGIME ONLY: ignoring sentiment (for comparison)")
    print("  3. SENTIMENT ONLY: ignoring regime (for comparison)")
    
    print("\nConfidence indicator (CONF):")
    print(f"  ✓ = ≥{MIN_TRADES_CONFIDENCE} trades (reliable)")
    print(f"  ✗ = <{MIN_TRADES_CONFIDENCE} trades (unreliable)")
    
    print("\nSignificance indicator (SIG?):")
    print(f"  ✅ = p<0.10 (statistically significant difference)")
    print(f"  ❌ = p≥0.10 (no significant difference)")
    
    pattern = os.path.join(output_folder, "trades_combined_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No combined files found in {output_folder}")
        return []
    
    print(f"\nFiles found: {len(files)}\n")
    
    results = []
    for f in files:
        r = analyze_strategy_combined(f, initial_capital, date_range=date_range)
        results.append(r)
    
    # Print each strategy
    for r in results:
        print_single_strategy_combined(r)
    
    # Print final summary table
    print_summary_table(results)
    
    # Interpretation guide
    print(f"\n{'='*200}")
    print("INTERPRETATION GUIDE:")
    print("\n  CONF (Confidence):")
    print("    ✓ = Reliable sample (≥50 trades) - trust these results")
    print("    ✗ = Unreliable sample (<50 trades) - don't trust these results")
    print("\n  SIG? (Statistical test):")
    print("    ✅ (p<0.10) = Difference is real, not random")
    print("    ❌ (p≥0.10) = Difference could be random chance")
    print("\n  KEY INSIGHTS:")
    print("    - If COMBINED STATE shows ✓✅ advantage: Use specific regime+sentiment filtering")
    print("    - If REGIME ONLY performs as well: Sentiment doesn't add value for this strategy")
    print("    - If SENTIMENT ONLY performs as well: Regime doesn't add value for this strategy")
    print("    - Use the simplest filter that captures the edge (Occam's Razor)")
    print(f"{'='*200}")
    
    return results


if __name__ == "__main__":
    analyze_all_strategies(date_range=DATE_RANGE_FILTER)