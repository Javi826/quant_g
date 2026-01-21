"""
sentiment_layer/sentiment_analyzer.py

Analyzes strategy performance by SENTIMENT STATE:
- extreme_fear
- fear
- neutral
- greed
- extreme_greed

Usage:
    python sentiment_analyzer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_layer.config import (
    OUTPUT_FOLDER, SENTIMENT_THRESHOLDS,
    INITIAL_CAPITAL, DATE_RANGE_FILTER
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


def analyze_by_sentiment(df: pd.DataFrame, initial_capital: float) -> dict:
    """
    Analyzes performance by sentiment state.
    """
    stats = {}
    
    for sentiment_state in df['sentiment_state'].unique():
        state_df = df[df['sentiment_state'] == sentiment_state].copy()
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
        
        stats[sentiment_state] = {
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


def analyze_strategy_sentiment(filepath: str, initial_capital: float, date_range: tuple = None) -> dict:
    """Analyzes a single strategy by sentiment state."""
    strategy = Path(filepath).stem.replace('trades_sentiment_', '')
    df = load_enriched_trades(filepath)
    
    # Apply date range filter if specified
    if date_range is not None:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['buy_time'] >= start_date) & (df['buy_time'] <= end_date)].copy()
    
    # Validate sentiment_state column exists
    if 'sentiment_state' not in df.columns:
        raise ValueError(f"File {filepath} missing 'sentiment_state' column")
    
    # Sort by time
    df = df.sort_values('buy_time').reset_index(drop=True)
    
    # Analyze by sentiment
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
        'sentiment_stats': sentiment_stats
    }


def format_significance(p_value: float) -> str:
    """Formats significance with green tick or red X."""
    if p_value < 0.1:
        return f"✅ (p={p_value:.3f})"
    else:
        return f"❌ (p={p_value:.2f})"


def print_single_strategy_sentiment(r: dict):
    """Prints sentiment analysis table for a single strategy."""
    print(f"\n{'='*200}")
    print(f"STRATEGY: {r['strategy']} (Total: {r['total_trades']} trades, Profit: ${r['total_profit']:.2f}, DD: {r['total_dd_pct']:.2f}%, WR: {r['total_win_rate']:.1f}%)")
    print(f"{'='*200}")
    
    # TABLE: BY SENTIMENT STATE
    print(f"\n{'─'*120}")
    print(f"BY SENTIMENT STATE")
    print(f"{'─'*120}")
    print(f"{'SENTIMENT':<20} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    sentiment_stats = r['sentiment_stats']
    sorted_sentiment = sorted(sentiment_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (state, stats) in enumerate(sorted_sentiment):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        
        # Calculate p-value
        if len(sorted_sentiment) < 2:
            p_str = "N/A"
        elif idx == 0:
            # Best vs 2nd best
            p_value = permutation_test(sorted_sentiment[0][1]['profits_list'], sorted_sentiment[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            # Others vs best
            p_value = permutation_test(stats['profits_list'], sorted_sentiment[0][1]['profits_list'])
            p_str = format_significance(p_value)
        
        print(f"{state:<20} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)
    print(f"{'TOTAL':<20} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    # Best sentiment comparison
    if len(sorted_sentiment) >= 2:
        best_sent, best_stats = sorted_sentiment[0]
        second_sent, second_stats = sorted_sentiment[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_sent} (${best_stats['profit']:.2f}) vs 2ND: {second_sent} (${second_stats['profit']:.2f}) | {sig_str}")


def print_summary_table(results: list):
    """Prints final summary table across all strategies."""
    print(f"\n{'='*200}")
    print(f"{'='*200}")
    print(f"SUMMARY - ALL STRATEGIES")
    print(f"{'='*200}")
    print(f"{'='*200}")
    
    # Summary: Best Sentiment State
    print(f"\n{'─'*145}")
    print("BEST SENTIMENT STATE PER STRATEGY")
    print(f"{'─'*145}")
    print(f"{'STRATEGY':<30} {'BEST_SENTIMENT':<20} {'CONF':>5} {'TRADES':>8} {'PROFIT':>10} {'2ND_BEST':<20} {'TRADES':>8} {'PROFIT':>10} {'SIGNIFICANT?':>15}")
    print("-" * 145)
    
    for r in results:
        sentiment_stats = r['sentiment_stats']
        if sentiment_stats and len(sentiment_stats) >= 2:
            sorted_sent = sorted(sentiment_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
            best_sent, best_stats = sorted_sent[0]
            second_sent, second_stats = sorted_sent[1]
            p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
            
            sig_str = format_significance(p_value)
            
            print(f"{r['strategy']:<30} {best_sent:<20} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {second_sent:<20} {second_stats['num_trades']:>8} {second_stats['profit']:>10.2f} {sig_str:>15}")
        elif sentiment_stats and len(sentiment_stats) == 1:
            best_sent, best_stats = list(sentiment_stats.items())[0]
            print(f"{r['strategy']:<30} {best_sent:<20} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {'(only one)':<20} {0:>8} {0.0:>10.2f} {'N/A':>15}")
    
    print("-" * 145)


def analyze_all_strategies(output_folder: str = None, initial_capital: float = None,
                           date_range: tuple = None) -> list:
    """Analyzes all strategies by sentiment state."""
    output_folder = output_folder or OUTPUT_FOLDER
    initial_capital = initial_capital or INITIAL_CAPITAL
    
    print("=" * 70)
    print("SENTIMENT ANALYZER - Performance by sentiment state")
    print("=" * 70)
    
    if date_range:
        print(f"\n⚠️  DATE RANGE FILTER ACTIVE: {date_range[0]} → {date_range[1]}")
    
    print(f"\nInitial capital: ${initial_capital}")
    
    print("\nSentiment states analyzed:")
    print("  - extreme_fear (0.00 - 0.25)")
    print("  - fear (0.25 - 0.45)")
    print("  - neutral (0.45 - 0.55)")
    print("  - greed (0.55 - 0.75)")
    print("  - extreme_greed (0.75 - 1.00)")
    
    print("\nConfidence indicator (CONF):")
    print(f"  ✓ = ≥{MIN_TRADES_CONFIDENCE} trades (reliable)")
    print(f"  ✗ = <{MIN_TRADES_CONFIDENCE} trades (unreliable)")
    
    print("\nSignificance indicator (SIGNIFICANT?):")
    print(f"  ✅ = p<0.10 (statistically significant difference)")
    print(f"  ❌ = p≥0.10 (no significant difference)")
    
    pattern = os.path.join(output_folder, "trades_sentiment_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {output_folder}")
        return []
    
    print(f"\nFiles found: {len(files)}\n")
    
    results = []
    for f in files:
        r = analyze_strategy_sentiment(f, initial_capital, date_range=date_range)
        results.append(r)
    
    # Print each strategy
    for r in results:
        print_single_strategy_sentiment(r)
    
    # Print final summary table
    print_summary_table(results)
    
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
    print("    - If best sentiment state is ✓✅: filter by that sentiment")
    print("    - Otherwise: don't filter, operate in all sentiment conditions")
    print(f"{'='*200}")
    
    return results


if __name__ == "__main__":
    analyze_all_strategies(date_range=DATE_RANGE_FILTER)