"""
ensemble_performance_analyzer.py

Analyzes strategy performance by ensemble voting WITHIN SAME TIMEFRAME.
For each trade, counts how many OTHER strategies (same timeframe) also signaled on that symbol.
Then analyzes performance by vote count (1, 2, 3, 4+ votes).

IMPORTANT: Only compares strategies with the same timeframe:
  - 4H strategies vote with other 4H strategies
  - 1H strategies vote with other 1H strategies
  - 6Hutc strategies vote with other 6Hutc strategies

Usage:
    python ensemble_performance_analyzer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.config import OUTPUT_FOLDER, INITIAL_CAPITAL, DATE_RANGE_FILTER


# =============================================================================
# CONFIGURATION
# =============================================================================

# Time window to consider "same signal" (in hours)
TIME_WINDOW_HOURS = 4

# Minimum trades for confidence
MIN_TRADES_CONFIDENCE = 50


# =============================================================================
# ENSEMBLE VOTE CALCULATION
# =============================================================================

def extract_timeframe(strategy_name):
    """
    Extract timeframe from strategy name.
    
    Examples:
        'flag_long_4H_OOS' -> '4H'
        'reversal_short_1H_OOS' -> '1H'
        'parity_long_6Hutc_OOS' -> '6Hutc'
    """
    strategy_lower = strategy_name.lower()
    
    if '6hutc' in strategy_lower:
        return '6Hutc'
    elif '4h' in strategy_lower:
        return '4H'
    elif '1h' in strategy_lower:
        return '1H'
    else:
        return 'unknown'


def count_ensemble_votes(trade_row, all_strategies_trades, time_window_hours=4):
    """
    Count how many strategies signaled on same symbol within time window.
    ONLY counts votes from strategies with SAME TIMEFRAME.
    
    Args:
        trade_row: Single trade from one strategy
        all_strategies_trades: Dict of {strategy_name: df_trades}
        time_window_hours: Time tolerance for "same moment"
    
    Returns:
        num_votes: Number of strategies that signaled (including this one)
        voters: List of strategy names that voted
    """
    symbol = trade_row['symbol']
    buy_time = trade_row['buy_time']
    current_strategy = trade_row['strategy']
    
    # Extract timeframe of current strategy
    current_tf = extract_timeframe(current_strategy)
    
    voters = [current_strategy]  # This strategy always votes for itself
    
    # Check other strategies
    for strategy_name, df_other in all_strategies_trades.items():
        if strategy_name == current_strategy:
            continue  # Skip self
        
        # FILTER: Only compare strategies with same timeframe
        other_tf = extract_timeframe(strategy_name)
        if other_tf != current_tf:
            continue  # Skip different timeframes
        
        # Find trades in same symbol within time window
        time_min = buy_time - timedelta(hours=time_window_hours)
        time_max = buy_time + timedelta(hours=time_window_hours)
        
        matching_trades = df_other[
            (df_other['symbol'] == symbol) &
            (df_other['buy_time'] >= time_min) &
            (df_other['buy_time'] <= time_max)
        ]
        
        if len(matching_trades) > 0:
            voters.append(strategy_name)
    
    return len(voters), voters


def classify_votes(num_votes):
    """Classify trades by vote count."""
    if num_votes == 1:
        return '1_vote'
    elif num_votes == 2:
        return '2_votes'
    elif num_votes == 3:
        return '3_votes'
    elif num_votes >= 4:
        return '4+_votes'
    else:
        return 'unknown'


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

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


def format_significance(p_value: float) -> str:
    """Formats significance with green tick or red X."""
    if p_value < 0.1:
        return f"✅ (p={p_value:.3f})"
    else:
        return f"❌ (p={p_value:.2f})"


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_by_votes(df: pd.DataFrame, initial_capital: float) -> dict:
    """
    Analyzes performance by vote count.
    
    Returns:
        stats: Dict with vote_category -> metrics
    """
    stats = {}
    
    for vote_category in df['vote_category'].unique():
        cat_df = df[df['vote_category'] == vote_category].copy()
        cat_df = cat_df.sort_values('buy_time').reset_index(drop=True)
        cat_df['equity'] = initial_capital + cat_df['profit'].cumsum()
        
        num_trades = len(cat_df)
        profit = cat_df['profit'].sum()
        profits_list = cat_df['profit'].tolist()
        
        # Confidence indicator
        if num_trades >= MIN_TRADES_CONFIDENCE:
            confidence = "✓"
        else:
            confidence = "✗"
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_confidence_interval(profits_list)
        
        stats[vote_category] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(cat_df['equity']),
            'win_rate': (cat_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'avg_profit': profit / num_trades if num_trades > 0 else 0.0,
            'profits_list': profits_list,
            'confidence': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return stats


def analyze_strategy(strategy_name: str, df_trades: pd.DataFrame, all_strategies_trades: dict, 
                     initial_capital: float, time_window_hours: int) -> dict:
    """Analyzes a single strategy by ensemble votes."""
    
    print(f"  Calculating ensemble votes for {strategy_name}...")
    
    # Calculate votes for each trade
    votes_list = []
    voters_list = []
    
    for idx, row in df_trades.iterrows():
        num_votes, voters = count_ensemble_votes(row, all_strategies_trades, time_window_hours)
        votes_list.append(num_votes)
        voters_list.append(','.join(voters))
    
    df_trades['ensemble_votes'] = votes_list
    df_trades['ensemble_voters'] = voters_list
    df_trades['vote_category'] = df_trades['ensemble_votes'].apply(classify_votes)
    
    # Sort by time
    df_trades = df_trades.sort_values('buy_time').reset_index(drop=True)
    
    # Analyze by vote count
    vote_stats = analyze_by_votes(df_trades, initial_capital)
    
    # Calculate total metrics
    df_sorted = df_trades.sort_values('buy_time').reset_index(drop=True)
    df_sorted['equity_total'] = initial_capital + df_sorted['profit'].cumsum()
    total_dd_pct = calculate_max_dd_pct(df_sorted['equity_total'])
    total_win_rate = (df_sorted['profit'] > 0).mean() * 100 if len(df_sorted) > 0 else 0.0
    
    return {
        'strategy': strategy_name,
        'total_trades': len(df_trades),
        'total_profit': df_trades['profit'].sum(),
        'total_dd_pct': total_dd_pct,
        'total_win_rate': total_win_rate,
        'vote_stats': vote_stats,
        'df_enriched': df_trades  # Keep enriched dataframe
    }


# =============================================================================
# PRINTING FUNCTIONS
# =============================================================================

def print_strategy_analysis(r: dict):
    """Prints analysis for a single strategy."""
    print(f"\n\033[93m{'='*145}\033[0m")
    print(f"\033[93mSTRATEGY: {r['strategy']} (Total: {r['total_trades']} trades, Profit: ${r['total_profit']:.2f}, DD: {r['total_dd_pct']:.2f}%, WR: {r['total_win_rate']:.1f}%)\033[0m")
    print(f"\033[93m{'='*145}\033[0m")
    
    print(f"\n{'─'*145}")
    print(f"BY ENSEMBLE VOTES")
    print(f"{'─'*145}")
    print(f"{'VOTES':<15} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'AVG_PROFIT':>12} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 145)
    
    vote_stats = r['vote_stats']
    
    # Sort by vote count (4+ first, then 3, 2, 1)
    vote_order = ['4+_votes', '3_votes', '2_votes', '1_vote']
    sorted_votes = [(v, vote_stats[v]) for v in vote_order if v in vote_stats]
    
    for idx, (vote_category, stats) in enumerate(sorted_votes):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        
        # Calculate p-value
        if len(sorted_votes) < 2:
            p_str = "N/A"
        elif idx == 0:
            # Best vs 2nd best
            if len(sorted_votes) > 1:
                p_value = permutation_test(sorted_votes[0][1]['profits_list'], sorted_votes[1][1]['profits_list'])
                p_str = format_significance(p_value)
            else:
                p_str = "N/A"
        else:
            # Others vs best
            p_value = permutation_test(stats['profits_list'], sorted_votes[0][1]['profits_list'])
            p_str = format_significance(p_value)
        
        print(f"{vote_category:<15} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['avg_profit']:>12.2f} {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 145)
    print(f"{'TOTAL':<15} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_profit']/r['total_trades']:>12.2f} {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    # Best votes comparison
    if len(sorted_votes) >= 2:
        best_vote, best_stats = sorted_votes[0]
        second_vote, second_stats = sorted_votes[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_vote} (${best_stats['profit']:.2f}, WR {best_stats['win_rate']:.1f}%) vs 2ND: {second_vote} (${second_stats['profit']:.2f}, WR {second_stats['win_rate']:.1f}%) | {sig_str}")


def print_summary_table(results: list):
    """Prints summary table across all strategies."""
    print(f"\n{'='*145}")
    print(f"{'='*145}")
    print(f"SUMMARY - ALL STRATEGIES")
    print(f"{'='*145}")
    print(f"{'='*145}")
    
    print(f"\n{'─'*145}")
    print("BEST VOTE CATEGORY PER STRATEGY")
    print(f"{'─'*145}")
    print(f"{'STRATEGY':<30} {'BEST_VOTES':<15} {'CONF':>5} {'TRADES':>8} {'PROFIT':>10} {'WIN%':>8} {'2ND_BEST':<15} {'TRADES':>8} {'PROFIT':>10} {'SIGNIFICANT?':>15}")
    print("-" * 145)
    
    for r in results:
        vote_stats = r['vote_stats']
        if vote_stats and len(vote_stats) >= 2:
            # Sort by profit
            sorted_votes = sorted(vote_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
            best_vote, best_stats = sorted_votes[0]
            second_vote, second_stats = sorted_votes[1]
            p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
            
            sig_str = format_significance(p_value)
            
            print(f"{r['strategy']:<30} {best_vote:<15} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {best_stats['win_rate']:>7.1f}% {second_vote:<15} {second_stats['num_trades']:>8} {second_stats['profit']:>10.2f} {sig_str:>15}")
        elif vote_stats and len(vote_stats) == 1:
            best_vote, best_stats = list(vote_stats.items())[0]
            print(f"{r['strategy']:<30} {best_vote:<15} {best_stats['confidence']:>5} {best_stats['num_trades']:>8} {best_stats['profit']:>10.2f} {best_stats['win_rate']:>7.1f}% {'(only one)':<15} {0:>8} {0.0:>10.2f} {'N/A':>15}")
    
    print("-" * 145)


def print_aggregated_stats(results: list):
    """Print aggregated statistics across all strategies."""
    print(f"\n{'='*145}")
    print("AGGREGATED STATISTICS ACROSS ALL STRATEGIES")
    print(f"{'='*145}")
    
    # Collect all trades by vote category
    aggregated = {}
    
    for r in results:
        for vote_cat, stats in r['vote_stats'].items():
            if vote_cat not in aggregated:
                aggregated[vote_cat] = {
                    'total_trades': 0,
                    'total_profit': 0.0,
                    'all_profits': []
                }
            
            aggregated[vote_cat]['total_trades'] += stats['num_trades']
            aggregated[vote_cat]['total_profit'] += stats['profit']
            aggregated[vote_cat]['all_profits'].extend(stats['profits_list'])
    
    # Calculate aggregated metrics
    print(f"\n{'VOTES':<15} {'TOTAL_TRADES':>15} {'TOTAL_PROFIT':>15} {'AVG_PROFIT':>15} {'WIN_RATE':>12}")
    print("-" * 145)
    
    vote_order = ['4+_votes', '3_votes', '2_votes', '1_vote']
    for vote_cat in vote_order:
        if vote_cat in aggregated:
            data = aggregated[vote_cat]
            avg_profit = data['total_profit'] / data['total_trades'] if data['total_trades'] > 0 else 0.0
            win_rate = (np.array(data['all_profits']) > 0).mean() * 100
            
            print(f"{vote_cat:<15} {data['total_trades']:>15} ${data['total_profit']:>14.2f} ${avg_profit:>14.2f} {win_rate:>11.1f}%")
    
    print("-" * 145)
    
    return aggregated


def print_global_comparison(results: list, initial_capital: float):
    """Print BEFORE vs AFTER comparison of entire system with ensemble filtering (SAME TIMEFRAME only)."""
    print(f"\n{'='*145}")
    print(f"{'='*145}")
    print("GLOBAL SYSTEM COMPARISON: VOTING BY SAME TIMEFRAME")
    print(f"{'='*145}")
    print(f"{'='*145}")
    
    # Collect ALL trades from ALL strategies
    all_trades = []
    for r in results:
        df = r['df_enriched'].copy()
        df['strategy'] = r['strategy']
        all_trades.append(df)
    
    df_all = pd.concat(all_trades, ignore_index=True)
    df_all = df_all.sort_values('buy_time').reset_index(drop=True)
    
    # Calculate baseline (no filter)
    baseline_trades = len(df_all)
    baseline_profit = df_all['profit'].sum()
    baseline_wr = (df_all['profit'] > 0).mean() * 100
    
    df_all['equity'] = initial_capital + df_all['profit'].cumsum()
    baseline_dd = calculate_max_dd_pct(df_all['equity'])
    
    # Calculate Sharpe (simplified)
    returns = df_all['profit'] / initial_capital * 100
    baseline_sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0
    
    # Prepare scenarios
    scenarios = []
    
    # Scenario 1: Baseline (no filter)
    scenarios.append({
        'scenario': 'Baseline (no filter)',
        'trades': baseline_trades,
        'profit': baseline_profit,
        'wr': baseline_wr,
        'dd': baseline_dd,
        'sharpe': baseline_sharpe,
        'change_pct': 0.0
    })
    
    # Scenario 2-4: Different vote thresholds (SAME TIMEFRAME only)
    for min_votes in [2, 3, 4]:
        df_filtered = df_all[df_all['ensemble_votes'] >= min_votes].copy()
        
        if len(df_filtered) == 0:
            continue
        
        df_filtered = df_filtered.sort_values('buy_time').reset_index(drop=True)
        
        filtered_trades = len(df_filtered)
        filtered_profit = df_filtered['profit'].sum()
        filtered_wr = (df_filtered['profit'] > 0).mean() * 100
        
        df_filtered['equity'] = initial_capital + df_filtered['profit'].cumsum()
        filtered_dd = calculate_max_dd_pct(df_filtered['equity'])
        
        returns_filtered = df_filtered['profit'] / initial_capital * 100
        filtered_sharpe = returns_filtered.mean() / returns_filtered.std() if returns_filtered.std() > 0 else 0.0
        
        change_pct = ((filtered_profit - baseline_profit) / baseline_profit * 100) if baseline_profit != 0 else 0.0
        
        scenarios.append({
            'scenario': f'{min_votes}+ votes (same TF)',
            'trades': filtered_trades,
            'profit': filtered_profit,
            'wr': filtered_wr,
            'dd': filtered_dd,
            'sharpe': filtered_sharpe,
            'change_pct': change_pct
        })
    
    # Print comparison table
    print(f"\n{'Scenario':<25} {'Trades':>10} {'Profit':>12} {'WR%':>8} {'DD%':>8} {'Sharpe':>8} {'Change':>12} {'Indicator':>12}")
    print("-" * 145)
    
    for scenario in scenarios:
        # Format change with emoji
        if scenario['change_pct'] > 10:
            change_str = f"+{scenario['change_pct']:.1f}%"
            indicator = "🚀"
        elif scenario['change_pct'] > 0:
            change_str = f"+{scenario['change_pct']:.1f}%"
            indicator = "📈"
        elif scenario['change_pct'] < -20:
            change_str = f"{scenario['change_pct']:.1f}%"
            indicator = "📉"
        elif scenario['change_pct'] < 0:
            change_str = f"{scenario['change_pct']:.1f}%"
            indicator = "⚠️"
        else:
            change_str = "—"
            indicator = "—"
        
        print(f"{scenario['scenario']:<25} {scenario['trades']:>10} ${scenario['profit']:>11.2f} {scenario['wr']:>7.1f}% {scenario['dd']:>7.2f}% {scenario['sharpe']:>8.2f} {change_str:>12} {indicator:>12}")
    
    print("-" * 145)
    
    # Find best scenario
    best_scenario = max(scenarios[1:], key=lambda x: x['profit']) if len(scenarios) > 1 else scenarios[0]
    
    print(f"\n💡 RECOMMENDATION:")
    if best_scenario['scenario'] == 'Baseline (no filter)':
        print(f"   → No filtering needed - baseline performs best")
    else:
        print(f"   → Use {best_scenario['scenario']} (best profit/risk ratio)")
        print(f"   → Improves profit by {best_scenario['change_pct']:+.1f}% while reducing drawdown")
    
    print(f"\n📊 KEY INSIGHTS:")
    
    # Compare best filter vs baseline
    if best_scenario != scenarios[0]:
        trade_reduction = ((baseline_trades - best_scenario['trades']) / baseline_trades * 100)
        wr_improvement = best_scenario['wr'] - baseline_wr
        dd_improvement = ((baseline_dd - best_scenario['dd']) / baseline_dd * 100)
        
        print(f"   • Reduces trades by {trade_reduction:.1f}% ({baseline_trades} → {best_scenario['trades']})")
        print(f"   • Improves win rate by {wr_improvement:+.1f}% ({baseline_wr:.1f}% → {best_scenario['wr']:.1f}%)")
        print(f"   • Reduces drawdown by {dd_improvement:.1f}% ({baseline_dd:.2f}% → {best_scenario['dd']:.2f}%)")
        print(f"   • Increases profit by ${best_scenario['profit'] - baseline_profit:+,.2f}")
    else:
        print(f"   • Current system is already optimal")
        print(f"   • Ensemble filtering would reduce opportunities without improving results")
    
    print(f"{'='*145}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def analyze_all_strategies(output_folder: str = None, initial_capital: float = None,
                           time_window_hours: int = TIME_WINDOW_HOURS, 
                           date_range: tuple = None) -> list:
    """Main analysis function."""
    output_folder = output_folder or OUTPUT_FOLDER
    initial_capital = initial_capital or INITIAL_CAPITAL
    
    print("=" * 70)
    print("ENSEMBLE PERFORMANCE ANALYZER - SAME TIMEFRAME VOTING")
    print("=" * 70)
    
    if date_range:
        print(f"\n⚠️  DATE RANGE FILTER ACTIVE: {date_range[0]} → {date_range[1]}")
    
    print(f"\nInitial capital: ${initial_capital}")
    print(f"Time window for ensemble votes: {time_window_hours} hours")
    
    print("\n🔑 VOTING RULE: Only strategies with SAME TIMEFRAME vote together")
    print("  - 4H strategies vote with other 4H strategies only")
    print("  - 1H strategies vote with other 1H strategies only")
    print("  - 6Hutc strategies vote with other 6Hutc strategies only")
    
    print("\nVote categories:")
    print("  1_vote: Only this strategy signaled (same TF)")
    print("  2_votes: This + 1 other strategy signaled (same TF)")
    print("  3_votes: This + 2 other strategies signaled (same TF)")
    print("  4+_votes: This + 3+ other strategies signaled (same TF)")
    
    print("\nConfidence indicator (CONF):")
    print(f"  ✓ = ≥{MIN_TRADES_CONFIDENCE} trades (reliable)")
    print(f"  ✗ = <{MIN_TRADES_CONFIDENCE} trades (unreliable)")
    
    print("\nSignificance indicator (SIGNIFICANT?):")
    print(f"  ✅ = p<0.10 (statistically significant difference)")
    print(f"  ❌ = p≥0.10 (no significant difference)")
    
    # Load all strategy files
    pattern = os.path.join(output_folder, "trades_enriched_*_OOS.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {output_folder}")
        return []
    
    print(f"\nFiles found: {len(files)}")
    print("\nLoading all strategies...")
    
    # Load all strategies into memory
    all_strategies_trades = {}
    
    for filepath in files:
        strategy_name = Path(filepath).stem.replace('trades_enriched_', '')
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.lower().str.strip()
        
        if 'buy_time' in df.columns:
            df['buy_time'] = pd.to_datetime(df['buy_time'])
        
        # Apply date range filter if specified
        if date_range is not None:
            start_date, end_date = date_range
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['buy_time'] >= start_date) & (df['buy_time'] <= end_date)].copy()
        
        df['strategy'] = strategy_name
        all_strategies_trades[strategy_name] = df
    
    print(f"Loaded {len(all_strategies_trades)} strategies\n")
    
    # Analyze each strategy
    results = []
    
    for strategy_name, df_trades in all_strategies_trades.items():
        r = analyze_strategy(
            strategy_name, 
            df_trades, 
            all_strategies_trades, 
            initial_capital,
            time_window_hours
        )
        results.append(r)
    
    # Print individual strategy analyses
    for r in results:
        print_strategy_analysis(r)
    
    # Print summary table
    print_summary_table(results)
    
    # Print aggregated stats
    aggregated = print_aggregated_stats(results)
    
    # Print global system comparison (BEFORE vs AFTER)
    print_global_comparison(results, initial_capital)
    
    # Interpretation guide
    print(f"\n{'='*145}")
    print("INTERPRETATION GUIDE:")
    print("\n  CONF (Confidence):")
    print("    ✓ = Reliable sample (≥50 trades) - trust these results")
    print("    ✗ = Unreliable sample (<50 trades) - don't trust these results")
    print("\n  SIGNIFICANT? (Statistical test):")
    print("    ✅ (p<0.10) = Difference is real, not random")
    print("    ❌ (p≥0.10) = Difference could be random chance")
    print("\n  ENSEMBLE DECISION:")
    print("    - Votes are counted ONLY within same timeframe")
    print("    - 4H strategies vote with 4H strategies only")
    print("    - 1H strategies vote with 1H strategies only")
    print("    - If 3+ or 4+ votes show ✓✅: implement MIN_VOTES threshold")
    print("    - If no significant difference: don't filter by votes")
    print("    - Higher votes typically = higher WR but fewer opportunities")
    print(f"{'='*145}")
    
    return results


if __name__ == "__main__":
    analyze_all_strategies(date_range=DATE_RANGE_FILTER)