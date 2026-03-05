"""
market_regime/regime_analyzer.py

Compares system performance in two scenarios:
1. WITHOUT TREND FILTER: All strategies operate on all trades
2. WITH TREND FILTER   : LONG strategies only in uptrend, SHORT strategies only in downtrend

Shows strategy-by-strategy and global portfolio comparison.

"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.config import OUTPUT_FOLDER, INITIAL_CAPITAL
from market_regime.config import DIRECTION_METHOD, DIRECTION_MA_PERIOD

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_strategy_type(strategy_name: str) -> str:
    """
    Detects if strategy is LONG or SHORT based on name.
    
    Returns: 'LONG' or 'SHORT'
    """
    name_lower = strategy_name.lower()
    
    # Explicit markers
    if '_long_' in name_lower or name_lower.endswith('_long'):
        return 'LONG'
    elif '_short_' in name_lower or name_lower.endswith('_short'):
        return 'SHORT'
    
    # Special cases
    if 'double_top' in name_lower:
        return 'LONG'  # double_top is a long strategy
    
    # Default fallback (shouldn't happen with proper naming)
    print(f"⚠️  WARNING: Cannot detect type for '{strategy_name}', assuming LONG")
    return 'LONG'


def calculate_strategy_metrics(df: pd.DataFrame, initial_capital: float) -> dict:
    """
    Calculates key metrics for a strategy.
    
    Returns dict with: num_trades, total_profit, net_gain_pct, max_dd_pct
    """
    if len(df) == 0:
        return {
            'num_trades': 0,
            'total_profit': 0.0,
            'net_gain_pct': 0.0,
            'max_dd_pct': 0.0
        }
    
    df = df.sort_values('buy_time').copy()
    df['cumulative_profit'] = df['profit'].cumsum()
    df['balance'] = initial_capital + df['cumulative_profit']
    
    # Net gain
    final_balance = df['balance'].iloc[-1]
    net_gain_pct = (final_balance - initial_capital) / initial_capital * 100
    
    # Max DD
    cummax = df['balance'].cummax()
    drawdown_pct = ((df['balance'] - cummax) / cummax * 100)
    max_dd_pct = drawdown_pct.min()
    
    return {
        'num_trades': len(df),
        'total_profit': df['profit'].sum(),
        'net_gain_pct': net_gain_pct,
        'max_dd_pct': max_dd_pct
    }


def load_enriched_trades(filepath: str) -> pd.DataFrame:
    """Loads enriched trades from Excel file."""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    return df


def classify_trend(row: pd.Series, direction_method: str = 'price_vs_ma') -> str:
    """Classifies trade trend based on direction method."""
    if direction_method == 'price_vs_ma':
        price_vs_ma_col = f'price_vs_ma_{DIRECTION_MA_PERIOD}'
        if price_vs_ma_col in row and not pd.isna(row[price_vs_ma_col]):
            return 'uptrend' if row[price_vs_ma_col] > 1.0 else 'downtrend'
    
    elif direction_method == 'ma_cross':
        # Import MA periods from config
        from market_regime.config import DIRECTION_MA_FAST, DIRECTION_MA_SLOW
        ma_cross_col = f'ma_{DIRECTION_MA_FAST}_vs_ma_{DIRECTION_MA_SLOW}'
        
        if ma_cross_col in row and not pd.isna(row[ma_cross_col]):
            return 'uptrend' if row[ma_cross_col] > 1.0 else 'downtrend'
    
    return 'unknown'


# =============================================================================
# MAIN COMPARISON LOGIC
# =============================================================================

def analyze_strategy_both_scenarios(filepath: str, initial_capital: float) -> dict:
    """
    Analyzes a single strategy in both scenarios (with and without trend filter).
    
    Returns dict with metrics for both scenarios.
    """
    strategy = Path(filepath).stem.replace('trades_enriched_', '')
    df = load_enriched_trades(filepath)
    
    # Detect strategy type
    strategy_type = detect_strategy_type(strategy)
    
    # Classify trades by trend
    df['trend'] = df.apply(lambda r: classify_trend(r, DIRECTION_METHOD), axis=1)
    
    # SCENARIO A: WITHOUT FILTER (all trades)
    metrics_without = calculate_strategy_metrics(df, initial_capital)
    
    # SCENARIO B: WITH FILTER (only matching trend)
    if strategy_type == 'LONG':
        df_filtered = df[df['trend'] == 'uptrend'].copy()
    else:  # SHORT
        df_filtered = df[df['trend'] == 'downtrend'].copy()
    
    metrics_with = calculate_strategy_metrics(df_filtered, initial_capital)
    
    return {
        'strategy': strategy,
        'type': strategy_type,
        'without_filter': metrics_without,
        'with_filter': metrics_with
    }


def calculate_global_portfolio(results: list, initial_capital: float, use_filter: bool = False) -> dict:
    """
    Calculates global portfolio metrics.
    
    Args:
        results: List of strategy results
        initial_capital: Capital per strategy
        use_filter: If True, uses filtered trades; if False, uses all trades
    """
    all_trades = []
    
    for r in results:
        filepath = r['filepath']
        strategy_type = r['type']
        
        df = load_enriched_trades(filepath)
        df['trend'] = df.apply(lambda row: classify_trend(row, DIRECTION_METHOD), axis=1)
        
        if use_filter:
            # Apply trend filter
            if strategy_type == 'LONG':
                df = df[df['trend'] == 'uptrend'].copy()
            else:  # SHORT
                df = df[df['trend'] == 'downtrend'].copy()
        
        all_trades.append(df[['buy_time', 'profit']].copy())
    
    # Combine all trades
    if not all_trades:
        return {
            'num_trades': 0,
            'total_profit': 0.0,
            'net_gain_pct': 0.0,
            'max_dd_pct': 0.0
        }
    
    combined_trades = pd.concat(all_trades, ignore_index=True)
    combined_trades = combined_trades.sort_values('buy_time').reset_index(drop=True)
    
    # Check if we have any trades after filtering
    if len(combined_trades) == 0:
        return {
            'num_trades': 0,
            'total_profit': 0.0,
            'net_gain_pct': 0.0,
            'max_dd_pct': 0.0
        }
    
    total_capital = initial_capital * len(results)
    
    # Calculate equity curve
    combined_trades['cumulative_profit'] = combined_trades['profit'].cumsum()
    combined_trades['balance'] = total_capital + combined_trades['cumulative_profit']
    
    # Net gain
    final_balance = combined_trades['balance'].iloc[-1]
    net_gain_pct = (final_balance - total_capital) / total_capital * 100
    
    # Max DD
    cummax = combined_trades['balance'].cummax()
    drawdown_pct = ((combined_trades['balance'] - cummax) / cummax * 100)
    max_dd_pct = drawdown_pct.min()
    
    return {
        'num_trades': len(combined_trades),
        'total_profit': combined_trades['profit'].sum(),
        'net_gain_pct': net_gain_pct,
        'max_dd_pct': max_dd_pct
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("TREND FILTERING COMPARISON")
    print("=" * 100)
    print(f"\nDirection detection: {DIRECTION_METHOD}")
    if DIRECTION_METHOD == 'price_vs_ma':
        print(f"Using: Price vs MA{DIRECTION_MA_PERIOD}")
    print(f"Initial capital per strategy: ${INITIAL_CAPITAL}")
    
    # Find all enriched trade files
    pattern = os.path.join(OUTPUT_FOLDER, "trades_enriched_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {OUTPUT_FOLDER}")
        return
    
    print(f"\nAnalyzing {len(files)} strategies...\n")
    
    # Analyze each strategy
    results = []
    for filepath in files:
        result = analyze_strategy_both_scenarios(filepath, INITIAL_CAPITAL)
        result['filepath'] = filepath
        results.append(result)
    
    # Calculate global portfolios
    global_without = calculate_global_portfolio(results, INITIAL_CAPITAL, use_filter=False)
    global_with = calculate_global_portfolio(results, INITIAL_CAPITAL, use_filter=True)
    
    # ==========================================================================
    # PRINT COMPARISON TABLE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY-BY-STRATEGY COMPARISON")
    print("=" * 80)
    
    # Create ultra-simplified comparison table
    comparison_rows = []
    
    for r in results:
        w = r['without_filter']
        f = r['with_filter']
        
        # Calculate % change in profit
        if w['total_profit'] != 0:
            profit_change_pct = ((f['total_profit'] - w['total_profit']) / abs(w['total_profit'])) * 100
        else:
            profit_change_pct = 0.0
        
        # Calculate % change in DD (improvement = less negative)
        if w['max_dd_pct'] != 0:
            dd_change_pct = ((f['max_dd_pct'] - w['max_dd_pct']) / abs(w['max_dd_pct'])) * 100
        else:
            dd_change_pct = 0.0
        
        # Indicators
        profit_indicator = "✅" if profit_change_pct > 5 else ("❌" if profit_change_pct < -5 else "=")
        dd_indicator = "✅" if dd_change_pct > 5 else ("❌" if dd_change_pct < -5 else "=")  # DD improvement = less negative = positive %
        
        comparison_rows.append({
            'Strategy': r['strategy'],
            'Type': r['type'],
            'ΔProfit%': profit_change_pct,
            'Profit': profit_indicator,
            'ΔDD%': dd_change_pct,
            'DD': dd_indicator
        })
    
    df_comp = pd.DataFrame(comparison_rows)
    
    # Format numeric columns
    df_comp['ΔProfit%'] = df_comp['ΔProfit%'].apply(lambda x: f"{x:+.1f}%")
    df_comp['ΔDD%'] = df_comp['ΔDD%'].apply(lambda x: f"{x:+.1f}%")
    
    # Print table
    print(df_comp.to_string(index=False))
    
    # Add global portfolio row
    if global_without['total_profit'] != 0:
        global_profit_change = ((global_with['total_profit'] - global_without['total_profit']) / abs(global_without['total_profit'])) * 100
    else:
        global_profit_change = 0.0
    
    if global_without['max_dd_pct'] != 0:
        global_dd_change = ((global_with['max_dd_pct'] - global_without['max_dd_pct']) / abs(global_without['max_dd_pct'])) * 100
    else:
        global_dd_change = 0.0
    
    global_profit_ind = "✅" if global_profit_change > 5 else ("❌" if global_profit_change < -5 else "=")
    global_dd_ind = "✅" if global_dd_change > 5 else ("❌" if global_dd_change < -5 else "=")
    
    print("\n" + "-" * 80)
    
    # Show detailed global metrics for verification
    print(f"\n{'DETAILED GLOBAL METRICS':^80}")
    print("-" * 80)
    print(f"WITHOUT FILTER:")
    print(f"  Total Profit: ${global_without['total_profit']:,.2f}")
    print(f"  Max DD:       {global_without['max_dd_pct']:.2f}%")
    print(f"  Num Trades:   {global_without['num_trades']:,}")
    print(f"\nWITH TREND FILTER:")
    print(f"  Total Profit: ${global_with['total_profit']:,.2f}")
    print(f"  Max DD:       {global_with['max_dd_pct']:.2f}%")
    print(f"  Num Trades:   {global_with['num_trades']:,}")
    print(f"\nCHANGE:")
    print(f"  Profit:       ${global_with['total_profit'] - global_without['total_profit']:,.2f} ({global_profit_change:+.1f}%)")
    print(f"  DD:           {global_with['max_dd_pct'] - global_without['max_dd_pct']:.2f} pts ({global_dd_change:+.1f}%)")
    print(f"  Trades:       {global_with['num_trades'] - global_without['num_trades']:,} ({(global_with['num_trades']/global_without['num_trades']-1)*100:+.1f}%)")
    print("-" * 80)
    
    print(f"{'GLOBAL PORTFOLIO':<30} {'BOTH':<6} "
          f"{global_profit_change:>+7.1f}% {global_profit_ind:>6} "
          f"{global_dd_change:>+7.1f}% {global_dd_ind:>6}")
    
    print("=" * 80)
    
    # ==========================================================================
    # SUMMARY STATISTICS
    # ==========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    # Count improvements
    improvements = sum(1 for r in results if r['with_filter']['net_gain_pct'] > r['without_filter']['net_gain_pct'])
    degradations = sum(1 for r in results if r['with_filter']['net_gain_pct'] < r['without_filter']['net_gain_pct'])
    unchanged = len(results) - improvements - degradations
    
    print(f"\nStrategies improved with filter:    {improvements}/{len(results)} ({improvements/len(results)*100:.1f}%)")
    print(f"Strategies degraded with filter:    {degradations}/{len(results)} ({degradations/len(results)*100:.1f}%)")
    print(f"Strategies unchanged:                {unchanged}/{len(results)} ({unchanged/len(results)*100:.1f}%)")
    
    # Average improvements
    avg_gain_improvement = np.mean([r['with_filter']['net_gain_pct'] - r['without_filter']['net_gain_pct'] for r in results])
    avg_dd_improvement = np.mean([r['with_filter']['max_dd_pct'] - r['without_filter']['max_dd_pct'] for r in results])
    
    print(f"\nAverage Net Gain improvement:        {avg_gain_improvement:+.2f}%")
    print(f"Average Max DD change:                {avg_dd_improvement:+.2f}%")
    
    # Global impact
    delta_global_gain = global_with['net_gain_pct'] - global_without['net_gain_pct']
    delta_global_dd = global_with['max_dd_pct'] - global_without['max_dd_pct']
    
    print(f"\n{'GLOBAL PORTFOLIO IMPACT':^50}")
    print("-" * 50)
    print(f"Without filter:  {global_without['net_gain_pct']:>6.2f}% gain, {global_without['max_dd_pct']:>6.2f}% DD")
    print(f"With filter:     {global_with['net_gain_pct']:>6.2f}% gain, {global_with['max_dd_pct']:>6.2f}% DD")
    print(f"Delta:           {delta_global_gain:>+6.2f}% gain, {delta_global_dd:>+6.2f}% DD")
    
    # Trades reduction
    trades_reduction_pct = (1 - global_with['num_trades'] / global_without['num_trades']) * 100
    print(f"\nTrades reduction:    {global_without['num_trades']:,} → {global_with['num_trades']:,} ({trades_reduction_pct:.1f}% less)")
    
    print("\n" + "=" * 100)
    
    # ==========================================================================
    # RECOMMENDATION
    # ==========================================================================
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    
    if delta_global_gain > 2.0 and (delta_global_dd > -1.0):
        print("\n✅ RECOMMEND USING TREND FILTER")
        print(f"   • Net Gain improves by {delta_global_gain:.2f}%")
        print(f"   • Max DD similar or better")
        print(f"   • {improvements} out of {len(results)} strategies improve")
    elif delta_global_gain < -2.0:
        print("\n❌ DO NOT USE TREND FILTER")
        print(f"   • Net Gain decreases by {abs(delta_global_gain):.2f}%")
        print(f"   • System performs better without filtering")
    else:
        print("\n⚠️  MARGINAL IMPACT")
        print(f"   • Net Gain change: {delta_global_gain:+.2f}%")
        print(f"   • Consider other factors (complexity, robustness, etc.)")
    
    print("=" * 100)


if __name__ == "__main__":
    main()