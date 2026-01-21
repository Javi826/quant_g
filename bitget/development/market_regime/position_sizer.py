"""
market_regime/position_sizer.py

Applies position sizing based on regime family and direction.
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
    OUTPUT_FOLDER, FAMILIES, STRATEGY_CONFIGS, 
    DIRECTION_MA_REFERENCE, INITIAL_CAPITAL, DATE_RANGE_FILTER,
    GLOBAL_REGIME_MULTIPLIERS, GLOBAL_DIRECTION_MULTIPLIERS
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


def extract_direction(strategy_name: str) -> str:
    """Extracts direction (long/short) from strategy name."""
    name_lower = strategy_name.lower()
    if 'long' in name_lower:
        return 'long'
    elif 'short' in name_lower:
        return 'short'
    else:
        return 'unknown'


def is_uptrend(row: pd.Series, ma_reference: str = 'ma_50') -> bool:
    """Check if market is in uptrend (price > MA)."""
    price_vs_ma_col = f'price_vs_{ma_reference}'
    price_vs_ma = row.get(price_vs_ma_col)
    
    if pd.isna(price_vs_ma):
        return False
    
    return price_vs_ma > 1.0


def calculate_regime_multiplier(row: pd.Series, strategy_config: dict) -> float:
    """
    Calculates regime multiplier based on strategy config.
    
    Args:
        row: Trade row with family classification
        strategy_config: Strategy config from STRATEGY_CONFIGS
    
    Returns:
        Multiplier (0.0 or 1.0) based on family and strategy config
    """
    family = row.get('family', 'unknown')
    
    # Get regime multipliers from strategy config
    regime_mult = {
        'trending': strategy_config.get('regime_trending', 1.0),
        'volatile': strategy_config.get('regime_volatile', 1.0),
        'ranging': strategy_config.get('regime_ranging', 1.0)
    }
    
    return regime_mult.get(family, 1.0)


def calculate_direction_multiplier(row: pd.Series, strategy_config: dict, 
                                  ma_reference: str = 'ma_50') -> float:
    """
    Calculates direction multiplier based on strategy config and direction_mode.
    
    Args:
        row: Trade row with metrics
        strategy_config: Strategy config from STRATEGY_CONFIGS
        ma_reference: Which MA to use ('ma_20', 'ma_50', 'ma_200')
    
    Returns:
        Multiplier (0.0 or 1.0) based on direction_mode
    """
    direction_mode = strategy_config.get('direction_mode', 'general')
    
    # If general mode, no filtering
    if direction_mode == 'general':
        return 1.0
    
    # Check if uptrend
    uptrend = is_uptrend(row, ma_reference)
    
    # Apply direction filter
    if direction_mode == 'long_only':
        # Only trade in uptrend
        return 1.0 if uptrend else 0.0
    elif direction_mode == 'short_only':
        # Only trade in downtrend
        return 0.0 if uptrend else 1.0
    else:
        return 1.0


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


def process_single_file(filepath: str, families: dict, initial_capital: float, date_range: tuple = None) -> dict:
    """Processes a single enriched file and returns results."""
    strategy = Path(filepath).stem.replace('trades_enriched_', '').replace('_OOS', '')
    
    # Get strategy config
    strategy_config = STRATEGY_CONFIGS.get(strategy)
    
    if strategy_config is None:
        print(f"⚠️  Strategy '{strategy}' not found in STRATEGY_CONFIGS, skipping...")
        return None
    
    if not strategy_config.get('active', True):
        print(f"⚠️  Strategy '{strategy}' is inactive, skipping...")
        return None
    
    df = load_enriched_trades(filepath)
    
    # Apply date range filter if specified
    if date_range is not None:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['buy_time'] >= start_date) & (df['buy_time'] <= end_date)].copy()
    
    # Extract direction from strategy name
    trade_direction = extract_direction(strategy)
    
    # Classify trades by family
    df['family'] = df.apply(lambda row: classify_trade(row, families), axis=1)
    
    # Apply regime multiplier
    df['regime_mult'] = df.apply(
        lambda row: calculate_regime_multiplier(row, strategy_config), 
        axis=1
    )
    
    # Apply direction multiplier
    df['direction_mult'] = df.apply(
        lambda row: calculate_direction_multiplier(row, strategy_config, DIRECTION_MA_REFERENCE),
        axis=1
    )
    
    # APPLY GLOBAL MULTIPLIERS
    # Global regime multiplier (based on family)
    df['global_regime_mult'] = df['family'].map(GLOBAL_REGIME_MULTIPLIERS).fillna(1.0)
    
    # Global direction multiplier (based on strategy direction)
    global_direction_mult = GLOBAL_DIRECTION_MULTIPLIERS.get(trade_direction, 1.0)
    df['global_direction_mult'] = global_direction_mult
    
    # Combined multiplier = strategy_regime * strategy_direction * global_regime * global_direction
    df['sizing_mult'] = df['regime_mult'] * df['direction_mult'] * df['global_regime_mult'] * df['global_direction_mult']
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
    win_rate_base = (df['profit'] > 0).mean() * 100
    win_rate_sized = (df[df['sizing_mult'] > 0]['profit'] > 0).mean() * 100 if trades_sizing > 0 else 0
    win_delta_pct = win_rate_sized - win_rate_base
    
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
        'direction': trade_direction,
        'generator': extract_generator(strategy),
        'num_trades': num_trades,
        'trades_sizing': trades_sizing,
        'profit_base': profit_base,
        'profit_sized': profit_sized,
        'delta_pct': delta_pct,
        'win_rate_base': win_rate_base,
        'win_rate_sized': win_rate_sized,
        'win_delta_pct': win_delta_pct,
        'dd_base_pct': dd_base_pct,
        'dd_sized_pct': dd_sized_pct,
        'dd_delta_pct': dd_delta_pct,
        'family_stats': family_stats,
        'df': df,
        'config': strategy_config
    }


def print_file_results(r: dict):
    """Prints results for a single file."""
    print(f"\n{'='*70}")
    print(f"STRATEGY: {r['strategy']}")
    print(f"{'='*70}")
    
    # Print config
    config = r.get('config', {})
    print(f"Config: T={config.get('regime_trending', 1.0)} R={config.get('regime_ranging', 1.0)} V={config.get('regime_volatile', 1.0)} | {config.get('direction_mode', 'general')}")
    
    # CHANGED: Use 🟢 when delta = 0
    profit_ok = "✅" if r['profit_sized'] > r['profit_base'] else ("🟢" if r['profit_sized'] == r['profit_base'] else "❌")
    dd_ok = "✅" if r['dd_sized_pct'] < r['dd_base_pct'] else ("🟢" if r['dd_sized_pct'] == r['dd_base_pct'] else "❌")
    
    print(f"Trades: {r['num_trades']} (base) → {r['trades_sizing']} (sizing)  |  Win%: {r['win_rate_base']:.1f}% → {r['win_rate_sized']:.1f}%")
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
    initial_capital: float = None,
    show_plots: bool = True
) -> list:
    """Applies position sizing to all enriched files."""
    output_folder = output_folder or OUTPUT_FOLDER
    families = families or FAMILIES
    initial_capital = initial_capital or INITIAL_CAPITAL
    
    print("=" * 70)
    print("POSITION SIZER - Regime-based position sizing")
    print("=" * 70)
    
    if DATE_RANGE_FILTER:
        print(f"\n⚠️  DATE RANGE FILTER ACTIVE: {DATE_RANGE_FILTER[0]} → {DATE_RANGE_FILTER[1]}")
    
    print("\nFamily classification:")
    for fam, rules in families.items():
        rules_str = ' & '.join([f"{m}{op}{v}" for m, (op, v) in rules.items()]) if rules else "(default)"
        print(f"  {fam:<12}: [{rules_str}]")
    
    print(f"\nDirection detection: price_vs_{DIRECTION_MA_REFERENCE}")
    print(f"  uptrend:   price > {DIRECTION_MA_REFERENCE}")
    print(f"  downtrend: price <= {DIRECTION_MA_REFERENCE}")
    
    print(f"\nGlobal regime multipliers:")
    for regime, mult in GLOBAL_REGIME_MULTIPLIERS.items():
        print(f"  {regime:<12}: x{mult:.2f}")
    
    print(f"\nGlobal direction multipliers:")
    for direction, mult in GLOBAL_DIRECTION_MULTIPLIERS.items():
        print(f"  {direction:<12}: x{mult:.2f}")
    
    print(f"\nActive strategies: {sum(1 for cfg in STRATEGY_CONFIGS.values() if cfg.get('active', True))}/{len(STRATEGY_CONFIGS)}")
    
    pattern = os.path.join(output_folder, "trades_enriched_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {output_folder}")
        return []
    
    print(f"\nFiles found: {len(files)}")
    
    results = []
    for f in files:
        r = process_single_file(f, families, initial_capital, date_range=DATE_RANGE_FILTER)
        if r is None:
            continue
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
        df_copy['direction'] = r['direction']
        all_dfs.append(df_copy)
    
    combined_df = pd.concat(all_dfs, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
    
    # =================================================================
    # DIAGNOSTIC: Direction/Trend Distribution
    # =================================================================
    print(f"\n{'='*70}")
    print("DIAGNOSTIC - Direction/Trend Distribution")
    print(f"{'='*70}")
    
    # Add trend column using the same logic as calculate_direction_multiplier
    price_vs_ma_col = f'price_vs_{DIRECTION_MA_REFERENCE}'
    combined_df['trend'] = combined_df.apply(
        lambda r: 'uptrend' if (not pd.isna(r.get(price_vs_ma_col)) and r[price_vs_ma_col] > 1.0) else 
                  'downtrend' if (not pd.isna(r.get(price_vs_ma_col))) else 
                  'unknown',
        axis=1
    )
    
    print(f"\nTrend detection: price_vs_{DIRECTION_MA_REFERENCE} > 1.0")
    
    print("\n1. Trades by Direction and Trend:")
    trend_dist = combined_df.groupby(['direction', 'trend']).size().reset_index(name='count')
    for _, row in trend_dist.iterrows():
        pct = (row['count'] / len(combined_df) * 100)
        print(f"   {row['direction']:<8} in {row['trend']:<10}: {row['count']:>5} trades ({pct:>5.1f}%)")
    
    print("\n2. Trades by Direction and Active/Filtered:")
    active_dist = combined_df.groupby(['direction']).apply(
        lambda x: pd.Series({
            'total': len(x),
            'active': (x['sizing_mult'] > 0).sum(),
            'filtered': (x['sizing_mult'] == 0).sum()
        })
    ).reset_index()
    for _, row in active_dist.iterrows():
        active_pct = (row['active'] / row['total'] * 100) if row['total'] > 0 else 0
        filtered_pct = (row['filtered'] / row['total'] * 100) if row['total'] > 0 else 0
        print(f"   {row['direction']:<8}: {row['total']:>5} total | {row['active']:>5} active ({active_pct:>5.1f}%) | {row['filtered']:>5} filtered ({filtered_pct:>5.1f}%)")
    
    print("\n3. Filtering breakdown by Direction+Trend:")
    filter_detail = combined_df.groupby(['direction', 'trend']).apply(
        lambda x: pd.Series({
            'total': len(x),
            'active': (x['sizing_mult'] > 0).sum(),
            'filtered': (x['sizing_mult'] == 0).sum()
        })
    ).reset_index()
    for _, row in filter_detail.iterrows():
        active_pct = (row['active'] / row['total'] * 100) if row['total'] > 0 else 0
        filtered_pct = (row['filtered'] / row['total'] * 100) if row['total'] > 0 else 0
        print(f"   {row['direction']:<8} {row['trend']:<10}: {row['active']:>5}/{row['total']:>5} active ({active_pct:>5.1f}%) | {row['filtered']:>5} filtered ({filtered_pct:>5.1f}%)")
    
    print(f"\n{'='*70}\n")
    
    # Calculate PORTFOLIO-LEVEL equity and DD
    combined_df['equity_base_portfolio'] = initial_capital + combined_df['profit'].cumsum()
    combined_df['equity_sized_portfolio'] = initial_capital + combined_df['profit_sized'].cumsum()
    
    portfolio_dd_base_pct = calculate_max_dd_pct(combined_df['equity_base_portfolio'])
    portfolio_dd_sized_pct = calculate_max_dd_pct(combined_df['equity_sized_portfolio'])
    portfolio_dd_delta_pct = portfolio_dd_sized_pct - portfolio_dd_base_pct
    
    # Calculate PORTFOLIO-LEVEL win rates
    portfolio_win_rate_base = (combined_df['profit'] > 0).mean() * 100
    portfolio_win_rate_sized = (combined_df[combined_df['sizing_mult'] > 0]['profit'] > 0).mean() * 100 if (combined_df['sizing_mult'] > 0).sum() > 0 else 0
    portfolio_win_delta_pct = portfolio_win_rate_sized - portfolio_win_rate_base
    
    # =================================================================
    # RESUMEN POR FAMILIA
    # =================================================================
    print(f"\n{'='*180}")
    print("RESUMEN POR FAMILIA (todas las estrategias agregadas)")
    print(f"{'='*180}")
    
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
            'win_rate_base': (fam_df['profit'] > 0).mean() * 100,
            'win_rate_sized': (fam_df[fam_df['sizing_mult'] > 0]['profit'] > 0).mean() * 100 if (fam_df['sizing_mult'] > 0).sum() > 0 else 0,
            'dd_base_pct': calculate_max_dd_pct(fam_df['equity_base']),
            'dd_sized_pct': calculate_max_dd_pct(fam_df['equity_sized'])
        }
    
    # Calculate total trades for percentage calculation
    total_trades_base_all = sum(agg['trades_base'] for agg in family_aggregates.values())
    
    print(f"\n{'FAMILY':<15} {'TRADES_BASE':>12} {'TRADES_%':>10} {'TRADES_SIZING':>14} {'PROFIT_BASE':>13} {'PROFIT_SIZING':>15} {'Δ%':>8} {'WIN%_BASE':>10} {'WIN%_SIZING':>12} {'ΔWin%':>8} {'DD%_BASE':>10} {'DD%_SIZING':>12} {'ΔDD%':>8}")
    print("-" * 180)
    
    total_trades_base = 0
    total_trades_sizing = 0
    total_profit_base = 0
    total_profit_sizing = 0
    
    for fam in sorted(family_aggregates.keys()):
        agg = family_aggregates[fam]
        trades_pct = (agg['trades_base'] / total_trades_base_all * 100) if total_trades_base_all > 0 else 0
        delta_pct = ((agg['profit_s'] - agg['profit_b']) / abs(agg['profit_b']) * 100) if agg['profit_b'] != 0 else 0
        win_delta_pct = agg['win_rate_sized'] - agg['win_rate_base']
        dd_delta_pct = agg['dd_sized_pct'] - agg['dd_base_pct']
        # CHANGED: Use 🟢 when delta = 0
        profit_ok = "✅" if agg['profit_s'] > agg['profit_b'] else ("🟢" if agg['profit_s'] == agg['profit_b'] else "❌")
        win_ok = "✅" if agg['win_rate_sized'] > agg['win_rate_base'] else ("🟢" if agg['win_rate_sized'] == agg['win_rate_base'] else "❌")
        dd_ok = "✅" if agg['dd_sized_pct'] < agg['dd_base_pct'] else ("🟢" if agg['dd_sized_pct'] == agg['dd_base_pct'] else "❌")
        
        print(f"{fam:<15} {agg['trades_base']:>12} {trades_pct:>9.1f}% {agg['trades_sizing']:>14} {agg['profit_b']:>13.2f} {agg['profit_s']:>15.2f} {profit_ok} {delta_pct:>+6.1f}% {agg['win_rate_base']:>9.1f}% {agg['win_rate_sized']:>11.1f}% {win_ok} {win_delta_pct:>+6.1f}% {agg['dd_base_pct']:>9.1f}% {agg['dd_sized_pct']:>11.1f}% {dd_ok} {dd_delta_pct:>+6.1f}%")
        
        total_trades_base += agg['trades_base']
        total_trades_sizing += agg['trades_sizing']
        total_profit_base += agg['profit_b']
        total_profit_sizing += agg['profit_s']
    
    print("-" * 180)
    
    # TOTAL: Use portfolio-level DD and WIN%
    total_trades_pct = 100.0
    total_delta_pct = ((total_profit_sizing - total_profit_base) / abs(total_profit_base) * 100) if total_profit_base != 0 else 0
    # CHANGED: Use 🟢 when delta = 0
    total_profit_ok = "✅" if total_profit_sizing > total_profit_base else ("🟢" if total_profit_sizing == total_profit_base else "❌")
    total_win_ok = "✅" if portfolio_win_rate_sized > portfolio_win_rate_base else ("🟢" if portfolio_win_rate_sized == portfolio_win_rate_base else "❌")
    total_dd_ok = "✅" if portfolio_dd_sized_pct < portfolio_dd_base_pct else ("🟢" if portfolio_dd_sized_pct == portfolio_dd_base_pct else "❌")
    print(f"{'TOTAL':<15} {total_trades_base:>12} {total_trades_pct:>9.1f}% {total_trades_sizing:>14} {total_profit_base:>13.2f} {total_profit_sizing:>15.2f} {total_profit_ok} {total_delta_pct:>+6.1f}% {portfolio_win_rate_base:>9.1f}% {portfolio_win_rate_sized:>11.1f}% {total_win_ok} {portfolio_win_delta_pct:>+6.1f}% {portfolio_dd_base_pct:>9.1f}% {portfolio_dd_sized_pct:>11.1f}% {total_dd_ok} {portfolio_dd_delta_pct:>+6.1f}%")
    
    # =================================================================
    # RESUMEN POR DIRECTION
    # =================================================================
    print(f"\n{'='*195}")
    print("RESUMEN POR DIRECTION (todas las estrategias agregadas)")
    print(f"{'='*195}")
    
    direction_aggregates = {}
    for direction in combined_df['direction'].unique():
        dir_df = combined_df[combined_df['direction'] == direction].copy()
        dir_df = dir_df.sort_values('buy_time').reset_index(drop=True)
        
        # Calculate equity curves
        dir_df['equity_base'] = initial_capital + dir_df['profit'].cumsum()
        dir_df['equity_sized'] = initial_capital + dir_df['profit_sized'].cumsum()
        
        direction_aggregates[direction] = {
            'trades_base': len(dir_df),
            'trades_sizing': (dir_df['sizing_mult'] > 0).sum(),
            'profit_b': dir_df['profit'].sum(),
            'profit_s': dir_df['profit_sized'].sum(),
            'win_rate_base': (dir_df['profit'] > 0).mean() * 100,
            'win_rate_sized': (dir_df[dir_df['sizing_mult'] > 0]['profit'] > 0).mean() * 100 if (dir_df['sizing_mult'] > 0).sum() > 0 else 0,
            'dd_base_pct': calculate_max_dd_pct(dir_df['equity_base']),
            'dd_sized_pct': calculate_max_dd_pct(dir_df['equity_sized'])
        }
    
    # Calculate total trades for percentage calculation
    dir_total_trades_base_all = sum(agg['trades_base'] for agg in direction_aggregates.values())
    
    print(f"\n{'DIRECTION':<15} {'TRADES_BASE':>12} {'TRADES_%':>10} {'TRADES_SIZING':>14} {'TRADES_ACTIVE%':>15} {'PROFIT_BASE':>13} {'PROFIT_SIZING':>15} {'Δ%':>8} {'WIN%_BASE':>10} {'WIN%_SIZING':>12} {'ΔWin%':>8} {'DD%_BASE':>10} {'DD%_SIZING':>12} {'ΔDD%':>8}")
    print("-" * 195)
    
    dir_total_trades_base = 0
    dir_total_trades_sizing = 0
    dir_total_profit_base = 0
    dir_total_profit_sizing = 0
    
    # Sort directions (long, short, unknown)
    sorted_dirs = sorted(direction_aggregates.keys(), key=lambda x: {'long': 0, 'short': 1, 'unknown': 2}.get(x, 3))
    
    for direction in sorted_dirs:
        agg = direction_aggregates[direction]
        trades_pct = (agg['trades_base'] / dir_total_trades_base_all * 100) if dir_total_trades_base_all > 0 else 0
        trades_active_pct = (agg['trades_sizing'] / agg['trades_base'] * 100) if agg['trades_base'] > 0 else 0
        delta_pct = ((agg['profit_s'] - agg['profit_b']) / abs(agg['profit_b']) * 100) if agg['profit_b'] != 0 else 0
        win_delta_pct = agg['win_rate_sized'] - agg['win_rate_base']
        dd_delta_pct = agg['dd_sized_pct'] - agg['dd_base_pct']
        # CHANGED: Use 🟢 when delta = 0
        profit_ok = "✅" if agg['profit_s'] > agg['profit_b'] else ("🟢" if agg['profit_s'] == agg['profit_b'] else "❌")
        win_ok = "✅" if agg['win_rate_sized'] > agg['win_rate_base'] else ("🟢" if agg['win_rate_sized'] == agg['win_rate_base'] else "❌")
        dd_ok = "✅" if agg['dd_sized_pct'] < agg['dd_base_pct'] else ("🟢" if agg['dd_sized_pct'] == agg['dd_base_pct'] else "❌")
        
        print(f"{direction:<15} {agg['trades_base']:>12} {trades_pct:>9.1f}% {agg['trades_sizing']:>14} {trades_active_pct:>14.1f}% {agg['profit_b']:>13.2f} {agg['profit_s']:>15.2f} {profit_ok} {delta_pct:>+6.1f}% {agg['win_rate_base']:>9.1f}% {agg['win_rate_sized']:>11.1f}% {win_ok} {win_delta_pct:>+6.1f}% {agg['dd_base_pct']:>9.1f}% {agg['dd_sized_pct']:>11.1f}% {dd_ok} {dd_delta_pct:>+6.1f}%")
        
        dir_total_trades_base += agg['trades_base']
        dir_total_trades_sizing += agg['trades_sizing']
        dir_total_profit_base += agg['profit_b']
        dir_total_profit_sizing += agg['profit_s']
    
    print("-" * 195)
    
    # TOTAL: Use portfolio-level DD and WIN%
    dir_total_trades_pct = 100.0
    dir_total_trades_active_pct = (dir_total_trades_sizing / dir_total_trades_base * 100) if dir_total_trades_base > 0 else 0
    dir_total_delta_pct = ((dir_total_profit_sizing - dir_total_profit_base) / abs(dir_total_profit_base) * 100) if dir_total_profit_base != 0 else 0
    # CHANGED: Use 🟢 when delta = 0
    dir_total_profit_ok = "✅" if dir_total_profit_sizing > dir_total_profit_base else ("🟢" if dir_total_profit_sizing == dir_total_profit_base else "❌")
    dir_total_win_ok = "✅" if portfolio_win_rate_sized > portfolio_win_rate_base else ("🟢" if portfolio_win_rate_sized == portfolio_win_rate_base else "❌")
    dir_total_dd_ok = "✅" if portfolio_dd_sized_pct < portfolio_dd_base_pct else ("🟢" if portfolio_dd_sized_pct == portfolio_dd_base_pct else "❌")
    print(f"{'TOTAL':<15} {dir_total_trades_base:>12} {dir_total_trades_pct:>9.1f}% {dir_total_trades_sizing:>14} {dir_total_trades_active_pct:>14.1f}% {dir_total_profit_base:>13.2f} {dir_total_profit_sizing:>15.2f} {dir_total_profit_ok} {dir_total_delta_pct:>+6.1f}% {portfolio_win_rate_base:>9.1f}% {portfolio_win_rate_sized:>11.1f}% {dir_total_win_ok} {portfolio_win_delta_pct:>+6.1f}% {portfolio_dd_base_pct:>9.1f}% {portfolio_dd_sized_pct:>11.1f}% {dir_total_dd_ok} {portfolio_dd_delta_pct:>+6.1f}%")
    
    # =================================================================
    # RESUMEN POR GENERADOR
    # =================================================================
    print(f"\n{'='*195}")
    print("RESUMEN POR GENERADOR (todas las estrategias agregadas)")
    print(f"{'='*195}")
    
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
            'win_rate_base': (gen_combined['profit'] > 0).mean() * 100,
            'win_rate_sized': (gen_combined[gen_combined['sizing_mult'] > 0]['profit'] > 0).mean() * 100 if (gen_combined['sizing_mult'] > 0).sum() > 0 else 0,
            'dd_base_pct': calculate_max_dd_pct(gen_combined['equity_base']),
            'dd_sized_pct': calculate_max_dd_pct(gen_combined['equity_sized'])
        }
    
    # Calculate total trades for percentage calculation
    gen_total_trades_base_all = sum(agg['trades_base'] for agg in generator_aggregates.values())
    
    print(f"\n{'GENERATOR':<15} {'TRADES_BASE':>12} {'TRADES_%':>10} {'TRADES_SIZING':>14} {'TRADES_ACTIVE%':>15} {'PROFIT_BASE':>13} {'PROFIT_SIZING':>15} {'Δ%':>8} {'WIN%_BASE':>10} {'WIN%_SIZING':>12} {'ΔWin%':>8} {'DD%_BASE':>10} {'DD%_SIZING':>12} {'ΔDD%':>8}")
    print("-" * 195)
    
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
        win_delta_pct = agg['win_rate_sized'] - agg['win_rate_base']
        dd_delta_pct = agg['dd_sized_pct'] - agg['dd_base_pct']
        # CHANGED: Use 🟢 when delta = 0
        profit_ok = "✅" if agg['profit_s'] > agg['profit_b'] else ("🟢" if agg['profit_s'] == agg['profit_b'] else "❌")
        win_ok = "✅" if agg['win_rate_sized'] > agg['win_rate_base'] else ("🟢" if agg['win_rate_sized'] == agg['win_rate_base'] else "❌")
        dd_ok = "✅" if agg['dd_sized_pct'] < agg['dd_base_pct'] else ("🟢" if agg['dd_sized_pct'] == agg['dd_base_pct'] else "❌")
        
        print(f"{gen:<15} {agg['trades_base']:>12} {trades_pct:>9.1f}% {agg['trades_sizing']:>14} {trades_active_pct:>14.1f}% {agg['profit_b']:>13.2f} {agg['profit_s']:>15.2f} {profit_ok} {delta_pct:>+6.1f}% {agg['win_rate_base']:>9.1f}% {agg['win_rate_sized']:>11.1f}% {win_ok} {win_delta_pct:>+6.1f}% {agg['dd_base_pct']:>9.1f}% {agg['dd_sized_pct']:>11.1f}% {dd_ok} {dd_delta_pct:>+6.1f}%")
        
        gen_total_trades_base += agg['trades_base']
        gen_total_trades_sizing += agg['trades_sizing']
        gen_total_profit_base += agg['profit_b']
        gen_total_profit_sizing += agg['profit_s']
    
    print("-" * 195)
    
    # TOTAL: Use portfolio-level DD and WIN%
    gen_total_trades_pct = 100.0
    gen_total_trades_active_pct = (gen_total_trades_sizing / gen_total_trades_base * 100) if gen_total_trades_base > 0 else 0
    gen_total_delta_pct = ((gen_total_profit_sizing - gen_total_profit_base) / abs(gen_total_profit_base) * 100) if gen_total_profit_base != 0 else 0
    # CHANGED: Use 🟢 when delta = 0
    gen_total_profit_ok = "✅" if gen_total_profit_sizing > gen_total_profit_base else ("🟢" if gen_total_profit_sizing == gen_total_profit_base else "❌")
    gen_total_win_ok = "✅" if portfolio_win_rate_sized > portfolio_win_rate_base else ("🟢" if portfolio_win_rate_sized == portfolio_win_rate_base else "❌")
    gen_total_dd_ok = "✅" if portfolio_dd_sized_pct < portfolio_dd_base_pct else ("🟢" if portfolio_dd_sized_pct == portfolio_dd_base_pct else "❌")
    print(f"{'TOTAL':<15} {gen_total_trades_base:>12} {gen_total_trades_pct:>9.1f}% {gen_total_trades_sizing:>14} {gen_total_trades_active_pct:>14.1f}% {gen_total_profit_base:>13.2f} {gen_total_profit_sizing:>15.2f} {gen_total_profit_ok} {gen_total_delta_pct:>+6.1f}% {portfolio_win_rate_base:>9.1f}% {portfolio_win_rate_sized:>11.1f}% {gen_total_win_ok} {portfolio_win_delta_pct:>+6.1f}% {portfolio_dd_base_pct:>9.1f}% {portfolio_dd_sized_pct:>11.1f}% {gen_total_dd_ok} {portfolio_dd_delta_pct:>+6.1f}%")
    
    # =================================================================
    # SUMMARY - ALL STRATEGIES
    # =================================================================
    print(f"\n{'='*145}")
    print("SUMMARY - ALL STRATEGIES")
    print(f"{'='*145}")
    
    print(f"\n{'STRATEGY':<30} {'TRADES_BASE':>12} {'TRADES_SIZING':>14} {'PROFIT_BASE':>13} {'PROFIT_SIZING':>15} {'Δ%':>7} {'WIN%_BASE':>10} {'WIN%_SIZING':>12} {'ΔWin%':>8} {'DD%_BASE':>9} {'DD%_SIZING':>11} {'ΔDD%':>8}")
    print("-" * 145)
    
    for r in results:
        # CHANGED: Use 🟢 when delta = 0
        profit_ok = "✅" if r['profit_sized'] > r['profit_base'] else ("🟢" if r['profit_sized'] == r['profit_base'] else "❌")
        win_ok = "✅" if r['win_rate_sized'] > r['win_rate_base'] else ("🟢" if r['win_rate_sized'] == r['win_rate_base'] else "❌")
        dd_ok = "✅" if r['dd_sized_pct'] < r['dd_base_pct'] else ("🟢" if r['dd_sized_pct'] == r['dd_base_pct'] else "❌")
        print(f"{r['strategy']:<30} {r['num_trades']:>12} {r['trades_sizing']:>14} {r['profit_base']:>13.2f} {r['profit_sized']:>12.2f} {profit_ok} {r['delta_pct']:>+6.1f}% {r['win_rate_base']:>9.1f}% {r['win_rate_sized']:>11.1f}% {win_ok} {r['win_delta_pct']:>+6.1f}% {r['dd_base_pct']:>9.1f}% {r['dd_sized_pct']:>10.1f}% {r['dd_delta_pct']:>+7.1f}% {dd_ok}")
    
    print("-" * 145)
    
    n = len(results)
    if n > 0:
        total_trades_base = sum(r['num_trades'] for r in results)
        total_trades_sizing = sum(r['trades_sizing'] for r in results)
        total_profit_b = sum(r['profit_base'] for r in results)
        total_profit_s = sum(r['profit_sized'] for r in results)
        total_delta = ((total_profit_s - total_profit_b) / abs(total_profit_b) * 100) if total_profit_b != 0 else 0
        
        # TOTAL: Use portfolio-level DD and WIN%
        # CHANGED: Use 🟢 when delta = 0
        profit_ok = "✅" if total_profit_s > total_profit_b else ("🟢" if total_profit_s == total_profit_b else "❌")
        win_ok = "✅" if portfolio_win_rate_sized > portfolio_win_rate_base else ("🟢" if portfolio_win_rate_sized == portfolio_win_rate_base else "❌")
        dd_ok = "✅" if portfolio_dd_sized_pct < portfolio_dd_base_pct else ("🟢" if portfolio_dd_sized_pct == portfolio_dd_base_pct else "❌")
        print(f"{'TOTAL':<30} {total_trades_base:>12} {total_trades_sizing:>14} {total_profit_b:>13.2f} {total_profit_s:>12.2f} {profit_ok} {total_delta:>+6.1f}% {portfolio_win_rate_base:>9.1f}% {portfolio_win_rate_sized:>11.1f}% {win_ok} {portfolio_win_delta_pct:>+6.1f}% {portfolio_dd_base_pct:>9.1f}% {portfolio_dd_sized_pct:>10.1f}% {portfolio_dd_delta_pct:>+7.1f}% {dd_ok}")
    
    print(f"\n{'='*145}")
    
    return results


if __name__ == "__main__":
    apply_sizing()