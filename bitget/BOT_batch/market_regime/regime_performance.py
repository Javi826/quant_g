#!/usr/bin/env python3
"""
market_regime/regime1_performance_OOS.py - MATCHES ENRICHER EXACTLY

Replicates enricher.py behavior:
- Drops NaN rows in critical metrics
- Calculates MA50 and price_vs_ma_50
- Uses closed candles only (no lookahead)

Now uses regime_common.py for shared functions.
"""

import os
import sys
import numpy as np
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared", "market_regime")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared")))

from regime_common import extract_timeframe, load_btc_for_timeframe, calc_all_metrics_at_time
from regime_common import classify_trade_by_family, load_trades, calculate_max_dd_pct
from regime_common import permutation_test, format_significance, analyze_by_dimension
from shared_config import REGIME_FAMILIES as FAMILIES, REGIME_HURST_WINDOW as HURST_WINDOW
from shared_config import REGIME_ER_WINDOW as ER_WINDOW, REGIME_ATR_WINDOW as ATR_WINDOW
from shared_config import REGIME_PE_WINDOW as PE_WINDOW, REGIME_PE_ORDER as PE_ORDER

BASE_DIR              = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OHLC_FOLDER           = os.path.join(BASE_DIR, "data", "crypto_2026_OOS")
MA_PERIOD             = 50
MIN_TRADES_CONFIDENCE = 50
ANALYZE_DIRECTION     = False


LOOKBACK_BARS = 100

_btc_cache = {}


def analyze_strategy(filepath, families, initial_capital):
    """Analyze single strategy - MATCHES enricher.py"""
    df = load_trades(filepath)
    strategy = df['strategy'].iloc[0]
    timeframe = extract_timeframe(df)
    btc_df = load_btc_for_timeframe(OHLC_FOLDER, timeframe, _btc_cache)
    
    df = load_trades(filepath)
    
    df['family'] = 'unknown'
    df['trend'] = 'unknown'
    df['hurst'] = np.nan
    df['efficiency_ratio'] = np.nan
    df['atr_pct'] = np.nan
    df['permutation_entropy'] = np.nan
    df['ma_50'] = np.nan
    df['price_vs_ma_50'] = np.nan
    
    for idx, trade in df.iterrows():
        metrics = calc_all_metrics_at_time(btc_df, trade['buy_time'], LOOKBACK_BARS, 
                                           MA_PERIOD, HURST_WINDOW, ER_WINDOW, 
                                           ATR_WINDOW, PE_WINDOW, PE_ORDER)
        if metrics:
            df.at[idx, 'hurst'] = metrics['hurst']
            df.at[idx, 'efficiency_ratio'] = metrics['efficiency_ratio']
            df.at[idx, 'atr_pct'] = metrics['atr_pct']
            df.at[idx, 'permutation_entropy'] = metrics['permutation_entropy']
            df.at[idx, 'ma_50'] = metrics['ma_50']
            df.at[idx, 'price_vs_ma_50'] = metrics['price_vs_ma_50']
            family = classify_trade_by_family(metrics, families)
            df.at[idx, 'family'] = family
            if not pd.isna(metrics['price_vs_ma_50']):
                df.at[idx, 'trend'] = 'uptrend' if metrics['price_vs_ma_50'] > 1.0 else 'downtrend'
    
    if ANALYZE_DIRECTION:
        critical_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy', 'ma_50', 'price_vs_ma_50']
    else:
        critical_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    
    if ANALYZE_DIRECTION:
        df['regime'] = df['family'] + '_' + df['trend']
    
    df = df.sort_values('buy_time').reset_index(drop=True)
    family_stats = analyze_by_dimension(df, 'family', initial_capital, MIN_TRADES_CONFIDENCE)
    
    if ANALYZE_DIRECTION:
        trend_stats = analyze_by_dimension(df, 'trend', initial_capital, MIN_TRADES_CONFIDENCE)
        regime_stats = analyze_by_dimension(df, 'regime', initial_capital, MIN_TRADES_CONFIDENCE)
    else:
        trend_stats = {}
        regime_stats = {}
    
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


def print_single_strategy_all_dimensions(r):
    """Print tables"""
    print(f"\n\033[93m{'='*145}\033[0m")
    print(f"\033[93mSTRATEGY: {r['strategy']} (Total: {r['total_trades']} trades, Profit: ${r['total_profit']:.2f}, DD: {r['total_dd_pct']:.2f}%, WR: {r['total_win_rate']:.1f}%)\033[0m")
    print(f"\033[93m{'='*145}\033[0m")
    
    print(f"\n{'─'*120}")
    print("BY FAMILY (trending/volatile/ranging)")
    print(f"{'─'*120}")
    print(f"{'FAMILY':<20} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    family_stats = r['family_stats']
    sorted_family = sorted(family_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (category, stats) in enumerate(sorted_family):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        if len(sorted_family) < 2:
            p_str = "N/A"
        elif idx == 0:
            p_value = permutation_test(sorted_family[0][1]['profits_list'], sorted_family[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            p_value = permutation_test(stats['profits_list'], sorted_family[0][1]['profits_list'])
            p_str = format_significance(p_value)
        print(f"{category:<20} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)
    print(f"{'TOTAL':<20} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    if len(sorted_family) >= 2:
        best_fam, best_stats = sorted_family[0]
        second_fam, second_stats = sorted_family[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_fam} (${best_stats['profit']:.2f}) vs 2ND: {second_fam} (${second_stats['profit']:.2f}) | {sig_str}")
    
    if not ANALYZE_DIRECTION:
        return
    
    print(f"\n{'─'*120}")
    print("BY DIRECTION (uptrend/downtrend)")
    print(f"{'─'*120}")
    print(f"{'DIRECTION':<20} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    trend_stats = r['trend_stats']
    sorted_trend = sorted(trend_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (category, stats) in enumerate(sorted_trend):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        if len(sorted_trend) < 2:
            p_str = "N/A"
        elif idx == 0:
            p_value = permutation_test(sorted_trend[0][1]['profits_list'], sorted_trend[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            p_value = permutation_test(stats['profits_list'], sorted_trend[0][1]['profits_list'])
            p_str = format_significance(p_value)
        print(f"{category:<20} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)
    print(f"{'TOTAL':<20} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    if len(sorted_trend) >= 2:
        best_dir, best_stats = sorted_trend[0]
        second_dir, second_stats = sorted_trend[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_dir} (${best_stats['profit']:.2f}) vs 2ND: {second_dir} (${second_stats['profit']:.2f}) | {sig_str}")
    
    print(f"\n{'─'*120}")
    print("BY REGIME (6 combined categories)")
    print(f"{'─'*120}")
    print(f"{'REGIME':<20} {'CONF':>5} {'TRADES':>10} {'PROFIT':>12} {'%PROFIT':>10} {'DD%':>10} {'WIN%':>10} {'P-VALUE':>15}")
    print("-" * 120)
    
    regime_stats = r['regime_stats']
    sorted_regime = sorted(regime_stats.items(), key=lambda x: x[1]['profit'], reverse=True)
    
    for idx, (category, stats) in enumerate(sorted_regime):
        profit_pct = (stats['profit'] / r['total_profit'] * 100) if r['total_profit'] != 0 else 0.0
        if len(sorted_regime) < 2:
            p_str = "N/A"
        elif idx == 0:
            p_value = permutation_test(sorted_regime[0][1]['profits_list'], sorted_regime[1][1]['profits_list'])
            p_str = format_significance(p_value)
        else:
            p_value = permutation_test(stats['profits_list'], sorted_regime[0][1]['profits_list'])
            p_str = format_significance(p_value)
        print(f"{category:<20} {stats['confidence']:>5} {stats['num_trades']:>10} {stats['profit']:>12.2f} {profit_pct:>9.1f}% {stats['dd_pct']:>10.2f} {stats['win_rate']:>10.1f} {p_str:>15}")
    
    print("-" * 120)
    print(f"{'TOTAL':<20} {'':>5} {r['total_trades']:>10} {r['total_profit']:>12.2f} {100.0:>9.1f}% {r['total_dd_pct']:>10.2f} {r['total_win_rate']:>10.1f} {'':>15}")
    
    if len(sorted_regime) >= 2:
        best_reg, best_stats = sorted_regime[0]
        second_reg, second_stats = sorted_regime[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        sig_str = format_significance(p_value)
        print(f"\n→ BEST: {best_reg} (${best_stats['profit']:.2f}) vs 2ND: {second_reg} (${second_stats['profit']:.2f}) | {sig_str}")

