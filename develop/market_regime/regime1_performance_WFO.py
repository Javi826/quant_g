#!/usr/bin/env python3
"""
develop/market_regime/regime1_performance_WFO.py 

Analyzes regime performance using Walk-Forward Optimization with two modes:
- ANCHORED: Expanding window from fixed start date
- UNANCHORED: Rolling window of fixed size

For each window, identifies the best performing FAMILY (and optionally DIRECTION),
then validates stability across windows.

Now uses regime_common.py for shared functions.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_common import extract_timeframe, load_btc_for_timeframe, calc_all_metrics_at_time
from regime_common import classify_trade_by_family, load_trades, calculate_max_dd_pct
from regime_common import analyze_by_dimension

# =============================================================================
# CONFIGURATION
# =============================================================================

# OOS Data Configuration
OOS_TRADES_FOLDER = '../brief_trades_2026'
OOS_OHLC_FOLDER   = '../data/crypto_2026_OOS'

# WFO Parameters
WFO_MODE          = 'unanchored'  # 'anchored' or 'unanchored'
WFO_WINDOW_MONTHS = 3             # Window size in months
OOS_START_DATE    = '2025-01-01'
OOS_END_DATE      = '2026-03-31'

# Analysis Parameters
MA_PERIOD             = 50
INITIAL_CAPITAL       = 800
ANALYZE_DIRECTION     = False

# =============================================================================

FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging': {}
}

HURST_WINDOW  = 100
ER_WINDOW     = 14
ATR_WINDOW    = 14
PE_WINDOW     = 50
PE_ORDER      = 3
LOOKBACK_BARS = 100

_btc_cache = {}


def analyze_strategy_window(filepath, ohlc_folder, families, initial_capital, start_date, end_date):
    """Analyze single strategy within a time window"""
    strategy = Path(filepath).stem.replace('all_trades_', '')
    
    timeframe = extract_timeframe(Path(filepath).name)
    btc_df = load_btc_for_timeframe(ohlc_folder, timeframe, _btc_cache)
    
    df = load_trades(filepath)
    
    df = df[(df['buy_time'] >= start_date) & (df['buy_time'] <= end_date)].copy()
    
    if len(df) == 0:
        return None
    
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
    
    critical_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy', 'ma_50', 'price_vs_ma_50'] if ANALYZE_DIRECTION else ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    
    if len(df) == 0:
        return None
    
    if ANALYZE_DIRECTION:
        df['regime'] = df['family'] + '_' + df['trend']
    
    df = df.sort_values('buy_time').reset_index(drop=True)
    
    family_stats = analyze_by_dimension(df, 'family', initial_capital)
    trend_stats = analyze_by_dimension(df, 'trend', initial_capital) if ANALYZE_DIRECTION else {}
    
    return {
        'strategy': strategy,
        'total_trades': len(df),
        'family_stats': family_stats,
        'trend_stats': trend_stats,
        'df': df
    }


def get_best_category(stats_dict):
    """Get best performing category (simplified - no p-value)"""
    if not stats_dict or len(stats_dict) < 1:
        return None, 0, 0.0
    
    sorted_stats = sorted(stats_dict.items(), key=lambda x: x[1]['profit'], reverse=True)
    best_cat, best_stats = sorted_stats[0]
    
    return best_cat, best_stats['num_trades'], best_stats['profit']


def generate_windows(start_date, end_date, window_months, mode):
    """Generate time windows for WFO"""
    windows = []
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    if mode == 'anchored':
        current_end = start + relativedelta(months=window_months)
        window_num = 1
        while current_end <= end:
            windows.append({
                'window': window_num,
                'start': start,
                'end': current_end
            })
            current_end += relativedelta(months=window_months)
            window_num += 1
        if windows[-1]['end'] < end:
            windows.append({
                'window': window_num,
                'start': start,
                'end': end
            })
    
    else:
        current_start = start
        window_num = 1
        while current_start < end:
            current_end = min(current_start + relativedelta(months=window_months), end)
            windows.append({
                'window': window_num,
                'start': current_start,
                'end': current_end
            })
            current_start = current_end
            window_num += 1
    
    return windows


def main():
    print("=" * 160)
    print("REGIME WFO (Walk-Forward Optimization) - OOS ANALYSIS")
    print("=" * 160)
    
    print(f"\nConfiguration:")
    print(f"  OOS Trades:       {OOS_TRADES_FOLDER}")
    print(f"  OOS OHLC:         {OOS_OHLC_FOLDER}")
    print(f"  WFO Mode:         {WFO_MODE.upper()}")
    print(f"  Window Size:      {WFO_WINDOW_MONTHS} months")
    print(f"  Period:           {OOS_START_DATE} to {OOS_END_DATE}")
    print(f"  Analyze Direction: {ANALYZE_DIRECTION}")
    
    windows = generate_windows(OOS_START_DATE, OOS_END_DATE, WFO_WINDOW_MONTHS, WFO_MODE)
    
    print(f"\n📅 Generated {len(windows)} windows:")
    for w in windows:
        print(f"   Window {w['window']}: {w['start'].strftime('%Y-%m-%d')} to {w['end'].strftime('%Y-%m-%d')}")
    
    pattern = str(Path(OOS_TRADES_FOLDER) / 'all_trades_*.xlsx')
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No files found in {OOS_TRADES_FOLDER}")
        return
    
    print(f"\n📂 Found {len(files)} strategy files")
    
    strategy_results = {}
    
    for filepath in files:
        strategy_name = Path(filepath).stem.replace('all_trades_', '')
        print(f"\n{'='*160}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'='*160}")
        
        window_results = []
        
        for w in windows:
            print(f"\n  Window {w['window']} ({w['start'].strftime('%Y-%m-%d')} to {w['end'].strftime('%Y-%m-%d')}):")
            
            result = analyze_strategy_window(filepath, OOS_OHLC_FOLDER, FAMILIES, INITIAL_CAPITAL, 
                                             w['start'], w['end'])
            
            if result is None:
                print(f"    ⚠️  No trades in this window")
                window_results.append(None)
                continue
            
            best_fam, fam_trades, fam_profit = get_best_category(result['family_stats'])
            
            print(f"    Total Trades: {result['total_trades']}")
            print(f"    Best FAMILY:  {best_fam} (profit: {fam_profit:.2f}, trades: {fam_trades})")
            
            if ANALYZE_DIRECTION:
                best_dir, dir_trades, dir_profit = get_best_category(result['trend_stats'])
                print(f"    Best DIRECTION: {best_dir} (profit: {dir_profit:.2f}, trades: {dir_trades})")
            
            window_results.append({
                'window': w['window'],
                'start': w['start'],
                'end': w['end'],
                'result': result,
                'best_family': best_fam,
                'family_profit': fam_profit,
                'family_trades': fam_trades
            })
        
        strategy_results[strategy_name] = window_results
    
    print(f"\n{'='*160}")
    print("WFO SUMMARY - FAMILY ACROSS WINDOWS")
    print(f"{'='*160}")
    
    print(f"\n{'STRATEGY':<30} ", end="")
    for w in windows:
        print(f"{'W'+str(w['window']):<12} ", end="")
    print(f"{'MOST_FREQ':<12} {'FREQ':>10}")
    print("-" * 160)
    
    for strategy_name, window_results in strategy_results.items():
        print(f"{strategy_name:<30} ", end="")
        
        families = []
        for wr in window_results:
            if wr is None:
                print(f"{'---':<12} ", end="")
            else:
                print(f"{wr['best_family'][:10]:<12} ", end="")
                families.append(wr['best_family'])
        
        if families:
            from collections import Counter
            freq_count = Counter(families)
            most_freq_family, freq = freq_count.most_common(1)[0]
            total_windows = len([wr for wr in window_results if wr is not None])
            freq_str = f"{freq}/{total_windows}"
            print(f"{most_freq_family:<12} {freq_str:>10}")
        else:
            print(f"{'---':<12} {'---':>10}")
    
    print("-" * 160)
    
    print(f"\n{'='*160}")
    print("INTERPRETATION:")
    print(f"{'='*160}")
    print("\n  WFO Mode:")
    if WFO_MODE == 'anchored':
        print("    ANCHORED - Expanding window from fixed start (cumulative data)")
    else:
        print("    UNANCHORED - Rolling window of fixed size (moving data)")
    print("\n  Columns:")
    print("    W1, W2, ... = Best family per window")
    print("    MOST_FREQ = Most frequent family across all windows")
    print("    FREQ = Number of times it appears / Total windows")
    print("\n  Stability:")
    print("    - Same family across windows → Stable regime")
    print("    - Changing family → Regime shift detected")
    print("    - High FREQ (e.g., 10/12) → Consistent regime")
    print(f"{'='*160}")


if __name__ == "__main__":
    main()