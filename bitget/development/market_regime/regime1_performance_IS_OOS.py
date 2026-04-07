#!/usr/bin/env python3
"""
regime1_compare_IS_vs_OOS.py - Compare IS vs OOS regime performance

Analyzes regime performance on both IS and OOS data, then compares results
to determine filtering decisions for production.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_metrics import calc_all_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

# IS (In-Sample) Configuration
IS_TRADES_FOLDER = '../brief_trades'
IS_OHLC_FOLDER   = '../data/crypto_2022_OOS'

# OOS (Out-of-Sample) Configuration
OOS_TRADES_FOLDER = '../brief_trades_2026'
OOS_OHLC_FOLDER   = '../data/crypto_2026_OOS'

# Analysis Parameters
MA_PERIOD             = 50
INITIAL_CAPITAL       = 800
MIN_TRADES_CONFIDENCE = 50
ANALYZE_DIRECTION     = False  # True: analyze FAMILY + DIRECTION + REGIME, False: analyze FAMILY only

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

def extract_timeframe(filename):
    """Extract timeframe from filename"""
    name = Path(filename).stem.replace('all_trades_', '')
    parts = name.split('_')
    if parts[-1].upper() in ['IS', 'OOS']:
        parts = parts[:-1]
    if parts:
        timeframe = parts[-1]
        if any(c.isdigit() for c in timeframe.upper()) and 'H' in timeframe.upper():
            return timeframe
    return '4H'

def load_btc_for_timeframe(ohlc_folder, timeframe):
    """Load BTC OHLC for specific timeframe"""
    cache_key = f"{ohlc_folder}_{timeframe}"
    if cache_key in _btc_cache:
        return _btc_cache[cache_key]
    
    filepath = Path(ohlc_folder) / f"BTCUSDT_{timeframe}.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"BTC OHLC not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    df = df.sort_values('ts').reset_index(drop=True)
    
    _btc_cache[cache_key] = df
    return df

def calc_all_metrics_at_time(btc_df, buy_time, lookback):
    """Calculate metrics at specific time"""
    closed_candles = btc_df[btc_df['ts'] < buy_time]
    if len(closed_candles) < lookback:
        return None
    idx = closed_candles.index[-1]
    start_idx = max(0, idx - lookback + 1)
    if idx - start_idx < 20:
        return None
    subset = btc_df.iloc[start_idx:idx + 1]
    ohlc = {
        'open': subset['open'].values.astype(np.float64),
        'high': subset['high'].values.astype(np.float64),
        'low': subset['low'].values.astype(np.float64),
        'close': subset['close'].values.astype(np.float64)
    }
    metrics = calc_all_metrics(ohlc, hurst_window=HURST_WINDOW, er_window=ER_WINDOW, 
                                atr_window=ATR_WINDOW, pe_window=PE_WINDOW, pe_order=PE_ORDER)
    current_close = float(btc_df.iloc[idx]['close'])
    if idx >= (MA_PERIOD - 1):
        ma_data = btc_df.iloc[idx - (MA_PERIOD - 1):idx + 1]['close'].values
        metrics['ma_50'] = float(np.mean(ma_data))
        metrics['price_vs_ma_50'] = current_close / metrics['ma_50']
    else:
        metrics['ma_50'] = np.nan
        metrics['price_vs_ma_50'] = np.nan
    return metrics

def classify_trade_by_family(metrics, families):
    """Classify trade into family"""
    for family_name, rules in families.items():
        if not rules:
            continue
        match = True
        for metric, (op, val) in rules.items():
            if metrics.get(metric) is None or pd.isna(metrics[metric]):
                match = False
                break
            if op == '>' and not (metrics[metric] > val):
                match = False
                break
            elif op == '<' and not (metrics[metric] < val):
                match = False
                break
        if match:
            return family_name
    for family_name, rules in families.items():
        if not rules:
            return family_name
    return 'unknown'

def load_trades(filepath):
    """Load trades"""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    else:
        raise ValueError("File missing buy_time column")
    return df

def calculate_max_dd_pct(equity_curve):
    """Calculate Maximum Drawdown"""
    if len(equity_curve) == 0:
        return 0.0
    cummax = equity_curve.cummax()
    drawdown_pct = np.where(cummax > 0, ((cummax - equity_curve) / cummax) * 100, 0.0)
    return float(np.max(drawdown_pct))

def permutation_test(profits1, profits2, n_permutations=1000):
    """Permutation test"""
    if len(profits1) < 10 or len(profits2) < 10:
        return 1.0
    observed_diff = np.mean(profits1) - np.mean(profits2)
    combined = profits1 + profits2
    n1 = len(profits1)
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    return count_extreme / n_permutations

def analyze_by_dimension(df, dimension, initial_capital):
    """Analyze performance by dimension"""
    stats = {}
    for category in df[dimension].unique():
        cat_df = df[df[dimension] == category].copy()
        cat_df = cat_df.sort_values('buy_time').reset_index(drop=True)
        cat_df['equity'] = initial_capital + cat_df['profit'].cumsum()
        num_trades = len(cat_df)
        profit = cat_df['profit'].sum()
        profits_list = cat_df['profit'].tolist()
        confidence = num_trades >= MIN_TRADES_CONFIDENCE
        stats[category] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(cat_df['equity']),
            'win_rate': (cat_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'profits_list': profits_list,
            'confidence': confidence
        }
    return stats

def analyze_strategy(filepath, ohlc_folder, families, initial_capital):
    """Analyze single strategy"""
    strategy = Path(filepath).stem.replace('all_trades_', '')
    
    timeframe = extract_timeframe(Path(filepath).name)
    btc_df = load_btc_for_timeframe(ohlc_folder, timeframe)
    
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
        metrics = calc_all_metrics_at_time(btc_df, trade['buy_time'], LOOKBACK_BARS)
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
    
    if ANALYZE_DIRECTION:
        df['regime'] = df['family'] + '_' + df['trend']
    
    df = df.sort_values('buy_time').reset_index(drop=True)
    
    family_stats = analyze_by_dimension(df, 'family', initial_capital)
    trend_stats = analyze_by_dimension(df, 'trend', initial_capital) if ANALYZE_DIRECTION else {}
    regime_stats = analyze_by_dimension(df, 'regime', initial_capital) if ANALYZE_DIRECTION else {}
    
    return {
        'strategy': strategy,
        'total_trades': len(df),
        'family_stats': family_stats,
        'trend_stats': trend_stats,
        'regime_stats': regime_stats
    }

def get_best_category(stats_dict):
    """Get best performing category with significance and confidence"""
    if not stats_dict or len(stats_dict) < 1:
        return None, None, None, False, False
    
    sorted_stats = sorted(stats_dict.items(), key=lambda x: x[1]['profit'], reverse=True)
    best_cat, best_stats = sorted_stats[0]
    
    is_confident = best_stats['confidence']
    
    is_significant = False
    if len(sorted_stats) >= 2:
        second_cat, second_stats = sorted_stats[1]
        p_value = permutation_test(best_stats['profits_list'], second_stats['profits_list'])
        is_significant = p_value < 0.10
    
    return best_cat, best_stats['num_trades'], best_stats['profit'], is_confident, is_significant

def compare_datasets(is_results, oos_results):
    """Compare IS vs OOS results and generate decision table"""
    
    print("\n" + "="*160)
    print("IS vs OOS COMPARISON - FAMILY")
    print("="*160)
    print(f"{'STRATEGY':<30} {'CONF':>5} {'IS_PROFIT':>12} {'OOS_PROFIT':>12} {'IS_BEST':<20} {'OOS_BEST':<20} "
          f"{'IS_SIG':>7} {'OOS_SIG':>8} {'MATCH':>6} {'DECISION':<30}")
    print("-"*160)
    
    for is_r in is_results:
        strategy = is_r['strategy']
        oos_r = next((r for r in oos_results if r['strategy'] == strategy), None)
        
        if not oos_r:
            continue
        
        is_best, is_trades, is_profit, is_conf, is_sig = get_best_category(is_r['family_stats'])
        oos_best, oos_trades, oos_profit, oos_conf, oos_sig = get_best_category(oos_r['family_stats'])
        
        # Combined confidence (both must be confident)
        conf_str = "✓" if (is_conf and oos_conf) else "✗"
        
        # Match symbol logic (green/orange/red)
        if is_best == oos_best:
            if is_sig and oos_sig:
                match = "🟢"  # Green: match + both significant
            elif oos_sig and not is_sig:
                match = "🟠"  # Orange: match + only OOS significant
            else:
                match = "🔴"  # Red: match but only IS sig or neither sig
        else:
            match = "🔴"  # Red: no match
        
        # Significance
        is_sig_str = "✅" if is_sig else "❌"
        oos_sig_str = "✅" if oos_sig else "❌"
        
        # Decision logic
        if is_best == oos_best and is_sig and oos_sig and is_conf and oos_conf:
            decision = f"Filter: {is_best}"
        elif oos_sig and oos_conf:
            decision = f"Filter: {oos_best} (OOS only)"
        elif is_sig and is_conf:
            decision = f"Filter: {is_best} (IS only)"
        else:
            decision = "NO FILTER"
        
        print(f"{strategy:<30} {conf_str:>5} {is_profit:>12.2f} {oos_profit:>12.2f} {is_best:<20} {oos_best:<20} "
              f"{is_sig_str:>7} {oos_sig_str:>8} {match:>6} {decision:<30}")
    
    print("-"*160)
    
    if not ANALYZE_DIRECTION:
        return  # Skip DIRECTION and REGIME comparison tables
    
    print("\n" + "="*160)
    print("IS vs OOS COMPARISON - DIRECTION")
    print("="*160)
    print(f"{'STRATEGY':<30} {'CONF':>5} {'IS_PROFIT':>12} {'OOS_PROFIT':>12} {'IS_BEST':<20} {'OOS_BEST':<20} "
          f"{'IS_SIG':>7} {'OOS_SIG':>8} {'MATCH':>6} {'DECISION':<30}")
    print("-"*160)
    
    for is_r in is_results:
        strategy = is_r['strategy']
        oos_r = next((r for r in oos_results if r['strategy'] == strategy), None)
        
        if not oos_r:
            continue
        
        is_best, is_trades, is_profit, is_conf, is_sig = get_best_category(is_r['trend_stats'])
        oos_best, oos_trades, oos_profit, oos_conf, oos_sig = get_best_category(oos_r['trend_stats'])
        
        # Combined confidence
        conf_str = "✓" if (is_conf and oos_conf) else "✗"
        
        # Match symbol logic
        if is_best == oos_best:
            if is_sig and oos_sig:
                match = "🟢"
            elif oos_sig and not is_sig:
                match = "🟠"
            else:
                match = "🔴"
        else:
            match = "🔴"
        
        # Significance
        is_sig_str = "✅" if is_sig else "❌"
        oos_sig_str = "✅" if oos_sig else "❌"
        
        # Decision logic
        if is_best == oos_best and is_sig and oos_sig and is_conf and oos_conf:
            decision = f"Filter: {is_best}"
        elif oos_sig and oos_conf:
            decision = f"Filter: {oos_best} (OOS only)"
        elif is_sig and is_conf:
            decision = f"Filter: {is_best} (IS only)"
        else:
            decision = "NO FILTER"
        
        print(f"{strategy:<30} {conf_str:>5} {is_profit:>12.2f} {oos_profit:>12.2f} {is_best:<20} {oos_best:<20} "
              f"{is_sig_str:>7} {oos_sig_str:>8} {match:>6} {decision:<30}")
    
    print("-"*160)
    
    print("\n" + "="*160)
    print("IS vs OOS COMPARISON - REGIME")
    print("="*160)
    print(f"{'STRATEGY':<30} {'CONF':>5} {'IS_PROFIT':>12} {'OOS_PROFIT':>12} {'IS_BEST':<20} {'OOS_BEST':<20} "
          f"{'IS_SIG':>7} {'OOS_SIG':>8} {'MATCH':>6} {'DECISION':<30}")
    print("-"*160)
    
    for is_r in is_results:
        strategy = is_r['strategy']
        oos_r = next((r for r in oos_results if r['strategy'] == strategy), None)
        
        if not oos_r:
            continue
        
        is_best, is_trades, is_profit, is_conf, is_sig = get_best_category(is_r['regime_stats'])
        oos_best, oos_trades, oos_profit, oos_conf, oos_sig = get_best_category(oos_r['regime_stats'])
        
        # Combined confidence
        conf_str = "✓" if (is_conf and oos_conf) else "✗"
        
        # Match symbol logic
        if is_best == oos_best:
            if is_sig and oos_sig:
                match = "🟢"
            elif oos_sig and not is_sig:
                match = "🟠"
            else:
                match = "🔴"
        else:
            match = "🔴"
        
        # Significance
        is_sig_str = "✅" if is_sig else "❌"
        oos_sig_str = "✅" if oos_sig else "❌"
        
        # Decision logic
        if is_best == oos_best and is_sig and oos_sig and is_conf and oos_conf:
            decision = f"Filter: {is_best}"
        elif oos_sig and oos_conf:
            decision = f"Filter: {oos_best} (OOS only)"
        elif is_sig and is_conf:
            decision = f"Filter: {is_best} (IS only)"
        else:
            decision = "NO FILTER"
        
        print(f"{strategy:<30} {conf_str:>5} {is_profit:>12.2f} {oos_profit:>12.2f} {is_best:<20} {oos_best:<20} "
              f"{is_sig_str:>7} {oos_sig_str:>8} {match:>6} {decision:<30}")
    
    print("-"*160)

def main():
    print("=" * 180)
    print("REGIME ANALYZER - IS vs OOS COMPARISON")
    print("=" * 180)
    
    print(f"\nConfiguration:")
    print(f"  IS Trades:  {IS_TRADES_FOLDER}")
    print(f"  IS OHLC:    {IS_OHLC_FOLDER}")
    print(f"  OOS Trades: {OOS_TRADES_FOLDER}")
    print(f"  OOS OHLC:   {OOS_OHLC_FOLDER}")
    print(f"  MA period:  MA{MA_PERIOD}")
    print(f"  Capital:    ${INITIAL_CAPITAL}")
    print(f"  Min trades: {MIN_TRADES_CONFIDENCE}")
    print(f"  Analyze Direction: {ANALYZE_DIRECTION}")
    
    # Analyze IS
    print(f"\n{'='*180}")
    print("PHASE 1: ANALYZING IS (In-Sample)")
    print(f"{'='*180}")
    
    is_pattern = str(Path(IS_TRADES_FOLDER) / 'all_trades_*.xlsx')
    is_files = sorted(glob(is_pattern))
    
    if not is_files:
        print(f"\n❌ No IS files found in {IS_TRADES_FOLDER}")
        return
    
    print(f"\n📂 Found {len(is_files)} IS strategy files")
    
    is_results = []
    for filepath in is_files:
        strategy = Path(filepath).stem.replace('all_trades_', '')
        print(f"   Processing IS: {strategy}...")
        result = analyze_strategy(filepath, IS_OHLC_FOLDER, FAMILIES, INITIAL_CAPITAL)
        is_results.append(result)
    
    # Analyze OOS
    print(f"\n{'='*180}")
    print("PHASE 2: ANALYZING OOS (Out-of-Sample)")
    print(f"{'='*180}")
    
    oos_pattern = str(Path(OOS_TRADES_FOLDER) / 'all_trades_*.xlsx')
    oos_files = sorted(glob(oos_pattern))
    
    if not oos_files:
        print(f"\n❌ No OOS files found in {OOS_TRADES_FOLDER}")
        return
    
    print(f"\n📂 Found {len(oos_files)} OOS strategy files")
    
    oos_results = []
    for filepath in oos_files:
        strategy = Path(filepath).stem.replace('all_trades_', '')
        print(f"   Processing OOS: {strategy}...")
        result = analyze_strategy(filepath, OOS_OHLC_FOLDER, FAMILIES, INITIAL_CAPITAL)
        oos_results.append(result)
    
    # Compare
    print(f"\n{'='*180}")
    print("PHASE 3: COMPARISON & FILTERING DECISIONS")
    print(f"{'='*180}")
    
    compare_datasets(is_results, oos_results)
    
    print(f"\n{'='*160}")
    print("DECISION RULES:")
    print(f"{'='*160}")
    print("\n  🟢 GREEN: Match + Both significant → Use this filter (IS and OOS agree strongly)")
    print("  🟠 ORANGE: Match + Only OOS significant → Use OOS filter (IS agrees but weak evidence)")
    print("  🔴 RED:")
    print("     - No match → Use OOS filter if ✓✅, or IS if only IS ✓✅, otherwise NO FILTER")
    print("     - Match but only IS significant → Use IS filter (OOS agrees but weak)")
    print("     - Match but neither significant → NO FILTER (both agree but weak evidence)")
    print("\n  Legend:")
    print("    CONF ✓ = Both IS and OOS confident (≥50 trades each)")
    print("    CONF ✗ = At least one unreliable (<50 trades)")
    print("    SIG ✅ = Significant (p<0.10)")
    print("    SIG ❌ = Not significant (p≥0.10)")
    print("    MATCH 🟢 = IS and OOS agree + both significant")
    print("    MATCH 🟠 = IS and OOS agree + only OOS significant")
    print("    MATCH 🔴 = No match OR weak evidence")
    print(f"{'='*160}")

if __name__ == "__main__":
    main()