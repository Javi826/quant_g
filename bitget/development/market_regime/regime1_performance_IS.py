#!/usr/bin/env python3
"""
regime1_performance_STANDALONE.py - MATCHES ENRICHER EXACTLY

Replicates enricher.py behavior:
- Drops NaN rows in critical metrics
- Calculates MA50 and price_vs_ma_50
- Uses closed candles only (no lookahead)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_metrics import calc_all_metrics

BASE_DIR              = '/home/javi/projects/quant/quant_g/bitget/development'
TRADES_FOLDER         = f'{BASE_DIR}/brief_trades_2026'
OHLC_FOLDER           = f'{BASE_DIR}/data/crypto_2026_OOS'  
MA_PERIOD             = 50
INITIAL_CAPITAL       = 800
MIN_TRADES_CONFIDENCE = 50

_btc_cache = {}  # Cache for BTC dataframes by timeframe

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

_btc_cache = {}  # Cache for BTC dataframes by timeframe

def extract_timeframe(filename):
    """Extract timeframe from filename - matches enricher.py"""
    name = Path(filename).stem.replace('all_trades_', '')
    parts = name.split('_')
    if parts[-1].upper() in ['IS', 'OOS']:
        parts = parts[:-1]
    if parts:
        timeframe = parts[-1]
        if any(c.isdigit() for c in timeframe.upper()) and 'H' in timeframe.upper():
            return timeframe
    print(f"    ⚠️  Could not extract timeframe from '{filename}', defaulting to 4H")
    return '4H'

def load_btc_for_timeframe(ohlc_folder, timeframe):
    """Load BTC OHLC for specific timeframe - matches enricher.py"""
    cache_key = f"BTCUSDT_{timeframe}"
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
    """Calculate metrics - MATCHES enricher.py exactly"""
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

def load_btc_1d(btc_file):
    """Load BTC data"""
    df = pd.read_parquet(btc_file)
    df.columns = df.columns.str.lower()
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    df = df.sort_values('ts').reset_index(drop=True)
    return df

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

def format_significance(p_value):
    """Format significance"""
    if p_value < 0.1:
        return f"✅ (p={p_value:.3f})"
    else:
        return f"❌ (p={p_value:.2f})"

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
        confidence = "✓" if num_trades >= MIN_TRADES_CONFIDENCE else "✗"
        stats[category] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(cat_df['equity']),
            'win_rate': (cat_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'profits_list': profits_list,
            'confidence': confidence
        }
    return stats

def analyze_strategy(filepath, families, initial_capital):
    """Analyze single strategy - MATCHES enricher.py"""
    strategy = Path(filepath).stem.replace('all_trades_', '')
    
    # Extract timeframe and load correct BTC file
    timeframe = extract_timeframe(Path(filepath).name)
    btc_df = load_btc_for_timeframe(OHLC_FOLDER, timeframe)
    
    print(f"   Processing {strategy} [{timeframe}]...")
    
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
    
    # CRITICAL: Drop NaN rows (matches enricher.py lines 244-256)
    critical_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy', 'ma_50', 'price_vs_ma_50']
    trades_before = len(df)
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    trades_after = len(df)
    if trades_before != trades_after:
        print(f"      Dropped {trades_before - trades_after} trades with NaN metrics")
    
    df['regime'] = df['family'] + '_' + df['trend']
    df = df.sort_values('buy_time').reset_index(drop=True)
    family_stats = analyze_by_dimension(df, 'family', initial_capital)
    trend_stats = analyze_by_dimension(df, 'trend', initial_capital)
    regime_stats = analyze_by_dimension(df, 'regime', initial_capital)
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

def print_summary_tables(results):
    """Print summaries"""
    print(f"\n{'='*145}")
    print(f"{'='*145}")
    print("SUMMARY - ALL STRATEGIES")
    print(f"{'='*145}")
    print(f"{'='*145}")
    
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

def main():
    print("=" * 70)
    print("REGIME ANALYZER - Performance across 3 dimensions (STANDALONE)")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Trades folder: {TRADES_FOLDER}")
    print(f"  OHLC folder:   {OHLC_FOLDER}")
    print(f"  MA period:     MA{MA_PERIOD}")
    print(f"  Capital:       ${INITIAL_CAPITAL}")
    
    print("\nDimensions analyzed:")
    print("  1. FAMILY: trending/volatile/ranging (ignoring BTC direction)")
    print("  2. DIRECTION: uptrend/downtrend (ignoring family)")
    print("  3. REGIME: 6 combined categories (full granularity)")
    
    print(f"\nConfidence indicator (CONF):")
    print(f"  ✓ = >={MIN_TRADES_CONFIDENCE} trades (reliable)")
    print(f"  ✗ = <{MIN_TRADES_CONFIDENCE} trades (unreliable)")
    
    print("\nSignificance indicator (SIGNIFICANT?):")
    print("  ✅ = p<0.10 (statistically significant difference)")
    print("  ❌ = p>=0.10 (no significant difference)")
    
    pattern = str(Path(TRADES_FOLDER) / 'all_trades_*.xlsx')
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No trades files found in {TRADES_FOLDER}")
        return
    
    print(f"\n📂 Found {len(files)} strategy files")
    print("\n🔍 Analyzing strategies...")
    
    results = []
    for filepath in files:
        result = analyze_strategy(filepath, FAMILIES, INITIAL_CAPITAL)
        results.append(result)
    
    for r in results:
        print_single_strategy_all_dimensions(r)
    
    print_summary_tables(results)
    
    print(f"\n{'='*145}")
    print("INTERPRETATION GUIDE:")
    print("\n  CONF (Confidence):")
    print("    ✓ = Reliable sample (>=50 trades) - trust these results")
    print("    ✗ = Unreliable sample (<50 trades) - don't trust these results")
    print("\n  SIGNIFICANT? (Statistical test):")
    print("    ✅ (p<0.10) = Difference is real, not random")
    print("    ❌ (p>=0.10) = Difference could be random chance")
    print("\n  FILTERING DECISION:")
    print("    - Only filter if BOTH: ✓ (reliable) AND ✅ (significant)")
    print("    - If FAMILY is ✓✅: filter by family only")
    print("    - If DIRECTION is ✓✅: filter by direction only")
    print("    - If REGIME is ✓✅: filter by specific regime")
    print("    - Otherwise: don't filter, operate in all conditions")
    print(f"{'='*145}")

if __name__ == "__main__":
    main()