#!/usr/bin/env python3
"""
regime_common.py - Shared functions for regime analysis scripts

Contains all common functions used across:
- regime1_performance_IS.py
- regime1_performance_WFO.py
- regime1_compare_IS_vs_OOS.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from regime_metrics import calc_all_metrics


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


def load_btc_for_timeframe(ohlc_folder, timeframe, cache):
    """Load BTC OHLC for specific timeframe with caching"""
    cache_key = f"{ohlc_folder}_{timeframe}"
    if cache_key in cache:
        return cache[cache_key]
    
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
    
    cache[cache_key] = df
    return df


def calc_all_metrics_at_time(btc_df, buy_time, lookback, ma_period, hurst_window, er_window, atr_window, pe_window, pe_order):
    """Calculate metrics at specific time - no lookahead"""
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
    metrics = calc_all_metrics(ohlc, hurst_window=hurst_window, er_window=er_window, 
                                atr_window=atr_window, pe_window=pe_window, pe_order=pe_order)
    current_close = float(btc_df.iloc[idx]['close'])
    if idx >= (ma_period - 1):
        ma_data = btc_df.iloc[idx - (ma_period - 1):idx + 1]['close'].values
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
    """Load trades from Excel file"""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    else:
        raise ValueError("File missing buy_time column")
    return df


def calculate_max_dd_pct(equity_curve):
    """Calculate Maximum Drawdown percentage"""
    if len(equity_curve) == 0:
        return 0.0
    cummax = equity_curve.cummax()
    drawdown_pct = np.where(cummax > 0, ((cummax - equity_curve) / cummax) * 100, 0.0)
    return float(np.max(drawdown_pct))


def permutation_test(profits1, profits2, n_permutations=1000, random_seed=42):

    if len(profits1) < 10 or len(profits2) < 10:
        return 1.0
    observed_diff = np.mean(profits1) - np.mean(profits2)
    combined = profits1 + profits2
    n1 = len(profits1)
    count_extreme = 0
    
    np.random.seed(random_seed)
    
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    return count_extreme / n_permutations


def format_significance(p_value):
    """Format significance indicator"""
    if p_value < 0.1:
        return f"✅ (p={p_value:.3f})"
    else:
        return f"❌ (p={p_value:.2f})"


def analyze_by_dimension(df, dimension, initial_capital, min_trades_confidence=None):
    """Analyze performance by dimension (family/direction/regime)
    
    Args:
        df: DataFrame with trades
        dimension: Column name to group by ('family', 'trend', 'regime')
        initial_capital: Starting capital
        min_trades_confidence: If provided, adds confidence indicator
    
    Returns:
        Dictionary with stats per category
    """
    stats = {}
    for category in df[dimension].unique():
        cat_df = df[df[dimension] == category].copy()
        cat_df = cat_df.sort_values('buy_time').reset_index(drop=True)
        cat_df['equity'] = initial_capital + cat_df['profit'].cumsum()
        num_trades = len(cat_df)
        profit = cat_df['profit'].sum()
        profits_list = cat_df['profit'].tolist()
        
        stats[category] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(cat_df['equity']),
            'win_rate': (cat_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'profits_list': profits_list
        }
        
        # Add confidence indicator if min_trades_confidence is provided
        if min_trades_confidence is not None:
            stats[category]['confidence'] = "✓" if num_trades >= min_trades_confidence else "✗"
    
    return stats