#!/usr/bin/env python3
"""
market_regime/regime_layers_comparison.py

Compare 4 filtering scenarios:
1. BASELINE: No filters (all trades)
2. REGIME 0: BTC MA filter only (global market direction)
3. REGIME 1: Family/Direction filter only (market structure)
4. BOTH: REGIME 0 + REGIME 1 combined (AND logic)

Usage:
    python regime_layers_comparison.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_metrics import calc_all_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
TRADES_FOLDER   = '../brief_trades_2026'
OHLC_FOLDER     = '../data/crypto_2026_OOS'
BTC_FILE        = '../data/crypto_2026_OOS/BTCUSDT_1Dutc.parquet'

# REGIME 0 parameters (BTC MA filter)
MA_PERIOD       = 5
LONG_TH         = 1.00
SHORT_TH        = 1.00

# REGIME 1 parameters
REGIME1_MA_PERIOD = 50
INITIAL_CAPITAL   = 800

# REGIME 1 filters per strategy (based on strategies_E1.py)
# family: list of allowed families ['trending', 'ranging', 'volatile'] or None (all)
# direction: 'uptrend', 'downtrend', or None (all)
REGIME1_FILTERS = {
    '02_reversal_long_4H': {
        'family':    ['ranging'],
        'direction': None,
    },
    '03_parity_long_4H': {
        'family':    ['trending'],
        'direction': None,
    },
    '04_reversal_short_4H': {
        'family':    ['trending', 'ranging', 'volatile'],
        'direction': None,
    },
    '06_reversal_long_1H': {
        'family':    ['trending'],
        'direction': None,
    },
    '07_reversal_short_1H': {
        'family':    ['trending', 'ranging', 'volatile'],
        'direction': None,
    },
    '08_reversal_long_6Hutc': {
        'family':    ['trending', 'ranging'],
        'direction': None,
    },
    '09_reversal_short_6Hutc': {
        'family':    ['trending', 'ranging', 'volatile'],
        'direction': None,
    },
    '10_parity_long_1H': {
        'family':    ['trending'],
        'direction': None,
    },
    '11_parity_short_1H': {
        'family':    ['trending', 'ranging', 'volatile'],
        'direction': None,
    },
    '12_parity_long_6Hutc': {
        'family':    ['trending', 'ranging'],
        'direction': None,
    },
    '13_orderblocks_short_4H': {
        'family':    ['ranging'],
        'direction': None,
    },
    '16_ranging_short_6Hutc': {
        'family':    ['trending', 'ranging', 'volatile'],
        'direction': None,
    },
    '17_flag_long_4H': {
        'family':    ['trending', 'ranging', 'volatile'],
        'direction': None,
    },
    '19_flag_short_4H': {
        'family':    ['trending', 'ranging', 'volatile'],
        'direction': None,
    },
    '20_flag_short_1H': {
        'family':    ['trending', 'ranging', 'volatile'],
        'direction': None,
    },
}

# Regime families classification
FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging': {}
}

# Regime metrics parameters
HURST_WINDOW  = 100
ER_WINDOW     = 14
ATR_WINDOW    = 14
PE_WINDOW     = 50
PE_ORDER      = 3
LOOKBACK_BARS = 100

# =============================================================================

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

def load_btc_1d():
    """Load BTC 1D for REGIME 0"""
    df = pd.read_parquet(BTC_FILE)
    df.columns = df.columns.str.lower()
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    df = df.sort_values('ts').reset_index(drop=True)
    df[f'ma{MA_PERIOD}'] = df['close'].rolling(window=MA_PERIOD).mean()
    return df

def get_btc_value_at_trade(btc_df, trade_time):
    """Get BTC close and MA at trade time - REGIME 0 (no lookahead)"""
    closed_candles = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed_candles) < MA_PERIOD:
        return None, None
    
    last_candle = closed_candles.iloc[-1]
    
    if pd.isna(last_candle[f'ma{MA_PERIOD}']):
        return None, None
    
    return last_candle['close'], last_candle[f'ma{MA_PERIOD}']

def calc_all_metrics_at_time(btc_df, buy_time, lookback):
    """Calculate metrics at specific time - REGIME 1 (no lookahead)"""
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
    if idx >= (REGIME1_MA_PERIOD - 1):
        ma_data = btc_df.iloc[idx - (REGIME1_MA_PERIOD - 1):idx + 1]['close'].values
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

def detect_strategy_type(strategy_name):
    """Detect if strategy is LONG or SHORT"""
    name_lower = strategy_name.lower()
    if '_long_' in name_lower or name_lower.endswith('_long'):
        return 'LONG'
    elif '_short_' in name_lower or name_lower.endswith('_short'):
        return 'SHORT'
    return 'LONG'

def load_trades(filepath):
    """Load trades"""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    else:
        raise ValueError(f"File {filepath} missing buy_time column")
    return df

def apply_regime0_filter(df, btc_1d_df, strategy_type):
    """Apply REGIME 0 filter (BTC MA)"""
    df['regime0_pass'] = False
    
    for idx, trade in df.iterrows():
        btc_close, ma_value = get_btc_value_at_trade(btc_1d_df, trade['buy_time'])
        
        if btc_close is None or ma_value is None:
            df.at[idx, 'regime0_pass'] = True
            continue
        
        if strategy_type == 'LONG':
            df.at[idx, 'regime0_pass'] = btc_close > ma_value * LONG_TH
        else:  # SHORT
            df.at[idx, 'regime0_pass'] = btc_close < ma_value * SHORT_TH
    
    return df

def apply_regime1_filter(df, btc_timeframe_df, strategy_name):
    """Apply REGIME 1 filter (Family + Direction)"""
    df['family'] = 'unknown'
    df['direction'] = 'unknown'
    df['regime1_pass'] = False
    
    # Get filter config for this strategy
    filter_config = REGIME1_FILTERS.get(strategy_name, {'family': None, 'direction': None})
    target_families = filter_config.get('family')  # Now a list or None
    target_direction = filter_config.get('direction')
    
    # If no filters configured, pass all
    if target_families is None and target_direction is None:
        df['regime1_pass'] = True
        return df
    
    for idx, trade in df.iterrows():
        metrics = calc_all_metrics_at_time(btc_timeframe_df, trade['buy_time'], LOOKBACK_BARS)
        
        if metrics is None:
            continue
        
        # Classify family
        family = classify_trade_by_family(metrics, FAMILIES)
        df.at[idx, 'family'] = family
        
        # Classify direction
        if not pd.isna(metrics['price_vs_ma_50']):
            direction = 'uptrend' if metrics['price_vs_ma_50'] > 1.0 else 'downtrend'
            df.at[idx, 'direction'] = direction
        else:
            continue
        
        # Check if passes filter
        # Family: None = all families, or list with OR logic
        if target_families is None:
            family_pass = True
        elif isinstance(target_families, list):
            family_pass = family in target_families
        else:
            # Backwards compatibility: single string
            family_pass = family == target_families
        
        # Direction: None = all directions
        direction_pass = (target_direction is None) or (direction == target_direction)
        
        df.at[idx, 'regime1_pass'] = family_pass and direction_pass
    
    return df

def calculate_portfolio_metrics(df_list, initial_capital):
    """Calculate portfolio metrics from list of strategy dataframes"""
    if not df_list:
        return {'num_trades': 0, 'total_profit': 0.0, 'net_gain_pct': 0.0, 'max_dd_pct': 0.0}
    
    all_trades = pd.concat(df_list, ignore_index=True)
    all_trades = all_trades.sort_values('buy_time').reset_index(drop=True)
    
    if len(all_trades) == 0:
        return {'num_trades': 0, 'total_profit': 0.0, 'net_gain_pct': 0.0, 'max_dd_pct': 0.0}
    
    total_capital = initial_capital * len(df_list)
    
    all_trades['cumulative_profit'] = all_trades['profit'].cumsum()
    all_trades['balance'] = total_capital + all_trades['cumulative_profit']
    
    final_balance = all_trades['balance'].iloc[-1]
    net_gain_pct = (final_balance - total_capital) / total_capital * 100
    
    cummax = all_trades['balance'].cummax()
    drawdown_pct = ((all_trades['balance'] - cummax) / cummax * 100)
    max_dd_pct = drawdown_pct.min()
    
    return {
        'num_trades': len(all_trades),
        'total_profit': all_trades['profit'].sum(),
        'net_gain_pct': net_gain_pct,
        'max_dd_pct': max_dd_pct
    }

def main():
    print("=" * 100)
    print("REGIME LAYERS COMPARISON")
    print("=" * 100)
    
    print(f"\nConfiguration:")
    print(f"  Trades folder: {TRADES_FOLDER}")
    print(f"  OHLC folder:   {OHLC_FOLDER}")
    print(f"  BTC 1D file:   {BTC_FILE}")
    print(f"  REGIME 0:      MA{MA_PERIOD} (LONG: {LONG_TH}, SHORT: {SHORT_TH})")
    print(f"  REGIME 1:      MA{REGIME1_MA_PERIOD} + Family/Direction filters")
    
    # Load BTC 1D for REGIME 0
    print("\n📂 Loading BTC 1D data...")
    btc_1d_df = load_btc_1d()
    print(f"✅ Loaded {len(btc_1d_df)} daily bars")
    
    # Find all trades files
    pattern = str(Path(TRADES_FOLDER) / 'all_trades_*.xlsx')
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No trades files found in {TRADES_FOLDER}")
        return
    
    print(f"\n📂 Found {len(files)} strategy files")
    
    # Process each strategy
    print("\n🔍 Processing strategies...")
    
    baseline_dfs = []
    regime0_dfs = []
    regime1_dfs = []
    both_dfs = []
    
    for filepath in files:
        strategy = Path(filepath).stem.replace('all_trades_', '')
        strategy_type = detect_strategy_type(strategy)
        timeframe = extract_timeframe(Path(filepath).name)
        
        print(f"   Processing {strategy} [{timeframe}]...")
        
        # Load trades
        df = load_trades(filepath)
        
        # Load BTC timeframe data for REGIME 1
        btc_timeframe_df = load_btc_for_timeframe(OHLC_FOLDER, timeframe)
        
        # Apply filters
        df = apply_regime0_filter(df, btc_1d_df, strategy_type)
        df = apply_regime1_filter(df, btc_timeframe_df, strategy)
        
        # Scenario 1: BASELINE (all trades)
        baseline_dfs.append(df[['buy_time', 'profit']].copy())
        
        # Scenario 2: REGIME 0 only
        regime0_dfs.append(df[df['regime0_pass']][['buy_time', 'profit']].copy())
        
        # Scenario 3: REGIME 1 only
        regime1_dfs.append(df[df['regime1_pass']][['buy_time', 'profit']].copy())
        
        # Scenario 4: BOTH (AND logic)
        both_dfs.append(df[df['regime0_pass'] & df['regime1_pass']][['buy_time', 'profit']].copy())
    
    # Calculate portfolio metrics for each scenario
    num_strategies = len(files)
    
    baseline_metrics = calculate_portfolio_metrics(baseline_dfs, INITIAL_CAPITAL)
    regime0_metrics = calculate_portfolio_metrics(regime0_dfs, INITIAL_CAPITAL)
    regime1_metrics = calculate_portfolio_metrics(regime1_dfs, INITIAL_CAPITAL)
    both_metrics = calculate_portfolio_metrics(both_dfs, INITIAL_CAPITAL)
    
    # ==========================================================================
    # STRATEGY-BY-STRATEGY COMPARISON
    # ==========================================================================
    print("\n" + "=" * 100)
    print("STRATEGY-BY-STRATEGY COMPARISON")
    print("=" * 100)
    
    # Header
    print(f"\n{'Strategy':<30} {'Type':<8} {'REGIME 0':>12} {'REGIME 1':>12} {'BOTH':>12}")
    print("-" * 100)
    
    for i, filepath in enumerate(files):
        strategy = Path(filepath).stem.replace('all_trades_', '')
        strategy_type = detect_strategy_type(strategy)
        
        # Get metrics for each scenario
        baseline_profit = baseline_dfs[i]['profit'].sum() if len(baseline_dfs[i]) > 0 else 0
        regime0_profit = regime0_dfs[i]['profit'].sum() if len(regime0_dfs[i]) > 0 else 0
        regime1_profit = regime1_dfs[i]['profit'].sum() if len(regime1_dfs[i]) > 0 else 0
        both_profit = both_dfs[i]['profit'].sum() if len(both_dfs[i]) > 0 else 0
        
        # Calculate % changes vs baseline
        r0_change = ((regime0_profit - baseline_profit) / abs(baseline_profit) * 100) if baseline_profit != 0 else 0
        r1_change = ((regime1_profit - baseline_profit) / abs(baseline_profit) * 100) if baseline_profit != 0 else 0
        both_change = ((both_profit - baseline_profit) / abs(baseline_profit) * 100) if baseline_profit != 0 else 0
        
        # Format values with fixed width
        r0_str = f"{r0_change:+7.1f}%".replace('.', ',')
        r1_str = f"{r1_change:+7.1f}%".replace('.', ',')
        both_str = f"{both_change:+7.1f}%".replace('.', ',')
        
        # Apply colors
        if r0_change > 5:
            r0_final = f"\033[92m{r0_str}\033[0m"
        elif r0_change < -5:
            r0_final = f"\033[91m{r0_str}\033[0m"
        else:
            r0_final = r0_str
        
        if r1_change > 5:
            r1_final = f"\033[92m{r1_str}\033[0m"
        elif r1_change < -5:
            r1_final = f"\033[91m{r1_str}\033[0m"
        else:
            r1_final = r1_str
        
        if both_change > 5:
            both_final = f"\033[92m{both_str}\033[0m"
        elif both_change < -5:
            both_final = f"\033[91m{both_str}\033[0m"
        else:
            both_final = both_str
        
        # Manual padding to account for ANSI codes
        r0_padded = ' ' * (12 - len(r0_str)) + r0_final
        r1_padded = ' ' * (12 - len(r1_str)) + r1_final
        both_padded = ' ' * (12 - len(both_str)) + both_final
        
        print(f"{strategy:<30} {strategy_type:<8} {r0_padded} {r1_padded} {both_padded}")
    
    print("-" * 100)
    
    # ==========================================================================
    # PRINT SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 120)
    print("LAYER COMPARISON - GLOBAL PORTFOLIO")
    print("=" * 120)
    
    print(f"\n{'Metric':<25} {'BASELINE':>15} {'REGIME 0':>15} {'REGIME 1':>15} {'BOTH (R0+R1)':>15} {'BEST':>15}")
    print("-" * 120)
    
    # Trades
    trades_vals = [baseline_metrics['num_trades'], regime0_metrics['num_trades'], 
                   regime1_metrics['num_trades'], both_metrics['num_trades']]
    best_trades_idx = trades_vals.index(max(trades_vals))
    best_trades = ['BASELINE', 'REGIME 0', 'REGIME 1', 'BOTH'][best_trades_idx]
    
    t_base = f"{baseline_metrics['num_trades']:,}".replace(',', '.')
    t_r0 = f"{regime0_metrics['num_trades']:,}".replace(',', '.')
    t_r1 = f"{regime1_metrics['num_trades']:,}".replace(',', '.')
    t_both = f"{both_metrics['num_trades']:,}".replace(',', '.')
    
    print(f"{'Trades':<25} {t_base:>15} {t_r0:>15} {t_r1:>15} {t_both:>15} {best_trades:>15}")
    
    # Total Profit
    profit_vals = [baseline_metrics['total_profit'], regime0_metrics['total_profit'],
                   regime1_metrics['total_profit'], both_metrics['total_profit']]
    best_profit_idx = profit_vals.index(max(profit_vals))
    best_profit = ['BASELINE', 'REGIME 0', 'REGIME 1', 'BOTH'][best_profit_idx]
    
    p_base = f"{baseline_metrics['total_profit']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    p_r0 = f"{regime0_metrics['total_profit']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    p_r1 = f"{regime1_metrics['total_profit']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    p_both = f"{both_metrics['total_profit']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    
    print(f"{'Total Profit':<25} {p_base:>15} {p_r0:>15} {p_r1:>15} {p_both:>15} {best_profit:>15}")
    
    # Net Gain %
    gain_vals = [baseline_metrics['net_gain_pct'], regime0_metrics['net_gain_pct'],
                 regime1_metrics['net_gain_pct'], both_metrics['net_gain_pct']]
    best_gain_idx = gain_vals.index(max(gain_vals))
    best_gain = ['BASELINE', 'REGIME 0', 'REGIME 1', 'BOTH'][best_gain_idx]
    
    g_base = f"{baseline_metrics['net_gain_pct']:.2f}".replace('.', ',') + '%'
    g_r0 = f"{regime0_metrics['net_gain_pct']:.2f}".replace('.', ',') + '%'
    g_r1 = f"{regime1_metrics['net_gain_pct']:.2f}".replace('.', ',') + '%'
    g_both = f"{both_metrics['net_gain_pct']:.2f}".replace('.', ',') + '%'
    
    print(f"{'Net Gain %':<25} {g_base:>15} {g_r0:>15} {g_r1:>15} {g_both:>15} {best_gain:>15}")
    
    # Max DD % (best = closest to 0, i.e., highest value since they're negative)
    dd_vals = [baseline_metrics['max_dd_pct'], regime0_metrics['max_dd_pct'],
               regime1_metrics['max_dd_pct'], both_metrics['max_dd_pct']]
    best_dd_idx = dd_vals.index(max(dd_vals))
    best_dd = ['BASELINE', 'REGIME 0', 'REGIME 1', 'BOTH'][best_dd_idx]
    
    d_base = f"{baseline_metrics['max_dd_pct']:.2f}".replace('.', ',') + '%'
    d_r0 = f"{regime0_metrics['max_dd_pct']:.2f}".replace('.', ',') + '%'
    d_r1 = f"{regime1_metrics['max_dd_pct']:.2f}".replace('.', ',') + '%'
    d_both = f"{both_metrics['max_dd_pct']:.2f}".replace('.', ',') + '%'
    
    print(f"{'Max Drawdown %':<25} {d_base:>15} {d_r0:>15} {d_r1:>15} {d_both:>15} {best_dd:>15}")
    
    print("-" * 120)
    
    # Improvements vs BASELINE - TABLE FORMAT (regimes in columns)
    print("\nIMPROVEMENT vs BASELINE:")
    print(f"\n{'Improvement':<25} {'REGIME 0':>15} {'REGIME 1':>15} {'BOTH (R0+R1)':>15}")
    print("-" * 75)
    
    # Net Gain Change (absolute difference in percentage points)
    r0_gain_change = regime0_metrics['net_gain_pct'] - baseline_metrics['net_gain_pct']
    r1_gain_change = regime1_metrics['net_gain_pct'] - baseline_metrics['net_gain_pct']
    both_gain_change = both_metrics['net_gain_pct'] - baseline_metrics['net_gain_pct']
    
    r0_gain_str = f"{r0_gain_change:+.1f}".replace('.', ',') + '%'
    r1_gain_str = f"{r1_gain_change:+.1f}".replace('.', ',') + '%'
    both_gain_str = f"{both_gain_change:+.1f}".replace('.', ',') + '%'
    
    print(f"{'Net Gain Change':<25} {r0_gain_str:>15} {r1_gain_str:>15} {both_gain_str:>15}")
    
    # Max DD Change (absolute difference in percentage points)
    r0_dd_change = regime0_metrics['max_dd_pct'] - baseline_metrics['max_dd_pct']
    r1_dd_change = regime1_metrics['max_dd_pct'] - baseline_metrics['max_dd_pct']
    both_dd_change = both_metrics['max_dd_pct'] - baseline_metrics['max_dd_pct']
    
    r0_dd_str = f"{r0_dd_change:+.1f}".replace('.', ',') + '%'
    r1_dd_str = f"{r1_dd_change:+.1f}".replace('.', ',') + '%'
    both_dd_str = f"{both_dd_change:+.1f}".replace('.', ',') + '%'
    
    print(f"{'Max DD Change':<25} {r0_dd_str:>15} {r1_dd_str:>15} {both_dd_str:>15}")
    
    print("-" * 75)
    
    print("=" * 120)


if __name__ == "__main__":
    main()