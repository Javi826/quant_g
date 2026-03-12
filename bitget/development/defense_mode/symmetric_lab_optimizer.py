#!/usr/bin/env python3
"""
defense_mode/symmetric_lab_optimizer.py

Optimize BTC 1D Filter Rules on LAB, Validate on LIVE
Tests multiple MA thresholds to find best profit configuration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob


def load_btc_1d():
    """Load BTC 1D data"""
    btc_file = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/BTCUSDT_1Dutc.parquet')
    
    if not btc_file.exists():
        raise FileNotFoundError(f"BTC 1D file not found: {btc_file}")
    
    df = pd.read_parquet(btc_file)
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Calculate MAs
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    return df


def get_btc_regime_at_time(btc_df, trade_time):
    """Get BTC price and MAs at trade time"""
    closed_candles = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed_candles) == 0:
        return None
    
    last_candle = closed_candles.iloc[-1]
    
    # Check if all MAs are available
    if pd.isna(last_candle['ma5']) or pd.isna(last_candle['ma10']) or pd.isna(last_candle['ma20']) or pd.isna(last_candle['ma50']):
        return None
    
    return {
        'close': last_candle['close'],
        'ma5': last_candle['ma5'],
        'ma10': last_candle['ma10'],
        'ma20': last_candle['ma20'],
        'ma50': last_candle['ma50']
    }


def should_allow_trade(direction, btc_regime, ma_type, threshold):
    """
    Decide if trade should be allowed
    ma_type: 'ma20' or 'ma50'
    threshold: multiplier (e.g., 1.0 = exact MA, 0.98 = 2% below, 1.02 = 2% above)
    """
    if btc_regime is None:
        return True
    
    ma_value = btc_regime[ma_type] * threshold
    
    if btc_regime['close'] > ma_value:
        # Bullish - only LONG
        return direction == 'LONG'
    else:
        # Bearish - only SHORT
        return direction == 'SHORT'


def load_all_lab_trades():
    """Load all lab trades"""
    lab_folder = Path('/home/javi/projects/quant/quant_g/bitget/development/brief_trades')
    files = glob(str(lab_folder / 'all_trades_*.xlsx'))
    
    all_trades = []
    for filepath in files:
        df = pd.read_excel(filepath)
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        df['buy_time'] = pd.to_datetime(df['buy_time'])
        all_trades.append(df)
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined.sort_values('buy_time').reset_index(drop=True)


def load_live_trades(filepath):
    """Load live trades from enriched file"""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.upper()
    
    ts_col = None
    for col in ['OPEN_AT', 'BUY_TIME', 'ENTRY_TIME']:
        if col in df.columns:
            ts_col = col
            break
    
    if not ts_col:
        raise ValueError(f"No timestamp column found in {filepath}")
    
    df[ts_col] = pd.to_datetime(df[ts_col])
    df.rename(columns={ts_col: 'open_time'}, inplace=True)
    
    return df.sort_values('open_time').reset_index(drop=True)


def calculate_max_dd(trades_list):
    """Calculate max drawdown from list of profits"""
    if len(trades_list) == 0:
        return 0, 0
    
    cumulative = 0
    peak = 0
    max_dd = 0
    
    for profit in trades_list:
        cumulative += profit
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    
    dd_pct = (max_dd / peak * 100) if peak > 0 else 0
    
    return -max_dd if max_dd > 0 else 0, -dd_pct if dd_pct > 0 else 0


def evaluate_rule(df_trades, btc_df, ma_type, threshold):
    """Evaluate a single rule on trades"""
    # Sort by time first
    df_trades = df_trades.sort_values('buy_time').reset_index(drop=True)
    
    all_trades_data = []
    filtered_trades_data = []
    
    for idx, trade in df_trades.iterrows():
        direction = trade['position_type']
        profit = trade['profit']
        trade_time = trade['buy_time']
        
        # All trades
        all_trades_data.append({
            'profit': profit,
            'is_winner': profit > 0
        })
        
        btc_regime = get_btc_regime_at_time(btc_df, trade_time)
        
        if should_allow_trade(direction, btc_regime, ma_type, threshold):
            # Filtered trades
            filtered_trades_data.append({
                'profit': profit,
                'is_winner': profit > 0
            })
    
    # Calculate metrics for all trades
    total_trades = len(all_trades_data)
    total_profit = sum(t['profit'] for t in all_trades_data)
    total_winners = sum(1 for t in all_trades_data if t['is_winner'])
    total_wr = (total_winners / total_trades * 100) if total_trades > 0 else 0
    total_dd, total_dd_pct = calculate_max_dd([t['profit'] for t in all_trades_data])
    
    # Calculate metrics for filtered trades
    filtered_trades = len(filtered_trades_data)
    filtered_profit = sum(t['profit'] for t in filtered_trades_data)
    filtered_winners = sum(1 for t in filtered_trades_data if t['is_winner'])
    filtered_wr = (filtered_winners / filtered_trades * 100) if filtered_trades > 0 else 0
    filtered_dd, filtered_dd_pct = calculate_max_dd([t['profit'] for t in filtered_trades_data])
    
    return {
        'total_trades': total_trades,
        'total_profit': total_profit,
        'total_wr': total_wr,
        'total_dd': total_dd,
        'total_dd_pct': total_dd_pct,
        'filtered_trades': filtered_trades,
        'filtered_profit': filtered_profit,
        'filtered_wr': filtered_wr,
        'filtered_dd': filtered_dd,
        'filtered_dd_pct': filtered_dd_pct,
        'profit_change': filtered_profit - total_profit,
        'wr_change': filtered_wr - total_wr,
        'dd_change': filtered_dd - total_dd,
        'dd_pct_change': filtered_dd_pct - total_dd_pct
    }


def main():
    print("="*100)
    print("BTC 1D FILTER OPTIMIZATION - LAB ONLY")
    print("="*100)
    
    # Load BTC 1D
    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc_1d()
    print(f"✅ Loaded {len(btc_df)} daily bars")
    
    # Load LAB trades
    print("\n📂 Loading LAB trades...")
    df_lab = load_all_lab_trades()
    print(f"✅ Loaded {len(df_lab)} LAB trades")
    
    # Define rules to test
    rules = []
    
    # MA5 rules
    for threshold in [0.95, 0.98, 1.00, 1.02, 1.05]:
        rules.append(('ma5', threshold))
    
    # MA10 rules
    for threshold in [0.95, 0.98, 1.00, 1.02, 1.05]:
        rules.append(('ma10', threshold))
    
    # MA20 rules
    for threshold in [0.95, 0.98, 1.00, 1.02, 1.05]:
        rules.append(('ma20', threshold))
    
    # MA50 rules
    for threshold in [0.95, 0.98, 1.00, 1.02, 1.05]:
        rules.append(('ma50', threshold))
    
    print(f"\n🔍 Testing {len(rules)} rules on LAB...")
    
    # Evaluate all rules on LAB
    lab_results = []
    
    for ma_type, threshold in rules:
        print(f"   Testing {ma_type.upper()} * {threshold:.2f}...", end='\r')
        result = evaluate_rule(df_lab, btc_df, ma_type, threshold)
        result['ma_type'] = ma_type
        result['threshold'] = threshold
        lab_results.append(result)
    
    print()
    
    # Sort by profit (best first)
    lab_results = sorted(lab_results, key=lambda x: x['filtered_profit'], reverse=True)
    
    # Display results
    print("\n" + "="*130)
    print("LAB RESULTS - ALL RULES (Sorted by AFTER Profit)")
    print("="*130)
    
    print(f"\n{'#':>3} {'Rule':<12} {'Trades':>8} {'After':>8} {'Δ':>7} "
          f"{'WR%':>7} {'After':>7} {'Δ':>6} "
          f"{'Profit':>10} {'After':>10} {'Δ':>10} "
          f"{'DD%':>7} {'After':>7} {'Δ':>6}")
    print("-"*130)
    
    for rank, result in enumerate(lab_results, 1):
        rule_label = f"{result['ma_type'].upper()}{result['threshold']:.2f}"
        
        print(f"{rank:>3} {rule_label:<12} "
              f"{result['total_trades']:>8} {result['filtered_trades']:>8} {result['filtered_trades']-result['total_trades']:>7} "
              f"{result['total_wr']:>6.1f}% {result['filtered_wr']:>6.1f}% {result['wr_change']:>5.1f}p "
              f"{result['total_profit']:>10.2f} {result['filtered_profit']:>10.2f} {result['profit_change']:>10.2f} "
              f"{result['total_dd_pct']:>6.1f}% {result['filtered_dd_pct']:>6.1f}% {result['dd_pct_change']:>5.1f}p")
    
    # Best rule details
    best_rule = lab_results[0]
    
    print("\n" + "="*130)
    print("BEST RULE (Highest AFTER Profit)")
    print("="*130)
    
    print(f"\nRule: BTC > {best_rule['ma_type'].upper()} * {best_rule['threshold']:.2f}")
    print(f"      if True → LONG only, else SHORT only")
    
    print(f"\n{'Metric':<20} {'BEFORE':>15} {'AFTER':>15} {'CHANGE':>15}")
    print("-"*70)
    print(f"{'Trades':<20} {best_rule['total_trades']:>15} {best_rule['filtered_trades']:>15} "
          f"{best_rule['filtered_trades'] - best_rule['total_trades']:>15}")
    print(f"{'Win Rate %':<20} {best_rule['total_wr']:>15.1f} {best_rule['filtered_wr']:>15.1f} "
          f"{best_rule['wr_change']:>+15.1f}")
    print(f"{'Profit':<20} {best_rule['total_profit']:>15.2f} {best_rule['filtered_profit']:>15.2f} "
          f"{best_rule['profit_change']:>+15.2f}")
    print(f"{'Max Drawdown %':<20} {best_rule['total_dd_pct']:>15.1f} {best_rule['filtered_dd_pct']:>15.1f} "
          f"{best_rule['dd_pct_change']:>+15.1f}")
    
    profit_pct_change = (best_rule['profit_change']/abs(best_rule['total_profit'])*100) if best_rule['total_profit'] != 0 else 0
    
    if best_rule['profit_change'] > 0:
        print(f"\n✅ PROFIT IMPROVES: +${best_rule['profit_change']:.2f} ({profit_pct_change:+.1f}%)")
    else:
        print(f"\n⚠️  PROFIT DECREASES: ${best_rule['profit_change']:.2f} ({profit_pct_change:+.1f}%)")
    
    if best_rule['dd_pct_change'] > 0:  # DD improved (less negative %)
        print(f"✅ MAX DD IMPROVES: {abs(best_rule['dd_pct_change']):.1f}pp better")
    elif best_rule['dd_pct_change'] < 0:  # DD worsened (more negative %)
        print(f"❌ MAX DD WORSENS: {abs(best_rule['dd_pct_change']):.1f}pp worse")
    else:
        print(f"➖ MAX DD UNCHANGED")
    
    print("\n" + "="*130)
    print("\n💡 Next step: Test this rule on LIVE data")
    print("="*130)


if __name__ == "__main__":
    main()