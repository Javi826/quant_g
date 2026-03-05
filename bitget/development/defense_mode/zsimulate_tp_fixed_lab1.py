#!/usr/bin/env python3
"""
Optimize Independent Rules for LONG and SHORT in LAB
Find best MA threshold for each direction separately
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


def get_btc_value(btc_df, trade_time, ma_type):
    """Get BTC close and MA at trade time"""
    closed_candles = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed_candles) == 0:
        return None, None
    
    last_candle = closed_candles.iloc[-1]
    
    if pd.isna(last_candle[ma_type]):
        return None, None
    
    return last_candle['close'], last_candle[ma_type]


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


def calculate_max_dd(profits):
    """Calculate max drawdown %"""
    if len(profits) == 0:
        return 0
    
    cumulative = 0
    peak = 0
    max_dd = 0
    
    for profit in profits:
        cumulative += profit
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    
    dd_pct = (max_dd / peak * 100) if peak > 0 else 0
    return -dd_pct if dd_pct > 0 else 0


def evaluate_rule_for_direction(df_trades, btc_df, ma_type, threshold, target_direction):
    """
    Evaluate rule for specific direction
    target_direction: 'LONG' or 'SHORT'
    
    For LONG: if BTC > MA*threshold → allow, else skip
    For SHORT: if BTC < MA*threshold → allow, else skip
    """
    df_trades = df_trades.sort_values('buy_time').reset_index(drop=True)
    
    # Filter only trades of target direction
    df_dir = df_trades[df_trades['position_type'] == target_direction].copy()
    
    all_profits = []
    filtered_profits = []
    
    for idx, trade in df_dir.iterrows():
        profit = trade['profit']
        all_profits.append(profit)
        
        btc_close, ma_value = get_btc_value(btc_df, trade['buy_time'], ma_type)
        
        if btc_close is None or ma_value is None:
            filtered_profits.append(profit)
            continue
        
        ma_threshold = ma_value * threshold
        
        # Apply rule based on direction
        if target_direction == 'LONG':
            # LONG: allow if BTC > MA*threshold
            if btc_close > ma_threshold:
                filtered_profits.append(profit)
        else:
            # SHORT: allow if BTC < MA*threshold
            if btc_close < ma_threshold:
                filtered_profits.append(profit)
    
    # Calculate metrics
    total_trades = len(all_profits)
    total_profit = sum(all_profits)
    total_wr = sum(1 for p in all_profits if p > 0) / total_trades * 100 if total_trades > 0 else 0
    total_dd = calculate_max_dd(all_profits)
    
    filtered_trades = len(filtered_profits)
    filtered_profit = sum(filtered_profits)
    filtered_wr = sum(1 for p in filtered_profits if p > 0) / filtered_trades * 100 if filtered_trades > 0 else 0
    filtered_dd = calculate_max_dd(filtered_profits)
    
    return {
        'total_trades': total_trades,
        'total_profit': total_profit,
        'total_wr': total_wr,
        'total_dd': total_dd,
        'filtered_trades': filtered_trades,
        'filtered_profit': filtered_profit,
        'filtered_wr': filtered_wr,
        'filtered_dd': filtered_dd
    }


def main():
    print("="*110)
    print("OPTIMIZE SHORT RULE (LONG fixed at MA5*1.02)")
    print("="*110)
    
    # Load BTC 1D
    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc_1d()
    print(f"✅ Loaded {len(btc_df)} daily bars")
    
    # Load LAB trades
    print("\n📂 Loading LAB trades...")
    df_lab = load_all_lab_trades()
    print(f"✅ Loaded {len(df_lab)} LAB trades")
    
    # Count LONG vs SHORT
    long_trades = len(df_lab[df_lab['position_type'] == 'LONG'])
    short_trades = len(df_lab[df_lab['position_type'] == 'SHORT'])
    print(f"   LONG:  {long_trades} trades")
    print(f"   SHORT: {short_trades} trades")
    
    print(f"\n✅ LONG rule FIXED: BTC > MA5*1.02")
    
    # Evaluate LONG with fixed rule (MA5*1.02)
    long_result = evaluate_rule_for_direction(df_lab, btc_df, 'ma5', 1.02, 'LONG')
    
    print(f"   LONG profit: ${long_result['total_profit']:.2f} → ${long_result['filtered_profit']:.2f} "
          f"({long_result['filtered_profit']-long_result['total_profit']:+.2f})")
    
    # Define SHORT rules to test
    short_rules = []
    for ma_type in ['ma5', 'ma10', 'ma20', 'ma50']:
        for threshold in [0.95, 0.98, 1.00, 1.02, 1.05]:
            short_rules.append((ma_type, threshold))
    
    print(f"\n🔍 Testing {len(short_rules)} rules for SHORT...")
    
    # Evaluate SHORT rules
    short_results = []
    for ma_type, threshold in short_rules:
        result = evaluate_rule_for_direction(df_lab, btc_df, ma_type, threshold, 'SHORT')
        result['ma_type'] = ma_type
        result['threshold'] = threshold
        short_results.append(result)
    
    # Sort by filtered profit
    short_results = sorted(short_results, key=lambda x: x['filtered_profit'], reverse=True)
    
    # Display SHORT results (top 10)
    print("\n" + "="*110)
    print("TOP 10 RULES FOR SHORT")
    print("="*110)
    
    print(f"\n{'#':>3} {'Rule':<12} {'Trades':>8} {'After':>8} {'Δ':>7} "
          f"{'WR%':>7} {'After':>7} {'Δ':>6} "
          f"{'Profit':>10} {'After':>10} {'Δ':>10} "
          f"{'DD%':>7} {'After':>7} {'Δ':>6}")
    print("-"*110)
    
    for rank, r in enumerate(short_results[:10], 1):
        rule_label = f"{r['ma_type'].upper()}{r['threshold']:.2f}"
        
        print(f"{rank:>3} {rule_label:<12} "
              f"{r['total_trades']:>8} {r['filtered_trades']:>8} {r['filtered_trades']-r['total_trades']:>7} "
              f"{r['total_wr']:>6.1f}% {r['filtered_wr']:>6.1f}% {r['filtered_wr']-r['total_wr']:>5.1f}p "
              f"{r['total_profit']:>10.2f} {r['filtered_profit']:>10.2f} {r['filtered_profit']-r['total_profit']:>10.2f} "
              f"{r['total_dd']:>6.1f}% {r['filtered_dd']:>6.1f}% {r['filtered_dd']-r['total_dd']:>5.1f}p")
    
    # Best SHORT rule
    best_short = short_results[0]
    
    print("\n" + "="*110)
    print("BEST COMBINATION")
    print("="*110)
    
    long_improvement = long_result['filtered_profit'] - long_result['total_profit']
    short_improvement = best_short['filtered_profit'] - best_short['total_profit']
    total_improvement = long_improvement + short_improvement
    
    print(f"\nLONG:  BTC > MA5*1.02")
    print(f"  Profit: ${long_result['total_profit']:.2f} → ${long_result['filtered_profit']:.2f} ({long_improvement:+.2f})")
    
    print(f"\nSHORT: BTC < {best_short['ma_type'].upper()}*{best_short['threshold']:.2f}")
    print(f"  Profit: ${best_short['total_profit']:.2f} → ${best_short['filtered_profit']:.2f} ({short_improvement:+.2f})")
    
    print(f"\nCOMBINED IMPROVEMENT: {total_improvement:+.2f}")
    
    print("\n" + "="*110)
    print("\n💡 Next step: Validate this combination on LIVE data")
    print("="*110)


if __name__ == "__main__":
    main()