#!/usr/bin/env python3
"""
market_regime/zasymetric_lab_exhaustive.py
Find Best LONG + SHORT Rule Combination by Testing All Pairs
Tests all 400 combinations (20 LONG × 20 SHORT) on LAB data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

LAB_TRADES_FOLDER = "../brief_trades"
BTC_FILE          = "../data/crypto_2022_OOS/BTCUSDT_1Dutc.parquet"

# MA thresholds to test
MA_TYPES          = ['ma5', 'ma10', 'ma20', 'ma50']
THRESHOLDS        = [0.95, 0.98, 1.00, 1.02, 1.05]

# =============================================================================
def load_btc_1d():
    """Load BTC 1D data"""
    btc_file = Path(BTC_FILE)
    
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
    for ma_type in MA_TYPES:
        period = int(ma_type.replace('ma', ''))
        df[ma_type] = df['close'].rolling(window=period).mean()
    
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
    lab_folder = Path(LAB_TRADES_FOLDER)
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


def evaluate_direction(df_trades, btc_df, ma_type, threshold, direction):
    """
    Evaluate rule for specific direction
    
    For LONG: if BTC > MA*threshold → allow, else skip
    For SHORT: if BTC < MA*threshold → allow, else skip
    """
    df_dir = df_trades[df_trades['position_type'] == direction].copy()
    
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
        
        # Apply rule
        if direction == 'LONG':
            if btc_close > ma_threshold:
                filtered_profits.append(profit)
        else:  # SHORT
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


def print_combination_details(rank, combo):
    """Print detailed results for a combination"""
    long_rule = f"{combo['long_ma'].upper()}*{combo['long_th']:.2f}"
    short_rule = f"{combo['short_ma'].upper()}*{combo['short_th']:.2f}"
    
    lr = combo['long_result']
    sr = combo['short_result']
    
    # Calculate totals
    total_before_trades = lr['total_trades'] + sr['total_trades']
    total_after_trades = lr['filtered_trades'] + sr['filtered_trades']
    total_before_profit = lr['total_profit'] + sr['total_profit']
    total_after_profit = lr['filtered_profit'] + sr['filtered_profit']
    total_before_wr = (lr['total_wr'] * lr['total_trades'] + sr['total_wr'] * sr['total_trades']) / total_before_trades if total_before_trades > 0 else 0
    total_after_wr = (lr['filtered_wr'] * lr['filtered_trades'] + sr['filtered_wr'] * sr['filtered_trades']) / total_after_trades if total_after_trades > 0 else 0
    
    # Weighted DD
    total_before_dd = (abs(lr['total_dd']) * abs(lr['total_profit']) + abs(sr['total_dd']) * abs(sr['total_profit'])) / (abs(lr['total_profit']) + abs(sr['total_profit'])) if (abs(lr['total_profit']) + abs(sr['total_profit'])) > 0 else 0
    total_after_dd = (abs(lr['filtered_dd']) * abs(lr['filtered_profit']) + abs(sr['filtered_dd']) * abs(sr['filtered_profit'])) / (abs(lr['filtered_profit']) + abs(sr['filtered_profit'])) if (abs(lr['filtered_profit']) + abs(sr['filtered_profit'])) > 0 else 0
    total_before_dd = -total_before_dd
    total_after_dd = -total_after_dd
    
    print(f"\n#{rank} {'='*130}")
    print(f"LONG:  BTC > {long_rule}")
    print(f"SHORT: BTC < {short_rule}")
    print("="*110)
    
    # Header
    print(f"\n{'Direction':<10} {'TRADES':<30} {'PROFIT':<45} {'WIN RATE':<20} {'MAX DD':<20}")
    print(f"{'':10} {'Before':<8} {'After':<8} {'Δ':<12} {'Before':<12} {'After':<12} {'Δ':<10} {'%Δ':<8} "
          f"{'Before':<8} {'After':<10} {'Before':<8} {'After':<10}")
    print("-"*110)
    
    # LONG
    long_profit_change = lr['filtered_profit'] - lr['total_profit']
    long_profit_pct = (long_profit_change / abs(lr['total_profit']) * 100) if lr['total_profit'] != 0 else 0
    
    print(f"{'LONG':<10} {lr['total_trades']:<8} {lr['filtered_trades']:<8} {lr['filtered_trades']-lr['total_trades']:<12} "
          f"${lr['total_profit']:<11.2f} ${lr['filtered_profit']:<11.2f} ${long_profit_change:<9.2f} {long_profit_pct:>+6.1f}%  "
          f"{lr['total_wr']:<7.1f}% {lr['filtered_wr']:<9.1f}% "
          f"{lr['total_dd']:<7.1f}% {lr['filtered_dd']:<9.1f}%")
    
    # SHORT
    short_profit_change = sr['filtered_profit'] - sr['total_profit']
    short_profit_pct = (short_profit_change / abs(sr['total_profit']) * 100) if sr['total_profit'] != 0 else 0
    
    print(f"{'SHORT':<10} {sr['total_trades']:<8} {sr['filtered_trades']:<8} {sr['filtered_trades']-sr['total_trades']:<12} "
          f"${sr['total_profit']:<11.2f} ${sr['filtered_profit']:<11.2f} ${short_profit_change:<9.2f} {short_profit_pct:>+6.1f}%  "
          f"{sr['total_wr']:<7.1f}% {sr['filtered_wr']:<9.1f}% "
          f"{sr['total_dd']:<7.1f}% {sr['filtered_dd']:<9.1f}%")
    
    print("-"*110)
    
    # TOTAL
    total_trades_change = total_after_trades - total_before_trades
    total_profit_change = total_after_profit - total_before_profit
    total_profit_pct = (total_profit_change / abs(total_before_profit) * 100) if total_before_profit != 0 else 0
    
    print(f"{'TOTAL':<10} {total_before_trades:<8} {total_after_trades:<8} {total_trades_change:<12} "
          f"${total_before_profit:<11.2f} ${total_after_profit:<11.2f} ${total_profit_change:<9.2f} {total_profit_pct:>+6.1f}%  "
          f"{total_before_wr:<7.1f}% {total_after_wr:<9.1f}% "
          f"{total_before_dd:<7.1f}% {total_after_dd:<9.1f}%")
    
    print(f"\n💰 COMBINED PROFIT: ${combo['combined_profit']:,.2f}")

def print_summary_table(combinations):
    """Print summary table of top 3 combinations"""
    print("\n" + "="*110)
    print("SUMMARY - TOP 3 COMBINATIONS")
    print("="*110)
    
    print(f"\n{'#':>3} {'LONG RULE':<15} {'SHORT RULE':<15} {'TRADES':<12} {'PROFIT':<15} {'WIN RATE':<12} {'MAX DD':<12}")
    print(f"{'':>3} {'':15} {'':15} {'After':>12} {'After':>15} {'After':>12} {'After':>12}")
    print("-"*110)
    
    for rank, combo in enumerate(combinations[:3], 1):
        long_rule = f"{combo['long_ma'].upper()}{combo['long_th']:.2f}"
        short_rule = f"{combo['short_ma'].upper()}{combo['short_th']:.2f}"
        
        lr = combo['long_result']
        sr = combo['short_result']
        
        # Calculate totals
        total_after_trades = lr['filtered_trades'] + sr['filtered_trades']
        total_after_profit = lr['filtered_profit'] + sr['filtered_profit']
        total_after_wr = (lr['filtered_wr'] * lr['filtered_trades'] + sr['filtered_wr'] * sr['filtered_trades']) / total_after_trades if total_after_trades > 0 else 0
        
        # Weighted DD
        total_after_dd = (abs(lr['filtered_dd']) * abs(lr['filtered_profit']) + abs(sr['filtered_dd']) * abs(sr['filtered_profit'])) / (abs(lr['filtered_profit']) + abs(sr['filtered_profit'])) if (abs(lr['filtered_profit']) + abs(sr['filtered_profit'])) > 0 else 0
        total_after_dd = -total_after_dd
        
        print(f"{rank:>3} {long_rule:<15} {short_rule:<15} {total_after_trades:>12} ${total_after_profit:>14.2f} {total_after_wr:>11.1f}% {total_after_dd:>11.1f}%")
    
    print("-"*110)
def main():
    print("="*140)
    print("FIND BEST LONG + SHORT COMBINATION (Testing All Pairs)")
    print("="*140)
    
    # Load BTC 1D
    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc_1d()
    print(f"✅ Loaded {len(btc_df)} daily bars")
    
    # Load LAB trades
    print("\n📂 Loading LAB trades...")
    df_lab = load_all_lab_trades()
    print(f"✅ Loaded {len(df_lab)} LAB trades")
    
    long_count = len(df_lab[df_lab['position_type'] == 'LONG'])
    short_count = len(df_lab[df_lab['position_type'] == 'SHORT'])
    print(f"   LONG:  {long_count} trades")
    print(f"   SHORT: {short_count} trades")
    
    # Define rules
    rules = []
    for ma_type in MA_TYPES:
        for threshold in THRESHOLDS:
            rules.append((ma_type, threshold))
    
    print(f"\n🔍 Testing {len(rules)} LONG rules × {len(rules)} SHORT rules = {len(rules)**2} combinations...")
    
    # Test all combinations
    combinations = []
    total_combos = len(rules) ** 2
    current = 0
    
    for long_ma, long_th in rules:
        for short_ma, short_th in rules:
            current += 1
            print(f"   Progress: {current}/{total_combos} ({current/total_combos*100:.1f}%)...", end='\r')
            
            # Evaluate LONG
            long_result = evaluate_direction(df_lab, btc_df, long_ma, long_th, 'LONG')
            
            # Evaluate SHORT
            short_result = evaluate_direction(df_lab, btc_df, short_ma, short_th, 'SHORT')
            
            # Combined profit
            combined_profit = long_result['filtered_profit'] + short_result['filtered_profit']
            
            combinations.append({
                'long_ma': long_ma,
                'long_th': long_th,
                'short_ma': short_ma,
                'short_th': short_th,
                'long_result': long_result,
                'short_result': short_result,
                'combined_profit': combined_profit
            })
    
    print()
    
    # Sort by combined profit
    combinations = sorted(combinations, key=lambda x: x['combined_profit'], reverse=True)
    
    # Display top 5
    print("\n" + "="*140)
    print("TOP 3 BEST COMBINATIONS (by Total Profit)")
    print("="*140)
    
    for rank, combo in enumerate(combinations[:3], 1):
        print_combination_details(rank, combo)
    
    print_summary_table(combinations)
    
    # Best combination summary
    best = combinations[0]
    
    print("\n" + "="*140)
    print("BEST COMBINATION SUMMARY")
    print("="*140)
    
    print(f"\n✅ LONG:  BTC > {best['long_ma'].upper()}*{best['long_th']:.2f}")
    print(f"✅ SHORT: BTC < {best['short_ma'].upper()}*{best['short_th']:.2f}")
    print(f"\n💰 TOTAL PROFIT: ${best['combined_profit']:,.2f}")
    
    print("\n" + "="*140)
    print("\n💡 Next step: Validate this combination on LIVE data")
    print("="*140)


if __name__ == "__main__":
    main()