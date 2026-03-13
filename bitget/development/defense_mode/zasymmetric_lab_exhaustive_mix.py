#!/usr/bin/env python3
"""
3-Phase BTC Filter Optimization:
Phase 1: Find best MA combination (400 combos)
Phase 2: Find best Hurst combination (400 combos)
Phase 3: Compare winners + hybrid
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
    
    # Calculate all MAs
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    return df


def calculate_hurst(prices, window=10):
    """Calculate Hurst exponent using R/S analysis"""
    if len(prices) < window:
        return 0.5
    
    prices = np.array(prices[-window:])
    
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    
    if len(log_returns) == 0:
        return 0.5
    
    # Mean and std
    mean_return = np.mean(log_returns)
    std_return = np.std(log_returns)
    
    if std_return == 0:
        return 0.5
    
    # Cumulative deviations
    cumsum = np.cumsum(log_returns - mean_return)
    
    # Range
    R = np.max(cumsum) - np.min(cumsum)
    
    # Rescaled range
    RS = R / std_return if std_return > 0 else 0
    
    # Hurst exponent approximation
    if RS > 0:
        hurst = np.log(RS) / np.log(len(log_returns))
    else:
        hurst = 0.5
    
    # Clamp between 0 and 1
    hurst = max(0.0, min(1.0, hurst))
    
    return hurst


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


# =============================================================================
# PHASE 1: MA RULES
# =============================================================================

def get_btc_ma_regime(btc_df, trade_time, ma_type, threshold):
    """Get BTC regime using MA method"""
    closed_candles = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed_candles) == 0:
        return None
    
    last_candle = closed_candles.iloc[-1]
    
    if pd.isna(last_candle[ma_type]):
        return None
    
    btc_close = last_candle['close']
    ma_value = last_candle[ma_type]
    ma_threshold = ma_value * threshold
    
    if btc_close > ma_threshold:
        return 'LONGS'
    elif btc_close < ma_threshold:
        return 'SHORTS'
    else:
        return 'INACTIVE'


def evaluate_ma_combination(df_trades, btc_df, long_ma, long_th, short_ma, short_th):
    """Evaluate MA combination"""
    results = {}
    
    for direction in ['LONG', 'SHORT']:
        df_dir = df_trades[df_trades['position_type'] == direction].copy()
        
        filtered_profits = []
        
        for idx, trade in df_dir.iterrows():
            profit = trade['profit']
            
            if direction == 'LONG':
                regime = get_btc_ma_regime(btc_df, trade['buy_time'], long_ma, long_th)
                if regime == 'LONGS':
                    filtered_profits.append(profit)
            else:  # SHORT
                regime = get_btc_ma_regime(btc_df, trade['buy_time'], short_ma, short_th)
                if regime == 'SHORTS':
                    filtered_profits.append(profit)
        
        results[direction] = {
            'trades': len(filtered_profits),
            'profit': sum(filtered_profits),
            'wr': sum(1 for p in filtered_profits if p > 0) / len(filtered_profits) * 100 if len(filtered_profits) > 0 else 0,
            'dd': calculate_max_dd(filtered_profits)
        }
    
    combined_profit = results['LONG']['profit'] + results['SHORT']['profit']
    combined_trades = results['LONG']['trades'] + results['SHORT']['trades']
    
    return results, combined_profit, combined_trades


# =============================================================================
# PHASE 2: HURST RULES
# =============================================================================

def get_btc_hurst_regime(btc_df, trade_time, hurst_th, slope_th):
    """Get BTC regime using Hurst method"""
    closed_candles = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed_candles) < 10:
        return None
    
    # Get last 10 closes
    recent_closes = closed_candles['close'].values[-10:]
    
    # Calculate Hurst
    hurst = calculate_hurst(recent_closes, window=10)
    
    # Calculate slope (5-day change)
    if len(recent_closes) >= 5:
        slope = (recent_closes[-1] - recent_closes[-5]) / recent_closes[-5]
    else:
        slope = 0
    
    # Classify
    if hurst > hurst_th:
        if slope > slope_th:
            return 'LONGS'
        elif slope < -slope_th:
            return 'SHORTS'
        else:
            return 'INACTIVE'
    else:
        return 'INACTIVE'


def evaluate_hurst_combination(df_trades, btc_df, long_hurst_th, long_slope_th, short_hurst_th, short_slope_th):
    """Evaluate Hurst combination"""
    results = {}
    
    for direction in ['LONG', 'SHORT']:
        df_dir = df_trades[df_trades['position_type'] == direction].copy()
        
        filtered_profits = []
        
        for idx, trade in df_dir.iterrows():
            profit = trade['profit']
            
            if direction == 'LONG':
                regime = get_btc_hurst_regime(btc_df, trade['buy_time'], long_hurst_th, long_slope_th)
                if regime == 'LONGS':
                    filtered_profits.append(profit)
            else:  # SHORT
                regime = get_btc_hurst_regime(btc_df, trade['buy_time'], short_hurst_th, short_slope_th)
                if regime == 'SHORTS':
                    filtered_profits.append(profit)
        
        results[direction] = {
            'trades': len(filtered_profits),
            'profit': sum(filtered_profits),
            'wr': sum(1 for p in filtered_profits if p > 0) / len(filtered_profits) * 100 if len(filtered_profits) > 0 else 0,
            'dd': calculate_max_dd(filtered_profits)
        }
    
    combined_profit = results['LONG']['profit'] + results['SHORT']['profit']
    combined_trades = results['LONG']['trades'] + results['SHORT']['trades']
    
    return results, combined_profit, combined_trades


# =============================================================================
# PHASE 3: HYBRID (BEST MA + BEST HURST)
# =============================================================================

def evaluate_hybrid_combination(df_trades, btc_df, best_ma_config, best_hurst_config):
    """Evaluate hybrid: MA AND Hurst must both agree"""
    results = {}
    
    long_ma, long_ma_th = best_ma_config['long']
    short_ma, short_ma_th = best_ma_config['short']
    long_hurst_th, long_slope_th = best_hurst_config['long']
    short_hurst_th, short_slope_th = best_hurst_config['short']
    
    for direction in ['LONG', 'SHORT']:
        df_dir = df_trades[df_trades['position_type'] == direction].copy()
        
        filtered_profits = []
        
        for idx, trade in df_dir.iterrows():
            profit = trade['profit']
            
            if direction == 'LONG':
                ma_regime = get_btc_ma_regime(btc_df, trade['buy_time'], long_ma, long_ma_th)
                hurst_regime = get_btc_hurst_regime(btc_df, trade['buy_time'], long_hurst_th, long_slope_th)
                
                if ma_regime == 'LONGS' and hurst_regime == 'LONGS':
                    filtered_profits.append(profit)
            else:  # SHORT
                ma_regime = get_btc_ma_regime(btc_df, trade['buy_time'], short_ma, short_ma_th)
                hurst_regime = get_btc_hurst_regime(btc_df, trade['buy_time'], short_hurst_th, short_slope_th)
                
                if ma_regime == 'SHORTS' and hurst_regime == 'SHORTS':
                    filtered_profits.append(profit)
        
        results[direction] = {
            'trades': len(filtered_profits),
            'profit': sum(filtered_profits),
            'wr': sum(1 for p in filtered_profits if p > 0) / len(filtered_profits) * 100 if len(filtered_profits) > 0 else 0,
            'dd': calculate_max_dd(filtered_profits)
        }
    
    combined_profit = results['LONG']['profit'] + results['SHORT']['profit']
    combined_trades = results['LONG']['trades'] + results['SHORT']['trades']
    
    return results, combined_profit, combined_trades


def main():
    print("="*110)
    print("3-PHASE BTC FILTER OPTIMIZATION")
    print("="*110)
    
    # Load data
    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc_1d()
    print(f"✅ Loaded {len(btc_df)} daily bars")
    
    print("\n📂 Loading LAB trades...")
    df_lab = load_all_lab_trades()
    print(f"✅ Loaded {len(df_lab)} LAB trades")
    
    long_count = len(df_lab[df_lab['position_type'] == 'LONG'])
    short_count = len(df_lab[df_lab['position_type'] == 'SHORT'])
    print(f"   LONG:  {long_count} trades")
    print(f"   SHORT: {short_count} trades")
    
    # Calculate baseline
    baseline_long_profit = df_lab[df_lab['position_type'] == 'LONG']['profit'].sum()
    baseline_short_profit = df_lab[df_lab['position_type'] == 'SHORT']['profit'].sum()
    baseline_total_profit = baseline_long_profit + baseline_short_profit
    baseline_total_wr = (df_lab['profit'] > 0).mean() * 100
    baseline_dd = calculate_max_dd(df_lab['profit'].tolist())
    
    print(f"\n📊 Baseline (no filter):")
    print(f"   Profit: ${baseline_total_profit:,.2f}")
    print(f"   WR: {baseline_total_wr:.1f}%")
    print(f"   DD: {baseline_dd:.1f}%")
    
    # =============================================================================
    # PHASE 1: FIND BEST MA COMBINATION
    # =============================================================================
    print("\n" + "="*110)
    print("PHASE 1: TESTING MA RULES")
    print("="*110)
    
    ma_rules = []
    for ma_type in ['ma5', 'ma10', 'ma20', 'ma50']:
        for threshold in [0.95, 0.98, 1.00, 1.02, 1.05]:
            ma_rules.append((ma_type, threshold))
    
    print(f"\n🔍 Testing {len(ma_rules)} LONG rules × {len(ma_rules)} SHORT rules = {len(ma_rules)**2} combinations...")
    
    best_ma_combo = None
    best_ma_profit = -float('inf')
    total_combos = len(ma_rules) ** 2
    current = 0
    
    for long_ma, long_th in ma_rules:
        for short_ma, short_th in ma_rules:
            current += 1
            print(f"   Progress: {current}/{total_combos} ({current/total_combos*100:.1f}%)...", end='\r')
            
            results, combined_profit, combined_trades = evaluate_ma_combination(
                df_lab, btc_df, long_ma, long_th, short_ma, short_th
            )
            
            if combined_profit > best_ma_profit:
                best_ma_profit = combined_profit
                best_ma_combo = {
                    'long': (long_ma, long_th),
                    'short': (short_ma, short_th),
                    'results': results,
                    'profit': combined_profit,
                    'trades': combined_trades
                }
    
    print()
    
    long_ma, long_ma_th = best_ma_combo['long']
    short_ma, short_ma_th = best_ma_combo['short']
    
    print(f"\n✅ Best MA combination:")
    print(f"   LONG:  BTC > {long_ma.upper()}*{long_ma_th:.2f}")
    print(f"   SHORT: BTC < {short_ma.upper()}*{short_ma_th:.2f}")
    print(f"   Profit: ${best_ma_combo['profit']:,.2f}")
    print(f"   Trades: {best_ma_combo['trades']}")
    
    long_r = best_ma_combo['results']['LONG']
    short_r = best_ma_combo['results']['SHORT']
    combined_wr = (long_r['wr'] * long_r['trades'] + short_r['wr'] * short_r['trades']) / best_ma_combo['trades'] if best_ma_combo['trades'] > 0 else 0
    combined_dd = (abs(long_r['dd']) * abs(long_r['profit']) + abs(short_r['dd']) * abs(short_r['profit'])) / (abs(long_r['profit']) + abs(short_r['profit'])) if (abs(long_r['profit']) + abs(short_r['profit'])) > 0 else 0
    combined_dd = -combined_dd
    
    print(f"   WR: {combined_wr:.1f}%")
    print(f"   DD: {combined_dd:.1f}%")
    
    # =============================================================================
    # PHASE 2: FIND BEST HURST COMBINATION
    # =============================================================================
    print("\n" + "="*110)
    print("PHASE 2: TESTING HURST RULES")
    print("="*110)
    
    hurst_rules = []
    for hurst_th in [0.48, 0.50, 0.52, 0.55, 0.58]:
        for slope_th in [0.01, 0.02, 0.03, 0.04]:
            hurst_rules.append((hurst_th, slope_th))
    
    print(f"\n🔍 Testing {len(hurst_rules)} LONG rules × {len(hurst_rules)} SHORT rules = {len(hurst_rules)**2} combinations...")
    
    best_hurst_combo = None
    best_hurst_profit = -float('inf')
    total_combos = len(hurst_rules) ** 2
    current = 0
    
    for long_hurst_th, long_slope_th in hurst_rules:
        for short_hurst_th, short_slope_th in hurst_rules:
            current += 1
            print(f"   Progress: {current}/{total_combos} ({current/total_combos*100:.1f}%)...", end='\r')
            
            results, combined_profit, combined_trades = evaluate_hurst_combination(
                df_lab, btc_df, long_hurst_th, long_slope_th, short_hurst_th, short_slope_th
            )
            
            if combined_profit > best_hurst_profit:
                best_hurst_profit = combined_profit
                best_hurst_combo = {
                    'long': (long_hurst_th, long_slope_th),
                    'short': (short_hurst_th, short_slope_th),
                    'results': results,
                    'profit': combined_profit,
                    'trades': combined_trades
                }
    
    print()
    
    long_hurst_th, long_slope_th = best_hurst_combo['long']
    short_hurst_th, short_slope_th = best_hurst_combo['short']
    
    print(f"\n✅ Best Hurst combination:")
    print(f"   LONG:  Hurst>{long_hurst_th:.2f} + slope>{long_slope_th:.2f}")
    print(f"   SHORT: Hurst>{short_hurst_th:.2f} + slope<-{short_slope_th:.2f}")
    print(f"   Profit: ${best_hurst_combo['profit']:,.2f}")
    print(f"   Trades: {best_hurst_combo['trades']}")
    
    long_r = best_hurst_combo['results']['LONG']
    short_r = best_hurst_combo['results']['SHORT']
    combined_wr = (long_r['wr'] * long_r['trades'] + short_r['wr'] * short_r['trades']) / best_hurst_combo['trades'] if best_hurst_combo['trades'] > 0 else 0
    combined_dd = (abs(long_r['dd']) * abs(long_r['profit']) + abs(short_r['dd']) * abs(short_r['profit'])) / (abs(long_r['profit']) + abs(short_r['profit'])) if (abs(long_r['profit']) + abs(short_r['profit'])) > 0 else 0
    combined_dd = -combined_dd
    
    print(f"   WR: {combined_wr:.1f}%")
    print(f"   DD: {combined_dd:.1f}%")
    
    # =============================================================================
    # PHASE 3: TEST HYBRID (BEST MA + BEST HURST)
    # =============================================================================
    print("\n" + "="*110)
    print("PHASE 3: TESTING HYBRID (BEST MA + BEST HURST)")
    print("="*110)
    
    print(f"\n🔍 Testing: Best MA AND Best Hurst combined...")
    
    hybrid_results, hybrid_profit, hybrid_trades = evaluate_hybrid_combination(
        df_lab, btc_df, best_ma_combo, best_hurst_combo
    )
    
    long_r = hybrid_results['LONG']
    short_r = hybrid_results['SHORT']
    hybrid_wr = (long_r['wr'] * long_r['trades'] + short_r['wr'] * short_r['trades']) / hybrid_trades if hybrid_trades > 0 else 0
    hybrid_dd = (abs(long_r['dd']) * abs(long_r['profit']) + abs(short_r['dd']) * abs(short_r['profit'])) / (abs(long_r['profit']) + abs(short_r['profit'])) if (abs(long_r['profit']) + abs(short_r['profit'])) > 0 else 0
    hybrid_dd = -hybrid_dd
    
    print(f"\n✅ Hybrid combination:")
    print(f"   LONG:  (BTC > {long_ma.upper()}*{long_ma_th:.2f}) AND (Hurst>{long_hurst_th:.2f} + slope>{long_slope_th:.2f})")
    print(f"   SHORT: (BTC < {short_ma.upper()}*{short_ma_th:.2f}) AND (Hurst>{short_hurst_th:.2f} + slope<-{short_slope_th:.2f})")
    print(f"   Profit: ${hybrid_profit:,.2f}")
    print(f"   Trades: {hybrid_trades}")
    print(f"   WR: {hybrid_wr:.1f}%")
    print(f"   DD: {hybrid_dd:.1f}%")
    
    # =============================================================================
    # FINAL COMPARISON TABLE
    # =============================================================================
    print("\n" + "="*110)
    print("FINAL COMPARISON")
    print("="*110)
    
    print(f"\n{'Method':<30} {'Trades':>10} {'Profit':>15} {'WR%':>10} {'DD%':>10} {'Improvement':>15}")
    print("-"*110)
    
    # Baseline
    print(f"{'Baseline (no filter)':<30} {len(df_lab):>10} ${baseline_total_profit:>13.2f} {baseline_total_wr:>9.1f}% {baseline_dd:>9.1f}% {'—':>15}")
    
    # Best MA
    ma_improvement = (best_ma_combo['profit'] - baseline_total_profit) / abs(baseline_total_profit) * 100
    ma_emoji = "🚀" if ma_improvement > 10 else "📈" if ma_improvement > 0 else "📉"
    ma_wr = (best_ma_combo['results']['LONG']['wr'] * best_ma_combo['results']['LONG']['trades'] + 
             best_ma_combo['results']['SHORT']['wr'] * best_ma_combo['results']['SHORT']['trades']) / best_ma_combo['trades']
    ma_dd = -(abs(best_ma_combo['results']['LONG']['dd']) * abs(best_ma_combo['results']['LONG']['profit']) + 
             abs(best_ma_combo['results']['SHORT']['dd']) * abs(best_ma_combo['results']['SHORT']['profit'])) / (
             abs(best_ma_combo['results']['LONG']['profit']) + abs(best_ma_combo['results']['SHORT']['profit']))
    
    print(f"{'Best MA':<30} {best_ma_combo['trades']:>10} ${best_ma_combo['profit']:>13.2f} {ma_wr:>9.1f}% {ma_dd:>9.1f}% {ma_improvement:>+13.1f}% {ma_emoji}")
    
    # Best Hurst
    hurst_improvement = (best_hurst_combo['profit'] - baseline_total_profit) / abs(baseline_total_profit) * 100
    hurst_emoji = "🚀" if hurst_improvement > 10 else "📈" if hurst_improvement > 0 else "📉"
    hurst_wr = (best_hurst_combo['results']['LONG']['wr'] * best_hurst_combo['results']['LONG']['trades'] + 
                best_hurst_combo['results']['SHORT']['wr'] * best_hurst_combo['results']['SHORT']['trades']) / best_hurst_combo['trades']
    hurst_dd = -(abs(best_hurst_combo['results']['LONG']['dd']) * abs(best_hurst_combo['results']['LONG']['profit']) + 
                abs(best_hurst_combo['results']['SHORT']['dd']) * abs(best_hurst_combo['results']['SHORT']['profit'])) / (
                abs(best_hurst_combo['results']['LONG']['profit']) + abs(best_hurst_combo['results']['SHORT']['profit']))
    
    print(f"{'Best Hurst':<30} {best_hurst_combo['trades']:>10} ${best_hurst_combo['profit']:>13.2f} {hurst_wr:>9.1f}% {hurst_dd:>9.1f}% {hurst_improvement:>+13.1f}% {hurst_emoji}")
    
    # Hybrid
    hybrid_improvement = (hybrid_profit - baseline_total_profit) / abs(baseline_total_profit) * 100
    hybrid_emoji = "🔥" if hybrid_improvement > 20 else "🚀" if hybrid_improvement > 10 else "📈" if hybrid_improvement > 0 else "📉"
    
    print(f"{'Best MA + Best Hurst':<30} {hybrid_trades:>10} ${hybrid_profit:>13.2f} {hybrid_wr:>9.1f}% {hybrid_dd:>9.1f}% {hybrid_improvement:>+13.1f}% {hybrid_emoji}")
    
    print("-"*110)
    
    # Winner
    winner_data = [
        ('Best MA', best_ma_combo['profit']),
        ('Best Hurst', best_hurst_combo['profit']),
        ('Hybrid', hybrid_profit)
    ]
    
    winner_name, winner_profit = max(winner_data, key=lambda x: x[1])
    winner_improvement = (winner_profit - baseline_total_profit) / abs(baseline_total_profit) * 100
    
    print(f"\n💡 WINNER: {winner_name}")
    print(f"   Profit: ${winner_profit:,.2f}")
    print(f"   Improvement: {winner_improvement:+.1f}%")
    
    if winner_name == 'Best MA':
        print(f"\n📋 Production Config:")
        print(f"   LONG:  BTC > {long_ma.upper()}*{long_ma_th:.2f}")
        print(f"   SHORT: BTC < {short_ma.upper()}*{short_ma_th:.2f}")
    elif winner_name == 'Best Hurst':
        print(f"\n📋 Production Config:")
        print(f"   LONG:  Hurst>{long_hurst_th:.2f} + slope>{long_slope_th:.2f}")
        print(f"   SHORT: Hurst>{short_hurst_th:.2f} + slope<-{short_slope_th:.2f}")
    else:  # Hybrid
        print(f"\n📋 Production Config:")
        print(f"   LONG:  (BTC > {long_ma.upper()}*{long_ma_th:.2f}) AND (Hurst>{long_hurst_th:.2f} + slope>{long_slope_th:.2f})")
        print(f"   SHORT: (BTC < {short_ma.upper()}*{short_ma_th:.2f}) AND (Hurst>{short_hurst_th:.2f} + slope<-{short_slope_th:.2f})")
    
    print("\n" + "="*110)


if __name__ == "__main__":
    main()