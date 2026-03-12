#!/usr/bin/env python3
"""
Analyze REGIME 0 Impact Per Strategy - Simple Version
Shows profit before/after applying asymmetric rule (LONG: MA5*1.02, SHORT: MA5*1.00)
"""

import pandas as pd
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
    df['ma5'] = df['close'].rolling(window=5).mean()
    
    return df


def get_btc_values(btc_df, trade_time):
    """Get BTC close and MA5 at trade time"""
    closed = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed) == 0:
        return None, None
    
    last = closed.iloc[-1]
    
    if pd.isna(last['ma5']):
        return None, None
    
    return last['close'], last['ma5']


def extract_strategy_name(filepath):
    """Extract strategy name from filename"""
    filename = Path(filepath).stem
    if filename.startswith('all_trades_'):
        return filename.replace('all_trades_', '')
    return filename


def load_all_lab_trades():
    """Load all lab trades with strategy column from filename"""
    lab_folder = Path('/home/javi/projects/quant/quant_g/bitget/development/brief_trades')
    files = glob(str(lab_folder / 'all_trades_*.xlsx'))
    
    all_trades = []
    for filepath in files:
        df = pd.read_excel(filepath)
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        df['buy_time'] = pd.to_datetime(df['buy_time'])
        
        strategy_name = extract_strategy_name(filepath)
        df['strategy'] = strategy_name
        
        all_trades.append(df)
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined.sort_values('buy_time').reset_index(drop=True)


def evaluate_strategy(df_strategy, btc_df, long_th, short_th):
    """
    Evaluate strategy with asymmetric rules
    LONG:  BTC > MA5*long_th
    SHORT: BTC < MA5*short_th
    """
    all_profits = []
    filtered_profits = []
    
    for _, trade in df_strategy.iterrows():
        direction = trade['position_type']
        profit = trade['profit']
        
        all_profits.append(profit)
        
        btc_close, ma5 = get_btc_values(btc_df, trade['buy_time'])
        
        if btc_close is None or ma5 is None:
            filtered_profits.append(profit)
            continue
        
        # Apply asymmetric rules
        if direction == 'LONG':
            if btc_close > ma5 * long_th:
                filtered_profits.append(profit)
        else:  # SHORT
            if btc_close < ma5 * short_th:
                filtered_profits.append(profit)
    
    total_profit = sum(all_profits)
    filtered_profit = sum(filtered_profits)
    
    return {
        'before': total_profit,
        'after': filtered_profit,
        'change': filtered_profit - total_profit
    }


def main():
    print("="*100)
    print("REGIME 0 IMPACT ANALYSIS - SIMPLE VIEW")
    print("Rule: LONG: BTC > MA5*1.02  |  SHORT: BTC < MA5*1.00")
    print("="*100)
    
    # Load BTC 1D
    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc_1d()
    print(f"✅ Loaded {len(btc_df)} daily bars")
    
    # Load LAB trades
    print("\n📂 Loading LAB trades...")
    df_lab = load_all_lab_trades()
    print(f"✅ Loaded {len(df_lab)} LAB trades")
    
    # Get unique strategies
    strategies = sorted(df_lab['strategy'].unique())
    print(f"📊 Found {len(strategies)} unique strategies\n")
    
    # Store results
    results = []
    
    # Analyze each strategy
    for strategy in strategies:
        df_strategy = df_lab[df_lab['strategy'] == strategy]
        result = evaluate_strategy(df_strategy, btc_df, 1.02, 1.00)
        
        results.append({
            'strategy': strategy,
            'before': result['before'],
            'after': result['after'],
            'change': result['change']
        })
    
    # Print results
    print("="*100)
    print(f"{'Strategy':<35} {'BEFORE':<15} {'AFTER':<15} {'Δ':<15} {'%Δ':<10}")
    print("-"*100)
    
    for r in sorted(results, key=lambda x: x['change'], reverse=True):
        pct_change = (r['change'] / abs(r['before']) * 100) if r['before'] != 0 else 0
        
        print(f"{r['strategy']:<35} ${r['before']:>12.2f} ${r['after']:>12.2f} ${r['change']:>+12.2f} {pct_change:>+8.1f}%")
    
    # Total
    print("-"*100)
    
    total_before = sum(r['before'] for r in results)
    total_after = sum(r['after'] for r in results)
    total_change = total_after - total_before
    total_pct = (total_change / abs(total_before) * 100) if total_before != 0 else 0
    
    print(f"{'TOTAL SYSTEM':<35} ${total_before:>12.2f} ${total_after:>12.2f} ${total_change:>+12.2f} {total_pct:>+8.1f}%")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()