#!/usr/bin/env python3
"""
Compare Asymmetric Rules on 00 (raw) vs E1 (with LAYER 1)
LONG:  BTC > MA5*1.02 → allow
SHORT: BTC < MA5*1.00 → allow
"""

import pandas as pd
from pathlib import Path


def load_btc_1d():
    """Load BTC 1D data"""
    btc_file = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/BTCUSDT_1Dutc.parquet')
    df = pd.read_parquet(btc_file)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df.get('timestamp', df.index))
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


def load_trades(filepath):
    """Load trades file"""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.upper()
    
    ts_col = next((c for c in ['OPEN_AT', 'BUY_TIME', 'ENTRY_TIME'] if c in df.columns), None)
    if not ts_col:
        raise ValueError(f"No timestamp column in {filepath}")
    
    df[ts_col] = pd.to_datetime(df[ts_col])
    df.rename(columns={ts_col: 'open_time'}, inplace=True)
    return df.sort_values('open_time').reset_index(drop=True)


def calculate_dd(profits):
    """Calculate max drawdown %"""
    if len(profits) == 0:
        return 0
    cumulative = 0
    peak = 0
    max_dd = 0
    for p in profits:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return -(max_dd / peak * 100) if peak > 0 else 0


def evaluate_file(filepath, btc_df):
    """
    Evaluate single file with asymmetric rules
    LONG:  BTC > MA5*1.02 → allow
    SHORT: BTC < MA5*1.00 → allow
    """
    df = load_trades(filepath)
    
    all_profits = []
    filtered_profits = []
    
    for _, trade in df.iterrows():
        direction = trade['DIRECTION']
        profit = trade['PROFIT']
        
        all_profits.append(profit)
        
        btc_close, ma5 = get_btc_values(btc_df, trade['open_time'])
        
        if btc_close is None or ma5 is None:
            filtered_profits.append(profit)
            continue
        
        # Apply asymmetric rules
        if direction == 'LONG':
            # LONG: allow if BTC > MA5*1.02
            if btc_close > ma5 * 1.02:
                filtered_profits.append(profit)
        else:
            # SHORT: allow if BTC < MA5*1.00
            if btc_close < ma5 * 1.00:
                filtered_profits.append(profit)
    
    total_trades = len(all_profits)
    total_profit = sum(all_profits)
    total_wr = sum(1 for p in all_profits if p > 0) / total_trades * 100 if total_trades > 0 else 0
    total_dd = calculate_dd(all_profits)
    
    filtered_trades = len(filtered_profits)
    filtered_profit = sum(filtered_profits)
    filtered_wr = sum(1 for p in filtered_profits if p > 0) / filtered_trades * 100 if filtered_trades > 0 else 0
    filtered_dd = calculate_dd(filtered_profits)
    
    return {
        'file': Path(filepath).stem,
        'before_trades': total_trades,
        'before_profit': total_profit,
        'before_wr': total_wr,
        'before_dd': total_dd,
        'after_trades': filtered_trades,
        'after_profit': filtered_profit,
        'after_wr': filtered_wr,
        'after_dd': filtered_dd
    }


def main():
    print("="*120)
    print("ASYMMETRIC RULES: 00 (raw) vs E1 (with LAYER 1)")
    print("LONG: BTC > MA5*1.02 → allow  |  SHORT: BTC < MA5*1.00 → allow")
    print("="*120)
    
    # Load BTC
    btc_df = load_btc_1d()
    
    # Find files
    folder = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode')
    files_00 = sorted(folder.glob('bot_trades_00_*.xlsx'))
    files_e1 = sorted(folder.glob('bot_trades_E1_*.xlsx'))
    
    # Evaluate 00 files
    print("\n" + "="*120)
    print("00 FILES (RAW - NO LAYER 1)")
    print("="*120)
    
    print(f"\n{'FILE':<20} {'BEFORE T':>10} {'WR%':>8} {'PROFIT':>12} {'DD%':>8} "
          f"{'AFTER T':>10} {'WR%':>8} {'PROFIT':>12} {'DD%':>8} {'Δ PROFIT':>12}")
    print("-"*120)
    
    results_00 = []
    for f in files_00:
        r = evaluate_file(f, btc_df)
        results_00.append(r)
        
        profit_change = r['after_profit'] - r['before_profit']
        
        print(f"{r['file']:<20} {r['before_trades']:>10} {r['before_wr']:>7.1f}% {r['before_profit']:>12.2f} {r['before_dd']:>7.1f}% "
              f"{r['after_trades']:>10} {r['after_wr']:>7.1f}% {r['after_profit']:>12.2f} {r['after_dd']:>7.1f}% "
              f"{profit_change:>+12.2f}")
    
    # Totals 00
    total_00_before_profit = sum(r['before_profit'] for r in results_00)
    total_00_after_profit = sum(r['after_profit'] for r in results_00)
    total_00_change = total_00_after_profit - total_00_before_profit
    
    print("-"*120)
    print(f"{'TOTAL 00':<20} {sum(r['before_trades'] for r in results_00):>10} {'':>8} {total_00_before_profit:>12.2f} {'':>8} "
          f"{sum(r['after_trades'] for r in results_00):>10} {'':>8} {total_00_after_profit:>12.2f} {'':>8} "
          f"{total_00_change:>+12.2f}")
    
    # Evaluate E1 files
    print("\n" + "="*120)
    print("E1 FILES (WITH LAYER 1)")
    print("="*120)
    
    print(f"\n{'FILE':<20} {'BEFORE T':>10} {'WR%':>8} {'PROFIT':>12} {'DD%':>8} "
          f"{'AFTER T':>10} {'WR%':>8} {'PROFIT':>12} {'DD%':>8} {'Δ PROFIT':>12}")
    print("-"*120)
    
    results_e1 = []
    for f in files_e1:
        r = evaluate_file(f, btc_df)
        results_e1.append(r)
        
        profit_change = r['after_profit'] - r['before_profit']
        
        print(f"{r['file']:<20} {r['before_trades']:>10} {r['before_wr']:>7.1f}% {r['before_profit']:>12.2f} {r['before_dd']:>7.1f}% "
              f"{r['after_trades']:>10} {r['after_wr']:>7.1f}% {r['after_profit']:>12.2f} {r['after_dd']:>7.1f}% "
              f"{profit_change:>+12.2f}")
    
    # Totals E1
    total_e1_before_profit = sum(r['before_profit'] for r in results_e1)
    total_e1_after_profit = sum(r['after_profit'] for r in results_e1)
    total_e1_change = total_e1_after_profit - total_e1_before_profit
    
    print("-"*120)
    print(f"{'TOTAL E1':<20} {sum(r['before_trades'] for r in results_e1):>10} {'':>8} {total_e1_before_profit:>12.2f} {'':>8} "
          f"{sum(r['after_trades'] for r in results_e1):>10} {'':>8} {total_e1_after_profit:>12.2f} {'':>8} "
          f"{total_e1_change:>+12.2f}")
    
    # Summary comparison
    print("\n" + "="*120)
    print("SUMMARY")
    print("="*120)
    
    print(f"\n{'GROUP':<20} {'BEFORE PROFIT':>20} {'AFTER PROFIT':>20} {'CHANGE':>20} {'% CHANGE':>15}")
    print("-"*80)
    print(f"{'00 (raw)':<20} {total_00_before_profit:>20.2f} {total_00_after_profit:>20.2f} {total_00_change:>+20.2f} "
          f"{total_00_change/abs(total_00_before_profit)*100:>+14.1f}%")
    print(f"{'E1 (+ LAYER 1)':<20} {total_e1_before_profit:>20.2f} {total_e1_after_profit:>20.2f} {total_e1_change:>+20.2f} "
          f"{total_e1_change/abs(total_e1_before_profit)*100 if total_e1_before_profit != 0 else 0:>+14.1f}%")
    
    # Verdict
    print("\n" + "="*120)
    print("VERDICT")
    print("="*120)
    
    print(f"\nLAB Results (asymmetric rules):")
    print(f"  LONG:  BTC > MA5*1.02 → -$31.55")
    print(f"  SHORT: BTC < MA5*1.00 → +$1037.74")
    print(f"  TOTAL: +$1006.20")
    
    if total_00_change > 0:
        print(f"\n00 (raw):        ✅ Asymmetric rules improve by ${total_00_change:.2f}")
    else:
        print(f"\n00 (raw):        ❌ Asymmetric rules worsen by ${total_00_change:.2f}")
    
    if total_e1_change > 0:
        print(f"E1 (+ LAYER 1):  ✅ Asymmetric rules add value: +${total_e1_change:.2f}")
        print(f"\n💡 RECOMMENDATION: Use BOTH layers with asymmetric rules")
        print(f"   - LAYER 1: Regime per strategy (4H)")
        print(f"   - LAYER 2: Asymmetric LONG/SHORT rules (1D)")
        print(f"     • LONG:  allow if BTC > MA5*1.02")
        print(f"     • SHORT: allow if BTC < MA5*1.00")
    else:
        print(f"E1 (+ LAYER 1):  ❌ Asymmetric rules do NOT add value: ${total_e1_change:.2f}")
        print(f"\n💡 RECOMMENDATION: Use LAYER 1 only")
    
    print("\n" + "="*120)


if __name__ == "__main__":
    main()