#!/usr/bin/env python3
"""
Compare Two Rules on LIVE Data (00 and E1)
Rule A: Symmetric  MA5*1.00 / MA5*1.00
Rule B: Asymmetric MA5*1.02 / MA5*1.00
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


def evaluate_file_with_rule(filepath, btc_df, rule_name, long_th, short_th):
    """
    Evaluate file with specific rule
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
        
        # Apply rule
        if direction == 'LONG':
            if btc_close > ma5 * long_th:
                filtered_profits.append(profit)
        else:  # SHORT
            if btc_close < ma5 * short_th:
                filtered_profits.append(profit)
    
    total_profit = sum(all_profits)
    filtered_profit = sum(filtered_profits)
    
    return {
        'file': Path(filepath).stem,
        'rule': rule_name,
        'before_profit': total_profit,
        'after_profit': filtered_profit,
        'change': filtered_profit - total_profit,
        'before_trades': len(all_profits),
        'after_trades': len(filtered_profits)
    }


def main():
    print("="*100)
    print("COMPARE TWO RULES ON LIVE DATA")
    print("="*100)
    print("\nRule A (Symmetric):  LONG: BTC > MA5*1.00  |  SHORT: BTC < MA5*1.00")
    print("Rule B (Asymmetric): LONG: BTC > MA5*1.02  |  SHORT: BTC < MA5*1.00")
    print("="*100)
    
    # Load BTC
    btc_df = load_btc_1d()
    
    # Find files
    folder = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode')
    files_00 = sorted(folder.glob('bot_trades_00_*.xlsx'))
    files_e1 = sorted(folder.glob('bot_trades_E1_*.xlsx'))
    
    # Store all results
    all_results = []
    
    # Evaluate 00 files
    print("\n" + "="*100)
    print("00 FILES (RAW - NO LAYER 1)")
    print("="*100)
    
    for f in files_00:
        result_a = evaluate_file_with_rule(f, btc_df, 'A', 1.00, 1.00)
        result_b = evaluate_file_with_rule(f, btc_df, 'B', 1.02, 1.00)
        all_results.extend([result_a, result_b])
    
    # Print 00 results
    print(f"\n{'FILE':<25} {'RULE':<6} {'TRADES':<15} {'BEFORE':<15} {'AFTER':<15} {'CHANGE':<15}")
    print("-"*100)
    
    results_00_a = [r for r in all_results if '00' in r['file'] and r['rule'] == 'A']
    results_00_b = [r for r in all_results if '00' in r['file'] and r['rule'] == 'B']
    
    for i in range(len(results_00_a)):
        ra = results_00_a[i]
        rb = results_00_b[i]
        
        print(f"{ra['file']:<25} {ra['rule']:<6} {ra['before_trades']:>5}→{ra['after_trades']:<6} "
              f"${ra['before_profit']:>12.2f} ${ra['after_profit']:>12.2f} ${ra['change']:>+12.2f}")
        print(f"{'':<25} {rb['rule']:<6} {rb['before_trades']:>5}→{rb['after_trades']:<6} "
              f"${rb['before_profit']:>12.2f} ${rb['after_profit']:>12.2f} ${rb['change']:>+12.2f}")
        print()
    
    # Totals 00
    total_00_a = sum(r['after_profit'] for r in results_00_a)
    total_00_b = sum(r['after_profit'] for r in results_00_b)
    change_00_a = sum(r['change'] for r in results_00_a)
    change_00_b = sum(r['change'] for r in results_00_b)
    
    print("-"*100)
    print(f"{'TOTAL 00 - Rule A':<25} {'A':<6} {'':<15} {'':<15} ${total_00_a:>12.2f} ${change_00_a:>+12.2f}")
    print(f"{'TOTAL 00 - Rule B':<25} {'B':<6} {'':<15} {'':<15} ${total_00_b:>12.2f} ${change_00_b:>+12.2f}")
    
    # Evaluate E1 files
    print("\n" + "="*100)
    print("E1 FILES (WITH LAYER 1)")
    print("="*100)
    
    results_e1 = []
    for f in files_e1:
        result_a = evaluate_file_with_rule(f, btc_df, 'A', 1.00, 1.00)
        result_b = evaluate_file_with_rule(f, btc_df, 'B', 1.02, 1.00)
        results_e1.extend([result_a, result_b])
    
    # Print E1 results
    print(f"\n{'FILE':<25} {'RULE':<6} {'TRADES':<15} {'BEFORE':<15} {'AFTER':<15} {'CHANGE':<15}")
    print("-"*100)
    
    results_e1_a = [r for r in results_e1 if r['rule'] == 'A']
    results_e1_b = [r for r in results_e1 if r['rule'] == 'B']
    
    for i in range(len(results_e1_a)):
        ra = results_e1_a[i]
        rb = results_e1_b[i]
        
        print(f"{ra['file']:<25} {ra['rule']:<6} {ra['before_trades']:>5}→{ra['after_trades']:<6} "
              f"${ra['before_profit']:>12.2f} ${ra['after_profit']:>12.2f} ${ra['change']:>+12.2f}")
        print(f"{'':<25} {rb['rule']:<6} {rb['before_trades']:>5}→{rb['after_trades']:<6} "
              f"${rb['before_profit']:>12.2f} ${rb['after_profit']:>12.2f} ${rb['change']:>+12.2f}")
        print()
    
    # Totals E1
    total_e1_a = sum(r['after_profit'] for r in results_e1_a)
    total_e1_b = sum(r['after_profit'] for r in results_e1_b)
    change_e1_a = sum(r['change'] for r in results_e1_a)
    change_e1_b = sum(r['change'] for r in results_e1_b)
    
    print("-"*100)
    print(f"{'TOTAL E1 - Rule A':<25} {'A':<6} {'':<15} {'':<15} ${total_e1_a:>12.2f} ${change_e1_a:>+12.2f}")
    print(f"{'TOTAL E1 - Rule B':<25} {'B':<6} {'':<15} {'':<15} ${total_e1_b:>12.2f} ${change_e1_b:>+12.2f}")
    
    # Final summary
    print("\n" + "="*100)
    print("FINAL COMPARISON")
    print("="*100)
    
    print(f"\n{'DATASET':<20} {'RULE A (Symmetric)':<30} {'RULE B (Asymmetric)':<30} {'WINNER':<10}")
    print("-"*100)
    print(f"{'00 (raw)':<20} ${total_00_a:>12.2f} ({change_00_a:>+10.2f})    "
          f"${total_00_b:>12.2f} ({change_00_b:>+10.2f})    "
          f"{'A' if total_00_a > total_00_b else 'B' if total_00_b > total_00_a else 'TIE':<10}")
    print(f"{'E1 (+ LAYER 1)':<20} ${total_e1_a:>12.2f} ({change_e1_a:>+10.2f})    "
          f"${total_e1_b:>12.2f} ({change_e1_b:>+10.2f})    "
          f"{'A' if total_e1_a > total_e1_b else 'B' if total_e1_b > total_e1_a else 'TIE':<10}")
    
    # Overall winner
    total_a = total_00_a + total_e1_a
    total_b = total_00_b + total_e1_b
    
    print("\n" + "="*100)
    print("VERDICT")
    print("="*100)
    
    print(f"\nRule A (Symmetric):  ${total_a:,.2f}")
    print(f"Rule B (Asymmetric): ${total_b:,.2f}")
    
    if total_a > total_b:
        diff = total_a - total_b
        print(f"\n🏆 WINNER: Rule A (Symmetric MA5*1.00)")
        print(f"   Beats Rule B by ${diff:,.2f}")
        print(f"\n💡 RECOMMENDATION: Switch to symmetric rule")
        print(f"   LONG:  BTC > MA5*1.00")
        print(f"   SHORT: BTC < MA5*1.00")
    elif total_b > total_a:
        diff = total_b - total_a
        print(f"\n🏆 WINNER: Rule B (Asymmetric MA5*1.02/1.00)")
        print(f"   Beats Rule A by ${diff:,.2f}")
        print(f"\n💡 RECOMMENDATION: Keep current asymmetric rule")
        print(f"   LONG:  BTC > MA5*1.02")
        print(f"   SHORT: BTC < MA5*1.00")
    else:
        print(f"\n⚖️  TIE: Both rules perform equally")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()