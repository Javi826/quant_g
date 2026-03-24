#!/usr/bin/env python3
"""
Triple Comparison: LAB vs LIVE 00 vs LIVE E1
Validates if LAB predictions match real trading results
Periods are determined by LIVE files
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

LAB_FOLDER = '/home/javi/projects/quant/quant_g/bitget/development/brief_trades'
LIVE_FOLDER = '/home/javi/projects/quant/quant_g/bitget/development/defense_mode'

# =============================================================================


def load_all_lab_trades():
    """Load all LAB trades from all_trades_*.xlsx files"""
    pattern = str(Path(LAB_FOLDER) / 'all_trades_*.xlsx')
    from glob import glob
    files = glob(pattern)
    
    all_trades = []
    for filepath in files:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.lower()
        
        # Extract strategy name from filename
        strategy = Path(filepath).stem.replace('all_trades_', '')
        df['strategy'] = strategy
        
        all_trades.append(df)
    
    combined = pd.concat(all_trades, ignore_index=True)
    
    # Use sell_time to match weekly analysis
    combined['sell_time'] = pd.to_datetime(combined['sell_time'])
    
    return combined.sort_values('sell_time').reset_index(drop=True)


def load_live_file(filepath):
    """Load single LIVE trade file"""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.upper()
    
    # Use CLOSE_AT (equivalent to sell_time)
    if 'CLOSE_AT' not in df.columns:
        raise ValueError(f"No CLOSE_AT column found in {filepath}")
    
    df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
    df.rename(columns={'CLOSE_AT': 'timestamp'}, inplace=True)
    
    return df.sort_values('timestamp').reset_index(drop=True)


def analyze_period(lab_df, live_00_df, live_e1_df, period_name):
    """Analyze LAB vs LIVE 00 vs LIVE E1 for a specific period"""
    
    # LAB metrics (no filters)
    lab_trades = len(lab_df)
    lab_profit = lab_df['profit'].sum() if lab_trades > 0 else 0
    lab_wr = (lab_df['profit'] > 0).sum() / lab_trades * 100 if lab_trades > 0 else 0
    
    # LIVE 00 metrics (no LAYER 1)
    live_00_trades = len(live_00_df)
    live_00_profit = live_00_df['PROFIT'].sum() if live_00_trades > 0 else 0
    live_00_wr = (live_00_df['PROFIT'] > 0).sum() / live_00_trades * 100 if live_00_trades > 0 else 0
    
    # LIVE E1 metrics (with LAYER 1)
    live_e1_trades = len(live_e1_df)
    live_e1_profit = live_e1_df['PROFIT'].sum() if live_e1_trades > 0 else 0
    live_e1_wr = (live_e1_df['PROFIT'] > 0).sum() / live_e1_trades * 100 if live_e1_trades > 0 else 0
    
    # Date ranges (using sell_time)
    lab_start = lab_df['sell_time'].min() if lab_trades > 0 else None
    lab_end = lab_df['sell_time'].max() if lab_trades > 0 else None
    
    live_start = live_e1_df['timestamp'].min() if live_e1_trades > 0 else None
    live_end = live_e1_df['timestamp'].max() if live_e1_trades > 0 else None
    
    return {
        'period': period_name,
        'start': live_start,
        'end': live_end,
        
        'lab_trades': lab_trades,
        'lab_profit': lab_profit,
        'lab_wr': lab_wr,
        
        'live_00_trades': live_00_trades,
        'live_00_profit': live_00_profit,
        'live_00_wr': live_00_wr,
        
        'live_e1_trades': live_e1_trades,
        'live_e1_profit': live_e1_profit,
        'live_e1_wr': live_e1_wr
    }


def main():
    print("=" * 140)
    print("TRIPLE COMPARISON: LAB vs LIVE 00 (no LAYER 1) vs LIVE E1 (with LAYER 1)")
    print("=" * 140)
    
    # Load LAB trades
    print("\n📂 Loading LAB trades...")
    lab_df = load_all_lab_trades()
    print(f"✅ Loaded {len(lab_df):,} LAB trades")
    print(f"   Date range (sell_time): {lab_df['sell_time'].min().date()} → {lab_df['sell_time'].max().date()}")
    
    # Find LIVE files (00 and E1 pairs)
    live_folder = Path(LIVE_FOLDER)
    
    periods = ['jan', 'feb', 'mar']
    live_pairs = {}
    
    print("\n📂 Checking LIVE files...")
    for period in periods:
        file_00 = live_folder / f'bot_trades_00_{period}.xlsx'
        file_e1 = live_folder / f'bot_trades_E1_{period}.xlsx'
        
        if file_00.exists() and file_e1.exists():
            print(f"   ✅ Found pair: {file_00.name} + {file_e1.name}")
            live_pairs[period] = {
                '00': file_00,
                'E1': file_e1
            }
        else:
            if not file_00.exists():
                print(f"   ❌ Missing {file_00.name}")
            if not file_e1.exists():
                print(f"   ❌ Missing {file_e1.name}")
    
    if not live_pairs:
        print("\n❌ No complete LIVE file pairs found")
        return
    
    # Analyze each period
    print("\n🔍 Analyzing periods...")
    results = []
    
    for period, files in live_pairs.items():
        print(f"\n   Processing {period.upper()}...")
        
        # Load LIVE files
        live_00_df = load_live_file(files['00'])
        live_e1_df = load_live_file(files['E1'])
        
        # Get date range from LIVE E1 (most filtered, defines the period)
        live_start = live_e1_df['timestamp'].min()
        live_end = live_e1_df['timestamp'].max()
        
        print(f"   Period: {live_start.date()} → {live_end.date()}")
        
        # Filter LAB to same period (using sell_time)
        lab_filtered = lab_df[
            (lab_df['sell_time'] >= live_start) & 
            (lab_df['sell_time'] <= live_end)
        ].copy()
        
        print(f"   LAB trades in period: {len(lab_filtered):,}")
        print(f"   LIVE 00 trades: {len(live_00_df):,}")
        print(f"   LIVE E1 trades: {len(live_e1_df):,}")
        
        # Analyze
        result = analyze_period(lab_filtered, live_00_df, live_e1_df, period)
        results.append(result)
    
    # Print comparison table
    print("\n" + "=" * 140)
    print("COMPARISON TABLE - BY PERIOD")
    print("=" * 140)
    
    for r in results:
        date_range = f"{r['start'].date()} → {r['end'].date()}"
        
        print(f"\n{'─'*140}")
        print(f"PERIOD: {r['period'].upper()} ({date_range})")
        print(f"{'─'*140}")
        print(f"{'SOURCE':<20} {'TRADES':<15} {'WIN RATE %':<15} {'PROFIT':<20} {'vs LAB Δ':<20}")
        print("-" * 140)
        
        # LAB row (baseline)
        print(f"{'LAB (no filters)':<20} {r['lab_trades']:<15,} {r['lab_wr']:<14.1f}% ${r['lab_profit']:<19,.2f} {'—':<20}")
        
        # LIVE 00 row
        delta_00_trades = r['live_00_trades'] - r['lab_trades']
        delta_00_profit = r['live_00_profit'] - r['lab_profit']
        trades_00_pct = (delta_00_trades / r['lab_trades'] * 100) if r['lab_trades'] > 0 else 0
        profit_00_pct = (delta_00_profit / abs(r['lab_profit']) * 100) if r['lab_profit'] != 0 else 0
        
        print(f"{'LIVE 00 (no L1)':<20} {r['live_00_trades']:<15,} {r['live_00_wr']:<14.1f}% ${r['live_00_profit']:<19,.2f} "
              f"{delta_00_trades:+,} T ({trades_00_pct:+.0f}%)")
        print(f"{'':>20} {'':>15} {'':>15} {'':>20} ${delta_00_profit:+,.2f} ({profit_00_pct:+.0f}%)")
        
        # LIVE E1 row
        delta_e1_trades = r['live_e1_trades'] - r['lab_trades']
        delta_e1_profit = r['live_e1_profit'] - r['lab_profit']
        trades_e1_pct = (delta_e1_trades / r['lab_trades'] * 100) if r['lab_trades'] > 0 else 0
        profit_e1_pct = (delta_e1_profit / abs(r['lab_profit']) * 100) if r['lab_profit'] != 0 else 0
        
        print(f"{'LIVE E1 (with L1)':<20} {r['live_e1_trades']:<15,} {r['live_e1_wr']:<14.1f}% ${r['live_e1_profit']:<19,.2f} "
              f"{delta_e1_trades:+,} T ({trades_e1_pct:+.0f}%)")
        print(f"{'':>20} {'':>15} {'':>15} {'':>20} ${delta_e1_profit:+,.2f} ({profit_e1_pct:+.0f}%)")
    
    # Summary totals
    print("\n" + "=" * 140)
    print("SUMMARY - ALL PERIODS COMBINED")
    print("=" * 140)
    
    total_lab_trades = sum(r['lab_trades'] for r in results)
    total_lab_profit = sum(r['lab_profit'] for r in results)
    total_lab_wr = sum(r['lab_wr'] * r['lab_trades'] for r in results) / total_lab_trades if total_lab_trades > 0 else 0
    
    total_00_trades = sum(r['live_00_trades'] for r in results)
    total_00_profit = sum(r['live_00_profit'] for r in results)
    total_00_wr = sum(r['live_00_wr'] * r['live_00_trades'] for r in results) / total_00_trades if total_00_trades > 0 else 0
    
    total_e1_trades = sum(r['live_e1_trades'] for r in results)
    total_e1_profit = sum(r['live_e1_profit'] for r in results)
    total_e1_wr = sum(r['live_e1_wr'] * r['live_e1_trades'] for r in results) / total_e1_trades if total_e1_trades > 0 else 0
    
    print(f"\n{'SOURCE':<20} {'TOTAL TRADES':<20} {'AVG WIN RATE':<20} {'TOTAL PROFIT':<20}")
    print("-" * 140)
    print(f"{'LAB (no filters)':<20} {total_lab_trades:<20,} {total_lab_wr:<19.1f}% ${total_lab_profit:<19,.2f}")
    print(f"{'LIVE 00 (no L1)':<20} {total_00_trades:<20,} {total_00_wr:<19.1f}% ${total_00_profit:<19,.2f}")
    print(f"{'LIVE E1 (with L1)':<20} {total_e1_trades:<20,} {total_e1_wr:<19.1f}% ${total_e1_profit:<19,.2f}")
    
    # Deltas
    print(f"\n{'DELTA':<20} {'TRADES':<20} {'WIN RATE':<20} {'PROFIT':<20}")
    print("-" * 140)
    
    delta_00_trades = total_00_trades - total_lab_trades
    delta_00_wr = total_00_wr - total_lab_wr
    delta_00_profit = total_00_profit - total_lab_profit
    trades_00_pct = (delta_00_trades / total_lab_trades * 100) if total_lab_trades > 0 else 0
    
    print(f"{'00 vs LAB':<20} {delta_00_trades:+,} ({trades_00_pct:+.1f}%) {delta_00_wr:+.1f}% {'':<8} ${delta_00_profit:+,.2f}")
    
    delta_e1_trades = total_e1_trades - total_lab_trades
    delta_e1_wr = total_e1_wr - total_lab_wr
    delta_e1_profit = total_e1_profit - total_lab_profit
    trades_e1_pct = (delta_e1_trades / total_lab_trades * 100) if total_lab_trades > 0 else 0
    
    print(f"{'E1 vs LAB':<20} {delta_e1_trades:+,} ({trades_e1_pct:+.1f}%) {delta_e1_wr:+.1f}% {'':<8} ${delta_e1_profit:+,.2f}")
    
    # Interpretation
    print("\n" + "=" * 140)
    print("INTERPRETATION")
    print("=" * 140)
    
    # LAB reliability check (compare LAB vs LIVE 00)
    profit_diff_pct = abs(delta_00_profit / total_lab_profit * 100) if total_lab_profit != 0 else 0
    
    print(f"\n1. LAB RELIABILITY (LAB vs LIVE 00 - both without filters):")
    print(f"   LAB predicted:  ${total_lab_profit:,.2f}")
    print(f"   LIVE 00 actual: ${total_00_profit:,.2f}")
    print(f"   Difference:     ${delta_00_profit:+,.2f} ({profit_diff_pct:.1f}%)")
    
    if profit_diff_pct < 10:
        print(f"   ✅ LAB is RELIABLE - predictions match reality within 10%")
    elif profit_diff_pct < 25:
        print(f"   ⚠️  LAB has MODERATE error - {profit_diff_pct:.1f}% difference")
    else:
        print(f"   ❌ LAB is UNRELIABLE - {profit_diff_pct:.1f}% difference is too high")
        print(f"      Possible causes: slippage, execution differences, data issues")
    
    # LAYER 1 effectiveness check
    layer1_effect = total_e1_profit - total_00_profit
    
    print(f"\n2. LAYER 1 EFFECTIVENESS (LIVE E1 vs LIVE 00):")
    print(f"   Without LAYER 1: ${total_00_profit:,.2f}")
    print(f"   With LAYER 1:    ${total_e1_profit:,.2f}")
    print(f"   Effect:          ${layer1_effect:+,.2f}")
    
    if layer1_effect > 0:
        print(f"   ✅ LAYER 1 improves performance by ${layer1_effect:,.2f}")
    else:
        print(f"   ❌ LAYER 1 hurts performance by ${abs(layer1_effect):,.2f}")
    
    print("\n" + "=" * 140)


if __name__ == "__main__":
    main()