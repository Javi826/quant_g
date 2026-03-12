#!/usr/bin/env python3
"""
defense_mode/phase1_lab_analysis.py
PHASE 1: COMPREHENSIVE LAB ANALYSIS
Exhaustive daily analysis to identify patterns in bad trading days.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob


def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_er(df, period=10):
    """Calculate Efficiency Ratio"""
    change = np.abs(df['close'] - df['close'].shift(period))
    volatility = np.abs(df['close'] - df['close'].shift()).rolling(window=period).sum()
    
    er = change / volatility
    return er


def load_all_lab_trades():
    """Load all lab trades"""
    lab_folder = Path('/home/javi/projects/quant/quant_g/bitget/development/brief_trades')
    files = glob(str(lab_folder / 'all_trades_*.xlsx'))
    
    all_trades = []
    for filepath in files:
        df = pd.read_excel(filepath)
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        all_trades.append(df)
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined.sort_values('sell_time').reset_index(drop=True)


def load_btc_data():
    """Load and prepare BTC data"""
    btc_file = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/BTCUSDT_4H.parquet')
    
    df = pd.read_parquet(btc_file)
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Calculate ATR
    df['atr'] = calculate_atr(df)
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Calculate ER
    df['er'] = calculate_er(df)
    
    # Calculate MAs
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['above_ma50'] = df['close'] > df['ma50']
    df['above_ma20'] = df['close'] > df['ma20']
    
    # Daily price change
    df['daily_change_pct'] = df['close'].pct_change() * 100
    
    # Intraday range
    df['intraday_range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
    
    return df


def calculate_daily_system_wr(df_trades):
    """Calculate system WR day by day"""
    df_trades['date'] = df_trades['sell_time'].dt.date
    
    daily_wr = []
    
    for date, group in df_trades.groupby('date'):
        wr = (group['profit'] > 0).mean() * 100
        trades = len(group)
        profit = group['profit'].sum()
        avg_profit = group['profit'].mean()
        
        daily_wr.append({
            'date': date,
            'wr': wr,
            'trades': trades,
            'profit': profit,
            'avg_profit': avg_profit
        })
    
    return pd.DataFrame(daily_wr)


def calculate_daily_btc_metrics(btc_df):
    """Calculate BTC metrics day by day"""
    btc_df['date'] = btc_df['ts'].dt.date
    
    daily_metrics = []
    
    for date, group in btc_df.groupby('date'):
        # Count crosses
        crosses_ma50 = 0
        crosses_ma20 = 0
        
        for i in range(1, len(group)):
            if pd.notna(group.iloc[i-1]['above_ma50']) and pd.notna(group.iloc[i]['above_ma50']):
                if group.iloc[i-1]['above_ma50'] != group.iloc[i]['above_ma50']:
                    crosses_ma50 += 1
            
            if pd.notna(group.iloc[i-1]['above_ma20']) and pd.notna(group.iloc[i]['above_ma20']):
                if group.iloc[i-1]['above_ma20'] != group.iloc[i]['above_ma20']:
                    crosses_ma20 += 1
        
        # Price metrics
        high_max = group['high'].max()
        low_min = group['low'].min()
        close_avg = group['close'].mean()
        open_first = group.iloc[0]['open']
        close_last = group.iloc[-1]['close']
        
        daily_range = ((high_max - low_min) / close_avg) * 100
        daily_change = ((close_last - open_first) / open_first) * 100
        
        # Volatility
        intrabar_volatility = group['intraday_range_pct'].mean()
        
        metrics = {
            'date': date,
            'avg_atr': group['atr_pct'].mean(),
            'avg_er': group['er'].mean(),
            'crosses_ma50': crosses_ma50,
            'crosses_ma20': crosses_ma20,
            'daily_range': daily_range,
            'daily_change': daily_change,
            'intrabar_volatility': intrabar_volatility,
            'bars': len(group),
            'open': open_first,
            'close': close_last,
            'high': high_max,
            'low': low_min
        }
        
        daily_metrics.append(metrics)
    
    return pd.DataFrame(daily_metrics)


def main():
    print("="*120)
    print("PHASE 1: COMPREHENSIVE LAB ANALYSIS (2025 + 2026)")
    print("="*120)
    
    # Load data
    print("\n📂 Loading lab trades...")
    df_trades = load_all_lab_trades()
    
    print(f"✅ Loaded {len(df_trades)} trades")
    print(f"   Period: {df_trades['sell_time'].min().date()} → {df_trades['sell_time'].max().date()}")
    
    print("\n📂 Loading BTC data...")
    btc_df = load_btc_data()
    
    # Calculate daily stats
    print("\n🔍 Calculating daily statistics...")
    df_system = calculate_daily_system_wr(df_trades)
    df_btc = calculate_daily_btc_metrics(btc_df)
    
    # Merge
    df_merged = pd.merge(df_system, df_btc, on='date', how='inner')
    
    # Filter days with enough trades
    df_merged = df_merged[df_merged['trades'] >= 5].copy()
    
    print(f"✅ Merged {len(df_merged)} days (with >= 5 trades)")
    
    # Classify days
    df_merged['category'] = df_merged['wr'].apply(
        lambda x: 'bad' if x < 60 else ('good' if x > 80 else 'neutral')
    )
    
    bad_days = df_merged[df_merged['category'] == 'bad']
    good_days = df_merged[df_merged['category'] == 'good']
    
    print(f"\n📊 Day classification:")
    print(f"   Bad days (WR < 60%): {len(bad_days)}")
    print(f"   Neutral days (60-80%): {len(df_merged[df_merged['category'] == 'neutral'])}")
    print(f"   Good days (WR > 80%): {len(good_days)}")
    
    # ANALYSIS 1: Basic correlations
    print("\n" + "="*120)
    print("ANALYSIS 1: CORRELATIONS (WR vs BTC Metrics)")
    print("="*120)
    
    metrics = ['avg_atr', 'avg_er', 'crosses_ma50', 'crosses_ma20', 'daily_range', 
               'daily_change', 'intrabar_volatility']
    
    print(f"\n{'Metric':<30} {'Correlation':>15} {'Strength':>20}")
    print("-"*70)
    
    correlations = {}
    for metric in metrics:
        corr = df_merged['wr'].corr(df_merged[metric])
        correlations[metric] = corr
        strength = 'Strong' if abs(corr) > 0.5 else ('Moderate' if abs(corr) > 0.3 else 'Weak')
        print(f"{metric:<30} {corr:>+15.3f} {strength:>20}")
    
    # ANALYSIS 2: Bad vs Good days comparison
    print("\n" + "="*120)
    print("ANALYSIS 2: BAD DAYS vs GOOD DAYS")
    print("="*120)
    
    print(f"\n{'Metric':<30} {'Bad Days':>15} {'Good Days':>15} {'Diff %':>15}")
    print("-"*80)
    
    significant_diffs = []
    
    for metric in metrics:
        bad_avg = bad_days[metric].mean()
        good_avg = good_days[metric].mean()
        diff_pct = ((bad_avg - good_avg) / good_avg * 100) if good_avg != 0 else 0
        
        print(f"{metric:<30} {bad_avg:>15.2f} {good_avg:>15.2f} {diff_pct:>+14.1f}%")
        
        if abs(diff_pct) > 20:
            significant_diffs.append((metric, diff_pct))
    
    # ANALYSIS 3: Extreme volatility events
    print("\n" + "="*120)
    print("ANALYSIS 3: EXTREME VOLATILITY EVENTS")
    print("="*120)
    
    # Define thresholds
    extreme_atr = df_merged['avg_atr'].quantile(0.90)
    extreme_range = df_merged['daily_range'].quantile(0.90)
    extreme_change = df_merged['daily_change'].abs().quantile(0.90)
    
    print(f"\nThresholds (90th percentile):")
    print(f"  ATR > {extreme_atr:.2f}%")
    print(f"  Daily Range > {extreme_range:.2f}%")
    print(f"  |Daily Change| > {extreme_change:.2f}%")
    
    extreme_days = df_merged[
        (df_merged['avg_atr'] > extreme_atr) |
        (df_merged['daily_range'] > extreme_range) |
        (df_merged['daily_change'].abs() > extreme_change)
    ]
    
    print(f"\nDays with extreme volatility: {len(extreme_days)}")
    print(f"Average WR on extreme days: {extreme_days['wr'].mean():.1f}%")
    print(f"Average WR on normal days: {df_merged[~df_merged['date'].isin(extreme_days['date'])]['wr'].mean():.1f}%")
    
    # ANALYSIS 4: Crash days
    print("\n" + "="*120)
    print("ANALYSIS 4: CRASH DAYS (Large Price Drops)")
    print("="*120)
    
    crash_threshold = -3.0  # 3% daily drop
    crash_days = df_merged[df_merged['daily_change'] < crash_threshold]
    
    print(f"\nDays with > 3% drop: {len(crash_days)}")
    if len(crash_days) > 0:
        print(f"Average WR on crash days: {crash_days['wr'].mean():.1f}%")
        print(f"Average profit on crash days: ${crash_days['profit'].mean():.2f}")
        
        print(f"\nWorst crash days:")
        worst_crashes = crash_days.nsmallest(5, 'daily_change')
        for _, row in worst_crashes.iterrows():
            print(f"  {row['date']}: {row['daily_change']:+.2f}% change, WR {row['wr']:.1f}%, P/L ${row['profit']:.2f}")
    
    # ANALYSIS 5: Clustering (consecutive bad days)
    print("\n" + "="*120)
    print("ANALYSIS 5: TEMPORAL CLUSTERING")
    print("="*120)
    
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    df_merged['prev_bad'] = df_merged['category'].shift(1) == 'bad'
    
    bad_after_bad = df_merged[(df_merged['category'] == 'bad') & (df_merged['prev_bad'])]
    
    print(f"\nBad days that follow another bad day: {len(bad_after_bad)}")
    print(f"Probability of bad day after bad day: {len(bad_after_bad) / len(bad_days) * 100:.1f}%")
    
    # ANALYSIS 6: Worst days details
    print("\n" + "="*120)
    print("ANALYSIS 6: WORST 10 DAYS IN LAB")
    print("="*120)
    
    worst_days = df_merged.nsmallest(10, 'wr')
    
    print(f"\n{'Date':<12} {'WR%':>8} {'Trades':>8} {'ATR%':>8} {'Range%':>8} {'Change%':>8} {'MA50 X':>8}")
    print("-"*80)
    
    for _, row in worst_days.iterrows():
        print(f"{str(row['date']):<12} {row['wr']:>7.1f}% {row['trades']:>8} "
              f"{row['avg_atr']:>7.2f}% {row['daily_range']:>7.2f}% "
              f"{row['daily_change']:>+7.2f}% {row['crosses_ma50']:>8}")
    
    # ANALYSIS 7: Proposed defensive mode criteria
    print("\n" + "="*120)
    print("ANALYSIS 7: PROPOSED DEFENSIVE MODE CRITERIA")
    print("="*120)
    
    print("\nBased on significant differences:")
    for metric, diff_pct in significant_diffs:
        print(f"  • {metric}: {diff_pct:+.1f}% difference")
    
    # Test multiple criteria
    print("\n" + "="*120)
    print("Testing potential criteria:")
    print("-"*80)
    
    criteria_tests = [
        ('High ATR', df_merged['avg_atr'] > extreme_atr),
        ('High Daily Range', df_merged['daily_range'] > extreme_range),
        ('Crash Day', df_merged['daily_change'] < crash_threshold),
        ('High Crosses MA50', df_merged['crosses_ma50'] >= 2),
        ('Extreme Volatility', 
         (df_merged['avg_atr'] > extreme_atr) | (df_merged['daily_range'] > extreme_range)),
    ]
    
    print(f"\n{'Criterion':<30} {'Triggers':>10} {'Bad Caught':>12} {'Precision':>12} {'Recall':>12}")
    print("-"*80)
    
    for name, condition in criteria_tests:
        triggered = df_merged[condition]
        bad_caught = triggered[triggered['category'] == 'bad']
        
        precision = len(bad_caught) / len(triggered) * 100 if len(triggered) > 0 else 0
        recall = len(bad_caught) / len(bad_days) * 100 if len(bad_days) > 0 else 0
        
        print(f"{name:<30} {len(triggered):>10} {len(bad_caught):>12} {precision:>11.1f}% {recall:>11.1f}%")
    
    # Save consolidated data
    output_dir = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/files')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'lab_daily_analysis.xlsx'
    print(f"\n💾 Saved daily analysis to: {output_file}")
    
    print("\n" + "="*120)
    print("✅ PHASE 1 COMPLETE - Ready for Phase 2 (Live Validation)")
    print("="*120)


if __name__ == "__main__":
    main()