#!/usr/bin/env python3
"""
Same Week Analysis: Current Week WR vs Current Week BTC Metrics
Identifies which BTC conditions occur DURING low WR weeks.
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
    
    # Calculate MA50
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['above_ma50'] = df['close'] > df['ma50']
    
    # Calculate MA20
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['above_ma20'] = df['close'] > df['ma20']
    
    # Calculate trend duration (for MA50)
    df['trend_duration'] = 0
    current_state = None
    duration = 0
    
    for idx in range(len(df)):
        if pd.isna(df.loc[idx, 'ma50']):
            continue
        
        state = df.loc[idx, 'above_ma50']
        
        if state == current_state:
            duration += 1
        else:
            duration = 1
            current_state = state
        
        df.loc[idx, 'trend_duration'] = duration
    
    return df


def calculate_weekly_system_wr(df_trades):
    """Calculate system WR week by week"""
    df_trades['week'] = df_trades['sell_time'].dt.to_period('W')
    
    weekly_wr = []
    
    for week, group in df_trades.groupby('week'):
        wr = (group['profit'] > 0).mean() * 100
        trades = len(group)
        
        weekly_wr.append({
            'week': str(week),
            'wr': wr,
            'trades': trades
        })
    
    return pd.DataFrame(weekly_wr)


def calculate_weekly_btc_metrics(btc_df):
    """Calculate BTC metrics week by week"""
    btc_df['week'] = btc_df['ts'].dt.to_period('W')
    
    weekly_metrics = []
    
    for week, group in btc_df.groupby('week'):
        # Count MA50 crosses
        crosses_ma50 = 0
        for i in range(1, len(group)):
            if pd.notna(group.iloc[i-1]['above_ma50']) and pd.notna(group.iloc[i]['above_ma50']):
                if group.iloc[i-1]['above_ma50'] != group.iloc[i]['above_ma50']:
                    crosses_ma50 += 1
        
        # Count MA20 crosses
        crosses_ma20 = 0
        for i in range(1, len(group)):
            if pd.notna(group.iloc[i-1]['above_ma20']) and pd.notna(group.iloc[i]['above_ma20']):
                if group.iloc[i-1]['above_ma20'] != group.iloc[i]['above_ma20']:
                    crosses_ma20 += 1
        
        # Price range
        high_max = group['high'].max()
        low_min = group['low'].min()
        close_avg = group['close'].mean()
        price_range = ((high_max - low_min) / close_avg) * 100 if close_avg > 0 else 0
        
        metrics = {
            'week': str(week),
            'avg_atr': group['atr_pct'].mean(),
            'avg_er': group['er'].mean(),
            'avg_duration': group['trend_duration'].mean(),
            'crosses_ma50': crosses_ma50,
            'crosses_ma20': crosses_ma20,
            'price_range': price_range
        }
        
        weekly_metrics.append(metrics)
    
    return pd.DataFrame(weekly_metrics)


def main():
    print("="*120)
    print("SAME WEEK ANALYSIS: WR vs BTC METRICS (SAME WEEK)")
    print("="*120)
    
    # Load data
    print("\n📂 Loading trades...")
    df_trades = load_all_lab_trades()
    
    print("📂 Loading BTC data...")
    btc_df = load_btc_data()
    
    # Calculate weekly stats
    print("\n🔍 Calculating weekly statistics...")
    df_system = calculate_weekly_system_wr(df_trades)
    df_btc = calculate_weekly_btc_metrics(btc_df)
    
    # Merge on same week
    df_merged = pd.merge(df_system, df_btc, on='week', how='inner')
    
    print(f"✅ Merged {len(df_merged)} weeks")
    
    # Add category column
    df_merged['category'] = df_merged['wr'].apply(
        lambda x: 'bad' if x < 70 else ('good' if x > 80 else 'neutral')
    )
    
    # Save consolidated data
    output_dir = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/files')
    output_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    output_file = output_dir / 'consolidated_weekly_data.xlsx'
    df_merged.to_excel(output_file, index=False)
    print(f"\n💾 Saved consolidated data to: {output_file}")
    
    # Display merged data
    print("\n" + "="*120)
    print("WEEKLY DATA: WR + BTC METRICS (SAME WEEK)")
    print("="*120)
    
    print(f"\n{'Week':<20} {'WR%':>8} {'Trades':>8} {'ATR%':>8} {'ER':>8} {'Dur':>8} {'MA50 X':>8} {'MA20 X':>8} {'Range%':>8}")
    print("-"*130)
    
    for _, row in df_merged.iterrows():
        print(f"{row['week']:<20} {row['wr']:>7.1f}% {row['trades']:>8} "
              f"{row['avg_atr']:>7.2f}% {row['avg_er']:>7.3f} "
              f"{row['avg_duration']:>7.1f} {row['crosses_ma50']:>8} "
              f"{row['crosses_ma20']:>8} {row['price_range']:>7.2f}%")
    
    # Correlations
    print("\n" + "="*120)
    print("CORRELATIONS (WR vs BTC Metrics - SAME WEEK)")
    print("="*120)
    
    corr_atr = df_merged['wr'].corr(df_merged['avg_atr'])
    corr_er = df_merged['wr'].corr(df_merged['avg_er'])
    corr_duration = df_merged['wr'].corr(df_merged['avg_duration'])
    corr_crosses_ma50 = df_merged['wr'].corr(df_merged['crosses_ma50'])
    corr_crosses_ma20 = df_merged['wr'].corr(df_merged['crosses_ma20'])
    corr_range = df_merged['wr'].corr(df_merged['price_range'])
    
    print(f"\n{'Metric':<40} {'Correlation':>15} {'Strength':>20}")
    print("-"*80)
    print(f"{'WR vs ATR':<40} {corr_atr:>+15.3f} {('Strong' if abs(corr_atr) > 0.5 else 'Moderate' if abs(corr_atr) > 0.3 else 'Weak'):>20}")
    print(f"{'WR vs ER':<40} {corr_er:>+15.3f} {('Strong' if abs(corr_er) > 0.5 else 'Moderate' if abs(corr_er) > 0.3 else 'Weak'):>20}")
    print(f"{'WR vs Duration':<40} {corr_duration:>+15.3f} {('Strong' if abs(corr_duration) > 0.5 else 'Moderate' if abs(corr_duration) > 0.3 else 'Weak'):>20}")
    print(f"{'WR vs MA50 Crosses':<40} {corr_crosses_ma50:>+15.3f} {('Strong' if abs(corr_crosses_ma50) > 0.5 else 'Moderate' if abs(corr_crosses_ma50) > 0.3 else 'Weak'):>20}")
    print(f"{'WR vs MA20 Crosses':<40} {corr_crosses_ma20:>+15.3f} {('Strong' if abs(corr_crosses_ma20) > 0.5 else 'Moderate' if abs(corr_crosses_ma20) > 0.3 else 'Weak'):>20}")
    print(f"{'WR vs Price Range':<40} {corr_range:>+15.3f} {('Strong' if abs(corr_range) > 0.5 else 'Moderate' if abs(corr_range) > 0.3 else 'Weak'):>20}")
    
    # Compare bad vs good weeks
    bad_weeks = df_merged[df_merged['wr'] < 70]
    good_weeks = df_merged[df_merged['wr'] > 80]
    
    print("\n" + "="*120)
    print("COMPARISON: BAD WEEKS (WR<70%) vs GOOD WEEKS (WR>80%)")
    print("="*120)
    
    print(f"\n{'Metric':<40} {'Bad Weeks':>15} {'Good Weeks':>15} {'Difference':>20}")
    print("-"*100)
    
    for metric in ['avg_atr', 'avg_er', 'avg_duration', 'crosses_ma50', 'crosses_ma20', 'price_range']:
        bad_avg = bad_weeks[metric].mean()
        good_avg = good_weeks[metric].mean()
        diff = bad_avg - good_avg
        diff_pct = (diff / good_avg * 100) if good_avg != 0 else 0
        
        print(f"{metric:<40} {bad_avg:>15.2f} {good_avg:>15.2f} {diff:>+10.2f} ({diff_pct:+.1f}%)")
    
    # Defensive mode detection (same week)
    print("\n" + "="*120)
    print("DEFENSIVE MODE DETECTION CRITERIA (SAME WEEK)")
    print("="*120)
    
    print(f"\nBad weeks (WR < 70%) characteristics:")
    print(f"  Count: {len(bad_weeks)} weeks")
    print(f"  ATR range: {bad_weeks['avg_atr'].min():.2f}% - {bad_weeks['avg_atr'].max():.2f}%")
    print(f"  ER range: {bad_weeks['avg_er'].min():.3f} - {bad_weeks['avg_er'].max():.3f}")
    print(f"  Duration range: {bad_weeks['avg_duration'].min():.1f} - {bad_weeks['avg_duration'].max():.1f} bars")
    print(f"  MA50 Crosses range: {bad_weeks['crosses_ma50'].min():.0f} - {bad_weeks['crosses_ma50'].max():.0f}")
    print(f"  MA20 Crosses range: {bad_weeks['crosses_ma20'].min():.0f} - {bad_weeks['crosses_ma20'].max():.0f}")
    print(f"  Range: {bad_weeks['price_range'].min():.2f}% - {bad_weeks['price_range'].max():.2f}%")
    
    # Calculate threshold for real-time detection
    atr_25 = bad_weeks['avg_atr'].quantile(0.25)
    atr_75 = bad_weeks['avg_atr'].quantile(0.75)
    er_25 = bad_weeks['avg_er'].quantile(0.25)
    er_75 = bad_weeks['avg_er'].quantile(0.75)
    duration_median = bad_weeks['avg_duration'].median()
    crosses_ma50_median = bad_weeks['crosses_ma50'].median()
    crosses_ma20_median = bad_weeks['crosses_ma20'].median()
    
    print(f"\n💡 DEFENSIVE MODE REAL-TIME DETECTION:")
    print(f"   During the week, if rolling metrics show:")
    print(f"   - ATR: {atr_25:.2f}% - {atr_75:.2f}%")
    print(f"   - ER: {er_25:.3f} - {er_75:.3f}")
    print(f"   - Avg Duration: < {duration_median:.1f} bars")
    print(f"   - MA50 Crosses: >= {crosses_ma50_median:.0f}")
    print(f"   - MA20 Crosses: >= {crosses_ma20_median:.0f}")
    print(f"   → Likely a bad week in progress")
    
    print("\n" + "="*120)


if __name__ == "__main__":
    main()