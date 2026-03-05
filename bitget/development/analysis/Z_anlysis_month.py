#!/usr/bin/env python3
"""
Weekly Win Rate Analysis - Lab Trades
Shows WR week by week to identify if Feb 15-28 was particularly bad.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt


def load_all_lab_trades():
    """Load and combine all lab trades"""
    
    lab_folder = Path('/home/javi/projects/quant/quant_g/bitget/development/brief_trades')
    files = glob(str(lab_folder / 'all_trades_*.xlsx'))
    
    all_trades = []
    
    for filepath in files:
        df = pd.read_excel(filepath)
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        all_trades.append(df)
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined.sort_values('sell_time').reset_index(drop=True)


def calculate_weekly_wr(df):
    """Calculate WR week by week"""
    
    # Create week column
    df['week'] = df['sell_time'].dt.to_period('W')
    
    weekly_stats = []
    
    for week, group in df.groupby('week'):
        week_start = group['sell_time'].min()
        week_end = group['sell_time'].max()
        
        total = len(group)
        winners = (group['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        total_profit = group['profit'].sum()
        avg_profit = group['profit'].mean()
        
        weekly_stats.append({
            'week': str(week),
            'start_date': week_start.date(),
            'end_date': week_end.date(),
            'trades': total,
            'wr': wr,
            'total_profit': total_profit,
            'avg_profit': avg_profit
        })
    
    return pd.DataFrame(weekly_stats)


def calculate_daily_wr(df, start_date, end_date):
    """Calculate WR day by day for specific period"""
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    df_period = df[(df['sell_time'] >= start) & (df['sell_time'] <= end)].copy()
    df_period['date'] = df_period['sell_time'].dt.date
    
    daily_stats = []
    
    for date, group in df_period.groupby('date'):
        total = len(group)
        winners = (group['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        total_profit = group['profit'].sum()
        
        daily_stats.append({
            'date': date,
            'trades': total,
            'wr': wr,
            'total_profit': total_profit
        })
    
    return pd.DataFrame(daily_stats)


def plot_weekly_wr(df_weekly):
    """Plot weekly WR with Feb 15-28 highlighted"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    weeks = df_weekly['week'].values
    x_pos = np.arange(len(weeks))
    
    # Identify Feb weeks
    feb_15_28_mask = df_weekly['start_date'].apply(
        lambda x: pd.to_datetime(x) >= pd.to_datetime('2026-02-15')
    ) & df_weekly['end_date'].apply(
        lambda x: pd.to_datetime(x) <= pd.to_datetime('2026-02-28')
    )
    
    colors = ['red' if m else 'steelblue' for m in feb_15_28_mask]
    
    # Plot 1: Win Rate
    ax1.bar(x_pos, df_weekly['wr'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=73, color='green', linestyle='--', linewidth=2, label='Lab Average (73%)')
    ax1.axhline(y=52, color='red', linestyle='--', linewidth=2, label='Live Feb 15-28 (52%)')
    ax1.set_title('Weekly Win Rate - Lab Trades (Red = Feb 15-28 period)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Win Rate %')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{w}\n{s}" for w, s in zip(weeks, df_weekly['start_date'])], 
                        rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Plot 2: Total Profit
    profit_colors = ['green' if p > 0 else 'red' for p in df_weekly['total_profit']]
    
    ax2.bar(x_pos, df_weekly['total_profit'], color=profit_colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Weekly Total Profit', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Profit $')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{w}\n{s}" for w, s in zip(weeks, df_weekly['start_date'])], 
                        rotation=45, ha='right', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    print("="*100)
    print("WEEKLY WIN RATE ANALYSIS - LAB TRADES")
    print("="*100)
    
    # Load all trades
    print("\n📂 Loading all lab trades...")
    df = load_all_lab_trades()
    print(f"✅ Loaded {len(df)} total trades")
    print(f"   Date range: {df['sell_time'].min().date()} → {df['sell_time'].max().date()}")
    
    # Weekly analysis
    print("\n" + "="*100)
    print("WEEKLY STATISTICS")
    print("="*100)
    
    df_weekly = calculate_weekly_wr(df)
    
    print(f"\n{'Week':<12} {'Start':<12} {'End':<12} {'Trades':>8} {'WR%':>8} {'Profit':>12} {'Avg':>10}")
    print("-"*90)
    
    for _, row in df_weekly.iterrows():
        print(f"{row['week']:<12} {str(row['start_date']):<12} {str(row['end_date']):<12} "
              f"{row['trades']:>8} {row['wr']:>7.1f}% ${row['total_profit']:>11.2f} ${row['avg_profit']:>9.2f}")
    
    # Highlight Feb 15-28
    feb_weeks = df_weekly[
        (pd.to_datetime(df_weekly['start_date']) >= pd.to_datetime('2026-02-15')) &
        (pd.to_datetime(df_weekly['end_date']) <= pd.to_datetime('2026-02-28'))
    ]
    
    if len(feb_weeks) > 0:
        print("\n" + "="*100)
        print("FEB 15-28 WEEKS (Your live trading period)")
        print("="*100)
        
        avg_wr_feb = feb_weeks['wr'].mean()
        avg_profit_feb = feb_weeks['total_profit'].mean()
        
        print(f"\nAverage WR (Feb 15-28 weeks): {avg_wr_feb:.1f}%")
        print(f"Average Profit per week: ${avg_profit_feb:.2f}")
        
        # Compare with overall
        overall_wr = (df['profit'] > 0).mean() * 100
        
        print(f"\nOverall lab WR: {overall_wr:.1f}%")
        print(f"Feb 15-28 WR: {avg_wr_feb:.1f}%")
        print(f"Difference: {avg_wr_feb - overall_wr:+.1f}pp")
        
        if avg_wr_feb < overall_wr - 5:
            print("\n⚠️  Feb 15-28 had WORSE than average WR")
            print("   You entered during a bad period")
        elif avg_wr_feb > overall_wr + 5:
            print("\n✅ Feb 15-28 had BETTER than average WR")
        else:
            print("\n➡️  Feb 15-28 was AVERAGE")
    
    # Daily analysis for Feb
    print("\n" + "="*100)
    print("DAILY BREAKDOWN: Feb 15-28")
    print("="*100)
    
    df_daily = calculate_daily_wr(df, '2026-02-15', '2026-02-28')
    
    print(f"\n{'Date':<12} {'Trades':>8} {'WR%':>8} {'Profit':>12}")
    print("-"*50)
    
    for _, row in df_daily.iterrows():
        print(f"{str(row['date']):<12} {row['trades']:>8} {row['wr']:>7.1f}% ${row['total_profit']:>11.2f}")
    
    # Plot
    print("\n📊 Generating plots...")
    plot_weekly_wr(df_weekly)
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()