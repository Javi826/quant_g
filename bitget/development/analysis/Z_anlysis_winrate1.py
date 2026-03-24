#!/usr/bin/env python3
"""
Trading Analysis by Day of Week - Opening Time
Analyze WR and profit impact of excluding each day
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob


def load_all_lab_trades():
    """Load and combine all lab trades, return trades and initial capital"""
    
    lab_folder = Path('/home/javi/projects/quant/quant_g/bitget/development/brief_trades')
    files = glob(str(lab_folder / 'all_trades_*.xlsx'))
    
    if not files:
        print("⚠️  No trade files found in brief_trades/")
        return pd.DataFrame(), 0
    
    all_trades = []
    
    for filepath in files:
        df = pd.read_excel(filepath)
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        
        # Add buy_time if not present (use sell_time as fallback)
        if 'buy_time' not in df.columns:
            df['buy_time'] = df['sell_time']
        else:
            df['buy_time'] = pd.to_datetime(df['buy_time'])
        
        all_trades.append(df)
    
    combined = pd.concat(all_trades, ignore_index=True)
    
    # Calculate initial capital: 800 per strategy file
    initial_capital = 800 * len(files)
    
    print(f"   Strategies found: {len(files)}")
    print(f"   Initial capital: {initial_capital:,.0f} ({len(files)} × 800)")
    
    return combined.sort_values('sell_time').reset_index(drop=True), initial_capital


def calculate_payoff_ratio(df):
    """Calculate Payoff Ratio (Avg Win / Avg Loss)"""
    
    wins = df[df['profit'] > 0]['profit']
    losses = df[df['profit'] < 0]['profit']
    
    if len(wins) == 0 or len(losses) == 0:
        return None
    
    avg_win = wins.mean()
    avg_loss = losses.mean()
    payoff = avg_win / abs(avg_loss)
    
    return payoff


def analyze_by_day_of_week(df):
    """Analyze performance by day of week (opening time)"""
    
    # Add day of week column based on buy_time
    df['day_of_week'] = df['buy_time'].dt.day_name()
    
    # Define order of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    day_stats = []
    
    for day in day_order:
        subset = df[df['day_of_week'] == day]
        
        if len(subset) == 0:
            continue
        
        total = len(subset)
        winners = (subset['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        total_profit = subset['profit'].sum()
        avg_profit = subset['profit'].mean()
        payoff = calculate_payoff_ratio(subset)
        
        day_stats.append({
            'Day': day,
            'Trades': total,
            'WR%': round(wr, 1),
            'Total_Profit': round(total_profit, 2),
            'Avg_Profit': round(avg_profit, 2),
            'Payoff_Ratio': round(payoff, 2) if payoff else 0
        })
    
    return pd.DataFrame(day_stats)


def get_session(hour_utc):
    """Determine trading session based on UTC hour"""
    if 0 <= hour_utc < 8:
        return 'Asia'
    elif 8 <= hour_utc < 16:
        return 'Europe'
    else:  # 16 <= hour_utc < 24
        return 'America'


def analyze_by_session(df):
    """Analyze performance by trading session (opening time)"""
    
    # Add session column based on buy_time hour
    df['hour_utc'] = df['buy_time'].dt.hour
    df['session'] = df['hour_utc'].apply(get_session)
    
    # Define order of sessions
    session_order = ['Asia', 'Europe', 'America']
    
    session_stats = []
    
    for session in session_order:
        subset = df[df['session'] == session]
        
        if len(subset) == 0:
            continue
        
        total = len(subset)
        winners = (subset['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        total_profit = subset['profit'].sum()
        avg_profit = subset['profit'].mean()
        payoff = calculate_payoff_ratio(subset)
        
        session_stats.append({
            'Session': session,
            'Trades': total,
            'WR%': round(wr, 1),
            'Total_Profit': round(total_profit, 2),
            'Avg_Profit': round(avg_profit, 2),
            'Payoff_Ratio': round(payoff, 2) if payoff else 0
        })
    
    return pd.DataFrame(session_stats)


def analyze_system_excluding_session(df, initial_capital):
    """Analyze system performance when excluding each session"""
    
    # Add session column based on buy_time hour
    df['hour_utc'] = df['buy_time'].dt.hour
    df['session'] = df['hour_utc'].apply(get_session)
    
    # Define order of sessions
    session_order = ['Asia', 'Europe', 'America']
    
    results = []
    
    # Full system (no exclusions)
    total = len(df)
    winners = (df['profit'] > 0).sum()
    wr = (winners / total * 100) if total > 0 else 0
    total_profit = df['profit'].sum()
    net_gain_pct = (total_profit / initial_capital) * 100
    payoff = calculate_payoff_ratio(df)
    
    results.append({
        'Excluded_Session': 'NONE (Full)',
        'Trades': total,
        'WR%': round(wr, 1),
        'Total_Profit': round(total_profit, 2),
        'Net_Gain_%': round(net_gain_pct, 2),
        'Payoff_Ratio': round(payoff, 2) if payoff else 0
    })
    
    # Exclude each session
    for session in session_order:
        subset = df[df['session'] != session]
        
        if len(subset) == 0:
            continue
        
        total = len(subset)
        winners = (subset['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        total_profit = subset['profit'].sum()
        net_gain_pct = (total_profit / initial_capital) * 100
        payoff = calculate_payoff_ratio(subset)
        
        results.append({
            'Excluded_Session': session,
            'Trades': total,
            'WR%': round(wr, 1),
            'Total_Profit': round(total_profit, 2),
            'Net_Gain_%': round(net_gain_pct, 2),
            'Payoff_Ratio': round(payoff, 2) if payoff else 0
        })
    
    return pd.DataFrame(results)


def analyze_system_excluding_day(df, initial_capital):
    """Analyze system performance when excluding each day"""
    
    # Add day of week column based on buy_time
    df['day_of_week'] = df['buy_time'].dt.day_name()
    
    # Define order of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    results = []
    
    # Full system (no exclusions)
    total = len(df)
    winners = (df['profit'] > 0).sum()
    wr = (winners / total * 100) if total > 0 else 0
    total_profit = df['profit'].sum()
    net_gain_pct = (total_profit / initial_capital) * 100
    payoff = calculate_payoff_ratio(df)
    
    results.append({
        'Excluded_Day': 'NONE (Full)',
        'Trades': total,
        'WR%': round(wr, 1),
        'Total_Profit': round(total_profit, 2),
        'Net_Gain_%': round(net_gain_pct, 2),
        'Payoff_Ratio': round(payoff, 2) if payoff else 0
    })
    
    # Exclude each day
    for day in day_order:
        subset = df[df['day_of_week'] != day]
        
        if len(subset) == 0:
            continue
        
        total = len(subset)
        winners = (subset['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        total_profit = subset['profit'].sum()
        net_gain_pct = (total_profit / initial_capital) * 100
        payoff = calculate_payoff_ratio(subset)
        
        results.append({
            'Excluded_Day': day,
            'Trades': total,
            'WR%': round(wr, 1),
            'Total_Profit': round(total_profit, 2),
            'Net_Gain_%': round(net_gain_pct, 2),
            'Payoff_Ratio': round(payoff, 2) if payoff else 0
        })
    
    return pd.DataFrame(results)


def print_table(df, title):
    """Print clean formatted table"""
    
    print("\n" + "="*100)
    print(f"{title}")
    print("="*100)
    print()
    
    # Configure display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df.to_string(index=False))
    print()


def main():
    print("\n" + "="*100)
    print("TRADING ANALYSIS BY DAY OF WEEK (OPENING TIME)")
    print("="*100)
    
    # Load trades
    print("\n📂 Loading trades...")
    df, initial_capital = load_all_lab_trades()
    
    if df.empty:
        print("❌ No trades found. Exiting.")
        return
    
    print(f"✅ Loaded {len(df):,} trades")
    print(f"   Period: {df['sell_time'].min().date()} → {df['sell_time'].max().date()}")
    
    # Analyze by day of week (opening)
    df_by_day = analyze_by_day_of_week(df)
    print_table(df_by_day, "WIN RATE BY DAY OF WEEK (OPENING)")
    
    # Best and worst days
    best_day_wr = df_by_day.loc[df_by_day['WR%'].idxmax()]
    worst_day_wr = df_by_day.loc[df_by_day['WR%'].idxmin()]
    
    print(f"Best Day (WR):   {best_day_wr['Day']} → WR: {best_day_wr['WR%']:.1f}% | Trades: {best_day_wr['Trades']}")
    print(f"Worst Day (WR):  {worst_day_wr['Day']} → WR: {worst_day_wr['WR%']:.1f}% | Trades: {worst_day_wr['Trades']}")
    
    best_day_profit = df_by_day.loc[df_by_day['Total_Profit'].idxmax()]
    worst_day_profit = df_by_day.loc[df_by_day['Total_Profit'].idxmin()]
    
    print(f"\nBest Day (Profit):  {best_day_profit['Day']} → Profit: {best_day_profit['Total_Profit']:.2f}")
    print(f"Worst Day (Profit): {worst_day_profit['Day']} → Profit: {worst_day_profit['Total_Profit']:.2f}")
    
    # Analyze system excluding each day
    df_exclusions = analyze_system_excluding_day(df, initial_capital)
    print_table(df_exclusions, "SYSTEM PERFORMANCE BY EXCLUDED DAY")
    
    # Calculate impact of excluding each day
    baseline_profit = df_exclusions[df_exclusions['Excluded_Day'] == 'NONE (Full)']['Total_Profit'].values[0]
    baseline_wr = df_exclusions[df_exclusions['Excluded_Day'] == 'NONE (Full)']['WR%'].values[0]
    
    print("📊 IMPACT ANALYSIS (BY DAY):")
    print("-" * 100)
    
    for _, row in df_exclusions.iterrows():
        if row['Excluded_Day'] == 'NONE (Full)':
            continue
        
        profit_diff = row['Total_Profit'] - baseline_profit
        wr_diff = row['WR%'] - baseline_wr
        
        status = "🟢 IMPROVED" if profit_diff > 0 else "🔴 WORSENED"
        
        print(f"{row['Excluded_Day']:<12} → Profit change: {profit_diff:+7.2f} | WR change: {wr_diff:+5.1f}% | {status}")
    
    # =========================================================================
    # SESSION ANALYSIS
    # =========================================================================
    
    # Analyze by session (opening)
    df_by_session = analyze_by_session(df)
    print_table(df_by_session, "WIN RATE BY SESSION (OPENING) - UTC Times: Asia 00-08, Europe 08-16, America 16-00")
    
    # Best and worst sessions
    best_session_wr = df_by_session.loc[df_by_session['WR%'].idxmax()]
    worst_session_wr = df_by_session.loc[df_by_session['WR%'].idxmin()]
    
    print(f"Best Session (WR):   {best_session_wr['Session']} → WR: {best_session_wr['WR%']:.1f}% | Trades: {best_session_wr['Trades']}")
    print(f"Worst Session (WR):  {worst_session_wr['Session']} → WR: {worst_session_wr['WR%']:.1f}% | Trades: {worst_session_wr['Trades']}")
    
    best_session_profit = df_by_session.loc[df_by_session['Total_Profit'].idxmax()]
    worst_session_profit = df_by_session.loc[df_by_session['Total_Profit'].idxmin()]
    
    print(f"\nBest Session (Profit):  {best_session_profit['Session']} → Profit: {best_session_profit['Total_Profit']:.2f}")
    print(f"Worst Session (Profit): {worst_session_profit['Session']} → Profit: {worst_session_profit['Total_Profit']:.2f}")
    
    # Analyze system excluding each session
    df_exclusions_session = analyze_system_excluding_session(df, initial_capital)
    print_table(df_exclusions_session, "SYSTEM PERFORMANCE BY EXCLUDED SESSION")
    
    # Calculate impact of excluding each session
    print("📊 IMPACT ANALYSIS (BY SESSION):")
    print("-" * 100)
    
    for _, row in df_exclusions_session.iterrows():
        if row['Excluded_Session'] == 'NONE (Full)':
            continue
        
        profit_diff = row['Total_Profit'] - baseline_profit
        wr_diff = row['WR%'] - baseline_wr
        
        status = "🟢 IMPROVED" if profit_diff > 0 else "🔴 WORSENED"
        
        print(f"{row['Excluded_Session']:<12} → Profit change: {profit_diff:+7.2f} | WR change: {wr_diff:+5.1f}% | {status}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()