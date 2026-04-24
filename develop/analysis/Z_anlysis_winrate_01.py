#devolop/analysis/Z_analysis_winrate_01.py
"""
Clean Trading Analysis - Weekly & Monthly Metrics
Focused on actionable insights with readable tables
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt


def load_all_lab_trades():
    """Load and combine all lab trades, return trades and initial capital"""
    
    lab_folder = Path(os.path.join(os.path.dirname(__file__), "..", "brief_trades"))
    files = glob(str(lab_folder / 'all_trades_*.csv'))
    
    if not files:
        print("⚠️  No trade files found in brief_trades/")
        return pd.DataFrame(), 0
    
    all_trades = []
    
    for filepath in files:
        df = pd.read_csv(filepath)
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        all_trades.append(df)
    
    combined = pd.concat(all_trades, ignore_index=True)
    
    # Calculate initial capital: 800 per strategy file
    initial_capital = 800 * len(files)
    
    print(f"   Strategies found: {len(files)}")
    print(f"   Initial capital: {initial_capital:,.0f} ({len(files)} × 800)")
    
    return combined.sort_values('sell_time').reset_index(drop=True), initial_capital


def calculate_streaks(df):
    """Calculate max winning and losing streaks"""
    
    wins = (df['profit'] > 0).astype(int)
    
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0
    
    for win in wins:
        if win == 1:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)
    
    return max_win_streak, max_loss_streak


def calculate_weekly_negative_streaks(df_weekly):
    """Calculate max consecutive weeks with negative profit"""
    
    profits = df_weekly['Profit'].values
    
    max_negative_streak = 0
    current_negative = 0
    
    for profit in profits:
        if profit < 0:
            current_negative += 1
            max_negative_streak = max(max_negative_streak, current_negative)
        else:
            current_negative = 0
    
    return max_negative_streak


def analyze_weekly(df, initial_capital):
    """Weekly analysis with key metrics"""
    
    df['week'] = df['sell_time'].dt.to_period('W')
    
    # Calculate cumulative profit per week
    weekly_data = []
    cumulative_profit = 0
    
    for week, group in df.groupby('week'):
        week_start = group['sell_time'].min()
        
        total = len(group)
        winners = (group['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        week_profit = group['profit'].sum()
        cumulative_profit += week_profit
        
        # Calculate Profit_% as simple ROI (profit / fixed initial capital)
        profit_pct = (week_profit / initial_capital) * 100
        
        weekly_data.append({
            'Week_Start': week_start.strftime('%Y-%m-%d'),
            'Trades': total,
            'WR%': round(wr, 1),
            'Profit': round(week_profit, 2),
            'Profit_%': round(profit_pct, 2),
            'Cumulative': cumulative_profit
        })
    
    df_weekly = pd.DataFrame(weekly_data)
    
    # Consistency score (% weeks better than previous week)
    weekly_returns = pd.Series(df_weekly['Cumulative']).pct_change()
    consistency = (weekly_returns > 0).mean() * 100 if len(weekly_returns) > 1 else 0
    
    # Calculate average Profit_%
    avg_profit_pct = df_weekly['Profit_%'].mean()
    
    # Remove cumulative column from output (only used for consistency calc)
    df_weekly = df_weekly.drop(columns=['Cumulative'])
    
    return df_weekly, consistency, avg_profit_pct


def analyze_monthly(df, initial_capital):
    """Monthly analysis with key metrics"""
    
    df['month'] = df['sell_time'].dt.to_period('M')
    
    # Calculate cumulative profit per month
    monthly_data = []
    cumulative_profit = 0
    
    for month, group in df.groupby('month'):
        total = len(group)
        winners = (group['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        month_profit = group['profit'].sum()
        cumulative_profit += month_profit
        
        # Calculate Profit_% as simple ROI (profit / fixed initial capital)
        profit_pct = (month_profit / initial_capital) * 100
        
        monthly_data.append({
            'Month': str(month),
            'Trades': total,
            'WR%': round(wr, 1),
            'Profit': round(month_profit, 2),
            'Profit_%': round(profit_pct, 2),
            'Cumulative': cumulative_profit
        })
    
    df_monthly = pd.DataFrame(monthly_data)
    
    # Consistency score (% months better than previous month)
    monthly_returns = pd.Series(df_monthly['Cumulative']).pct_change()
    consistency = (monthly_returns > 0).mean() * 100 if len(monthly_returns) > 1 else 0
    
    # Calculate average Profit_%
    avg_profit_pct = df_monthly['Profit_%'].mean()
    
    # Remove cumulative column from output (only used for consistency calc)
    df_monthly = df_monthly.drop(columns=['Cumulative'])
    
    return df_monthly, consistency, avg_profit_pct


def analyze_by_direction(df):
    """Analyze LONG vs SHORT performance"""
    
    if 'position_type' not in df.columns:
        return None
    
    total_profit_all = df['profit'].sum()
    
    direction_stats = []
    
    for direction in ['LONG', 'SHORT']:
        subset = df[df['position_type'] == direction]
        
        if len(subset) == 0:
            continue
        
        total = len(subset)
        winners = (subset['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0
        
        total_profit = subset['profit'].sum()
        avg_profit = subset['profit'].mean()
        profit_pct = (total_profit / total_profit_all * 100) if total_profit_all != 0 else 0
        
        direction_stats.append({
            'Direction': direction,
            'Trades': total,
            'WR%': round(wr, 1),
            'Total_Profit': round(total_profit, 2),
            'Profit_%': round(profit_pct, 1),
            'Avg_Profit': round(avg_profit, 2)
        })
    
    return pd.DataFrame(direction_stats) if direction_stats else None


def calculate_payoff_ratio(df):
    """Calculate Payoff Ratio (Avg Win / Avg Loss)"""
    
    wins = df[df['profit'] > 0]['profit']
    losses = df[df['profit'] < 0]['profit']
    
    if len(wins) == 0 or len(losses) == 0:
        return None, None, None
    
    avg_win = wins.mean()
    avg_loss = losses.mean()
    payoff = avg_win / abs(avg_loss)
    
    return avg_win, avg_loss, payoff


def calculate_weekly_payoff(df):
    """Calculate average Payoff Ratio across all weeks"""
    
    df['week'] = df['sell_time'].dt.to_period('W')
    
    weekly_payoffs = []
    
    for week, group in df.groupby('week'):
        avg_win, avg_loss, payoff = calculate_payoff_ratio(group)
        if payoff is not None:
            weekly_payoffs.append(payoff)
    
    return np.mean(weekly_payoffs) if weekly_payoffs else None


def calculate_monthly_payoff(df):
    """Calculate average Payoff Ratio across all months"""
    
    df['month'] = df['sell_time'].dt.to_period('M')
    
    monthly_payoffs = []
    
    for month, group in df.groupby('month'):
        avg_win, avg_loss, payoff = calculate_payoff_ratio(group)
        if payoff is not None:
            monthly_payoffs.append(payoff)
    
    return np.mean(monthly_payoffs) if monthly_payoffs else None


def print_table(df, title, footer_text=None):
    """Print clean formatted table with optional footer"""
    
    print("\n" + "="*100)
    print(f"{title}")
    print("="*100)
    print()
    
    # Configure display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df.to_string(index=False))
    
    if footer_text:
        print()
        print(footer_text)
    
    print()


def plot_wr_by_period(df_weekly, df_monthly):
    """Plot Win Rate by period (weekly and monthly)"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Weekly WR
    weeks = range(len(df_weekly))
    ax1.plot(weeks, df_weekly['WR%'], marker='o', linewidth=2, 
             color='steelblue', markersize=4)
    ax1.axhline(y=df_weekly['WR%'].mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Average: {df_weekly["WR%"].mean():.1f}%')
    ax1.axhline(y=60, color='red', linestyle='--', linewidth=1, 
                alpha=0.5, label='60% threshold')
    ax1.set_title('Weekly Win Rate', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Week Number')
    ax1.set_ylabel('Win Rate %')
    ax1.set_ylim([0, 100])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Monthly WR
    months = range(len(df_monthly))
    ax2.plot(months, df_monthly['WR%'], marker='o', linewidth=2, 
             color='darkorange', markersize=6)
    ax2.axhline(y=df_monthly['WR%'].mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Average: {df_monthly["WR%"].mean():.1f}%')
    ax2.axhline(y=60, color='red', linestyle='--', linewidth=1, 
                alpha=0.5, label='60% threshold')
    
    # Add month labels on x-axis
    ax2.set_xticks(months)
    ax2.set_xticklabels(df_monthly['Month'], rotation=45, ha='right')
    
    ax2.set_title('Monthly Win Rate', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Win Rate %')
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()


def main():
    print("\n" + "="*100)
    print("TRADING PERFORMANCE ANALYSIS")
    print("="*100)
    
    # Load trades
    print("\n📂 Loading trades...")
    df, initial_capital = load_all_lab_trades()
    
    if df.empty:
        print("❌ No trades found. Exiting.")
        return
    
    print(f"✅ Loaded {len(df):,} trades")
    print(f"   Period: {df['sell_time'].min().date()} → {df['sell_time'].max().date()}")
    
    # Overall metrics
    overall_wr = (df['profit'] > 0).mean() * 100
    total_profit = df['profit'].sum()
    avg_profit = df['profit'].mean()
    
    print("\n" + "="*100)
    print("OVERALL METRICS")
    print("="*100)
    print(f"Total Trades:    {len(df):,}")
    print(f"Win Rate:        {overall_wr:.1f}%")
    print(f"Total Profit:    {total_profit:,.2f}")
    print(f"Avg per Trade:   {avg_profit:.2f}")
    
    # Streaks
    max_win, max_loss = calculate_streaks(df)
    print(f"\nMax Win Streak:  {max_win} consecutive wins")
    print(f"Max Loss Streak: {max_loss} consecutive losses")
    
    # Weekly analysis
    df_weekly, weekly_consistency, avg_weekly_profit_pct = analyze_weekly(df, initial_capital)
    
    footer_weekly = f"Average Weekly Profit_%: {avg_weekly_profit_pct:.2f}%"
    print_table(df_weekly, "WEEKLY PERFORMANCE", footer_weekly)
    
    best_week_wr = df_weekly.loc[df_weekly['WR%'].idxmax()]
    worst_week_wr = df_weekly.loc[df_weekly['WR%'].idxmin()]
    
    print(f"Best Week (WR):  {best_week_wr['Week_Start']} → WR: {best_week_wr['WR%']:.1f}% | Profit: {best_week_wr['Profit']:.2f}")
    print(f"Worst Week (WR): {worst_week_wr['Week_Start']} → WR: {worst_week_wr['WR%']:.1f}% | Profit: {worst_week_wr['Profit']:.2f}")
    
    # Weekly negative streak
    max_negative_weeks = calculate_weekly_negative_streaks(df_weekly)
    print(f"\nMax Consecutive Weeks with Negative Profit: {max_negative_weeks} weeks")
    
    # Monthly analysis
    df_monthly, monthly_consistency, avg_monthly_profit_pct = analyze_monthly(df, initial_capital)
    
    footer_monthly = f"Average Monthly Profit_%: {avg_monthly_profit_pct:.2f}%"
    print_table(df_monthly, "MONTHLY PERFORMANCE", footer_monthly)
    
    best_month_wr = df_monthly.loc[df_monthly['WR%'].idxmax()]
    worst_month_wr = df_monthly.loc[df_monthly['WR%'].idxmin()]
    
    print(f"Best Month (WR):  {best_month_wr['Month']} → WR: {best_month_wr['WR%']:.1f}% | Profit: {best_month_wr['Profit']:.2f}")
    print(f"Worst Month (WR): {worst_month_wr['Month']} → WR: {worst_month_wr['WR%']:.1f}% | Profit: {worst_month_wr['Profit']:.2f}")
    
    # Consistency summary table
    consistency_data = {
        'Period': ['Weekly', 'Monthly'],
        'Total_Periods': [len(df_weekly), len(df_monthly)],
        'Profitable_%': [round(weekly_consistency, 1), round(monthly_consistency, 1)]
    }
    df_consistency = pd.DataFrame(consistency_data)
    print_table(df_consistency, "CONSISTENCY SUMMARY")
    
    # Direction analysis
    df_direction = analyze_by_direction(df)
    
    if df_direction is not None and not df_direction.empty:
        print_table(df_direction, "PERFORMANCE BY DIRECTION (LONG vs SHORT)")
    
    # =========================================================================
    # PAYOFF RATIO SUMMARY
    # =========================================================================
    
    # Overall Payoff
    avg_win_overall, avg_loss_overall, payoff_overall = calculate_payoff_ratio(df)
    
    # Weekly average Payoff
    payoff_weekly_avg = calculate_weekly_payoff(df)
    
    # Monthly average Payoff
    payoff_monthly_avg = calculate_monthly_payoff(df)
    
    # Build summary table
    payoff_summary = {
        'Period': ['Weekly', 'Monthly', 'Overall'],
        'Avg_Win': [
            round(avg_win_overall, 2) if avg_win_overall else 0,
            round(avg_win_overall, 2) if avg_win_overall else 0,
            round(avg_win_overall, 2) if avg_win_overall else 0
        ],
        'Avg_Loss': [
            round(avg_loss_overall, 2) if avg_loss_overall else 0,
            round(avg_loss_overall, 2) if avg_loss_overall else 0,
            round(avg_loss_overall, 2) if avg_loss_overall else 0
        ],
        'Payoff_Ratio': [
            round(payoff_weekly_avg, 2) if payoff_weekly_avg else 0,
            round(payoff_monthly_avg, 2) if payoff_monthly_avg else 0,
            round(payoff_overall, 2) if payoff_overall else 0
        ]
    }
    
    df_payoff = pd.DataFrame(payoff_summary)
    
    # Add warning footer if Payoff < 1.0
    warning = None
    if payoff_overall and payoff_overall < 1.0:
        warning = f"⚠️  Payoff < 1.0 → System depends on high Win Rate (>{int((1/(1+payoff_overall))*100)}%) to remain profitable"
    
    print_table(df_payoff, "PAYOFF RATIO SUMMARY", warning)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100 + "\n")
    
    # Generate plot
    print("📊 Generating Win Rate chart...")
    
    plot_wr_by_period(df_weekly, df_monthly)
    
    plt.show()
    
    print("\n✅ Chart displayed")


if __name__ == "__main__":
    main()