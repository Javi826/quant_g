#!/usr/bin/env python3
"""
Clean Trading Analysis - Weekly & Monthly Metrics
Focused on actionable insights with readable tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt


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
        all_trades.append(df)

    combined = pd.concat(all_trades, ignore_index=True)
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


def calculate_weekly_consistency_metrics(df_weekly):
    """Calculate streak and consistency metrics from weekly profit series"""

    profits = df_weekly['Profit'].values
    mean_profit = profits.mean()

    # Gaps (positive weeks) between consecutive negative weeks
    gaps = []
    current_gap = 0

    for profit in profits:
        if profit < 0:
            if current_gap > 0:
                gaps.append(current_gap)
            current_gap = 0
        else:
            current_gap += 1

    avg_gap = np.mean(gaps) if gaps else None
    median_gap = np.median(gaps) if gaps else None

    # % weeks above mean profit
    pct_above_mean = (profits > mean_profit).mean() * 100

    return avg_gap, median_gap, pct_above_mean


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
    """Weekly analysis with key metrics, including cumulative WR"""

    df = df.copy()
    df['week'] = df['sell_time'].dt.to_period('W')

    weekly_data = []
    cumulative_profit = 0
    cumulative_winners = 0
    cumulative_trades = 0

    for week, group in df.groupby('week'):
        week_start = group['sell_time'].min()

        total = len(group)
        winners = (group['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0

        cumulative_winners += winners
        cumulative_trades += total
        cumulative_wr = (cumulative_winners / cumulative_trades * 100) if cumulative_trades > 0 else 0

        week_profit = group['profit'].sum()
        cumulative_profit += week_profit
        profit_pct = (week_profit / initial_capital) * 100

        weekly_data.append({
            'Week_Start':     week_start.strftime('%Y-%m-%d'),
            'Trades':         total,
            'WR%':            round(wr, 1),
            'Cumulative_WR%': round(cumulative_wr, 1),
            'Profit':         round(week_profit, 2),
            'Profit_%':       round(profit_pct, 2),
            'Cumulative':     cumulative_profit,
        })

    df_weekly = pd.DataFrame(weekly_data)

    weekly_returns = pd.Series(df_weekly['Cumulative']).pct_change()
    consistency = (weekly_returns > 0).mean() * 100 if len(weekly_returns) > 1 else 0
    avg_profit_pct = df_weekly['Profit_%'].mean()

    df_weekly = df_weekly.drop(columns=['Cumulative'])

    return df_weekly, consistency, avg_profit_pct


def analyze_monthly(df, initial_capital):
    """Monthly analysis with key metrics, including cumulative WR"""

    df = df.copy()
    df['month'] = df['sell_time'].dt.to_period('M')

    monthly_data = []
    cumulative_profit = 0
    cumulative_winners = 0
    cumulative_trades = 0

    for month, group in df.groupby('month'):
        total = len(group)
        winners = (group['profit'] > 0).sum()
        wr = (winners / total * 100) if total > 0 else 0

        cumulative_winners += winners
        cumulative_trades += total
        cumulative_wr = (cumulative_winners / cumulative_trades * 100) if cumulative_trades > 0 else 0

        month_profit = group['profit'].sum()
        cumulative_profit += month_profit
        profit_pct = (month_profit / initial_capital) * 100

        monthly_data.append({
            'Month':          str(month),
            'Trades':         total,
            'WR%':            round(wr, 1),
            'Cumulative_WR%': round(cumulative_wr, 1),
            'Profit':         round(month_profit, 2),
            'Profit_%':       round(profit_pct, 2),
            'Cumulative':     cumulative_profit,
        })

    df_monthly = pd.DataFrame(monthly_data)

    monthly_returns = pd.Series(df_monthly['Cumulative']).pct_change()
    consistency = (monthly_returns > 0).mean() * 100 if len(monthly_returns) > 1 else 0
    avg_profit_pct = df_monthly['Profit_%'].mean()

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
            'Direction':    direction,
            'Trades':       total,
            'WR%':          round(wr, 1),
            'Total_Profit': round(total_profit, 2),
            'Profit_%':     round(profit_pct, 1),
            'Avg_Profit':   round(avg_profit, 2),
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

    df = df.copy()
    df['week'] = df['sell_time'].dt.to_period('W')

    weekly_payoffs = []
    for _, group in df.groupby('week'):
        _, _, payoff = calculate_payoff_ratio(group)
        if payoff is not None:
            weekly_payoffs.append(payoff)

    return np.mean(weekly_payoffs) if weekly_payoffs else None


def calculate_monthly_payoff(df):
    """Calculate average Payoff Ratio across all months"""

    df = df.copy()
    df['month'] = df['sell_time'].dt.to_period('M')

    monthly_payoffs = []
    for _, group in df.groupby('month'):
        _, _, payoff = calculate_payoff_ratio(group)
        if payoff is not None:
            monthly_payoffs.append(payoff)

    return np.mean(monthly_payoffs) if monthly_payoffs else None


def print_table(df, title, footer_text=None):
    """Print clean formatted table with optional footer"""

    print("\n" + "=" * 100)
    print(f"{title}")
    print("=" * 100)
    print()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df.to_string(index=False))

    if footer_text:
        print()
        print(footer_text)

    print()


def plot_weekly_profit(df_weekly):
    """Plot weekly profit (bars) and cumulative profit (line) as separate figure"""

    fig, ax1 = plt.subplots(figsize=(14, 6))

    weeks = range(len(df_weekly))
    profits = df_weekly['Profit'].values
    cumulative = df_weekly['Profit'].cumsum().values

    colors = ['steelblue' if p >= 0 else 'tomato' for p in profits]
    ax1.bar(weeks, profits, color=colors, alpha=0.7, label='Weekly Profit')
    ax1.axhline(y=0, color='white', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Week Number')
    ax1.set_ylabel('Profit')
    ax1.set_title('Weekly Profit & Cumulative Profit', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(weeks, cumulative, color='gold', linewidth=2, marker='o',
             markersize=3, label='Cumulative Profit')
    ax2.set_ylabel('Cumulative Profit')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()


def plot_wr_by_period(df_weekly, df_monthly):
    """Plot Win Rate (period + cumulative) for weekly and monthly periods"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # --- Weekly ---
    weeks = range(len(df_weekly))
    avg_wr_weekly = df_weekly['WR%'].mean()

    ax1.plot(weeks, df_weekly['WR%'], marker='o', linewidth=2,
             color='steelblue', markersize=4, label='Weekly WR%')
    ax1.plot(weeks, df_weekly['Cumulative_WR%'], linewidth=2,
             color='orange', linestyle='--', markersize=0, label='Cumulative WR%')
    ax1.axhline(y=avg_wr_weekly, color='green', linestyle='--',
                linewidth=1.5, label=f'Avg WR: {avg_wr_weekly:.1f}%')
    ax1.axhline(y=60, color='red', linestyle='--', linewidth=1,
                alpha=0.5, label='60% threshold')

    ax1.set_title('Weekly Win Rate', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Week Number')
    ax1.set_ylabel('Win Rate %')
    ax1.set_ylim([0, 100])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Monthly ---
    months = range(len(df_monthly))
    avg_wr_monthly = df_monthly['WR%'].mean()

    ax2.plot(months, df_monthly['WR%'], marker='o', linewidth=2,
             color='darkorange', markersize=6, label='Monthly WR%')
    ax2.plot(months, df_monthly['Cumulative_WR%'], linewidth=2,
             color='cyan', linestyle='--', markersize=0, label='Cumulative WR%')
    ax2.axhline(y=avg_wr_monthly, color='green', linestyle='--',
                linewidth=1.5, label=f'Avg WR: {avg_wr_monthly:.1f}%')
    ax2.axhline(y=60, color='red', linestyle='--', linewidth=1,
                alpha=0.5, label='60% threshold')

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
    print("\n" + "=" * 100)
    print("TRADING PERFORMANCE ANALYSIS")
    print("=" * 100)

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

    print("\n" + "=" * 100)
    print("OVERALL METRICS")
    print("=" * 100)
    print(f"Total Trades:    {len(df):,}")
    print(f"Win Rate:        {overall_wr:.1f}%")
    print(f"Total Profit:    {total_profit:,.2f}")
    print(f"Avg per Trade:   {avg_profit:.2f}")

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

    max_negative_weeks = calculate_weekly_negative_streaks(df_weekly)
    print(f"\nMax Consecutive Weeks with Negative Profit: {max_negative_weeks} weeks")

    avg_gap, median_gap, pct_above_mean = calculate_weekly_consistency_metrics(df_weekly)
    print(f"Avg Positive Weeks Between Negatives:       {avg_gap:.1f} weeks" if avg_gap is not None else "Avg Positive Weeks Between Negatives:       N/A")
    print(f"Median Positive Weeks Between Negatives:    {median_gap:.1f} weeks" if median_gap is not None else "Median Positive Weeks Between Negatives:    N/A")
    print(f"Weeks Above Mean Weekly Profit:             {pct_above_mean:.1f}%")

    # Monthly analysis
    df_monthly, monthly_consistency, avg_monthly_profit_pct = analyze_monthly(df, initial_capital)

    footer_monthly = f"Average Monthly Profit_%: {avg_monthly_profit_pct:.2f}%"
    print_table(df_monthly, "MONTHLY PERFORMANCE", footer_monthly)

    best_month_wr = df_monthly.loc[df_monthly['WR%'].idxmax()]
    worst_month_wr = df_monthly.loc[df_monthly['WR%'].idxmin()]

    print(f"Best Month (WR):  {best_month_wr['Month']} → WR: {best_month_wr['WR%']:.1f}% | Profit: {best_month_wr['Profit']:.2f}")
    print(f"Worst Month (WR): {worst_month_wr['Month']} → WR: {worst_month_wr['WR%']:.1f}% | Profit: {worst_month_wr['Profit']:.2f}")

    # Consistency summary
    consistency_data = {
        'Period':         ['Weekly', 'Monthly'],
        'Total_Periods':  [len(df_weekly), len(df_monthly)],
        'Profitable_%':   [round(weekly_consistency, 1), round(monthly_consistency, 1)],
    }
    df_consistency = pd.DataFrame(consistency_data)
    print_table(df_consistency, "CONSISTENCY SUMMARY")

    # Direction analysis
    df_direction = analyze_by_direction(df)
    if df_direction is not None and not df_direction.empty:
        print_table(df_direction, "PERFORMANCE BY DIRECTION (LONG vs SHORT)")

    # Payoff ratio summary
    avg_win_overall, avg_loss_overall, payoff_overall = calculate_payoff_ratio(df)
    payoff_weekly_avg = calculate_weekly_payoff(df)
    payoff_monthly_avg = calculate_monthly_payoff(df)

    payoff_summary = {
        'Period': ['Weekly', 'Monthly', 'Overall'],
        'Avg_Win': [
            round(avg_win_overall, 2) if avg_win_overall else 0,
            round(avg_win_overall, 2) if avg_win_overall else 0,
            round(avg_win_overall, 2) if avg_win_overall else 0,
        ],
        'Avg_Loss': [
            round(avg_loss_overall, 2) if avg_loss_overall else 0,
            round(avg_loss_overall, 2) if avg_loss_overall else 0,
            round(avg_loss_overall, 2) if avg_loss_overall else 0,
        ],
        'Payoff_Ratio': [
            round(payoff_weekly_avg, 2) if payoff_weekly_avg else 0,
            round(payoff_monthly_avg, 2) if payoff_monthly_avg else 0,
            round(payoff_overall, 2) if payoff_overall else 0,
        ],
    }

    df_payoff = pd.DataFrame(payoff_summary)

    warning = None
    if payoff_overall and payoff_overall < 1.0:
        warning = f"⚠️  Payoff < 1.0 → System depends on high Win Rate (>{int((1 / (1 + payoff_overall)) * 100)}%) to remain profitable"

    print_table(df_payoff, "PAYOFF RATIO SUMMARY", warning)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100 + "\n")

    print("📊 Generating charts...")
    plot_wr_by_period(df_weekly, df_monthly)
    plot_weekly_profit(df_weekly)
    plt.show()
    print("\n✅ Charts displayed")


if __name__ == "__main__":
    main()