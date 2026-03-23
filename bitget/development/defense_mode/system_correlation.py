#!/usr/bin/env python3
"""
System Metrics Analysis: Internal System Features (Week N-1) vs Performance (Week N)
Tests if system's own past behavior predicts future performance better than BTC metrics

Features:
- Past WR/Profit (1w, 2w, rolling averages)
- Streaks (bad weeks, good weeks)
- Drawdown metrics
- Volatility metrics
- Sharpe ratio rolling

Targets:
1. WR% (Win Rate)
2. Profit Total ($)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from sklearn.feature_selection import mutual_info_regression


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


def calculate_weekly_system_performance(df_trades):
    """Calculate system WR and Profit week by week"""
    df_trades['week'] = df_trades['sell_time'].dt.to_period('W')
    
    weekly_perf = []
    
    for week, group in df_trades.groupby('week'):
        wr = (group['profit'] > 0).mean() * 100
        profit_total = group['profit'].sum()
        trades = len(group)
        wins = (group['profit'] > 0).sum()
        losses = (group['profit'] <= 0).sum()
        avg_win = group[group['profit'] > 0]['profit'].mean() if wins > 0 else 0
        avg_loss = group[group['profit'] <= 0]['profit'].mean() if losses > 0 else 0
        
        # Calculate max consecutive losses in this week
        max_losing_streak = 0
        current_streak = 0
        for profit in group['profit'].values:
            if profit <= 0:
                current_streak += 1
                max_losing_streak = max(max_losing_streak, current_streak)
            else:
                current_streak = 0
        
        weekly_perf.append({
            'week': str(week),
            'wr': wr,
            'profit': profit_total,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_losing_streak': max_losing_streak
        })
    
    return pd.DataFrame(weekly_perf)


def calculate_bad_weeks_streak(wr_series, threshold=70):
    """Calculate consecutive bad weeks streak"""
    streaks = []
    current_streak = 0
    
    for wr in wr_series:
        if pd.notna(wr) and wr < threshold:
            current_streak += 1
        else:
            current_streak = 0
        streaks.append(current_streak)
    
    return pd.Series(streaks, index=wr_series.index)


def calculate_good_weeks_streak(wr_series, threshold=80):
    """Calculate consecutive good weeks streak"""
    streaks = []
    current_streak = 0
    
    for wr in wr_series:
        if pd.notna(wr) and wr > threshold:
            current_streak += 1
        else:
            current_streak = 0
        streaks.append(current_streak)
    
    return pd.Series(streaks, index=wr_series.index)


def calculate_drawdown_metrics(profit_series):
    """Calculate drawdown percentage from equity peak"""
    cumulative_profit = profit_series.cumsum()
    running_max = cumulative_profit.expanding().max()
    drawdown = cumulative_profit - running_max
    drawdown_pct = (drawdown / running_max.abs()) * 100
    drawdown_pct = drawdown_pct.fillna(0)
    
    return drawdown_pct


def calculate_weeks_in_drawdown(profit_series):
    """Count consecutive weeks in drawdown"""
    cumulative_profit = profit_series.cumsum()
    running_max = cumulative_profit.expanding().max()
    in_drawdown = cumulative_profit < running_max
    
    weeks_in_dd = []
    current_dd_weeks = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_dd_weeks += 1
        else:
            current_dd_weeks = 0
        weeks_in_dd.append(current_dd_weeks)
    
    return pd.Series(weeks_in_dd, index=profit_series.index)


def calculate_sharpe_rolling(profit_series, window=4):
    """Calculate rolling Sharpe ratio"""
    mean = profit_series.rolling(window=window).mean()
    std = profit_series.rolling(window=window).std()
    sharpe = mean / std
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    return sharpe


def calculate_system_features(df_system):
    """Calculate all system-based features"""
    
    print("\n🔧 Calculating system features...")
    
    df = df_system.copy()
    
    # 1. PAST PERFORMANCE (lag features)
    print("   [1/6] Past performance metrics...")
    df['prev_wr_1w'] = df['wr'].shift(1)
    df['prev_wr_2w'] = df['wr'].shift(2)
    df['prev_profit_1w'] = df['profit'].shift(1)
    df['prev_profit_2w'] = df['profit'].shift(2)
    
    # Rolling averages
    df['wr_ma3'] = df['wr'].rolling(3).mean().shift(1)
    df['wr_ma5'] = df['wr'].rolling(5).mean().shift(1)
    df['profit_ma3'] = df['profit'].rolling(3).mean().shift(1)
    df['profit_ma5'] = df['profit'].rolling(5).mean().shift(1)
    
    # 2. STREAKS
    print("   [2/6] Streak metrics...")
    df['bad_weeks_streak'] = calculate_bad_weeks_streak(df['wr']).shift(1)
    df['good_weeks_streak'] = calculate_good_weeks_streak(df['wr']).shift(1)
    
    # Count bad/good weeks in last N weeks - FIX: Apply comparison to each value
    df['bad_weeks_last5'] = df['wr'].rolling(5).apply(lambda x: (x < 70).sum()).shift(1)
    df['good_weeks_last5'] = df['wr'].rolling(5).apply(lambda x: (x > 80).sum()).shift(1)
    
    # Profit streaks
    df['profit_positive_streak'] = (df['profit'] > 0).astype(int)
    positive_streaks = []
    current = 0
    for val in df['profit_positive_streak']:
        if val == 1:
            current += 1
        else:
            current = 0
        positive_streaks.append(current)
    df['profit_positive_streak'] = pd.Series(positive_streaks).shift(1)
    
    # 3. DRAWDOWN METRICS
    print("   [3/6] Drawdown metrics...")
    df['drawdown_pct'] = calculate_drawdown_metrics(df['profit']).shift(1)
    df['drawdown_weeks'] = calculate_weeks_in_drawdown(df['profit']).shift(1)
    
    # Max drawdown in rolling window
    df['max_dd_4w'] = df['drawdown_pct'].rolling(4).min().shift(1)
    
    # 4. VOLATILITY METRICS
    print("   [4/6] Volatility metrics...")
    df['profit_std_4w'] = df['profit'].rolling(4).std().shift(1)
    df['profit_std_8w'] = df['profit'].rolling(8).std().shift(1)
    df['wr_std_4w'] = df['wr'].rolling(4).std().shift(1)
    df['wr_std_8w'] = df['wr'].rolling(8).std().shift(1)
    
    # Profit change
    df['profit_change_1w'] = (df['profit'].shift(1) - df['profit'].shift(2))
    df['profit_change_pct'] = (df['profit_change_1w'] / df['profit'].shift(2).abs()) * 100
    
    # 5. SHARPE/SORTINO
    print("   [5/6] Risk-adjusted metrics...")
    df['sharpe_4w'] = calculate_sharpe_rolling(df['profit'], 4).shift(1)
    df['sharpe_8w'] = calculate_sharpe_rolling(df['profit'], 8).shift(1)
    
    # Sortino (downside deviation only)
    def calculate_sortino(series, window=4):
        mean = series.rolling(window).mean()
        downside = series[series < 0].rolling(window).std()
        sortino = mean / downside
        sortino = sortino.replace([np.inf, -np.inf], np.nan)
        return sortino
    
    df['sortino_4w'] = calculate_sortino(df['profit'], 4).shift(1)
    
    # 6. TRADE CHARACTERISTICS
    print("   [6/6] Trade characteristics...")
    df['prev_trades_count'] = df['trades'].shift(1)
    df['prev_max_losing_streak'] = df['max_losing_streak'].shift(1)
    df['trades_change'] = df['trades'].shift(1) - df['trades'].shift(2)
    
    # Win/Loss ratio
    df['prev_win_loss_ratio'] = (df['wins'] / df['losses']).replace([np.inf, -np.inf], np.nan).shift(1)
    
    # Average win/loss
    df['prev_avg_win'] = df['avg_win'].shift(1)
    df['prev_avg_loss'] = df['avg_loss'].shift(1)
    df['prev_win_loss_avg_ratio'] = (df['avg_win'] / df['avg_loss'].abs()).replace([np.inf, -np.inf], np.nan).shift(1)
    
    print("   ✅ All system features calculated")
    
    return df


def interpret_correlation(corr):
    """Interpret correlation strength"""
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        return "Very Strong"
    elif abs_corr >= 0.5:
        return "Strong"
    elif abs_corr >= 0.3:
        return "Moderate"
    elif abs_corr >= 0.1:
        return "Weak"
    else:
        return "Very Weak"


def interpret_mi(mi):
    """Interpret mutual information strength"""
    if mi >= 0.5:
        return "Very Strong"
    elif mi >= 0.3:
        return "Strong"
    elif mi >= 0.15:
        return "Moderate"
    elif mi >= 0.05:
        return "Weak"
    else:
        return "Very Weak"


def calculate_correlations_for_target(df_system, feature_cols, target_col, target_name):
    """Calculate correlations and MI for a specific target"""
    
    print(f"\n🔍 Calculating correlations for target: {target_name}...")
    
    results = []
    
    for i, feature in enumerate(feature_cols, 1):
        if i % 5 == 0:
            print(f"   Progress: {i}/{len(feature_cols)} features...")
        
        # Remove rows with NaN
        valid_data = df_system[[target_col, feature]].dropna()
        
        if len(valid_data) < 5:
            continue
        
        # Calculate Pearson correlation
        corr = valid_data[target_col].corr(valid_data[feature])
        
        # Calculate Mutual Information
        X = valid_data[[feature]].values
        y = valid_data[target_col].values
        
        try:
            mi = mutual_info_regression(X, y, random_state=42)[0]
        except:
            mi = np.nan
        
        if pd.notna(corr):
            results.append({
                'feature': feature,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'mutual_info': mi,
                'corr_strength': interpret_correlation(corr),
                'mi_strength': interpret_mi(mi) if pd.notna(mi) else 'N/A',
                'n_samples': len(valid_data)
            })
    
    return pd.DataFrame(results)


def display_results(df_results, target_name):
    """Display correlation results for a target"""
    
    # Sort by mutual information
    df_results = df_results.sort_values('mutual_info', ascending=False)
    
    print("\n" + "="*160)
    print(f"SYSTEM FEATURES vs {target_name} (Week N)")
    print("="*160)
    
    print(f"\n{'Rank':<6} {'Feature':<35} {'Correlation':>12} {'Corr Str':<15} {'Mutual Info':>12} {'MI Str':<15} {'Samples':>8}")
    print("-"*160)
    
    for rank, (_, row) in enumerate(df_results.head(25).iterrows(), 1):
        feature = row['feature']
        corr = row['correlation']
        corr_str = row['corr_strength']
        mi = row['mutual_info']
        mi_str = row['mi_strength']
        n_samples = row['n_samples']
        
        print(f"{rank:<6} {feature:<35} {corr:>+12.3f} {corr_str:<15} {mi:>12.3f} {mi_str:<15} {n_samples:>8}")
    
    # Summary
    print("\n" + "="*160)
    print(f"SUMMARY: {target_name}")
    print("="*160)
    
    strong_mi = df_results[df_results['mutual_info'] > 0.3]
    strong_positive = df_results[df_results['correlation'] > 0.5]
    strong_negative = df_results[df_results['correlation'] < -0.5]
    moderate = df_results[(df_results['abs_correlation'] >= 0.3) & (df_results['abs_correlation'] < 0.5)]
    
    print(f"\n✅ Strong Predictors (MI > 0.3): {len(strong_mi)}")
    if len(strong_mi) > 0:
        for _, row in strong_mi.iterrows():
            print(f"   • {row['feature']:<35} MI: {row['mutual_info']:.3f}, Corr: {row['correlation']:>+.3f}")
    
    print(f"\n✅ Strong Positive Correlation (corr > +0.5): {len(strong_positive)}")
    if len(strong_positive) > 0:
        for _, row in strong_positive.iterrows():
            print(f"   • {row['feature']:<35} Corr: {row['correlation']:>+.3f}, MI: {row['mutual_info']:.3f}")
    
    print(f"\n❌ Strong Negative Correlation (corr < -0.5): {len(strong_negative)}")
    if len(strong_negative) > 0:
        for _, row in strong_negative.iterrows():
            print(f"   • {row['feature']:<35} Corr: {row['correlation']:>+.3f}, MI: {row['mutual_info']:.3f}")
    
    print(f"\n🟡 Moderate Correlation (|corr| 0.3-0.5): {len(moderate)}")
    if len(moderate) > 0:
        for _, row in moderate.head(10).iterrows():
            print(f"   • {row['feature']:<35} Corr: {row['correlation']:>+.3f}, MI: {row['mutual_info']:.3f}")
    
    return df_results


def main():
    print("="*160)
    print("SYSTEM METRICS ANALYSIS: Internal Features vs Future Performance")
    print("="*160)
    
    # Load data
    print("\n📂 Loading trades...")
    df_trades = load_all_lab_trades()
    print(f"✅ Loaded {len(df_trades)} trades")
    
    # Calculate weekly performance
    print("\n🔍 Calculating weekly performance...")
    df_system = calculate_weekly_system_performance(df_trades)
    print(f"✅ {len(df_system)} weeks with trades")
    
    # Filter weeks with minimum trades
    min_trades = 20
    df_system = df_system[df_system['trades'] >= min_trades]
    print(f"✅ Filtered to {len(df_system)} weeks (≥{min_trades} trades/week)")
    
    # Calculate system features
    df_system = calculate_system_features(df_system)
    
    # Get feature columns (exclude original columns)
    exclude_cols = ['week', 'wr', 'profit', 'trades', 'wins', 'losses', 'avg_win', 'avg_loss', 'max_losing_streak']
    feature_cols = [col for col in df_system.columns if col not in exclude_cols]
    
    print(f"\n✅ Calculated {len(feature_cols)} system features")
    
    # Calculate correlations for WR
    print("\n" + "="*160)
    print("TARGET 1: WIN RATE %")
    print("="*160)
    df_results_wr = calculate_correlations_for_target(df_system, feature_cols, 'wr', 'WR%')
    df_results_wr = display_results(df_results_wr, 'WR%')
    
    # Calculate correlations for Profit
    print("\n" + "="*160)
    print("TARGET 2: PROFIT TOTAL")
    print("="*160)
    df_results_profit = calculate_correlations_for_target(df_system, feature_cols, 'profit', 'Profit')
    df_results_profit = display_results(df_results_profit, 'Profit')
    
    # Comparison
    print("\n" + "="*160)
    print("COMPARISON: SYSTEM FEATURES vs BTC FEATURES")
    print("="*160)
    
    best_wr_mi = df_results_wr['mutual_info'].max()
    best_profit_mi = df_results_profit['mutual_info'].max()
    
    best_wr_corr = df_results_wr['abs_correlation'].max()
    best_profit_corr = df_results_profit['abs_correlation'].max()
    
    print(f"\nBest Mutual Information (System Features):")
    print(f"   WR%:    {best_wr_mi:.3f}")
    print(f"   Profit: {best_profit_mi:.3f}")
    
    print(f"\nBest Absolute Correlation (System Features):")
    print(f"   WR%:    {best_wr_corr:.3f}")
    print(f"   Profit: {best_profit_corr:.3f}")
    
    print(f"\n💡 COMPARISON WITH BTC METRICS:")
    print(f"   BTC best MI (WR):     0.160")
    print(f"   System best MI (WR):  {best_wr_mi:.3f}  {'✅ BETTER' if best_wr_mi > 0.160 else '❌ WORSE'}")
    print(f"")
    print(f"   BTC best MI (Profit):    0.186")
    print(f"   System best MI (Profit): {best_profit_mi:.3f}  {'✅ BETTER' if best_profit_mi > 0.186 else '❌ WORSE'}")
    
    # Save results
    output_dir = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/files')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'system_features_correlations.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results_wr.to_excel(writer, sheet_name='WR_Correlations', index=False)
        df_results_profit.to_excel(writer, sheet_name='Profit_Correlations', index=False)
        df_system.to_excel(writer, sheet_name='Weekly_Data', index=False)
    
    print(f"\n💾 Saved results to: {output_file}")
    
    # Top actionable features
    print("\n" + "="*160)
    print("TOP 10 MOST PREDICTIVE SYSTEM FEATURES")
    print("="*160)
    
    print("\n🎯 FOR PREDICTING WR%:")
    top_wr = df_results_wr.head(10)
    for rank, (_, row) in enumerate(top_wr.iterrows(), 1):
        print(f"   {rank}. {row['feature']:<35} MI: {row['mutual_info']:.3f}, Corr: {row['correlation']:>+.3f}")
    
    print("\n💰 FOR PREDICTING PROFIT:")
    top_profit = df_results_profit.head(10)
    for rank, (_, row) in enumerate(top_profit.iterrows(), 1):
        print(f"   {rank}. {row['feature']:<35} MI: {row['mutual_info']:.3f}, Corr: {row['correlation']:>+.3f}")
    
    print("\n" + "="*160)
    print("\n💡 KEY INSIGHTS:")
    print("   • System's own past behavior should be a better predictor than BTC metrics")
    print("   • Look for features with MI > 0.3 or |Corr| > 0.5")
    print("   • If prev_wr, wr_ma3, or bad_weeks_streak show strong correlation → USE THEM")
    print("   • Combine best system features with best BTC features for defensive mode")
    print("\n" + "="*160)


if __name__ == "__main__":
    main()