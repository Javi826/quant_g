#!/usr/bin/env python3
"""
Comprehensive Lagged Correlation Analysis: 52 BTC Metrics (Week N-1) vs Multiple Targets (Week N)
Tests which BTC metrics from LAST WEEK predict THIS WEEK's performance
Using both Pearson Correlation and Mutual Information

Targets:
1. WR% (Win Rate)
2. Profit Total ($)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression


# ============================================================================
# CONFIGURATION
# ============================================================================
BTC_TIMEFRAME = '1Dutc'  # Options: '4H' or '1Dutc'
# ============================================================================


def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_realized_volatility(df, period=42):
    """Calculate realized volatility (annualized)"""
    log_returns = np.log(df['close'] / df['close'].shift())
    
    # Adjust for timeframe
    if BTC_TIMEFRAME == '1Dutc':
        annualization_factor = np.sqrt(252)  # Daily
    else:  # 4H
        annualization_factor = np.sqrt(252 * 6)  # 4H bars
    
    rv = log_returns.rolling(window=period).std() * annualization_factor
    return rv * 100


def calculate_bb_width(df, period=20, std_dev=2):
    """Calculate Bollinger Band Width"""
    ma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    bb_width = ((upper - lower) / ma) * 100
    return bb_width


def calculate_er(df, period=10):
    """Calculate Efficiency Ratio"""
    change = np.abs(df['close'] - df['close'].shift(period))
    volatility = np.abs(df['close'] - df['close'].shift()).rolling(window=period).sum()
    er = change / volatility
    return er


def calculate_adx(df, period=14):
    """Calculate ADX and DI"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di


def calculate_rsi(df, period=14):
    """Calculate RSI"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return histogram, signal_line


def calculate_stochastic(df, period=14):
    """Calculate Stochastic Oscillator"""
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=3).mean()
    return k, d


def calculate_choppiness(df, period=14):
    """Calculate Choppiness Index"""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(window=period).sum()
    chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(period)
    
    return chop


def calculate_hurst(df, period=100):
    """Calculate Hurst Exponent"""
    def hurst_exp(ts):
        if len(ts) < 2:
            return np.nan
        
        lags = range(2, min(20, len(ts)))
        tau = []
        
        for lag in lags:
            pp = np.subtract(ts[lag:], ts[:-lag])
            tau.append(np.std(pp))
        
        if len(tau) < 2:
            return np.nan
            
        tau = np.array(tau)
        lags = np.array(list(range(2, 2+len(tau))))
        
        valid_mask = (tau > 0) & (lags > 0)
        if valid_mask.sum() < 2:
            return np.nan
            
        reg = np.polyfit(np.log(lags[valid_mask]), np.log(tau[valid_mask]), 1)
        return reg[0]
    
    hurst_values = df['close'].rolling(window=period).apply(hurst_exp, raw=True)
    return hurst_values


def calculate_approximate_entropy(df, period=100, m=2, r=0.2):
    """Calculate Approximate Entropy"""
    def apen(ts, m, r):
        if len(ts) < m + 1:
            return np.nan
        
        def _maxdist(x_i, x_j, m):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x = [[ts[j] for j in range(i, i + m)] for i in range(len(ts) - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j, m) <= r]) / (len(ts) - m + 1.0) for x_i in x]
            return sum(np.log(C)) / (len(ts) - m + 1.0)
        
        return abs(_phi(m + 1) - _phi(m))
    
    std = df['close'].rolling(window=period).std()
    apen_values = df['close'].rolling(window=period).apply(
        lambda x: apen(x.values, m, r * x.std()) if x.std() > 0 else np.nan,
        raw=False
    )
    return apen_values


def calculate_permutation_entropy(df, period=100, order=3, delay=1):
    """Calculate Permutation Entropy"""
    def perm_entropy(ts, order, delay):
        if len(ts) < order * delay:
            return np.nan
        
        permutations = []
        for i in range(len(ts) - (order - 1) * delay):
            segment = [ts[i + j * delay] for j in range(order)]
            permutations.append(tuple(np.argsort(segment)))
        
        if len(permutations) == 0:
            return np.nan
        
        p = pd.Series(permutations).value_counts(normalize=True)
        return entropy(p)
    
    pe_values = df['close'].rolling(window=period).apply(
        lambda x: perm_entropy(x.values, order, delay),
        raw=False
    )
    return pe_values


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
    """Load and prepare BTC data with all 52 metrics"""
    
    # Select file based on timeframe
    if BTC_TIMEFRAME == '1Dutc':
        btc_file = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/BTCUSDT_1Dutc.parquet')
    else:  # 4H
        btc_file = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/BTCUSDT_4H.parquet')
    
    print(f"   Using BTC {BTC_TIMEFRAME} data: {btc_file.name}")
    
    df = pd.read_parquet(btc_file)
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    
    df = df.sort_values('ts').reset_index(drop=True)
    
    print("\n🔧 Calculating metrics (this may take a few minutes)...")
    
    # CATEGORY 1: Volatility (8)
    print("   [1/7] Volatility metrics...")
    df['atr_14'] = calculate_atr(df, 14)
    df['atr_pct_14'] = (df['atr_14'] / df['close']) * 100
    df['atr_7'] = calculate_atr(df, 7)
    df['atr_pct_7'] = (df['atr_7'] / df['close']) * 100
    df['atr_21'] = calculate_atr(df, 21)
    df['atr_pct_21'] = (df['atr_21'] / df['close']) * 100
    df['realized_vol'] = calculate_realized_volatility(df)
    df['bb_width_20'] = calculate_bb_width(df, 20, 2)
    df['bb_width_10'] = calculate_bb_width(df, 10, 1.5)
    
    # CATEGORY 2: Trend/Efficiency (10)
    print("   [2/7] Trend/Efficiency metrics...")
    df['er_10'] = calculate_er(df, 10)
    df['er_20'] = calculate_er(df, 20)
    df['adx_14'], df['plus_di_14'], df['minus_di_14'] = calculate_adx(df, 14)
    df['adx_7'], df['plus_di_7'], df['minus_di_7'] = calculate_adx(df, 7)
    df['dmi'] = abs(df['plus_di_14'] - df['minus_di_14'])
    df['choppiness'] = calculate_choppiness(df)
    
    # CATEGORY 3: Momentum (8)
    print("   [3/7] Momentum metrics...")
    df['rsi_14'] = calculate_rsi(df, 14)
    df['rsi_7'] = calculate_rsi(df, 7)
    df['macd_hist'], df['macd_signal'] = calculate_macd(df)
    df['roc_7'] = ((df['close'] - df['close'].shift(7)) / df['close'].shift(7)) * 100
    df['roc_14'] = ((df['close'] - df['close'].shift(14)) / df['close'].shift(14)) * 100
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df)
    
    # CATEGORY 4: Moving Averages (12)
    print("   [4/7] Moving Average metrics...")
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma100'] = df['close'].rolling(window=100).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    df['dist_ma5'] = ((df['close'] - df['ma5']) / df['ma5']) * 100
    df['dist_ma10'] = ((df['close'] - df['ma10']) / df['ma10']) * 100
    df['dist_ma20'] = ((df['close'] - df['ma20']) / df['ma20']) * 100
    df['dist_ma50'] = ((df['close'] - df['ma50']) / df['ma50']) * 100
    df['dist_ma100'] = ((df['close'] - df['ma100']) / df['ma100']) * 100
    
    # MA Alignment Score
    conditions = [
        (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20']) & (df['ma20'] > df['ma50']),
        (df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20']) & (df['ma20'] < df['ma50'])
    ]
    df['ma_alignment'] = np.select(conditions, [1, -1], default=0)
    
    df['golden_cross_dist'] = ((df['ma50'] - df['ma200']) / df['ma200']) * 100
    
    # CATEGORY 5: Volume (4)
    print("   [5/7] Volume metrics...")
    if 'volume' in df.columns:
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] / df['volume_ma20']
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_trend'] = df['volume_ma5'] / df['volume_ma20']
    else:
        df['volume_spike'] = np.nan
        df['volume_trend'] = np.nan
    
    # CATEGORY 6: Candle Patterns (6)
    print("   [6/7] Candle pattern metrics...")
    df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
    
    # CATEGORY 7: Fractality (4)
    print("   [7/7] Fractality metrics (slowest)...")
    df['hurst'] = calculate_hurst(df, 100)
    df['approx_entropy'] = calculate_approximate_entropy(df, 100)
    df['perm_entropy'] = calculate_permutation_entropy(df, 100)
    
    print("   ✅ All metrics calculated")
    
    return df


def calculate_weekly_system_performance(df_trades):
    """Calculate system WR and Profit week by week"""
    df_trades['week'] = df_trades['sell_time'].dt.to_period('W')
    
    weekly_perf = []
    
    for week, group in df_trades.groupby('week'):
        wr = (group['profit'] > 0).mean() * 100
        profit_total = group['profit'].sum()
        trades = len(group)
        
        weekly_perf.append({
            'week': str(week),
            'wr': wr,
            'profit': profit_total,
            'trades': trades
        })
    
    return pd.DataFrame(weekly_perf)


def calculate_weekly_btc_metrics(btc_df):
    """Calculate BTC metrics week by week"""
    btc_df['week'] = btc_df['ts'].dt.to_period('W')
    
    weekly_metrics = []
    
    for week, group in btc_df.groupby('week'):
        
        # Helper function for counting crosses
        def count_crosses(series):
            crosses = 0
            for i in range(1, len(series)):
                if pd.notna(series.iloc[i-1]) and pd.notna(series.iloc[i]):
                    if series.iloc[i-1] != series.iloc[i]:
                        crosses += 1
            return crosses
        
        # Calculate all 52 metrics aggregated weekly
        metrics = {
            'week': str(week),
            
            # Volatility (8)
            'atr_pct_14': group['atr_pct_14'].mean(),
            'atr_pct_7': group['atr_pct_7'].mean(),
            'atr_pct_21': group['atr_pct_21'].mean(),
            'realized_vol': group['realized_vol'].mean(),
            'bb_width_20': group['bb_width_20'].mean(),
            'bb_width_10': group['bb_width_10'].mean(),
            'price_range': ((group['high'].max() - group['low'].min()) / group['close'].mean()) * 100,
            'tr_avg': ((group['high'] - group['low']).mean() / group['close'].mean()) * 100,
            
            # Trend/Efficiency (10)
            'er_10': group['er_10'].mean(),
            'er_20': group['er_20'].mean(),
            'adx_14': group['adx_14'].mean(),
            'adx_7': group['adx_7'].mean(),
            'plus_di': group['plus_di_14'].mean(),
            'minus_di': group['minus_di_14'].mean(),
            'trend_consistency': (group['close'] > group['open']).mean() * 100,
            'dmi': group['dmi'].mean(),
            'choppiness': group['choppiness'].mean(),
            
            # Calculate trend duration
            'trend_duration': 0,  # Will calculate separately
            
            # Momentum (8)
            'rsi_14': group['rsi_14'].mean(),
            'rsi_7': group['rsi_7'].mean(),
            'macd_hist': group['macd_hist'].mean(),
            'macd_crosses': count_crosses(group['macd_hist'] > 0),
            'roc_7': group['roc_7'].mean(),
            'roc_14': group['roc_14'].mean(),
            'stoch_k': group['stoch_k'].mean(),
            'stoch_d': group['stoch_d'].mean(),
            
            # Moving Averages (12)
            'dist_ma5': group['dist_ma5'].mean(),
            'dist_ma10': group['dist_ma10'].mean(),
            'dist_ma20': group['dist_ma20'].mean(),
            'dist_ma50': group['dist_ma50'].mean(),
            'dist_ma100': group['dist_ma100'].mean(),
            'ma5_crosses': count_crosses(group['close'] > group['ma5']),
            'ma10_crosses': count_crosses(group['close'] > group['ma10']),
            'ma20_crosses': count_crosses(group['close'] > group['ma20']),
            'ma50_crosses': count_crosses(group['close'] > group['ma50']),
            'ma_alignment': group['ma_alignment'].mean(),
            'golden_cross_dist': group['golden_cross_dist'].mean(),
            'price_ma_position': 0,  # Will calculate
            
            # Volume (4)
            'volume_spike': group['volume_spike'].mean(),
            'volume_trend': group['volume_trend'].mean(),
            'obv_change': 0,  # Placeholder
            'vwap_dist': 0,  # Placeholder
            
            # Candle Patterns (6)
            'consecutive_green': 0,  # Will calculate
            'consecutive_red': 0,  # Will calculate
            'upper_wick': group['upper_wick'].mean(),
            'lower_wick': group['lower_wick'].mean(),
            'body_size': group['body_size'].mean(),
            'doji_ratio': (group['body_size'] < 0.1).mean() * 100,
            
            # Fractality (4)
            'hurst': group['hurst'].mean(),
            'approx_entropy': group['approx_entropy'].mean(),
            'sample_entropy': np.nan,  # Placeholder
            'perm_entropy': group['perm_entropy'].mean(),
        }
        
        # Calculate consecutive candles
        max_green = 0
        max_red = 0
        current_green = 0
        current_red = 0
        
        for _, row in group.iterrows():
            if row['close'] > row['open']:
                current_green += 1
                current_red = 0
                max_green = max(max_green, current_green)
            elif row['close'] < row['open']:
                current_red += 1
                current_green = 0
                max_red = max(max_red, current_red)
            else:
                current_green = 0
                current_red = 0
        
        metrics['consecutive_green'] = max_green
        metrics['consecutive_red'] = max_red
        
        weekly_metrics.append(metrics)
    
    return pd.DataFrame(weekly_metrics)


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


def calculate_correlations_for_target(df_merged, metric_cols, target_col, target_name):
    """Calculate correlations and MI for a specific target"""
    
    print(f"\n🔍 Calculating correlations for target: {target_name}...")
    
    results = []
    
    for i, metric in enumerate(metric_cols, 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(metric_cols)} metrics...")
        
        # Shift metric by 1 week (N-1)
        lagged_col = f'{metric}_lag1'
        if lagged_col not in df_merged.columns:
            df_merged[lagged_col] = df_merged[metric].shift(1)
        
        # Remove rows with NaN
        valid_data = df_merged[[target_col, lagged_col]].dropna()
        
        if len(valid_data) < 5:
            continue
        
        # Calculate Pearson correlation
        corr = valid_data[target_col].corr(valid_data[lagged_col])
        
        # Calculate Mutual Information
        X = valid_data[[lagged_col]].values
        y = valid_data[target_col].values
        
        try:
            mi = mutual_info_regression(X, y, random_state=42)[0]
        except:
            mi = np.nan
        
        if pd.notna(corr):
            results.append({
                'metric': metric,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'mutual_info': mi,
                'corr_strength': interpret_correlation(corr),
                'mi_strength': interpret_mi(mi) if pd.notna(mi) else 'N/A'
            })
    
    return pd.DataFrame(results)


def display_results(df_results, target_name):
    """Display correlation results for a target"""
    
    # Sort by mutual information
    df_results = df_results.sort_values('mutual_info', ascending=False)
    
    print("\n" + "="*160)
    print(f"CORRELATIONS: BTC {BTC_TIMEFRAME} Metrics (Week N-1) vs {target_name} (Week N)")
    print("="*160)
    
    print(f"\n{'Rank':<6} {'Metric':<30} {'Correlation':>12} {'Corr Str':<15} {'Mutual Info':>12} {'MI Str':<15}")
    print("-"*160)
    
    for rank, (_, row) in enumerate(df_results.head(20).iterrows(), 1):
        metric = row['metric']
        corr = row['correlation']
        corr_str = row['corr_strength']
        mi = row['mutual_info']
        mi_str = row['mi_strength']
        
        print(f"{rank:<6} {metric:<30} {corr:>+12.3f} {corr_str:<15} {mi:>12.3f} {mi_str:<15}")
    
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
            print(f"   • {row['metric']:<30} MI: {row['mutual_info']:.3f}, Corr: {row['correlation']:>+.3f}")
    
    print(f"\n✅ Strong Positive Correlation (corr > +0.5): {len(strong_positive)}")
    if len(strong_positive) > 0:
        for _, row in strong_positive.iterrows():
            print(f"   • {row['metric']:<30} Corr: {row['correlation']:>+.3f}, MI: {row['mutual_info']:.3f}")
    
    print(f"\n❌ Strong Negative Correlation (corr < -0.5): {len(strong_negative)}")
    if len(strong_negative) > 0:
        for _, row in strong_negative.iterrows():
            print(f"   • {row['metric']:<30} Corr: {row['correlation']:>+.3f}, MI: {row['mutual_info']:.3f}")
    
    print(f"\n🟡 Moderate Correlation (|corr| 0.3-0.5): {len(moderate)}")
    if len(moderate) > 0:
        for _, row in moderate.head(5).iterrows():
            print(f"   • {row['metric']:<30} Corr: {row['correlation']:>+.3f}, MI: {row['mutual_info']:.3f}")
    
    return df_results


def main():
    print("="*160)
    print("COMPREHENSIVE LAGGED ANALYSIS: 52 BTC METRICS vs MULTIPLE TARGETS")
    print(f"BTC Timeframe: {BTC_TIMEFRAME}")
    print("Targets: 1) WR%, 2) Profit Total")
    print("="*160)
    
    # Load data
    print("\n📂 Loading trades...")
    df_trades = load_all_lab_trades()
    print(f"✅ Loaded {len(df_trades)} trades")
    
    print(f"\n📂 Loading BTC {BTC_TIMEFRAME} data...")
    btc_df = load_btc_data()
    print(f"✅ Loaded {len(btc_df)} bars")
    
    # Calculate weekly stats
    print("\n🔍 Calculating weekly performance (WR + Profit)...")
    df_system = calculate_weekly_system_performance(df_trades)
    print(f"✅ {len(df_system)} weeks with trades")
    
    print("\n🔍 Aggregating BTC metrics to weekly...")
    df_btc = calculate_weekly_btc_metrics(btc_df)
    print(f"✅ {len(df_btc)} weeks with BTC data")
    
    # Merge
    df_merged = pd.merge(df_system, df_btc, on='week', how='inner')
    print(f"\n✅ Merged {len(df_merged)} weeks")
    
    # Filter weeks with minimum trades
    min_trades = 20
    df_merged = df_merged[df_merged['trades'] >= min_trades]
    print(f"✅ Filtered to {len(df_merged)} weeks (≥{min_trades} trades/week)")
    
    # Get all metric columns
    metric_cols = [col for col in df_btc.columns if col not in ['week']]
    
    # Calculate correlations for WR
    print("\n" + "="*160)
    print("TARGET 1: WIN RATE %")
    print("="*160)
    df_results_wr = calculate_correlations_for_target(df_merged, metric_cols, 'wr', 'WR%')
    df_results_wr = display_results(df_results_wr, 'WR%')
    
    # Calculate correlations for Profit
    print("\n" + "="*160)
    print("TARGET 2: PROFIT TOTAL")
    print("="*160)
    df_results_profit = calculate_correlations_for_target(df_merged, metric_cols, 'profit', 'Profit')
    df_results_profit = display_results(df_results_profit, 'Profit')
    
    # Comparison
    print("\n" + "="*160)
    print("COMPARISON: WHICH TARGET HAS BETTER PREDICTORS?")
    print("="*160)
    
    best_wr_mi = df_results_wr['mutual_info'].max()
    best_profit_mi = df_results_profit['mutual_info'].max()
    
    best_wr_corr = df_results_wr['abs_correlation'].max()
    best_profit_corr = df_results_profit['abs_correlation'].max()
    
    print(f"\nBest Mutual Information:")
    print(f"   WR%:    {best_wr_mi:.3f}")
    print(f"   Profit: {best_profit_mi:.3f}")
    
    if best_profit_mi > best_wr_mi:
        print(f"   ✅ PROFIT is more predictable (MI diff: +{best_profit_mi - best_wr_mi:.3f})")
    elif best_wr_mi > best_profit_mi:
        print(f"   ✅ WR% is more predictable (MI diff: +{best_wr_mi - best_profit_mi:.3f})")
    else:
        print(f"   ⚖️  Both equally predictable")
    
    print(f"\nBest Absolute Correlation:")
    print(f"   WR%:    {best_wr_corr:.3f}")
    print(f"   Profit: {best_profit_corr:.3f}")
    
    if best_profit_corr > best_wr_corr:
        print(f"   ✅ PROFIT has stronger correlation (diff: +{best_profit_corr - best_wr_corr:.3f})")
    elif best_wr_corr > best_profit_corr:
        print(f"   ✅ WR% has stronger correlation (diff: +{best_wr_corr - best_profit_corr:.3f})")
    else:
        print(f"   ⚖️  Both equally correlated")
    
    # Save results
    output_dir = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/files')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f'comprehensive_correlations_dual_targets_{BTC_TIMEFRAME}.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results_wr.to_excel(writer, sheet_name='WR_Correlations', index=False)
        df_results_profit.to_excel(writer, sheet_name='Profit_Correlations', index=False)
        df_merged.to_excel(writer, sheet_name='Weekly_Data', index=False)
    
    print(f"\n💾 Saved results to: {output_file}")
    
    print("\n" + "="*160)
    print("\n💡 CONCLUSION:")
    print("   If PROFIT shows stronger correlations → Use Profit as defensive mode trigger")
    print("   If WR% shows stronger correlations → Use WR% as defensive mode trigger")
    print("   If both are weak → BTC metrics alone may not be sufficient predictors")
    print("\n" + "="*160)


if __name__ == "__main__":
    main()