#devolop/analysis/compose_equities.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.linear_model import LinearRegression


FOLDER              = "../brief_equities"
INITIAL_CAPITAL     = 800
RESAMPLE_FREQ       = '1D'
BARS_PER_DAY        = 1
DATA_FOLDER         = "../../BOT_batch/data/crypto_2026_OOS"

# =============================================================================
# DETAILED METRICS CONFIGURATION
# =============================================================================

SELECTED_STRATEGIES_FOR_METRICS = [
    '03_parity_long_4H',
    '11_parity_short_1H',
# =============================================================================
#     '12_parity_long_6Hutc',
#     '13_orderblocks_short_4H',
#     #'16_ranging_short_6Hutc',
#     '17_flag_long_4H',
#     '19_flag_short_4H',
# =============================================================================
    #'20_flag_short_1H',
]

def total_return(df, capital):
    """Calculate net gain percentage"""
    return (df['balance'].iloc[-1] - capital) / capital * 100


def profit_factor(df):
    """Calculate profit factor (gains/losses ratio)"""
    returns = df['balance'].pct_change().dropna()
    gains   = returns[returns > 0].sum()
    losses  = -returns[returns < 0].sum()

    if losses == 0:
        return np.inf
    return gains / losses


def average_recovery_time(df):
    """
    Calculate average and max recovery time from drawdowns.
    Returns (avg_bars, max_bars) — convert to days dividing by BARS_PER_DAY.
    """
    bal        = df['balance'].values
    peaks      = np.maximum.accumulate(bal)
    underwater = bal < peaks

    recovery_times  = []
    last_peak_index = 0

    for i in range(1, len(bal)):
        if not underwater[i] and underwater[i - 1]:
            recovery_times.append(i - last_peak_index)
        if not underwater[i]:
            last_peak_index = i

    if len(recovery_times) == 0:
        return 0, 0
    return np.mean(recovery_times), np.max(recovery_times)


def equity_r_squared(df):
    """
    R-squared of equity curve vs straight line.
    Measures growth consistency.
    1.0 = perfect straight line, 0.5 = very erratic.
    """
    y = df['balance'].values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    return model.score(X, y)


def resample_equity(df_indexed):
    """
    Receives a DataFrame with DatetimeIndex and 'balance' column.
    Returns a new DataFrame resampled to RESAMPLE_FREQ.
    Uses ffill to avoid interpolating data.
    """
    common_index = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq=RESAMPLE_FREQ
    )
    df_r = df_indexed[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].ffill().bfill()
    df_r.index.name = 'timestamp'
    return df_r


def compute_metrics(equity_df, capital, name="Equity"):
    """
    Compute all metrics for a given equity curve.
    equity_df must have 'timestamp' and 'balance' columns.
    Assumes all rows are spaced RESAMPLE_FREQ apart (guaranteed by resample_equity).
    """
    df = equity_df.copy()
    df = df.sort_values('timestamp')

    returns    = df['balance'].pct_change().dropna()
    volatility = returns.std() * 100

    df['month']     = df['timestamp'].dt.to_period('M')
    monthly_returns = df.groupby('month')['balance'].last().pct_change()
    consistency     = (monthly_returns > 0).mean() * 100

    net_gain       = total_return(df, capital)
    pf             = profit_factor(df)
    rt_avg, rt_max = average_recovery_time(df)
    r2             = equity_r_squared(df)

    bal            = df['balance'].values
    cumulative_max = np.maximum.accumulate(bal)
    max_dd         = ((bal - cumulative_max) / cumulative_max * 100).min()

    return {
        "Curve":          name,
        "Volatility_pct": round(volatility, 2),
        "Monthly_pct":    round(consistency, 2),
        "Net_Gain_pct":   round(net_gain, 2),
        "Max_DD_pct":     round(max_dd, 2),
        "Profit_Factor":  round(pf, 3) if pf != np.inf else np.inf,
        "Rec_Time":       round(rt_avg / BARS_PER_DAY, 2),
        "Rec_Max":        round(rt_max / BARS_PER_DAY, 2),
        "R_Squared":      round(r2, 3)
    }


def build_combined_equity(dfs_list):
    """
    Given a list of DataFrames with DatetimeIndex and 'balance' column,
    returns a combined equity DataFrame with 'timestamp' and 'balance' columns.
    """
    start        = min(df.index.min() for df in dfs_list)
    end          = max(df.index.max() for df in dfs_list)
    common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

    resampled = []
    for df in dfs_list:
        df_r            = df[['balance']].reindex(common_index)
        df_r['balance'] = df_r['balance'].ffill().bfill()
        resampled.append(df_r['balance'])

    combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
    return pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})


def extract_numeric_id(segment):
    """Extract the numeric ID from a strategy segment string. e.g. '03_parity_long_4H' -> '03'"""
    for part in segment.split("_"):
        if part.isdigit():
            return part
    return segment


def shorten_curve_name(name):
    """
    Extract numeric IDs from curve name(s) and sort them numerically.
    e.g. 'equity_08_...+equity_02_...+equity_10_...' -> '02+08+10'
    """
    segments = name.strip().split("+")
    ids      = [extract_numeric_id(seg) for seg in segments]
    ids_sorted = sorted(ids, key=lambda x: int(x) if x.isdigit() else float('inf'))
    return "+".join(ids_sorted) if ids_sorted else name


def sort_combo_name(name):
    """
    Given a raw combination name (joined full strategy names with '+'),
    return a version with segments sorted numerically by their ID.
    e.g. '08_flag+02_parity+10_ob' -> '02_parity+08_flag+10_ob'
    """
    segments   = name.strip().split("+")
    seg_sorted = sorted(segments, key=lambda s: int(extract_numeric_id(s))
                        if extract_numeric_id(s).isdigit() else float('inf'))
    return "+".join(seg_sorted)


def print_metrics_table(metrics_list, title, shorten_names=False):
    """Print a formatted metrics table from a list of metric dicts."""
    df = pd.DataFrame(metrics_list)
    df['Curve'] = df['Curve'].astype(str)
    if shorten_names:
        df['Curve'] = df['Curve'].apply(shorten_curve_name)
    max_len     = df['Curve'].str.len().max()
    df['Curve'] = df['Curve'].apply(lambda x: x.ljust(max_len))
    print(f"\n{title}\n")
    print(df.to_string(index=False))


def plot_netgain_dd(equity_hist, capital, title="Net Gain % & DD"):
    """Plot net gain % and drawdown with BTC comparison"""
    timestamps = pd.to_datetime(equity_hist['timestamp'])
    balances   = np.array(equity_hist['balance'])

    net_gain_pct   = (balances - capital) / capital * 100
    cumulative_max = np.maximum.accumulate(balances)
    dd_pct         = (balances - cumulative_max) / cumulative_max * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Bitcoin line for comparison ---
    btc_file = os.path.join(DATA_FOLDER, "BTCUSDT_4H.parquet")
    btc_df   = pd.read_parquet(btc_file)

    if 'timestamp' not in btc_df.columns:
        if isinstance(btc_df.index, pd.DatetimeIndex):
            btc_df = btc_df.reset_index().rename(columns={'index': 'timestamp'})
        else:
            raise ValueError("BTC parquet missing 'timestamp' column or datetime index")

    btc_df                     = btc_df[['timestamp', 'close']]
    btc_df['timestamp']        = pd.to_datetime(btc_df['timestamp'])
    btc_df['btc_net_gain_pct'] = (btc_df['close'] / btc_df['close'].iloc[0] - 1) * 100

    btc_aligned = np.interp(
        timestamps.astype(np.int64) / 10**9,
        btc_df['timestamp'].astype(np.int64) / 10**9,
        btc_df['btc_net_gain_pct']
    )

    above_btc = net_gain_pct >= btc_aligned
    below_btc = net_gain_pct < btc_aligned

    ax1.fill_between(timestamps, net_gain_pct, 0, where=above_btc, alpha=0.2, color='green', interpolate=True)
    ax1.fill_between(timestamps, net_gain_pct, 0, where=below_btc, alpha=0.2, color='red',   interpolate=True)
    ax1.plot(timestamps, net_gain_pct, color='blue',       linewidth=1.2, label='Net Gain %')
    ax1.plot(btc_df['timestamp'], btc_df['btc_net_gain_pct'],
             color='darkorange', linewidth=0.6, linestyle='--', label='BTC %')

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net_Gain_pct", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(timestamps, dd_pct, color='lightcoral', linewidth=0.1, label='DD %')
    ax2.set_ylabel("Drawdown", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    textstr = (
        f'Final Net Gain: {net_gain_pct[-1]:.2f}%\n'
        f'Max DD: {dd_pct.min():.2f}%\n'
        f'BTC Final: {btc_df["btc_net_gain_pct"].iloc[-1]:.2f}%'
    )
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title)
    fig.autofmt_xdate()
    ax1.grid(True, linestyle='--', alpha=0.6)

    lines,  labels  = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    plt.show()


# -------------------------------------------------
# Read all files
# -------------------------------------------------
if __name__ == "__main__":
    dfs              = []
    file_names       = []
    metrics_table    = []
    correlation_data = {}
    
    for file_name in os.listdir(FOLDER):
        if not file_name.endswith(".xlsx"):
            continue
    
        path = os.path.join(FOLDER, file_name)
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"⚠️ Could not read {file_name}: {e}")
            continue
    
        if 'timestamp' not in df.columns or 'balance' not in df.columns:
            print(f"⚠️ {file_name} missing 'timestamp' or 'balance', skipping.")
            continue
    
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        df = resample_equity(df)
    
        short_name = os.path.splitext(file_name)[0]
    
        dfs.append(df)
        file_names.append(short_name)
        correlation_data[short_name] = df['balance'].pct_change()
    
        plot_netgain_dd(df.reset_index(), capital=INITIAL_CAPITAL,
                        title=f"Net Gain % & DD - {short_name}")
    
        metrics_table.append(
            compute_metrics(df.reset_index(), capital=INITIAL_CAPITAL, name=short_name)
        )
    
    named_dfs = dict(zip(file_names, dfs))
    
    # -------------------------------------------------
    # Combined portfolio (all strategies)
    # -------------------------------------------------
    if dfs:
        combined_df      = build_combined_equity(dfs)
        combined_capital = INITIAL_CAPITAL * len(dfs)
    
        plot_netgain_dd(combined_df, capital=combined_capital,
                        title="Net Gain % & DD - Combined Portfolio")
    
        metrics_table.append(
            compute_metrics(combined_df, capital=combined_capital, name="Combined Portfolio")
        )
    
        correlation_data["Combined Portfolio"] = combined_df.set_index('timestamp')['balance'].pct_change()
    
    # -------------------------------------------------
    # BTC metrics
    # -------------------------------------------------
    try:
        btc_path = os.path.join(DATA_FOLDER, "BTCUSDT_4H.parquet")
        btc_df   = pd.read_parquet(btc_path)
    
        if 'timestamp' not in btc_df.columns:
            if isinstance(btc_df.index, pd.DatetimeIndex):
                btc_df = btc_df.reset_index().rename(columns={'index': 'timestamp'})
    
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
        btc_df = btc_df.set_index('timestamp')[['close']].rename(columns={'close': 'balance'})
        btc_df = resample_equity(btc_df)
    
        metrics_table.append(
            compute_metrics(btc_df.reset_index(), capital=btc_df['balance'].iloc[0], name="BTCUSDT")
        )
    
        correlation_data["BTCUSDT"] = btc_df['balance'].pct_change()
    
    except Exception as e:
        print(f"⚠️ Error computing BTC metrics: {e}")
    
    # -------------------------------------------------
    # Selected strategies table + plot
    # -------------------------------------------------
    selected_metrics = [
        m for m in metrics_table
        if any(sel in m["Curve"] for sel in SELECTED_STRATEGIES_FOR_METRICS)
    ]
    
    if selected_metrics:
        selected_dfs = [named_dfs[m["Curve"]] for m in selected_metrics if m["Curve"] in named_dfs]
    
        if selected_dfs:
            combined_selected = build_combined_equity(selected_dfs)
            capital_selected  = INITIAL_CAPITAL * len(selected_dfs)
    
            selected_metrics.append(
                compute_metrics(combined_selected, capital=capital_selected, name="Combined Selected")
            )
    
            plot_netgain_dd(combined_selected, capital=capital_selected,
                            title="Net Gain % & DD - STRATEGIES SELECTED")
    
        print_metrics_table(selected_metrics, "📊 SELECTED STRATEGIES METRICS TABLE:")
    
    # -------------------------------------------------
    # Final table (all curves)
    # -------------------------------------------------
    metrics_table.sort(key=lambda m: int(extract_numeric_id(m["Curve"]))
                       if extract_numeric_id(m["Curve"]).isdigit() else float('inf'))
    print_metrics_table(metrics_table, "📊 FINAL METRICS TABLE (ALL CURVES):")
    
    # -------------------------------------------------
    # Combinations search
    # -------------------------------------------------
    combo_results = []
    
    for r in range(1, len(named_dfs) + 1):
        for combo in combinations(named_dfs.keys(), r):
            # Sort combo segments numerically before joining
            combo_sorted = tuple(sorted(combo, key=lambda s: int(extract_numeric_id(s))
                                        if extract_numeric_id(s).isdigit() else float('inf')))
            combo_dfs    = [named_dfs[name] for name in combo_sorted]
            combined     = build_combined_equity(combo_dfs)
            capital      = INITIAL_CAPITAL * len(combo_dfs)
    
            combo_results.append(
                compute_metrics(combined, capital=capital, name="+".join(combo_sorted))
            )
    
    # =============================================================================
    # TOP 5 COMBINATIONS BY KEY METRICS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("🏆 TOP 5 COMBINATIONS BY KEY METRICS")
    print("=" * 80)
    
    combo_df = pd.DataFrame(combo_results)
    
    top5_configs = [
        ("Net_Gain_pct",  False, "📈 TOP 5 COMBINATIONS BY NET GAIN (Highest):"),
        ("R_Squared",     False, "📐 TOP 5 COMBINATIONS BY R² (Most Consistent):"),
        ("Profit_Factor", False, "💰 TOP 5 COMBINATIONS BY PROFIT FACTOR (Highest):"),
        ("Rec_Time",      True,  "⏱️  TOP 5 COMBINATIONS BY RECOVERY TIME (Lowest):"),
        ("Max_DD_pct",    False, "📉 TOP 5 COMBINATIONS BY MAX DD (Lowest Drawdown):"),
    ]
    
    top5_results = {}
    
    for metric, ascending, label in top5_configs:
        df_filtered = combo_df if metric != "Profit_Factor" else combo_df[combo_df['Profit_Factor'] != np.inf]
        top5        = df_filtered.sort_values(metric, ascending=ascending).head(5).copy()
        print_metrics_table(top5.to_dict('records'), f"\n{label}", shorten_names=True)
        top5_results[metric] = top5
    
    print("\n" + "=" * 80)
    
    # -------------------------------------------------
    # Plot best combination per metric
    # -------------------------------------------------
    plot_configs = [
        ("Net_Gain_pct",  "Best Combination by Net Gain"),
        ("R_Squared",     "Best Combination by R² (Consistency)"),
        ("Profit_Factor", "Best Combination by Profit Factor"),
    ]
    
    for metric, plot_title in plot_configs:
        best_name  = top5_results[metric].iloc[0]["Curve"].strip()
        best_combo = best_name.split("+")
        best_dfs   = [named_dfs[name] for name in best_combo]
        best_df    = build_combined_equity(best_dfs)
        best_cap   = INITIAL_CAPITAL * len(best_dfs)
    
        plot_netgain_dd(best_df, capital=best_cap, title=f"{plot_title}: {best_name}")
    
    # =============================================================================
    # CORRELATION ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("📊 CORRELATION ANALYSIS")
    print("=" * 80)
    
    print("\n[1/2] Generating correlation heatmap...")
    
    returns_df         = pd.DataFrame({name: correlation_data[name] for name in file_names if name in correlation_data})
    correlation_matrix = returns_df.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix Between Strategies', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    print("\n[2/2] Identifying highly correlated pairs...")
    
    high_corr_pairs = [
        (correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
        for i in range(len(correlation_matrix.columns))
        for j in range(i + 1, len(correlation_matrix.columns))
        if correlation_matrix.iloc[i, j] > 0.7
    ]
    
    print("\n⚠️  PAIRS WITH HIGH POSITIVE CORRELATION (>0.7) - Consider reducing:\n")
    if high_corr_pairs:
        for s1, s2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
            print(f"   {s1} + {s2}: {corr:.3f}")
    else:
        print("   ✅ No pairs with high positive correlation")
    
    # =============================================================================
    # DRAWDOWN CORRELATION ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("📉 DRAWDOWN CORRELATION ANALYSIS")
    print("=" * 80)
    
    print("\n[1/2] Generating drawdown correlation heatmap...")
    
    dd_df = pd.DataFrame()
    for name, df in zip(file_names, dfs):
        bal    = df['balance'].values
        cummax = np.maximum.accumulate(bal)
        dd_df[name] = np.where(cummax > 0, ((cummax - bal) / cummax) * 100, 0.0)
    
    dd_correlation_matrix = dd_df.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(dd_correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Drawdown Correlation Matrix Between Strategies', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    print("\n[2/2] Identifying highly correlated drawdown pairs...")
    
    high_dd_corr_pairs = [
        (dd_correlation_matrix.columns[i], dd_correlation_matrix.columns[j], dd_correlation_matrix.iloc[i, j])
        for i in range(len(dd_correlation_matrix.columns))
        for j in range(i + 1, len(dd_correlation_matrix.columns))
        if dd_correlation_matrix.iloc[i, j] > 0.7
    ]
    
    print("\n⚠️  PAIRS WITH HIGH DRAWDOWN CORRELATION (>0.7) - Drawdowns happen together:\n")
    if high_dd_corr_pairs:
        for s1, s2, corr in sorted(high_dd_corr_pairs, key=lambda x: x[2], reverse=True):
            print(f"   {s1} + {s2}: {corr:.3f}")
    else:
        print("   ✅ No pairs with high drawdown correlation")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETED")
    print("=" * 80 + "\n")