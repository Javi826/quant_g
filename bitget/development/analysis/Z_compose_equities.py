import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


FOLDER              = "../brief_equities_2025"
INITIAL_CAPITAL     = 800
RESAMPLE_FREQ       = '4h'
DATA_FOLDER         = "../data/crypto_OOS_2025"

# -------------------------------------------------
# --- Metrics Functions
# -------------------------------------------------
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
    """Calculate average recovery time from drawdowns"""
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
        return 0
    return np.mean(recovery_times)

def equity_r_squared(df):
    """
    R² de la equity curve vs línea recta.
    Mide consistencia del crecimiento.
    
    R² = 1.0 → Línea recta perfecta (ideal)
    R² = 0.9 → Muy consistente
    R² = 0.7 → Algo de ruido
    R² = 0.5 → Muy errático
    """
    y = df['balance'].values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model.score(X, y)

# -------------------------------------------------
# Function to compute metrics
# -------------------------------------------------
def compute_metrics(equity_df, capital, name="Equity"):
    """Compute all metrics for a given equity curve"""
    df = equity_df.copy()
    df = df.sort_values('timestamp')

    returns = df['balance'].pct_change().dropna()
    volatility = returns.std() * 100

    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_returns = df.groupby('month')['balance'].last().pct_change()
    consistency = (monthly_returns > 0).mean() * 100

    net_gain = total_return(df, capital)
    pf = profit_factor(df)
    rt = average_recovery_time(df) / 6
    r2 = equity_r_squared(df)

    return {
        "Curve": name,
        "Volatility_pct": round(volatility, 2),
        "Monthly_pct": round(consistency, 2),
        "Net_Gain_pct": round(net_gain, 2),
        "Profit_Factor": round(pf, 3) if pf != np.inf else np.inf,
        "Rec_Time": round(rt, 2),
        "R_Squared": round(r2, 3)
    }

# -------------------------------------------------
# Plot function
# -------------------------------------------------
def plot_netgain_dd(equity_hist, capital, title="Net Gain % y DD"):
    """Plot net gain % and drawdown with BTC comparison"""
    initial_capital = capital
    timestamps = pd.to_datetime(equity_hist['timestamp'])
    balances = np.array(equity_hist['balance'])
    
    # Net Gain %
    net_gain_pct = (balances - initial_capital) / initial_capital * 100
    
    # Drawdown %
    cumulative_max = np.maximum.accumulate(balances)
    dd_pct = (balances - cumulative_max) / cumulative_max * 100
    
    fig, ax1 = plt.subplots(figsize=(12,6))
    
    # --- Bitcoin line for comparison ---
    btc_file = os.path.join(DATA_FOLDER, "BTCUSDT_4H.parquet")
    btc_df = pd.read_parquet(btc_file)

    if 'timestamp' not in btc_df.columns:
        if isinstance(btc_df.index, pd.DatetimeIndex):
            btc_df = btc_df.reset_index().rename(columns={'index': 'timestamp'})
        else:
            raise ValueError("BTC parquet missing 'timestamp' column or datetime index")

    btc_df = btc_df[['timestamp', 'close']]
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df['btc_net_gain_pct'] = (btc_df['close'] / btc_df['close'].iloc[0] - 1) * 100

    # --- Compare with BTC for dynamic coloring ---
    btc_aligned = np.interp(
        timestamps.astype(np.int64) / 10**9,
        btc_df['timestamp'].astype(np.int64) / 10**9,
        btc_df['btc_net_gain_pct']
    )

    # Create masks for when we beat BTC or not
    above_btc = net_gain_pct >= btc_aligned
    below_btc = net_gain_pct < btc_aligned

    # Green area where we beat BTC
    ax1.fill_between(timestamps, net_gain_pct, 0, where=above_btc, alpha=0.2, color='green', interpolate=True)

    # Red area where we don't beat BTC
    ax1.fill_between(timestamps, net_gain_pct, 0, where=below_btc, alpha=0.2, color='red', interpolate=True)

    # Blue line (always on top)
    ax1.plot(timestamps, net_gain_pct, color='blue', linewidth=1.2, label='Net Gain %')

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net_Gain_pct", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot BTC with orange dashed line
    ax1.plot(btc_df['timestamp'], btc_df['btc_net_gain_pct'], 
             color='darkorange', linewidth=0.6, linestyle='--', label='BTC %')

    # Drawdown %
    ax2 = ax1.twinx()
    ax2.plot(timestamps, dd_pct, color='lightcoral', linewidth=0.1, label='DD %')
    ax2.set_ylabel("Drawdown", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Calculate values for labels
    final_net_gain = net_gain_pct[-1]
    max_dd = dd_pct.min()
    final_btc = btc_df['btc_net_gain_pct'].iloc[-1]
    
    # Add labels to plot
    textstr = f'Final Net Gain: {final_net_gain:.2f}%\nMax DD: {max_dd:.2f}%\nBTC Final: {final_btc:.2f}%'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title)
    fig.autofmt_xdate()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    
    plt.show()

# -------------------------------------------------
# Read all files
# -------------------------------------------------
dfs = []
file_names = []
metrics_table = []
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
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    dfs.append(df)

    short_name = os.path.splitext(file_name)[0]
    file_names.append(short_name)

    correlation_data[short_name] = df['balance'].pct_change()

    plot_netgain_dd(df.reset_index(), capital=INITIAL_CAPITAL,
                    title=f"Net Gain % & DD - {short_name}")

    metrics_table.append(
        compute_metrics(df.reset_index(), capital=INITIAL_CAPITAL, name=short_name)
    )

# -------------------------------------------------
# Combined portfolio
# -------------------------------------------------
if dfs:
    start = min(df.index.min() for df in dfs)
    end   = max(df.index.max() for df in dfs)
    common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

    resampled_balances = []
    for df in dfs:
        df_r = df[['balance']].reindex(common_index)
        df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
        resampled_balances.append(df_r['balance'])

    combined_balance = pd.concat(resampled_balances, axis=1).sum(axis=1)
    combined_df = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})

    combined_capital = INITIAL_CAPITAL * len(dfs)
    plot_netgain_dd(combined_df, capital=combined_capital,
                    title="Net Gain % & DD - Combined Portfolio")

    metrics_table.append(
        compute_metrics(combined_df, capital=combined_capital, name="Combined Portfolio")
    )

    correlation_data["Combined Portfolio"] = combined_df['balance'].pct_change()

# -------------------------------------------------
# BTC metrics
# -------------------------------------------------
try:
    btc_path = os.path.join(DATA_FOLDER, "BTCUSDT_4H.parquet")
    btc_df = pd.read_parquet(btc_path)

    if 'timestamp' not in btc_df.columns:
        if isinstance(btc_df.index, pd.DatetimeIndex):
            btc_df = btc_df.reset_index().rename(columns={'index': 'timestamp'})

    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df_metrics = btc_df[['timestamp','close']].copy()
    btc_df_metrics['balance'] = btc_df_metrics['close']

    metrics_table.append(
        compute_metrics(btc_df_metrics, capital=btc_df_metrics['balance'].iloc[0],
                        name="BTCUSDT")
    )

    correlation_data["BTCUSDT"] = btc_df_metrics['balance'].pct_change()

except Exception as e:
    print(f"⚠️ Error computing BTC metrics: {e}")

# -------------------------------------------------
# Final table
# -------------------------------------------------
metrics_df = pd.DataFrame(metrics_table)
metrics_df['Curve'] = metrics_df['Curve'].astype(str)

print("\n📊 FINAL METRICS TABLE (ALL CURVES):\n")
metrics_df_display = metrics_df.copy()
max_len = metrics_df_display['Curve'].str.len().max()
metrics_df_display['Curve'] = metrics_df_display['Curve'].apply(lambda x: x.ljust(max_len))

print(metrics_df_display.to_string(index=False))

# -------------------------------------------------
# COMBINATIONS SEARCH
# -------------------------------------------------
from itertools import combinations

combo_results = []
named_dfs = dict(zip(file_names, dfs))

for r in range(1, len(named_dfs) + 1):
    for combo in combinations(named_dfs.keys(), r):

        combo_dfs = [named_dfs[name] for name in combo]

        start = min(df.index.min() for df in combo_dfs)
        end   = max(df.index.max() for df in combo_dfs)
        common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

        resampled = []
        for df in combo_dfs:
            df_r = df[['balance']].reindex(common_index)
            df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
            resampled.append(df_r['balance'])

        combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
        combined_df = pd.DataFrame({
            'timestamp': common_index,
            'balance': combined_balance
        })

        capital = INITIAL_CAPITAL * len(combo_dfs)

        metrics = compute_metrics(
            combined_df,
            capital=capital,
            name="+".join(combo)
        )
        combo_results.append(metrics)

# =============================================================================
# TOP 10 COMBINATIONS BY DIFFERENT METRICS
# =============================================================================

print("\n" + "="*80)
print("🏆 TOP 10 COMBINATIONS BY KEY METRICS")
print("="*80)

combo_df = pd.DataFrame(combo_results)

# -------------------------------------------------
# TOP 10 BY NET GAIN (Descending)
# -------------------------------------------------
combo_netgain = combo_df.sort_values("Net_Gain_pct", ascending=False).head(10).copy()
combo_netgain_display = combo_netgain.copy()
max_len_netgain = combo_netgain_display['Curve'].str.len().max()
combo_netgain_display['Curve'] = combo_netgain_display['Curve'].apply(lambda x: x.ljust(max_len_netgain))

print("\n📈 TOP 10 COMBINATIONS BY NET GAIN (Highest):\n")
print(combo_netgain_display.to_string(index=False))

# -------------------------------------------------
# TOP 10 BY R² (Descending - higher is better)
# -------------------------------------------------
combo_r2 = combo_df.sort_values("R_Squared", ascending=False).head(10).copy()
combo_r2_display = combo_r2.copy()
max_len_r2 = combo_r2_display['Curve'].str.len().max()
combo_r2_display['Curve'] = combo_r2_display['Curve'].apply(lambda x: x.ljust(max_len_r2))

print("\n📐 TOP 10 COMBINATIONS BY R² (Most Consistent):\n")
print(combo_r2_display.to_string(index=False))

# -------------------------------------------------
# TOP 10 BY PROFIT FACTOR (Descending)
# -------------------------------------------------
combo_pf = combo_df[combo_df['Profit_Factor'] != np.inf].sort_values("Profit_Factor", ascending=False).head(10).copy()
combo_pf_display = combo_pf.copy()
max_len_pf = combo_pf_display['Curve'].str.len().max()
combo_pf_display['Curve'] = combo_pf_display['Curve'].apply(lambda x: x.ljust(max_len_pf))

print("\n💰 TOP 10 COMBINATIONS BY PROFIT FACTOR (Highest):\n")
print(combo_pf_display.to_string(index=False))

print("\n" + "="*80)

# -------------------------------------------------
# Best combination by Net Gain
# -------------------------------------------------
best_name_netgain = combo_netgain.iloc[0]["Curve"]
best_combo_netgain = best_name_netgain.split("+")

best_dfs_netgain = [named_dfs[name] for name in best_combo_netgain]

start = min(df.index.min() for df in best_dfs_netgain)
end   = max(df.index.max() for df in best_dfs_netgain)
common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

resampled = []
for df in best_dfs_netgain:
    df_r = df[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
    resampled.append(df_r['balance'])

combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
best_df_netgain = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})

best_capital = INITIAL_CAPITAL * len(best_dfs_netgain)

plot_netgain_dd(best_df_netgain, capital=best_capital,
                title=f"Best Combination by Net Gain: {best_name_netgain}")

# -------------------------------------------------
# Best combination by R²
# -------------------------------------------------
best_name_r2 = combo_r2.iloc[0]["Curve"]
best_combo_r2 = best_name_r2.split("+")

best_dfs_r2 = [named_dfs[name] for name in best_combo_r2]

start = min(df.index.min() for df in best_dfs_r2)
end   = max(df.index.max() for df in best_dfs_r2)
common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

resampled = []
for df in best_dfs_r2:
    df_r = df[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
    resampled.append(df_r['balance'])

combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
best_df_r2 = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})
best_capital = INITIAL_CAPITAL * len(best_dfs_r2)

plot_netgain_dd(best_df_r2, capital=best_capital,
                title=f"Best Combination by R² (Consistency): {best_name_r2}")

# -------------------------------------------------
# Best combination by Profit Factor
# -------------------------------------------------
best_name_pf = combo_pf.iloc[0]["Curve"]
best_combo_pf = best_name_pf.split("+")

best_dfs_pf = [named_dfs[name] for name in best_combo_pf]

start = min(df.index.min() for df in best_dfs_pf)
end   = max(df.index.max() for df in best_dfs_pf)
common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

resampled = []
for df in best_dfs_pf:
    df_r = df[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
    resampled.append(df_r['balance'])

combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
best_df_pf = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})
best_capital = INITIAL_CAPITAL * len(best_dfs_pf)

plot_netgain_dd(best_df_pf, capital=best_capital,
                title=f"Best Combination by Profit Factor: {best_name_pf}")

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("📊 CORRELATION ANALYSIS")
print("="*80)

# -------------------------------------------------
# 1. CORRELATION HEATMAP (VISUAL)
# -------------------------------------------------
print("\n[1/2] Generating correlation heatmap...")

# Create DataFrame with returns from each strategy
returns_df = pd.DataFrame()

for name in file_names:
    if name in correlation_data:
        returns_df[name] = correlation_data[name]

# Calculate correlation
correlation_matrix = returns_df.corr()

import seaborn as sns

plt.figure(figsize=(14, 12))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn_r',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title('Correlation Matrix Between Strategies', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 2. HIGH CORRELATION PAIRS (WARNING)
# -------------------------------------------------
print("\n[2/2] Identifying highly correlated pairs...")

high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        
        if corr_value > 0.7:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value))

print("\n⚠️  PAIRS WITH HIGH POSITIVE CORRELATION (>0.7) - Consider reducing:\n")
if high_corr_pairs:
    for strat1, strat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
        print(f"   {strat1} + {strat2}: {corr:.3f}")
else:
    print("   ✅ No pairs with high positive correlation")



print("\n" + "="*80)
print("✅ ANALYSIS COMPLETED")
print("="*80 + "\n")