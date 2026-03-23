#analysis/montecarlo_analysis.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
FOLDER              = "../brief_equities"
INITIAL_CAPITAL     = 800
RESAMPLE_FREQ       = '1D'
N_SIMULATIONS       = 2000

# =============================================================================
# HELPER: Resample equity to common frequency
# =============================================================================
def resample_equity(df_indexed):
    """
    Receives a DataFrame with DatetimeIndex and 'balance' column.
    Returns a new DataFrame resampled to RESAMPLE_FREQ with the index reset.
    
    Uses .last() to aggregate (takes last value of each period).
    This avoids interpolating/inventing data.
    """
    common_index = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq=RESAMPLE_FREQ
    )
    df_r = df_indexed[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].ffill().bfill()  # Forward fill for missing values
    df_r.index.name = 'timestamp'
    return df_r

# =============================================================================
# MONTE CARLO: Bootstrap resampling on DAILY returns
# =============================================================================
def run_montecarlo_dd_netgain(equity_series, initial_capital, n_simulations=2000):
    """
    Run Monte Carlo simulations using bootstrap (resample DAILY returns with replacement).
    
    This ensures results are independent of the equity frequency (1H, 4H, etc.).
    We resample days, not individual bars.
    
    Parameters:
    -----------
    equity_series : pd.Series
        Balance values indexed by timestamp
    initial_capital : float
        Initial capital
    n_simulations : int
        Number of Monte Carlo paths
    
    Returns:
    --------
    dict with keys:
        - 'max_dd': array of max drawdowns (%)
        - 'net_gain_pct': array of final net gains (%)
    """
    # Convert to daily equity (last value of each day)
    daily_equity = equity_series.resample('1D').last().dropna()
    
    # Calculate daily returns
    daily_returns = daily_equity.pct_change().dropna()
    
    if len(daily_returns) == 0:
        print("⚠️ No returns to simulate")
        return {'max_dd': np.array([]), 'net_gain_pct': np.array([])}
    
    n_days = len(daily_returns)
    
    print(f"\n📅 Daily equity points: {len(daily_equity)}")
    print(f"📅 Daily returns: {n_days}")
    
    mc_max_dd = []
    mc_net_gain_pct = []
    
    print(f"\n🎲 Running {n_simulations:,} Monte Carlo simulations...")
    
    for _ in tqdm(range(n_simulations), desc="Simulating"):
        # Bootstrap: resample DAILY returns with replacement
        shuffled_returns = np.random.choice(daily_returns.values, size=n_days, replace=True)
        
        # Reconstruct equity curve from daily returns
        mc_equity = initial_capital * np.cumprod(1 + shuffled_returns)
        
        # Calculate Max Drawdown
        cummax = np.maximum.accumulate(mc_equity)
        drawdown = (mc_equity - cummax) / cummax * 100
        max_dd = drawdown.min()
        
        # Calculate Net Gain %
        final_balance = mc_equity[-1]
        net_gain_pct = (final_balance - initial_capital) / initial_capital * 100
        
        mc_max_dd.append(max_dd)
        mc_net_gain_pct.append(net_gain_pct)
    
    return {
        'max_dd': np.array(mc_max_dd),
        'net_gain_pct': np.array(mc_net_gain_pct)
    }

# =============================================================================
# PERCENTILE ANALYSIS
# =============================================================================
def calculate_percentiles(data, metric_name, is_drawdown=False):
    """
    Calculate key percentiles for a metric.
    
    For drawdowns (negative values), we invert the logic:
    - P95 = 95th percentile of absolute values (worst case)
    - P5 = 5th percentile of absolute values (best case)
    
    This makes P95 represent "95% confidence worst case" for risk metrics.
    """
    if is_drawdown:
        # For DD: use absolute values, then negate
        abs_data = np.abs(data)
        percentiles = {
            'P5':  -np.percentile(abs_data, 5),   # Best case (smallest DD)
            'P25': -np.percentile(abs_data, 25),
            'P50': -np.percentile(abs_data, 50),  # Median
            'P75': -np.percentile(abs_data, 75),
            'P90': -np.percentile(abs_data, 90),
            'P95': -np.percentile(abs_data, 95),  # Worst case (largest DD)
            'P99': -np.percentile(abs_data, 99),  # Extreme case
            'Mean': np.mean(data),
            'Std': np.std(data)
        }
    else:
        # For Net Gain: standard percentiles
        percentiles = {
            'P5':  np.percentile(data, 5),
            'P25': np.percentile(data, 25),
            'P50': np.percentile(data, 50),
            'P75': np.percentile(data, 75),
            'P90': np.percentile(data, 90),
            'P95': np.percentile(data, 95),
            'P99': np.percentile(data, 99),
            'Mean': np.mean(data),
            'Std': np.std(data)
        }
    
    return percentiles

def print_percentile_table(percentiles_dd, percentiles_netgain, historical_dd, historical_netgain):
    """Print formatted percentile comparison table"""
    
    print("\n" + "="*100)
    print("📊 MONTE CARLO PERCENTILE ANALYSIS")
    print("="*100)
    
    print(f"\n{'Percentile':<15} {'Max DD %':<35} {'Net Gain %':<35}")
    print("-" * 100)
    
    # P5
    dd_p5 = percentiles_dd['P5']
    ng_p5 = percentiles_netgain['P5']
    print(f"{'P5':<15} {dd_p5:>10.2f} {'(95% scenarios are worse)':<22} {ng_p5:>10.2f} {'(95% scenarios are better)':<22}")
    
    # P95
    dd_p95 = percentiles_dd['P95']
    ng_p95 = percentiles_netgain['P95']
    print(f"{'P95':<15} {dd_p95:>10.2f} {'(5% scenarios are worse)':<22} {ng_p95:>10.2f} {'(5% scenarios are better)':<22}")
    
    print("-" * 100)
    print(f"{'HISTORICAL':<15} {historical_dd:>10.2f} {'(observed)':<22} {historical_netgain:>10.2f} {'(observed)':<22}")
    print("="*100)
    
    # Key insights
    print("\n📌 KEY INSIGHTS:")
    print(f"   Max DD P95:        {dd_p95:.2f}% (only 5% of scenarios have worse DD)")
    print(f"   Max DD Historical: {historical_dd:.2f}%")
    print(f"   Net Gain P5:       {ng_p5:.2f}% (only 5% of scenarios have lower gain)")
    print(f"   Net Gain P95:      {ng_p95:.2f}% (only 5% of scenarios have higher gain)")
    
    dd_ratio = abs(dd_p95 / historical_dd) if historical_dd != 0 else np.nan
    print(f"\n⚠️  DD P95 is {dd_ratio:.2f}x the historical DD")
    
    ng_range = ng_p95 - ng_p5
    print(f"📈 90% confidence interval for Net Gain: [{ng_p5:.1f}%, {ng_p95:.1f}%] (range: {ng_range:.1f}%)")
    print("="*100 + "\n")

# =============================================================================
# PLOTTING
# =============================================================================
def plot_montecarlo_distributions(mc_results, historical_dd, historical_netgain):
    """Plot distribution histograms for DD and Net Gain"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- MAX DD DISTRIBUTION ---
    ax1 = axes[0]
    
    dd_data = mc_results['max_dd']
    
    ax1.hist(dd_data, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.axvline(historical_dd, color='red', linestyle='--', linewidth=2, label=f'Historical: {historical_dd:.2f}%')
    ax1.axvline(np.percentile(dd_data, 95), color='darkred', linestyle='-', linewidth=2, label=f'P95: {np.percentile(dd_data, 95):.2f}%')
    ax1.axvline(np.percentile(dd_data, 50), color='orange', linestyle=':', linewidth=1.5, label=f'P50: {np.percentile(dd_data, 50):.2f}%')
    
    ax1.set_xlabel('Max Drawdown (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Monte Carlo: Max Drawdown Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- NET GAIN DISTRIBUTION ---
    ax2 = axes[1]
    
    ng_data = mc_results['net_gain_pct']
    
    ax2.hist(ng_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(historical_netgain, color='blue', linestyle='--', linewidth=2, label=f'Historical: {historical_netgain:.2f}%')
    ax2.axvline(np.percentile(ng_data, 5), color='darkred', linestyle='-', linewidth=2, label=f'P5: {np.percentile(ng_data, 5):.2f}%')
    ax2.axvline(np.percentile(ng_data, 95), color='darkgreen', linestyle='-', linewidth=2, label=f'P95: {np.percentile(ng_data, 95):.2f}%')
    ax2.axvline(np.percentile(ng_data, 50), color='orange', linestyle=':', linewidth=1.5, label=f'P50: {np.percentile(ng_data, 50):.2f}%')
    
    ax2.set_xlabel('Net Gain (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Monte Carlo: Net Gain Distribution', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
print(f"\n📂 Loading equity files from: {FOLDER}")
print(f"⚙️  Configuration: {N_SIMULATIONS:,} simulations, {RESAMPLE_FREQ} resampling")

dfs = []
file_names = []

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

    # Resample to common frequency
    df = resample_equity(df)

    dfs.append(df)

    short_name = os.path.splitext(file_name)[0]
    file_names.append(short_name)

print(f"✅ Loaded {len(dfs)} equity files")

# =============================================================================
# BUILD COMBINED PORTFOLIO
# =============================================================================
if not dfs:
    print("❌ No equity files found. Exiting.")
    sys.exit(1)

start = min(df.index.min() for df in dfs)
end   = max(df.index.max() for df in dfs)
common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

resampled_balances = []
for df in dfs:
    df_r = df[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
    resampled_balances.append(df_r['balance'])

combined_balance = pd.concat(resampled_balances, axis=1).sum(axis=1)
combined_capital = INITIAL_CAPITAL * len(dfs)

print(f"\n📊 Combined Portfolio:")
print(f"   Total capital: {combined_capital:,.0f}")
print(f"   Time periods:  {len(combined_balance):,}")
print(f"   Date range:    {common_index[0]} to {common_index[-1]}")

# Calculate historical metrics
historical_netgain = (combined_balance.iloc[-1] - combined_capital) / combined_capital * 100

cummax = np.maximum.accumulate(combined_balance.values)
historical_dd = ((combined_balance.values - cummax) / cummax * 100).min()

print(f"\n📈 Historical Metrics:")
print(f"   Net Gain:  {historical_netgain:.2f}%")
print(f"   Max DD:    {historical_dd:.2f}%")

# =============================================================================
# RUN MONTE CARLO
# =============================================================================
mc_results = run_montecarlo_dd_netgain(
    equity_series=combined_balance,
    initial_capital=combined_capital,
    n_simulations=N_SIMULATIONS
)

# Calculate percentiles
percentiles_dd = calculate_percentiles(mc_results['max_dd'], 'Max DD', is_drawdown=True)
percentiles_netgain = calculate_percentiles(mc_results['net_gain_pct'], 'Net Gain', is_drawdown=False)

# Print results
print_percentile_table(percentiles_dd, percentiles_netgain, historical_dd, historical_netgain)

# Plot distributions
plot_montecarlo_distributions(mc_results, historical_dd, historical_netgain)

print("\n✅ Monte Carlo analysis completed\n")