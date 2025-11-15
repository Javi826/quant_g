import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FOLDER = "brief_equities"
INITIAL_CAPITAL = 800  # capital for individual curves
RESAMPLE_FREQ = '4H'   # Common frequency for composition

DATA_FOLDER = "data/crypto_OOS"

# -------------------------------------------------
# --- Additional Metrics Functions
# -------------------------------------------------
def total_return(df, capital):
    return (df['balance'].iloc[-1] - capital) / capital * 100

def cagr(df, capital):
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    if days <= 0:
        return 0
    years = days / 365
    final_value = df['balance'].iloc[-1]
    return (final_value / capital) ** (1 / years) - 1

def positive_period_ratio(df):
    returns = df['balance'].pct_change().dropna()
    if len(returns) == 0:
        return 0
    return (returns > 0).mean() * 100

def profit_factor(df):
    returns = df['balance'].pct_change().dropna()
    gains   = returns[returns > 0].sum()
    losses  = -returns[returns < 0].sum()
    
    if losses == 0:
        return np.inf
    return gains / losses

def average_recovery_time(df):
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

# -------------------------------------------------
# Function to compute metrics (expanded)
# -------------------------------------------------
def compute_metrics(equity_df, capital, name="Equity"):
    df = equity_df.copy()
    df = df.sort_values('timestamp')

    # Volatility + consistency (original)
    returns = df['balance'].pct_change().dropna()
    volatility = returns.std() * 100
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_returns = df.groupby('month')['balance'].last().pct_change()
    consistency = (monthly_returns > 0).mean() * 100

    # New metrics
    tr  = total_return(df, capital)
    cg  = cagr(df, capital) * 100
    pos = positive_period_ratio(df)
    pf  = profit_factor(df)
    rt  = average_recovery_time(df) / 6

    return {
        "Curve": name,
        "Volatility_pct": round(volatility, 2),
        "Monthly_pct": round(consistency, 2),
        "Total_pct": round(tr, 2),
        "CAGR_pct": round(cg, 2),
        "PPR_pct": round(pos, 2),
        "Profit_Factor": round(pf, 3) if pf != np.inf else np.inf,
        "Rec_Time": round(rt, 2)
    }

# -------------------------------------------------
# Plotting function (unchanged)
# -------------------------------------------------
def plot_netgain_dd(equity_hist, capital, title="Net Gain % & DD"):
    timestamps = pd.to_datetime(equity_hist['timestamp'])
    balances = np.array(equity_hist['balance'])

    net_gain_pct = (balances - capital) / capital * 100
    cumulative_max = np.maximum.accumulate(balances)
    dd_pct = (balances - cumulative_max) / cumulative_max * 100

    fig, ax1 = plt.subplots(figsize=(12,6))

    ax1.plot(timestamps, net_gain_pct, color='blue', linewidth=1.2, label='Net Gain %')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net Gain %", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    try:
        btc_path = os.path.join(DATA_FOLDER, "BTCUSDT_4H.parquet")
        btc_df = pd.read_parquet(btc_path)

        if 'timestamp' not in btc_df.columns:
            if isinstance(btc_df.index, pd.DatetimeIndex):
                btc_df = btc_df.reset_index().rename(columns={'index': 'timestamp'})
            else:
                raise ValueError("BTC parquet does not have a timestamp column.")

        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
        btc_df = btc_df[['timestamp', 'close']]
        btc_df['btc_net_gain_pct'] = (btc_df['close'] / btc_df['close'].iloc[0] - 1) * 100

        ax1.plot(btc_df['timestamp'], btc_df['btc_net_gain_pct'],
                 color='black', linewidth=0.3, label='BTC %')

    except Exception as e:
        print(f"⚠️ Error loading BTC: {e}")

    ax2 = ax1.twinx()
    ax2.plot(timestamps, dd_pct, color='lightcoral', linewidth=0.1, label='Drawdown %')
    ax2.set_ylabel("Drawdown %", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    fig.suptitle(title)
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig.autofmt_xdate()
    plt.show()

# -------------------------------------------------
# Read all files (unchanged)
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
    file_names.append(file_name)

    # For correlation calculation
    correlation_data[file_name] = df['balance'].pct_change()

    plot_netgain_dd(df.reset_index(), capital=INITIAL_CAPITAL,
                    title=f"Net Gain % & DD - {file_name}")

    metrics_table.append(
        compute_metrics(df.reset_index(), capital=INITIAL_CAPITAL, name=file_name)
    )

# -------------------------------------------------
# Combined portfolio (unchanged)
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
# Final metrics table
# -------------------------------------------------
metrics_df = pd.DataFrame(metrics_table)

# Left align only 'Curve'
metrics_df['Curve'] = metrics_df['Curve'].astype(str)

print("\n📊 FINAL METRICS TABLE (ALL CURVES):\n")
print(metrics_df.to_string(index=False))

# -------------------------------------------------
# Correlation matrix
# -------------------------------------------------
print("\n📈 CORRELATION MATRIX (returns):\n")
corr_df = pd.DataFrame(correlation_data).corr().round(2)
print(corr_df.to_string())

