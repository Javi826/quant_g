import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FOLDER = "brief_equities"
INITIAL_CAPITAL = 10000
RESAMPLE_FREQ = '1D'  # Frecuencia común para composición

def plot_netgain_dd(equity_hist, title="Net Gain % & DD"):
    timestamps = pd.to_datetime(equity_hist['timestamp'])
    balances = np.array(equity_hist['balance'])

    net_gain_pct = (balances - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    cumulative_max = np.maximum.accumulate(balances)
    dd_pct = (balances - cumulative_max) / cumulative_max * 100

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(timestamps, net_gain_pct, color='blue', linewidth=1.2, label='Net Gain %')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net Gain %", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(timestamps, dd_pct, color='red', linewidth=0.3, label='Drawdown %')
    ax2.set_ylabel("Drawdown %", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    fig.suptitle(title)
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig.autofmt_xdate()
    plt.show()

# --- Leer todos los ficheros ---
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
    dfs.append(df)
    file_names.append(file_name)

    # Plot individual
    plot_netgain_dd(df.reset_index(), title=f"Net Gain % & DD - {file_name}")

# --- Composición ---
if dfs:
    # Crear índice común
    start = min(df.index.min() for df in dfs)
    end   = max(df.index.max() for df in dfs)
    common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

    # Resample/interpolate cada df a índice común
    resampled_balances = []
    for df in dfs:
        df_r = df[['balance']].reindex(common_index)
        df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
        resampled_balances.append(df_r['balance'])

    # Sumar las equities
    combined_balance = sum(resampled_balances)
    combined_df = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})

    # Plot composición
    plot_netgain_dd(combined_df, title="Net Gain % & DD - Combined Portfolio")
