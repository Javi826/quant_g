import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


FOLDER              = "brief_equities"
INITIAL_CAPITAL     = 800
RESAMPLE_FREQ       = '4h'
DATA_FOLDER         = "data/crypto_OOS"

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
# --- New Smoothness Metrics
# -------------------------------------------------

def ulcer_index(df):
    balance = df["balance"].values
    peaks = np.maximum.accumulate(balance)
    dd = (balance - peaks) / peaks * 100
    return np.sqrt(np.mean(dd**2))

def rmse_trend(df):
    df2 = df.reset_index(drop=True)
    X = np.arange(len(df2)).reshape(-1, 1)
    y = df2["balance"].values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    return np.sqrt(np.mean((y - trend) ** 2))

# -------------------------------------------------
# Function to compute metrics
# -------------------------------------------------
def compute_metrics(equity_df, capital, name="Equity"):
    df = equity_df.copy()
    df = df.sort_values('timestamp')

    returns = df['balance'].pct_change().dropna()
    volatility = returns.std() * 100

    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_returns = df.groupby('month')['balance'].last().pct_change()
    consistency = (monthly_returns > 0).mean() * 100

    tr  = total_return(df, capital)
    cg  = cagr(df, capital) * 100
    pos = positive_period_ratio(df)
    pf  = profit_factor(df)
    rt  = average_recovery_time(df) / 6

    ui = ulcer_index(df)
    rm = rmse_trend(df)

    return {
        "Curve": name,
        "Volatility_pct": round(volatility, 2),
        "Monthly_pct": round(consistency, 2),
        "Total_pct": round(tr, 2),
        "CAGR_pct": round(cg, 2),
        "PPR_pct": round(pos, 2),
        "Profit_Factor": round(pf, 3) if pf != np.inf else np.inf,
        "Rec_Time": round(rt, 2),
        "Ulcer_Index": round(ui, 3),
        "RMSE": round(rm, 3),
    }

# -------------------------------------------------
# Plot function (unchanged)
# -------------------------------------------------
def plot_netgain_dd(equity_hist, capital, title="Net Gain % y DD"):
    initial_capital = capital
    timestamps = pd.to_datetime(equity_hist['timestamp'])
    balances = np.array(equity_hist['balance'])
    
    # Net Gain %
    net_gain_pct = (balances - initial_capital) / initial_capital * 100
    
    # Drawdown %
    cumulative_max = np.maximum.accumulate(balances)
    dd_pct = (balances - cumulative_max) / cumulative_max * 100
    
    fig, ax1 = plt.subplots(figsize=(12,6))
    
    # --- Línea Bitcoin (antes de plotear Net Gain para poder comparar) ---
    btc_file = os.path.join(DATA_FOLDER, "BTCUSDT_4H.parquet")
    btc_df = pd.read_parquet(btc_file)

    if 'timestamp' not in btc_df.columns:
        if isinstance(btc_df.index, pd.DatetimeIndex):
            btc_df = btc_df.reset_index().rename(columns={'index': 'timestamp'})
        else:
            raise ValueError("El parquet de BTC no tiene columna 'timestamp' ni índice datetime.")

    btc_df = btc_df[['timestamp', 'close']]
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df['btc_net_gain_pct'] = (btc_df['close'] / btc_df['close'].iloc[0] - 1) * 100

    # --- Comparar con BTC para colorear dinámicamente el área ---
    # Alinear BTC con nuestros timestamps
    btc_aligned = np.interp(
        timestamps.astype(np.int64) / 10**9,  # convertir a segundos
        btc_df['timestamp'].astype(np.int64) / 10**9,
        btc_df['btc_net_gain_pct']
    )

    # Crear máscaras para cuando superamos o no a BTC
    above_btc = net_gain_pct >= btc_aligned
    below_btc = net_gain_pct < btc_aligned

    # Área verde donde superamos BTC
    ax1.fill_between(timestamps, net_gain_pct, 0, where=above_btc, alpha=0.2, color='green', interpolate=True)

    # Área roja donde NO superamos BTC
    ax1.fill_between(timestamps, net_gain_pct, 0, where=below_btc, alpha=0.2, color='red', interpolate=True)

    # Línea azul siempre (encima de las áreas)
    ax1.plot(timestamps, net_gain_pct, color='blue', linewidth=1.2, label='Net Gain %')

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net_Gain_pct", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Graficar BTC con línea naranja punteada
    ax1.plot(btc_df['timestamp'], btc_df['btc_net_gain_pct'], 
             color='darkorange', linewidth=0.6, linestyle='--', label='BTC %')

    # Drawdown %
    ax2 = ax1.twinx()
    ax2.plot(timestamps, dd_pct, color='lightcoral', linewidth=0.1, label='DD %')
    ax2.set_ylabel("Drawdown", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Calcular valores para etiquetas
    final_net_gain = net_gain_pct[-1]  # Último valor en lugar del máximo
    max_dd = dd_pct.min()
    final_btc = btc_df['btc_net_gain_pct'].iloc[-1]
    
    # Añadir etiquetas en el plot
    textstr = f'Final Net Gain: {final_net_gain:.2f}%\nMax DD: {max_dd:.2f}%\nBTC Final: {final_btc:.2f}%'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title)
    fig.autofmt_xdate()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Leyenda combinada
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
# Ajuste de la columna Curve para que quede alineada a la izquierda
metrics_df_display = metrics_df.copy()
max_len = metrics_df_display['Curve'].str.len().max()
metrics_df_display['Curve'] = metrics_df_display['Curve'].apply(lambda x: x.ljust(max_len))

print("\n📊 FINAL METRICS TABLE (ALL CURVES):\n")
print(metrics_df_display.to_string(index=False))


# -------------------------------------------------
# COMBINATIONS SEARCH (unchanged)
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

combo_df = pd.DataFrame(combo_results)
combo_df = combo_df.sort_values("CAGR_pct", ascending=False)

combo_df_display = combo_df.copy()
max_len_combo = combo_df_display['Curve'].str.len().max()
combo_df_display['Curve'] = combo_df_display['Curve'].apply(lambda x: x.ljust(max_len_combo))

print("\n🏆 MEJORES COMBINACIONES (ordenadas por CAGR):\n")
print(combo_df_display.to_string(index=False))



best_name = combo_df.iloc[0]["Curve"]
best_combo = best_name.split("+")

best_dfs = [named_dfs[name] for name in best_combo]

start = min(df.index.min() for df in best_dfs)
end   = max(df.index.max() for df in best_dfs)
common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

resampled = []
for df in best_dfs:
    df_r = df[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
    resampled.append(df_r['balance'])

combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
best_df = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})

best_capital = INITIAL_CAPITAL * len(best_dfs)

plot_netgain_dd(best_df, capital=best_capital,
                title=f"Best Combination: {best_name}")

# -------------------------------------------------
# Mejor combinación por CAGR
# -------------------------------------------------
best_name_cagr = combo_df.iloc[0]["Curve"]
best_combo_cagr = best_name_cagr.split("+")

best_dfs_cagr = [named_dfs[name] for name in best_combo_cagr]

start = min(df.index.min() for df in best_dfs_cagr)
end   = max(df.index.max() for df in best_dfs_cagr)
common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

resampled = []
for df in best_dfs_cagr:
    df_r = df[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
    resampled.append(df_r['balance'])

combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
best_df_cagr = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})
best_capital = INITIAL_CAPITAL * len(best_dfs_cagr)

plot_netgain_dd(best_df_cagr, capital=best_capital,
                title=f"Best Combination by CAGR: {best_name_cagr}")


# -------------------------------------------------
# Mejor combinación por Ulcer Index (menor)
# -------------------------------------------------
best_name_ui = combo_df.loc[combo_df['Ulcer_Index'].idxmin(), "Curve"]
best_combo_ui = best_name_ui.split("+")

best_dfs_ui = [named_dfs[name] for name in best_combo_ui]

start = min(df.index.min() for df in best_dfs_ui)
end   = max(df.index.max() for df in best_dfs_ui)
common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

resampled = []
for df in best_dfs_ui:
    df_r = df[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
    resampled.append(df_r['balance'])

combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
best_df_ui = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})
best_capital = INITIAL_CAPITAL * len(best_dfs_ui)

plot_netgain_dd(best_df_ui, capital=best_capital,
                title=f"Best Combination by Ulcer Index: {best_name_ui}")


# -------------------------------------------------
# Mejor combinación por RMSE (menor)
# -------------------------------------------------
best_name_rmse = combo_df.loc[combo_df['RMSE'].idxmin(), "Curve"]
best_combo_rmse = best_name_rmse.split("+")

best_dfs_rmse = [named_dfs[name] for name in best_combo_rmse]

start = min(df.index.min() for df in best_dfs_rmse)
end   = max(df.index.max() for df in best_dfs_rmse)
common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)

resampled = []
for df in best_dfs_rmse:
    df_r = df[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
    resampled.append(df_r['balance'])

combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
best_df_rmse = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})
best_capital = INITIAL_CAPITAL * len(best_dfs_rmse)

plot_netgain_dd(best_df_rmse, capital=best_capital,
                title=f"Best Combination by RMSE: {best_name_rmse}")

# -------------------------------------------------
# Mejor combinación por Profit Factor (mayor)
# -------------------------------------------------
best_name_pf = combo_df.loc[combo_df['Profit_Factor'].idxmax(), "Curve"]
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
# # -------------------------------------------------
# # Print final personalizado para una combinación específica
# # -------------------------------------------------
# 
# custom_combo_name = "equity_reversal_long+equity_double_top_long+equity_parity_long+equity_reversal_short"
# 
# # Filtrar métricas del combo seleccionado
# custom_metrics = combo_df.loc[combo_df['Curve'] == custom_combo_name]
# 
# if not custom_metrics.empty:
#     # Ajustamos la columna 'Curve' para que el texto empiece a la izquierda
#     custom_metrics_formatted = custom_metrics.copy()
#     custom_metrics_formatted['Curve'] = custom_metrics_formatted['Curve'].str.ljust(70)  # ajustar tamaño según convenga
#     
#     print("\n📊 METRICS TABLE - COMBINACIÓN PERSONALIZADA:\n")
#     print(custom_metrics_formatted.to_string(index=False))
# else:
#     print(f"⚠️ No se encontraron métricas para la combinación: {custom_combo_name}")
# 
# # -------------------------------------------------
# # Plot personalizado para la misma combinación
# # -------------------------------------------------
# 
# custom_combo_name = "equity_reversal_long+equity_double_top_long+equity_parity_long+equity_reversal_short"
# 
# # Filtrar la lista de nombres que componen la combinación
# custom_combo_list = custom_combo_name.split("+")
# 
# # Extraer los DataFrames correspondientes
# custom_dfs = [named_dfs[name] for name in custom_combo_list]
# 
# # Crear índice común
# start = min(df.index.min() for df in custom_dfs)
# end   = max(df.index.max() for df in custom_dfs)
# common_index = pd.date_range(start=start, end=end, freq=RESAMPLE_FREQ)
# 
# # Interpolar y combinar balances
# resampled = []
# for df in custom_dfs:
#     df_r = df[['balance']].reindex(common_index)
#     df_r['balance'] = df_r['balance'].interpolate(method='time').ffill().bfill()
#     resampled.append(df_r['balance'])
# 
# combined_balance = pd.concat(resampled, axis=1).sum(axis=1)
# custom_df = pd.DataFrame({'timestamp': common_index, 'balance': combined_balance})
# 
# custom_capital = INITIAL_CAPITAL * len(custom_dfs)
# 
# # Generar plot
# plot_netgain_dd(custom_df, capital=custom_capital,
#                 title=f"Custom Combination: {custom_combo_name}")
# =============================================================================
