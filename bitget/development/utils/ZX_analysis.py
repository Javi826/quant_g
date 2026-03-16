import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


# =============================================================================
# CONFIGURATION
# =============================================================================
# Parameter vs Metrics Plots
SHOW_PLOTS = False

# Mutual Information & Pearson Correlation Analysis
ANALYZE_MI_PEARSON = False

# Parameter Surface Analysis
ANALYZE_SURFACE = False
SURFACE_PARAM_X = 'TP_PCT'
SURFACE_PARAM_Y = 'SL_PCT'
SURFACE_METRIC  = 'Net_Gain_pct'  # Options: 'Net_Gain_pct', 'Sharpe', 'Win_Ratio', 'DD_pct'


# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================

def analyze_mutual_information_pearson(df_portfolio, parameters):

    if df_portfolio.empty or df_portfolio.shape[0] < 5:
        mi_series = pd.Series([None]*len(parameters), index=parameters)
        pearson_series = pd.Series([None]*len(parameters), index=parameters)
    else:
        y = df_portfolio["Net_Gain"].values
        X = df_portfolio[parameters].copy()
        discrete_flags = [X[col].dtype == bool or np.issubdtype(X[col].dtype, np.integer) for col in X.columns]

        X_mi = X.copy()
        for col in X_mi.columns:
            if X_mi[col].dtype == bool:
                X_mi[col] = X_mi[col].astype(int)

        mi_values = mutual_info_regression(X_mi, y, discrete_features=discrete_flags, random_state=42)
        mi_series = pd.Series(mi_values, index=parameters)

        pearson_values = []
        for col in X.columns:
            x_col = X[col].astype(int) if X[col].dtype == bool else X[col]
            if x_col.nunique() > 1:
                corr, _ = pearsonr(x_col, y)
            else:
                corr = np.nan
            pearson_values.append(corr)
        pearson_series = pd.Series(pearson_values, index=parameters)

    analysis_df = pd.DataFrame({
        'Mutual_Information': mi_series,
        'Pearson_Correlation': pearson_series
    }).sort_values(by='Mutual_Information', ascending=False)
    
    return mi_series, analysis_df


def calculate_monthly_metrics(equity_hist, initial_capital):

    if not equity_hist or len(equity_hist['timestamp']) == 0:
        return pd.DataFrame()
   
    df_eq = pd.DataFrame({
        'timestamp': pd.to_datetime(equity_hist['timestamp']),
        'balance': equity_hist['balance']
    })
   
    df_eq['month'] = df_eq['timestamp'].dt.to_period('M')
   
    monthly_stats = []
    for month, group in df_eq.groupby('month'):
        start_balance = group['balance'].iloc[0]
        end_balance = group['balance'].iloc[-1]
       
        monthly_gain = end_balance - start_balance
        monthly_gain_pct = (monthly_gain / start_balance) * 100
       
        cummax = group['balance'].expanding().max()
        dd = (group['balance'] - cummax) / cummax * 100
        max_dd = dd.min()
       
        monthly_stats.append({
            'Month': str(month),
            'Net_Gain_%': monthly_gain_pct,
            'Max_DD_%': max_dd,
            'Start_Bal': start_balance,
            'End_Bal': end_balance
        })
   
    return pd.DataFrame(monthly_stats)


def plot_netgain_dd(equity_hist, initial_capital, data_folder, title="Net Gain % y DD"):

    timestamps = pd.to_datetime(equity_hist['timestamp'])
    balances = np.array(equity_hist['balance'])
   
    net_gain_pct = (balances - initial_capital) / initial_capital * 100
    cumulative_max = np.maximum.accumulate(balances)
    dd_pct = (balances - cumulative_max) / cumulative_max * 100
   
    fig, ax1 = plt.subplots(figsize=(12,6))
   
    # Load and process BTC data
    btc_file = os.path.join(data_folder, "BTCUSDT_4H.parquet")
    btc_df = pd.read_parquet(btc_file)

    if 'timestamp' not in btc_df.columns:
        if isinstance(btc_df.index, pd.DatetimeIndex):
            btc_df = btc_df.reset_index().rename(columns={'index': 'timestamp'})
        else:
            raise ValueError("BTC parquet has no 'timestamp' column or datetime index.")

    btc_df = btc_df[['timestamp', 'close']]
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df['btc_net_gain_pct'] = (btc_df['close'] / btc_df['close'].iloc[0] - 1) * 100
   
    # Align BTC data with strategy timestamps
    btc_aligned = np.interp(
        timestamps.astype(np.int64) / 10**9,
        btc_df['timestamp'].astype(np.int64) / 10**9,
        btc_df['btc_net_gain_pct']
    )
   
    # Color areas based on performance vs BTC
    above_btc = net_gain_pct >= btc_aligned
    below_btc = net_gain_pct < btc_aligned
   
    ax1.fill_between(timestamps, net_gain_pct, 0, where=above_btc, alpha=0.1, color='green', interpolate=True)
    ax1.fill_between(timestamps, net_gain_pct, 0, where=below_btc, alpha=0.1, color='red', interpolate=True)
    ax1.plot(timestamps, net_gain_pct, color='blue', linewidth=1.2, label='Net Gain %')
   
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net_Gain_pct", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
   
    ax1.plot(btc_df['timestamp'], btc_df['btc_net_gain_pct'],
             color='darkorange', linewidth=0.6, linestyle='--', label='BTC %')
   
    # Drawdown on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(timestamps, dd_pct, color='lightcoral', linewidth=0.1, label='DD %')
    ax2.set_ylabel("Drawdown", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
   
    # Statistics text box
    final_net_gain = net_gain_pct[-1]
    max_dd = dd_pct.min()
    final_btc = btc_df['btc_net_gain_pct'].iloc[-1]
   
    textstr = (
        f'Net Gain STR: {final_net_gain:.2f}%\n'
        f'Net Gain BTC: {final_btc:.2f}%\n'
        f'Max DD        : {max_dd:.2f}%'
    )

    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
   
    fig.suptitle(title)
    fig.autofmt_xdate()
    ax1.grid(True, linestyle='--', alpha=0.6)
   
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
   
    plt.show()


def plot_parameter_vs_metrics(df_portfolio, parameters):

    metrics_to_plot = []
    if 'Net_Gain_pct' in df_portfolio.columns:
        metrics_to_plot.append('Net_Gain_pct')
    if 'Win_Ratio' in df_portfolio.columns:
        metrics_to_plot.append('Win_Ratio')
   
    for param in parameters:
        agg_dict = {metric: 'sum' if metric=='Net_Gain_pct' else 'mean' for metric in metrics_to_plot}
        grouped = df_portfolio.groupby(param).agg(agg_dict).reset_index()
        if 'Win_Ratio' in grouped.columns:
            grouped['Win_Ratio_scaled'] = grouped['Win_Ratio'] * 100
       
        plt.figure(figsize=(8,5))
        plt.plot(grouped[param], grouped['Net_Gain_pct'], marker='o', color='blue', label='Net_Gain_pct')
        if 'Win_Ratio_scaled' in grouped.columns:
            plt.plot(grouped[param], grouped['Win_Ratio_scaled'], marker='o', color='green', label='Win_Ratio x100')
        plt.xlabel(param)
        plt.ylabel('Value')
        plt.title(f"{param} vs Portfolio Metrics")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


def analyze_parameter_surface(df, parameters, param_x, param_y, metric):

    print("\n" + "="*60)
    print("PARAMETER SURFACE ANALYSIS")
    print("="*60)
    
    # Validate parameters exist
    if param_x not in df.columns or param_y not in df.columns:
        print(f"⚠️  Parameters {param_x} or {param_y} not found in dataframe")
        return False
    
    if metric not in df.columns:
        print(f"⚠️  Metric {metric} not found in dataframe")
        return False
    
    # Filter to single values for other parameters
    filtered_df = df.copy()
    other_params = [p for p in parameters if p not in [param_x, param_y]]
    
    # Select most common combination for other parameters
    if other_params:
        mode_params = {}
        for param in other_params:
            mode_val = filtered_df[param].mode()
            if len(mode_val) > 0:
                mode_params[param] = mode_val.iloc[0]
        
        # Filter dataframe
        for param, val in mode_params.items():
            filtered_df = filtered_df[filtered_df[param] == val]
        
        print(f"Fixed parameters: {mode_params}")
    
    if filtered_df.empty:
        print("⚠️  No data available for surface analysis after filtering")
        return False
    
    # Create pivot table
    try:
        pivot = filtered_df.pivot_table(
            index=param_y,
            columns=param_x,
            values=metric,
            aggfunc='mean'
        )
    except Exception as e:
        print(f"⚠️  Error creating pivot table: {e}")
        return False
    
    if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        print(f"⚠️  Insufficient grid points for surface analysis: {pivot.shape}")
        return False
    
    # Calculate flatness metrics
    surface_values = pivot.values.flatten()
    surface_values_clean = surface_values[~np.isnan(surface_values)]
    
    if len(surface_values_clean) < 3:
        print("⚠️  Too few valid data points for analysis")
        return False
    
    # Basic statistics
    std_surface = np.std(surface_values_clean)
    range_surface = np.max(surface_values_clean) - np.min(surface_values_clean)
    mean_surface = np.mean(surface_values_clean)
    positive_pct = (surface_values_clean > 0).sum() / len(surface_values_clean) * 100
    
    # Calculate gradient (average change between adjacent cells)
    gradients = []
    for i in range(pivot.shape[0] - 1):
        for j in range(pivot.shape[1] - 1):
            if not np.isnan(pivot.iloc[i, j]) and not np.isnan(pivot.iloc[i+1, j]):
                gradients.append(abs(pivot.iloc[i+1, j] - pivot.iloc[i, j]))
            if not np.isnan(pivot.iloc[i, j]) and not np.isnan(pivot.iloc[i, j+1]):
                gradients.append(abs(pivot.iloc[i, j+1] - pivot.iloc[i, j]))
    
    avg_gradient = np.mean(gradients) if gradients else np.nan
    
    # Find best point and plateau center
    best_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.shape)
    best_value = pivot.iloc[best_idx]
    best_x = pivot.columns[best_idx[1]]
    best_y = pivot.index[best_idx[0]]
    
    # Identify plateau region (values within 80% of max)
    threshold = best_value * 0.80
    plateau_mask = pivot.values >= threshold
    plateau_indices = np.argwhere(plateau_mask)
    
    if len(plateau_indices) > 0:
        plateau_center_idx = (
            int(np.median(plateau_indices[:, 0])),
            int(np.median(plateau_indices[:, 1]))
        )
        plateau_x = pivot.columns[plateau_center_idx[1]]
        plateau_y = pivot.index[plateau_center_idx[0]]
        plateau_value = pivot.iloc[plateau_center_idx]
    else:
        plateau_x, plateau_y, plateau_value = best_x, best_y, best_value
    
    # Print metrics
    print(f"\nSurface Dimensions: {pivot.shape[0]} × {pivot.shape[1]} = {pivot.shape[0] * pivot.shape[1]} points")
    print(f"\nFlatness Metrics:")
    print(f"  Std Dev        : {std_surface:.2f}")
    print(f"  Range          : {range_surface:.2f}")
    print(f"  Avg Gradient   : {avg_gradient:.2f}")
    print(f"  Mean Value     : {mean_surface:.2f}")
    print(f"  Positive Cells : {positive_pct:.1f}%")
    
    print(f"\nBest Point (Peak):")
    print(f"  {param_x}={best_x}, {param_y}={best_y} → {metric}={best_value:.2f}")
    
    print(f"\nRobust Point (Plateau Center):")
    print(f"  {param_x}={plateau_x}, {param_y}={plateau_y} → {metric}={plateau_value:.2f}")
    
    # Robustness assessment
    plateau_size = len(plateau_indices)
    plateau_pct = (plateau_size / pivot.size) * 100
    print(f"\nPlateau Size: {plateau_size} cells ({plateau_pct:.1f}% of surface)")
    
    if plateau_pct > 20 and avg_gradient < mean_surface * 0.3:
        print("✅ ROBUST: Wide plateau with smooth transitions")
    elif plateau_pct > 10:
        print("⚠️  MODERATE: Some robustness but verify neighboring values")
    else:
        print("🚩 FRAGILE: Narrow peak - high parameter sensitivity")
    
    # Heatmap visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine color scale limits
    actual_min = np.nanmin(surface_values_clean)
    actual_max = np.nanmax(surface_values_clean)
    
    # Set vcenter at 0 if data crosses zero, otherwise at mean
    if actual_min < 0 < actual_max:
        vcenter = 0
    else:
        vcenter = mean_surface
    
    # Ensure vmin < vcenter < vmax
    vmin = min(actual_min, vcenter - abs(actual_max - vcenter))
    vmax = max(actual_max, vcenter + abs(vcenter - actual_min))
    
    # Create heatmap
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    
    im = ax.imshow(
        pivot.values,
        cmap='RdYlGn',
        norm=norm,
        aspect='auto',
        interpolation='nearest'
    )
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Labels
    ax.set_xlabel(param_x, fontsize=12, fontweight='bold')
    ax.set_ylabel(param_y, fontsize=12, fontweight='bold')
    ax.set_title(f'Parameter Surface: {metric}', fontsize=14, fontweight='bold')
    
    # Add values to cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=9, fontweight='bold')
    
    # Mark best point (purple X)
    ax.plot(best_idx[1], best_idx[0], 'x', color='purple', markersize=15, 
           markeredgewidth=3, label='Peak (Max)')
    
    # Mark plateau center (blue +)
    if (plateau_x != best_x or plateau_y != best_y):
        ax.plot(plateau_center_idx[1], plateau_center_idx[0], '+', color='blue',
               markersize=15, markeredgewidth=3, label='Robust Point')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, rotation=270, labelpad=20, fontsize=11)
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Grid
    ax.set_xticks(np.arange(len(pivot.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60 + "\n")
    
    return True


# =============================================================================
# MAIN REPORTING FUNCTION
# =============================================================================

def report_backtesting(df, parameters, data_folder, initial_capital, save_excel=False):
    
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Calculate derived metrics
    # -------------------------------------------------------------------------
    df["Net_Gain_pct"] = df["Net_Gain"] / initial_capital * 100
    df["Gain_signal"] = df["Net_Gain"] / df["Num_Signals"]
    df.loc[df["Num_Signals"] == 0, "Gain_signal"] = np.nan

    df_portfolio = df.sort_values(by="Net_Gain", ascending=False).reset_index(drop=True)
   
    # -------------------------------------------------------------------------
    # Mutual Information + Pearson correlation (optional)
    # -------------------------------------------------------------------------
    if ANALYZE_MI_PEARSON:
        mi_series, analysis_df = analyze_mutual_information_pearson(df_portfolio, parameters)
    else:
        mi_series = pd.Series([None]*len(parameters), index=parameters)
        analysis_df = pd.DataFrame({
            'Mutual_Information': mi_series,
            'Pearson_Correlation': pd.Series([None]*len(parameters), index=parameters)
        })
   
    # -------------------------------------------------------------------------
    # Prepare portfolio dataframe with key metrics
    # -------------------------------------------------------------------------
    metric_columns = ['Net_Gain_pct', 'Win_Ratio', 'Sharpe', 'DD_pct', 'Num_Signals', 'duration_m']
    ordered_columns = parameters + [col for col in metric_columns if col in df_portfolio.columns]
    df_portfolio = df_portfolio[ordered_columns]
   
    # -------------------------------------------------------------------------
    # Best combinations per metric
    # -------------------------------------------------------------------------
    best_netgain = df_portfolio.loc[df_portfolio['Net_Gain_pct'].idxmax()]
    best_sharpe  = df_portfolio.loc[df_portfolio['Sharpe'].idxmax()]
    best_dd      = df_portfolio.loc[df_portfolio['DD_pct'].idxmin()]
   
    df_summary = pd.DataFrame([
        {'Metric':'Net_Gain_pct', **best_netgain},
        {'Metric':'Sharpe      ', **best_sharpe},
        {'Metric':'Lowest DD   ', **best_dd}
    ])
    df_summary['Num_Signals'] = df_summary['Num_Signals'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    df_summary = df_summary.round(2)
    print(df_summary.to_string(index=False))
   
    # -------------------------------------------------------------------------
    # Monthly metrics analysis
    # -------------------------------------------------------------------------
    best_row = df.loc[df["Net_Gain_pct"].idxmax()]
    equity_hist = best_row.get("sim_balance_history", None)
   
    if equity_hist:
        monthly_df = calculate_monthly_metrics(equity_hist, initial_capital)
       
        if not monthly_df.empty:
            winning_months = (monthly_df['Net_Gain_%'] > 0).sum()
            total_months   = len(monthly_df)
            winning_pct    = (winning_months / total_months) * 100 if total_months > 0 else 0
            
            print("\n" + "-"*60)
            print("MONTHLY STATISTICS")
            print("-"*60)
            print(f"Winning Months:       {winning_months} / {total_months} ({winning_pct:.2f}%)")
            print()
            
    # -------------------------------------------------------------------------
    # Symbol distribution analysis
    # -------------------------------------------------------------------------
    trade_log = best_row.get("trade_log", None)
    
    if trade_log is not None and not trade_log.empty:
        total_trades = len(trade_log)
        
        symbol_stats = trade_log.groupby('symbol').size().to_frame('Num_Trades')
        symbol_stats['Trades_pct'] = (symbol_stats['Num_Trades'] / total_trades * 100).round(1)
        symbol_stats = symbol_stats.sort_values('Trades_pct', ascending=False).reset_index()
        
        print("\n" + "-"*60)
        print("SYMBOL DISTRIBUTION (Best Net Gain Combination)")
        print("-"*60)
        print(symbol_stats.to_string(index=False))
 
    # -------------------------------------------------------------------------
    # Optional: Parameter vs Metrics plots
    # -------------------------------------------------------------------------
    if SHOW_PLOTS:
        plot_parameter_vs_metrics(df_portfolio, parameters)
           
    # -------------------------------------------------------------------------
    # Net Gain % and Drawdown plot with BTC comparison
    # -------------------------------------------------------------------------
    best_row = df.loc[df["Net_Gain_pct"].idxmax()]
    equity_hist = best_row.get("sim_balance_history", None)
    plot_netgain_dd(equity_hist, initial_capital, data_folder, title="Net_Gain_pct & DD - Best Net Gain")
    
    # -------------------------------------------------------------------------
    # Parameter Surface Analysis
    # -------------------------------------------------------------------------
    if ANALYZE_SURFACE:
        analyze_parameter_surface(df, parameters, SURFACE_PARAM_X, SURFACE_PARAM_Y, SURFACE_METRIC)
                 
    return df_portfolio, mi_series

def report_montecarlo(df_portfolio, param_names, initial_balance):
    import ast
    
    # -----------------------------
    # RESUMEN POR COMBINACIÓN
    # -----------------------------
    summary_results = []
    
    # ===== CONVERTIR LISTAS A STRINGS PARA drop_duplicates() =====
    df_temp = df_portfolio.copy()
    list_columns = []
    
    for col in param_names:
        # Detectar si la columna contiene listas
        if df_temp[col].apply(lambda x: isinstance(x, list)).any():
            list_columns.append(col)
            # Convertir listas a strings
            df_temp[col] = df_temp[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    # Ahora drop_duplicates funciona (todo son strings o valores normales)
    combos_present = df_temp[param_names].drop_duplicates().to_dict(orient='records')
    # ==============================================================

    for comb in combos_present:
        # ===== CREAR FILTRO CORRECTO PARA LISTAS =====
        filt = np.ones(len(df_portfolio), dtype=bool)
        
        for k, v in comb.items():
            col_values = df_portfolio[k]
            
            # Si es una columna con listas (estaba en list_columns)
            if k in list_columns:
                # Normalizar v a lista o None
                if v is None or pd.isna(v):
                    v_normalized = None
                elif isinstance(v, list):
                    v_normalized = v
                elif isinstance(v, str):
                    if v in ["None", "nan", "NaN"]:
                        v_normalized = None
                    else:
                        try:
                            v_normalized = ast.literal_eval(v)
                        except:
                            v_normalized = None
                else:
                    v_normalized = None
                
                # Comparar
                if v_normalized is None:
                    filt &= col_values.isna() | (col_values == None)
                else:
                    filt &= col_values.apply(lambda x: x == v_normalized if isinstance(x, list) else False)
            else:
                # Comparación normal (números, strings, etc.)
                filt &= (col_values == v)
        # ==============================================
        
        subset = df_portfolio[filt]

        # Skip si no hay datos para esta combinación
        if subset.empty:
            continue

        port_balances  = subset['Portfolio_Final_Balance'].dropna()
        port_dd        = subset['DD'].dropna() if 'DD' in subset.columns else pd.Series(dtype=float)
        port_win_ratio = subset['Win_Ratio'].dropna() if 'Win_Ratio' in subset.columns else pd.Series(dtype=float)
        port_sharpe    = subset['Sharpe'].dropna() if 'Sharpe' in subset.columns else pd.Series(dtype=float)

        if len(port_balances) > 0:
            port_gain_abs          = port_balances - initial_balance
            port_gain_pct          = (port_gain_abs / initial_balance) * 100
            port_net_gain_mean     = port_gain_abs.mean()
            port_net_gain_pct_mean = port_gain_pct.mean()
        else:
            port_net_gain_mean = np.nan
            port_net_gain_pct_mean = np.nan

        port_dd_mean        = port_dd.mean() if len(port_dd) > 0 else np.nan
        port_win_ratio_mean = port_win_ratio.mean() if len(port_win_ratio) > 0 else np.nan
        port_sharpe_mean    = port_sharpe.mean() if len(port_sharpe) > 0 else np.nan

        summary_results.append({
            **comb,
            "Net_Gain_m": port_net_gain_mean,
            "Net_Gain_pct_m": port_net_gain_pct_mean,
            "Win_Ratio_m": port_win_ratio_mean,
            "DD_m": port_dd_mean,
            "Sharpe_m": port_sharpe_mean,
            "Paths_IDX": subset['path_index'].nunique() if 'path_index' in subset.columns else np.nan,
            "Rows": len(subset)
        })

    df_summary = pd.DataFrame(summary_results).sort_values(by='Net_Gain_pct_m', ascending=False).reset_index(drop=True)

    # -----------------------------
    # HISTOGRAMAS
    # -----------------------------
    path_grouped = df_portfolio.groupby('path_index').agg({
        'Portfolio_Final_Balance': 'mean',
        'DD': 'mean',
        'Win_Ratio': 'mean'
    }).reset_index()
   
    path_grouped['Net_Gain_pct'] = (path_grouped['Portfolio_Final_Balance'] - initial_balance) / initial_balance * 100
   
    fig, axes = plt.subplots(2, 1, figsize=(22,10))
   
    # Histograma Net_Gain_pct
    data_gain = path_grouped['Net_Gain_pct'].dropna()
    n_bins = max(10, min(50, len(data_gain)))
    counts, bins, patches = axes[0].hist(data_gain, bins=n_bins, edgecolor='white')
   
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        patch.set_facecolor('green' if bin_center >= 0 else 'red')
   
    axes[0].set_xlabel('Net Gain pct Portafolio (path_IDX)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution: Net Gain pct per Path_IDX')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
   
    # Histograma DD (granate)
    data_dd = path_grouped['DD'].dropna()
    axes[1].hist(data_dd, bins=max(10,min(50,len(data_dd))), edgecolor='white', color='lightcoral')
    axes[1].set_xlabel('DD pct Portafolio (path_IDX)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution: Drawdown per Path_IDX')
    axes[1].grid(True, linestyle='--', alpha=0.5)
   
    # -----------------------------
    # ETIQUETA SIMPLIFICADA
    # -----------------------------
    prob_negative = (path_grouped['Net_Gain_pct'] < 0).mean() * 100
    textstr = f'Probability of Negative Path: {prob_negative:.2f}%'

    fig.text(
        0.75, 0.90, textstr,      
        fontsize=14,
        fontfamily='monospace',
        va='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.9)
    )
   
    plt.tight_layout()
    plt.show()
    plt.close()
       
    # -----------------------------
    # MEJORES COMBOS POR MÉTRICA
    # -----------------------------
    SHARPE_ADJUSTMENT_FACTOR = 1e6
    df_summary['Sharpe_m']   = df_summary['Sharpe_m'] / SHARPE_ADJUSTMENT_FACTOR
   
    best_netgain = df_summary.loc[df_summary['Net_Gain_pct_m'].idxmax()]
    best_sharpe  = df_summary.loc[df_summary['Sharpe_m'].idxmax()]
    best_dd      = df_summary.loc[df_summary['DD_m'].idxmin()]
   
    df_best = pd.DataFrame([
        {'Metric': 'Net_Gain_pct',   **best_netgain},
        {'Metric': 'Sharpe      ',   **best_sharpe},
        {'Metric': 'Lowest DD   ',  **best_dd}
    ])
   
    df_best = df_best.drop(columns=['Net_Gain_m', 'Rows'], errors='ignore')
    cols    = ['Metric'] + [c for c in df_best.columns if c != 'Metric']
    df_best = df_best[cols]
    df_best = df_best.round(2)

    print(df_best.to_string(index=False))

    median_gain   = np.percentile(path_grouped['Net_Gain_pct'].dropna(), 50)
    print(f"\nP50 Net_Gain_pct per Path    : {median_gain:.2f}%")
    std_gain      = path_grouped['Net_Gain_pct'].dropna().std()
    print(f"Std Dev Net_Gain_pct per Path: {std_gain:.2f}%")
    prob_negative = (path_grouped['Net_Gain_pct'] < 0).mean() * 100
    print(f"Probability of Negative Path : {prob_negative:.2f}%")
    
    # Luego calcular P5 y P50 del Win Rate
    win_rates = path_grouped['Win_Ratio'].dropna()
    p5_winrate = np.percentile(win_rates, 5)
    p50_winrate = np.percentile(win_rates, 50)
    
    print(f"\nP5  Win Rate per Path: {p5_winrate:.2f}%")
    print(f"P50 Win Rate per Path: {p50_winrate:.2f}%")

    return df_summary