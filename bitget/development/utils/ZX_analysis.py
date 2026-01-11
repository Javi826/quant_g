import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)



def report_backtesting(df, parameters, data_folder, initial_capital, show_plots=False, save_excel=False):
 
    df = df.copy()
    # -----------------------------
    # Métricas derivadas
    # -----------------------------
    df["Net_Gain_pct"]       = df["Net_Gain"] / initial_capital * 100
    df["Gain_signal"]        = df["Net_Gain"] / df["Num_Signals"]
    df.loc[df["Num_Signals"] == 0, "Gain_signal"] = np.nan

    df_portfolio = df.sort_values(by="Net_Gain", ascending=False).reset_index(drop=True)
   
    # -----------------------------
    # Mutual Information + Pearson correlation - COMENTADO
    # -----------------------------
    # if df_portfolio.empty or df_portfolio.shape[0] < 5:
    #     mi_series = pd.Series([None]*len(parameters), index=parameters)
    #     pearson_series = pd.Series([None]*len(parameters), index=parameters)
    # else:
    #     y = df_portfolio["Net_Gain"].values
    #     X = df_portfolio[parameters].copy()
    #     discrete_flags = [X[col].dtype == bool or np.issubdtype(X[col].dtype, np.integer) for col in X.columns]

    #     X_mi = X.copy()
    #     for col in X_mi.columns:
    #         if X_mi[col].dtype == bool:
    #             X_mi[col] = X_mi[col].astype(int)

    #     mi_values = mutual_info_regression(X_mi, y, discrete_features=discrete_flags, random_state=42)
    #     mi_series = pd.Series(mi_values, index=parameters)

    #     pearson_values = []
    #     for col in X.columns:
    #         x_col = X[col].astype(int) if X[col].dtype == bool else X[col]
    #         if x_col.nunique() > 1:
    #             corr, _ = pearsonr(x_col, y)
    #         else:
    #             corr = np.nan
    #         pearson_values.append(corr)
    #     pearson_series = pd.Series(pearson_values, index=parameters)

    # analysis_df = pd.DataFrame({
    #     'Mutual_Information': mi_series,
    #     'Pearson_Correlation': pearson_series
    # }).sort_values(by='Mutual_Information', ascending=False)
   
    # ===== REEMPLAZO TEMPORAL: Series vacías =====
    mi_series = pd.Series([None]*len(parameters), index=parameters)
    pearson_series = pd.Series([None]*len(parameters), index=parameters)
    analysis_df = pd.DataFrame({
        'Mutual_Information': mi_series,
        'Pearson_Correlation': pearson_series
    })
    # ==============================================
   
    # Incluir duration_m en las métricas mostradas (duration en minutos)
    metric_columns = ['Net_Gain_pct', 'Win_Ratio', 'Sharpe', 'DD_pct', 'Num_Signals', 'duration_m']

    ordered_columns = parameters + [col for col in metric_columns if col in df_portfolio.columns]
    df_portfolio = df_portfolio[ordered_columns]
   
    # -----------------------------
    # BEST COMBOS PER METRIC
    # -----------------------------
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
   
    # -----------------------------
    # MONTHLY METRICS TABLE
    # -----------------------------
    def calculate_monthly_metrics(equity_hist, initial_capital):
        """Calcula métricas mensuales a partir del historial de equity"""
        if not equity_hist or len(equity_hist['timestamp']) == 0:
            return pd.DataFrame()
       
        df_eq = pd.DataFrame({
            'timestamp': pd.to_datetime(equity_hist['timestamp']),
            'balance': equity_hist['balance']
        })
       
        # Agrupar por mes
        df_eq['month'] = df_eq['timestamp'].dt.to_period('M')
       
        monthly_stats = []
        for month, group in df_eq.groupby('month'):
            start_balance = group['balance'].iloc[0]
            end_balance = group['balance'].iloc[-1]
           
            # Net Gain del mes
            monthly_gain = end_balance - start_balance
            monthly_gain_pct = (monthly_gain / start_balance) * 100
           
            # Drawdown máximo del mes
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
   
    # Calcular métricas mensuales para la mejor combinación por Net_Gain
    best_row = df.loc[df["Net_Gain_pct"].idxmax()]
    equity_hist = best_row.get("sim_balance_history", None)
   
    if equity_hist:
        monthly_df = calculate_monthly_metrics(equity_hist, initial_capital)
       
        if not monthly_df.empty:
# =============================================================================
#             print("\n" + "="*60)
#             print("MONTHLY PERFORMANCE - Best Net_Gain Strategy")
#             print("="*60)
#            
#             # Formatear la tabla
#             monthly_display = monthly_df.copy()
#             monthly_display['Net_Gain_%'] = monthly_display['Net_Gain_%'].apply(lambda x: f"{x:.2f}")
#             monthly_display['Max_DD_%']   = monthly_display['Max_DD_%'].apply(lambda x: f"{x:.2f}")
#             monthly_display['Start_Bal']  = monthly_display['Start_Bal'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
#             monthly_display['End_Bal']    = monthly_display['End_Bal'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
# =============================================================================
           
            #print(monthly_display.to_string(index=False))
           
            # Estadísticas agregadas
            print("\n" + "-"*60)
            print("MONTHLY STATISTICS")
            print("-"*60)
            print(f"Average Monthly Gain: {monthly_df['Net_Gain_%'].mean():.2f}%")
            print(f"Best Month:           {monthly_df['Net_Gain_%'].max():.2f}%")
            print(f"Worst Month:          {monthly_df['Net_Gain_%'].min():.2f}%")
            print(f"Winning Months:       {(monthly_df['Net_Gain_%'] > 0).sum()} / {len(monthly_df)}")
            print(f"Average Monthly DD:   {monthly_df['Max_DD_%'].mean():.2f}%")
            print()
 
    # -----------------------------
    # PLOTS
    # -----------------------------
    if show_plots:
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
           
    # -----------------------------
    # PLOT: Net Gain % y DD vs Tiempo (con BTC siempre)
    # -----------------------------
    def plot_netgain_dd(equity_hist, initial_capital, title="Net Gain % y DD"):
        timestamps = pd.to_datetime(equity_hist['timestamp'])
        balances = np.array(equity_hist['balance'])
       
        # Net Gain %
        net_gain_pct = (balances - initial_capital) / initial_capital * 100
       
        # Drawdown %
        cumulative_max = np.maximum.accumulate(balances)
        dd_pct = (balances - cumulative_max) / cumulative_max * 100
       
        fig, ax1 = plt.subplots(figsize=(12,6))
       
        # --- Línea Bitcoin (antes de plotear Net Gain para poder comparar) ---
        DATA_FOLDER = data_folder
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
        btc_aligned = np.interp(
            timestamps.astype(np.int64) / 10**9,
            btc_df['timestamp'].astype(np.int64) / 10**9,
            btc_df['btc_net_gain_pct']
        )
   
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
   
        # Drawdown %
        ax2 = ax1.twinx()
        ax2.plot(timestamps, dd_pct, color='lightcoral', linewidth=0.1, label='DD %')
        ax2.set_ylabel("Drawdown", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
       
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

    # -----------------------------
    # Uso de la función
    # -----------------------------
    best_row    = df.loc[df["Net_Gain_pct"].idxmax()]
    equity_hist = best_row.get("sim_balance_history", None)
    plot_netgain_dd(equity_hist, initial_capital, title="Net_Gain_pct & DD - Best Net Gain")
         
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
        'DD': 'mean'
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

    return df_summary


