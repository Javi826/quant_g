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


def report_backtesting(df, 
                         parameters, 
                         initial_capital=10000, 
                         show_plots=False, 
                         save_excel=False):
    df = df.copy()

    # -----------------------------
    # Métricas derivadas
    # -----------------------------
    # Porcentaje sobre capital inicial
    df["Net_Gain_pct"] = df["Net_Gain"] / initial_capital * 100

    # Beneficio por señal
    df["Gain_signal"] = df["Net_Gain"] / df["Num_Signals"]
    df.loc[df["Num_Signals"] == 0, "Gain_signal"] = np.nan

    # Ordenar por ganancia neta
    df_portfolio = df.sort_values(by="Net_Gain", ascending=False).reset_index(drop=True)
    
    # -----------------------------
    # Mutual Information + Pearson correlation
    # -----------------------------
    if df_portfolio.empty or df_portfolio.shape[0] < 5:
        print("\n⚠️ df_portfolio empty or <5 filas. Mutual Information y Pearson skipped.\n")
        mi_series = pd.Series([None]*len(parameters), index=parameters)
        pearson_series = pd.Series([None]*len(parameters), index=parameters)
    else:
        y = df_portfolio["Net_Gain"].values
        X = df_portfolio[parameters].copy()

        # Flags de discreto (int o bool)
        discrete_flags = [
            X[col].dtype == bool or np.issubdtype(X[col].dtype, np.integer)
            for col in X.columns
        ]

        # Convertir booleans a int para MI
        X_mi = X.copy()
        for col in X_mi.columns:
            if X_mi[col].dtype == bool:
                X_mi[col] = X_mi[col].astype(int)

        # Mutual Information
        mi_values = mutual_info_regression(X_mi, y, discrete_features=discrete_flags, random_state=42)
        mi_series = pd.Series(mi_values, index=parameters)

        # Pearson correlation
        pearson_values = []
        for col in X.columns:
            x_col = X[col].astype(int) if X[col].dtype == bool else X[col]
            if x_col.nunique() > 1:
                corr, _ = pearsonr(x_col, y)
            else:
                corr = np.nan
            pearson_values.append(corr)
        pearson_series = pd.Series(pearson_values, index=parameters)

    # Mostrar análisis MI & Pearson
    analysis_df = pd.DataFrame({
        'Mutual_Information': mi_series,
        'Pearson_Correlation': pearson_series
    }).sort_values(by='Mutual_Information', ascending=False)
    
# =============================================================================
#     print("\n🔹 Analysis MI & Pearson:")
#     print(analysis_df.round(2).to_string(index=False))
# =============================================================================
         
    metric_columns = ['Net_Gain_pct', 'Win_Ratio', 'Sharpe', 'DD_pct', 'Num_Signals']
    
    # Reorder columns: parameters first, then metrics
    ordered_columns = parameters + [col for col in metric_columns if col in df_portfolio.columns]
    df_portfolio = df_portfolio[ordered_columns]
    
    # -----------------------------
    # BEST COMBOS PER METRIC
    # -----------------------------
    best_netgain = df_portfolio.loc[df_portfolio['Net_Gain_pct'].idxmax()]
    best_sharpe  = df_portfolio.loc[df_portfolio['Sharpe'].idxmax()]
    best_dd      = df_portfolio.loc[df_portfolio['DD_pct'].idxmin()]
    
    # Build summary table
    df_summary = pd.DataFrame([
        {'Metric':'Net_Gain_pct', **best_netgain},
        {'Metric':'Sharpe      ',     **best_sharpe},
        {'Metric':'Lowest DD   ', **best_dd}
    ])
    
    # Format numeric values
    df_summary['Num_Signals'] = df_summary['Num_Signals'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    df_summary = df_summary.round(2)

    print(df_summary.to_string(index=False))
  
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
            
            # Escalar Win_Ratio para graficar
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
    # -----------------------------
    # PLOT: Net Gain % y DD vs Tiempo
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
        
        ax1.plot(timestamps, net_gain_pct, color='blue', linewidth=1.0, label='Net Gain %')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Net_Gain_pct", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(timestamps, dd_pct, color='lightcoral', linewidth=0.2, label='DD %')
        ax2.set_ylabel("Drawdown", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        fig.suptitle(title)
        fig.autofmt_xdate()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Leyenda combinada
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
        
        plt.show()
    
    # -----------------------------
    # Uso en tu función
    # -----------------------------
    best_row = df.loc[df["Net_Gain_pct"].idxmax()]
    equity_hist = best_row.get("sim_balance_history", None)
    plot_netgain_dd(equity_hist, initial_capital, title="Net_Gain_pct & DD - Best Net Gain")
 
    best_row = df.loc[df["Sharpe"].idxmax()]
    equity_hist = best_row.get("sim_balance_history", None)

    plot_netgain_dd(equity_hist, initial_capital, title="Net_Gain_pct & DD - Best Sharpe")
         
    return df_portfolio, mi_series

def report_montecarlo(df_portfolio, param_names, initial_balance):

    # -----------------------------
    # RESUMEN POR COMBINACIÓN
    # -----------------------------
    summary_results = []
    combos_present = df_portfolio[param_names].drop_duplicates().to_dict(orient='records')

    for comb in combos_present:
        filt = np.ones(len(df_portfolio), dtype=bool)
        for k, v in comb.items():
            filt &= (df_portfolio[k] == v)
        subset = df_portfolio[filt]

        port_balances = subset['Portfolio_Final_Balance'].dropna()
        port_dd       = subset['DD'].dropna() if 'DD' in subset.columns else pd.Series(dtype=float)
        port_win_ratio = subset['Win_Ratio'].dropna() if 'Win_Ratio' in subset.columns else pd.Series(dtype=float)
        port_sharpe   = subset['Sharpe'].dropna() if 'Sharpe' in subset.columns else pd.Series(dtype=float)

        if len(port_balances) > 0:
            port_gain_abs = port_balances - initial_balance
            port_gain_pct = (port_gain_abs / initial_balance) * 100
            port_net_gain_mean = port_gain_abs.mean()
            port_net_gain_pct_mean = port_gain_pct.mean()
        else:
            port_net_gain_mean = np.nan
            port_net_gain_pct_mean = np.nan

        port_dd_mean = port_dd.mean() if len(port_dd) > 0 else np.nan
        port_win_ratio_mean = port_win_ratio.mean() if len(port_win_ratio) > 0 else np.nan
        port_sharpe_mean = port_sharpe.mean() if len(port_sharpe) > 0 else np.nan

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
        'DD': 'mean'  # Asegurarnos de tener el drawdown medio por path_index
    }).reset_index()
    
    # Calcular Net Gain %
    path_grouped['Net_Gain_pct'] = (path_grouped['Portfolio_Final_Balance'] - initial_balance) / initial_balance * 100
    
    # Subplots: 2 filas, 1 columna
    fig, axes = plt.subplots(2, 1, figsize=(22,10))  # altura mayor para que no se solapen
    
    # Histograma Net_Gain_pct
    data_gain = path_grouped['Net_Gain_pct'].dropna()
    axes[0].hist(data_gain, bins=max(10,min(50,len(data_gain))), edgecolor='white', color='#1f77b4')
    axes[0].set_xlabel('Net Gain pct Portafolio (path_IDX)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution: Net Gain pct per Path_IDX')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Histograma DD
    data_dd = path_grouped['DD'].dropna()
    axes[1].hist(data_dd, bins=max(10,min(50,len(data_dd))), edgecolor='white', color='#2ca02c')
    axes[1].set_xlabel('DD pct Portafolio (path_IDX)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution: Drawdown per Path_IDX')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()  # Ajusta automáticamente los espacios
    plt.show()
    plt.close()
    
    # -----------------------------
    # TOP 3 COMBOS
    # -----------------------------
    cols_to_show = [c for c in df_summary.columns if c not in ['Net_Gain_m','Rows']]
    
    # -----------------------------
    # MEJORES COMBOS POR MÉTRICA
    # -----------------------------
    SHARPE_ADJUSTMENT_FACTOR = 1e6
    df_summary['Sharpe_m'] = df_summary['Sharpe_m'] / SHARPE_ADJUSTMENT_FACTOR
    
    # Determinar las mejores combinaciones por métrica
    best_netgain = df_summary.loc[df_summary['Net_Gain_pct_m'].idxmax()]
    best_sharpe  = df_summary.loc[df_summary['Sharpe_m'].idxmax()]
    best_dd      = df_summary.loc[df_summary['DD_m'].idxmin()]
    
    # Construir tabla resumen
    df_best = pd.DataFrame([
        {'Metric': 'Net_Gain_pct',   **best_netgain},
        {'Metric': 'Sharpe      ',       **best_sharpe},
        {'Metric': 'Lowest DD   ',  **best_dd}
    ])
    
    # Eliminar columnas innecesarias si existen
    df_best = df_best.drop(columns=['Net_Gain_m', 'Rows'], errors='ignore')
    
    # Reordenar columnas: Metric primero
    cols = ['Metric'] + [c for c in df_best.columns if c != 'Metric']
    df_best = df_best[cols]
    
    # Redondear y formatear
    df_best = df_best.round(2)

    print(df_best.to_string(index=False))


    median_gain = np.percentile(path_grouped['Net_Gain_pct'].dropna(), 50)
    print(f"\nP50 Net_Gain_pct per Path    : {median_gain:.2f}%")

    # -----------------------------
    # Desviación estándar
    # -----------------------------
    std_gain = path_grouped['Net_Gain_pct'].dropna().std()
    print(f"Std Dev Net_Gain_pct per Path: {std_gain:.2f}%")

    # -----------------------------
    # Probabilidad de path negativo
    # -----------------------------
    prob_negative = (path_grouped['Net_Gain_pct'] < 0).mean() * 100
    print(f"Probability of Negative Path : {prob_negative:.2f}%")


    return df_summary



