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
        print("\n⚠️ df_portfolio vacío o con <5 filas. Mutual Information y Pearson skipped.")
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
    
    #print("\n🔹 Analysis MI & Pearson:")
    #print(analysis_df.round(2).to_string(index=False))
         
    metric_columns = [
        'Net_Gain_pct', 
        'Win_Ratio', 
        'Num_Signals', 
        'Gain_signal', 
        'DD_pct'
    ]

    # Reorganizar columnas: parámetros primero, luego métricas
    ordered_columns = parameters + [col for col in metric_columns if col in df_portfolio.columns]
    df_portfolio = df_portfolio[ordered_columns]

    # -----------------------------
    # TOP COMBOS
    # -----------------------------
    df_top = df_portfolio.sort_values(by='Net_Gain_pct', ascending=False).head(3).copy()
    df_top['Num_Signals'] = df_top['Num_Signals'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    
    print("\n🥇 Top 3 combos by Net_Gain_pct:")
    print(df_top.round(2).to_string(index=False))

    df_top_win = df_portfolio.sort_values(by='DD_pct', ascending=True).head(3).copy()
    df_top_win['Num_Signals'] = df_top_win['Num_Signals'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    
    print("\n🥇 Top 3 combos by DD_pct:")
    print(df_top_win.round(2).to_string(index=False))
    
    # -----------------------------
    # PLOTS
    # -----------------------------
    if show_plots:
        # Histograma DD_pct
        plt.figure(figsize=(10,6))
        plt.hist(df_portfolio['DD_pct'].dropna(), bins=20, color='#2ca02c', edgecolor='black', alpha=0.7)
        plt.xlabel("DD_pct")
        plt.ylabel("Frequency")
        plt.title("Distribution of Portfolio Drawdown")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()
        
        # Histograma Net_Gain_pct
        plt.figure(figsize=(10,6))
        plt.hist(df_portfolio['Net_Gain_pct'].dropna(), bins=20, color='#1f77b4', edgecolor='black', alpha=0.7)
        plt.xlabel("Net_Gain_pct (%)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Portfolio Net Gain %")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

        # Curvas por parámetro
        win_ratio_scale = 100
        colors = {'Net_Gain_pct':'blue', 'Win_Ratio':'green'}
        for param in parameters:
            grouped = df_portfolio.groupby(param).agg({
                'Net_Gain': 'sum',
                'Win_Ratio': 'mean'
            }).reset_index()
            grouped['Net_Gain_pct'] = grouped['Net_Gain'] / initial_capital * 100
            grouped['Win_Ratio_scaled'] = grouped['Win_Ratio'] * win_ratio_scale
            
            plt.figure(figsize=(8,5))
            plt.plot(grouped[param], grouped['Net_Gain_pct'], marker='o', color=colors['Net_Gain_pct'], label='Net_Gain_pct')
            plt.plot(grouped[param], grouped['Win_Ratio_scaled'], marker='o', color=colors['Win_Ratio'], label=f'Win_Ratio x {win_ratio_scale}')
            plt.xlabel(param)
            plt.ylabel('Value')
            plt.title(f"{param} vs Portfolio Metrics")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()
                
    return df_portfolio, mi_series

def report_montecarlo(df_portfolio, param_names, initial_balance):

    # -----------------------------
    # RESUMEN POR COMBINACIÓN
    # -----------------------------
    summary_results = []
    combos_present  = df_portfolio[param_names].drop_duplicates().to_dict(orient='records')

    for comb in combos_present:
        filt = np.ones(len(df_portfolio), dtype=bool)
        for k, v in comb.items():
            filt &= (df_portfolio[k] == v)
        subset = df_portfolio[filt]

        port_balances = subset['Portfolio_Final_Balance'].dropna()
        port_dd = subset['DD'].dropna()
        port_win_ratio = subset['Win_Ratio'].dropna() if 'Win_Ratio' in subset.columns else pd.Series(dtype=float)

        if len(port_balances) > 0:
            port_gain_abs          = port_balances - initial_balance
            port_gain_pct          = (port_gain_abs / initial_balance) * 100
            port_net_gain_mean     = port_gain_abs.mean()
            port_net_gain_pct_mean = port_gain_pct.mean()
        else:
            port_net_gain_mean     = np.nan
            port_net_gain_pct_mean = np.nan

        port_dd_mean = port_dd.mean() if len(port_dd) > 0 else np.nan
        port_win_ratio_mean = port_win_ratio.mean() if len(port_win_ratio) > 0 else np.nan

        summary_results.append({
            **comb,
            "Net_Gain_m": port_net_gain_mean,
            "Net_Gain_pct_m": port_net_gain_pct_mean,
            "Win_Ratio_m": port_win_ratio_mean,
            "DD_m": port_dd_mean,
            "Paths_IDX": subset['path_index'].nunique() if 'path_index' in subset.columns else np.nan,
            "Rows": len(subset)
        })

    df_summary = pd.DataFrame(summary_results).sort_values(by='Net_Gain_pct_m', ascending=False).reset_index(drop=True)

    # -----------------------------
    # HISTOGRAMS
    # -----------------------------
    group_cols = param_names
    combo_grouped = df_portfolio.groupby(group_cols)['Portfolio_Final_Balance'].mean().reset_index()
    combo_grouped['Net_Gain_pct_m'] = (combo_grouped['Portfolio_Final_Balance'] - initial_balance)/initial_balance*100

    plt.figure(figsize=(10,6))
    data1 = combo_grouped['Net_Gain_pct_m'].dropna()
    plt.hist(data1, bins=50, edgecolor='black', color='#2ca02c', alpha=0.7)
    plt.xlabel('Port Net Gain pct (mean per combination)')
    plt.ylabel('Frequency')
    plt.title('Distribution: Port Net Gain pct (mean per combination)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.close()

    path_grouped = df_portfolio.groupby('path_index')['Portfolio_Final_Balance'].mean().reset_index()
    path_grouped['Net_Gain_pct'] = (path_grouped['Portfolio_Final_Balance'] - initial_balance)/initial_balance*100

    plt.figure(figsize=(10,6))
    data2 = path_grouped['Net_Gain_pct'].dropna()
    plt.hist(data2, bins=max(10,min(50,len(data2))), edgecolor='white', color='#1f77b4')
    plt.xlabel('Net Gain pct Portafolio (path_IDX)')
    plt.ylabel('Frequency')
    plt.title('Distribution: Net Gain pct Portafolio per Path_IDX')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.close()

    cols_to_show = [c for c in df_summary.columns if c not in ['Net_Gain_m','Rows']]

    print('\n🎲 Top 3 combos per Net_Gain_pct_m:')
    print(df_summary[cols_to_show].head(3).round(2).to_string(index=False))

    print('\n🎲 Top 3 combos per Win_Ratio_m:')
    print(df_summary.sort_values(by='Win_Ratio_m', ascending=False).head(3)[cols_to_show].round(2).to_string(index=False))

    median_gain = np.percentile(path_grouped['Net_Gain_pct'].dropna(), 50)
    print(f"\n🎲 P50 Net_Gain_pct per Path: {median_gain:.2f}%")

    return df_summary


