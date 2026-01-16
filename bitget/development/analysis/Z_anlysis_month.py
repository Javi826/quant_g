#!/usr/bin/env python3
"""
Script autocontenido para análisis mensual de trades
Lee automáticamente el archivo más reciente de brief_trades/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def find_latest_file(folder_path):
    """Encuentra el archivo más reciente en la carpeta"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"La carpeta {folder_path} no existe")
    
    files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]
    
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos Excel en {folder_path}")
    
    files_with_time = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in files]
    latest_file = max(files_with_time, key=lambda x: x[1])[0]
    
    return os.path.join(folder_path, latest_file)


def load_trades(file_path):
    """Carga el archivo de trades"""
    print(f"📂 Cargando archivo: {os.path.basename(file_path)}")
    
    # Intentar leer la primera hoja
    df = pd.read_excel(file_path, sheet_name=0)
    
    print(f"✅ Cargadas {len(df)} trades")
    return df


def calculate_monthly_metrics(df):
    """Calcula métricas mensuales de profit y win ratio"""
    
    # Convertir sell_time a datetime si no lo está
    if 'sell_time' not in df.columns:
        raise ValueError("El archivo debe tener una columna 'sell_time'")
    
    df['sell_time'] = pd.to_datetime(df['sell_time'], errors='coerce')
    
    # Filtrar trades válidas (con sell_time y profit)
    df_valid = df[df['sell_time'].notna() & df['profit'].notna()].copy()
    
    if len(df_valid) == 0:
        raise ValueError("No hay trades válidas para analizar")
    
    # Crear columna de mes
    df_valid['month'] = df_valid['sell_time'].dt.to_period('M')
    
    # Agrupar por mes
    monthly_stats = []
    
    for month, group in df_valid.groupby('month'):
        total_trades = len(group)
        winning_trades = (group['profit'] > 0).sum()
        losing_trades = (group['profit'] < 0).sum()
        
        win_ratio = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = group['profit'].sum()
        avg_profit = group['profit'].mean()
        
        avg_win = group[group['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = group[group['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
        
        monthly_stats.append({
            'Month': str(month),
            'Total_Trades': total_trades,
            'Winning_Trades': winning_trades,
            'Losing_Trades': losing_trades,
            'Win_Ratio_%': win_ratio,
            'Total_Profit': total_profit,
            'Avg_Profit': avg_profit,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss
        })
    
    return pd.DataFrame(monthly_stats)


def plot_monthly_metrics(df_monthly):
    """Genera gráficos de las métricas mensuales"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('📊 Monthly Trading Metrics', fontsize=16, fontweight='bold')
    
    months = df_monthly['Month'].values
    x_pos = np.arange(len(months))
    
    # 1. Profit Mensual (barras verdes/rojas)
    profits = df_monthly['Total_Profit'].values
    colors = ['green' if p > 0 else 'red' for p in profits]
    
    axes[0, 0].bar(x_pos, profits, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Total Profit por Mes')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Profit')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(months, rotation=45, ha='right')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Win Ratio Mensual (línea)
    axes[0, 1].plot(x_pos, df_monthly['Win_Ratio_%'], marker='o', 
                    color='blue', linewidth=2, markersize=8)
    axes[0, 1].fill_between(x_pos, df_monthly['Win_Ratio_%'], alpha=0.3, color='blue')
    axes[0, 1].set_title('Win Ratio % por Mes')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Win Ratio %')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(months, rotation=45, ha='right')
    axes[0, 1].axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # 3. Número de Trades por Mes
    axes[1, 0].bar(x_pos, df_monthly['Total_Trades'], color='steelblue', 
                   alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Número de Trades por Mes')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Trades')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(months, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Avg Win vs Avg Loss
    x_categories = np.arange(len(months))
    width = 0.35
    
    axes[1, 1].bar(x_categories - width/2, df_monthly['Avg_Win'], width, 
                   label='Avg Win', color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].bar(x_categories + width/2, df_monthly['Avg_Loss'], width, 
                   label='Avg Loss', color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Avg Win vs Avg Loss por Mes')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Profit')
    axes[1, 1].set_xticks(x_categories)
    axes[1, 1].set_xticklabels(months, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.show()


def print_summary(df_monthly):
    """Imprime resumen de métricas mensuales"""
    
    print("\n" + "="*80)
    print("📈 MONTHLY TRADING METRICS")
    print("="*80)
    
    # Formatear DataFrame para mejor visualización
    df_display = df_monthly.copy()
    df_display['Win_Ratio_%'] = df_display['Win_Ratio_%'].apply(lambda x: f"{x:.2f}")
    df_display['Total_Profit'] = df_display['Total_Profit'].apply(lambda x: f"{x:,.2f}")
    df_display['Avg_Profit'] = df_display['Avg_Profit'].apply(lambda x: f"{x:.2f}")
    df_display['Avg_Win'] = df_display['Avg_Win'].apply(lambda x: f"{x:.2f}")
    df_display['Avg_Loss'] = df_display['Avg_Loss'].apply(lambda x: f"{x:.2f}")
    
    print(df_display.to_string(index=False))
    
    print("\n" + "-"*80)
    print("📊 OVERALL STATISTICS")
    print("-"*80)
    
    total_trades = df_monthly['Total_Trades'].sum()
    total_winning = df_monthly['Winning_Trades'].sum()
    total_losing = df_monthly['Losing_Trades'].sum()
    overall_win_ratio = (total_winning / total_trades * 100) if total_trades > 0 else 0
    
    total_profit = df_monthly['Total_Profit'].sum()
    avg_monthly_profit = df_monthly['Total_Profit'].mean()
    
    winning_months = (df_monthly['Total_Profit'] > 0).sum()
    losing_months = (df_monthly['Total_Profit'] < 0).sum()
    monthly_win_ratio = (winning_months / len(df_monthly) * 100)
    
    print(f"Total Trades          : {total_trades:,}")
    print(f"Winning Trades        : {total_winning:,}")
    print(f"Losing Trades         : {total_losing:,}")
    print(f"Overall Win Ratio     : {overall_win_ratio:.2f}%")
    print(f"\nTotal Profit          : {total_profit:,.2f}")
    print(f"Avg Monthly Profit    : {avg_monthly_profit:,.2f}")
    print(f"\nWinning Months        : {winning_months} / {len(df_monthly)}")
    print(f"Losing Months         : {losing_months} / {len(df_monthly)}")
    print(f"Monthly Win Ratio     : {monthly_win_ratio:.2f}%")
    
    best_month = df_monthly.loc[df_monthly['Total_Profit'].idxmax()]
    worst_month = df_monthly.loc[df_monthly['Total_Profit'].idxmin()]
    
    print(f"\nBest Month            : {best_month['Month']} ({best_month['Total_Profit']:,.2f})")
    print(f"Worst Month           : {worst_month['Month']} ({worst_month['Total_Profit']:,.2f})")
    print("="*80 + "\n")


def main():
    """Función principal"""
    
    # Definir ruta de la carpeta (relativa al script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trades_folder = os.path.join(script_dir, '..', 'brief_trades')
    
    try:
        # Encontrar y cargar archivo más reciente
        latest_file = find_latest_file(trades_folder)
        df_trades = load_trades(latest_file)
        
        # Calcular métricas mensuales
        df_monthly = calculate_monthly_metrics(df_trades)
        
        # Mostrar resumen
        print_summary(df_monthly)
        
        # Mostrar gráficos
        plot_monthly_metrics(df_monthly)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()