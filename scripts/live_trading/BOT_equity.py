import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Colores ANSI
RED_BOLD     = "\033[1;91m"
GREEN_BOLD   = "\033[1;92m"
BLUE_BOLD    = "\033[1;94m"
YELLOW_BOLD  = "\033[0;93m"
RESET        = "\033[0m"


def equity_curve(
    excel_file='bot_trading_trades.xlsx',
    initial_capital=3671,
    show_plot=True,
    save_plot=False,
    output_file='equity_curve.png',
    return_data=False
):
    
    # Buscar archivo en bot_files si no existe en ruta directa
    if not os.path.exists(excel_file):
        bot_files_path = os.path.join("bot_files", os.path.basename(excel_file))
        if os.path.exists(bot_files_path):
            excel_file = bot_files_path
        else:
            print(f"{RED_BOLD}❌ File not found: {excel_file}{RESET}")
            return None
    
    # Leer Excel
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        print(f"{RED_BOLD}❌ Error reading Excel: {e}{RESET}")
        return None
    
    if df.empty:
        print(f"{YELLOW_BOLD}⚠️  No trades found in Excel file{RESET}")
        return None
    
    # Convertir fechas
    df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
    
    # Ordenar por fecha de cierre
    df = df.sort_values('CLOSE_AT').reset_index(drop=True)
    
    # Calcular profit acumulado
    df['cumulative_profit'] = df['PROFIT'].cumsum()
    
    # Calcular equity
    df['equity'] = initial_capital + df['cumulative_profit']
    
    # Crear DataFrame para la curva
    equity_df = pd.DataFrame({
        'date': df['CLOSE_AT'],
        'equity': df['equity'],
        'profit': df['PROFIT'],
        'cumulative_profit': df['cumulative_profit']
    })
    
    # Añadir punto inicial
    start_point = pd.DataFrame({
        'date': [df['CLOSE_AT'].min() - pd.Timedelta(days=1)],
        'equity': [initial_capital],
        'profit': [0],
        'cumulative_profit': [0]
    })
    equity_df = pd.concat([start_point, equity_df], ignore_index=True)
    
    # Mostrar estadísticas
    print()
    print(f"{BLUE_BOLD}{'=' * 70}{RESET}")
    print(f"{BLUE_BOLD}📈 EQUITY CURVE ANALYSIS{RESET}")
    print(f"{BLUE_BOLD}{'=' * 70}{RESET}")
    print(f"{BLUE_BOLD}💰 Initial Capital  :{RESET} ${initial_capital:,.2f}")
    print(f"{BLUE_BOLD}💵 Final Equity     :{RESET} ${equity_df['equity'].iloc[-1]:,.2f}")
    
    total_profit = equity_df['cumulative_profit'].iloc[-1]
    color = GREEN_BOLD if total_profit >= 0 else RED_BOLD
    print(f"{BLUE_BOLD}📊 Total P&L        :{RESET} {color}${total_profit:,.2f}{RESET}")
    print(f"{BLUE_BOLD}📈 Return %         :{RESET} {color}{(total_profit/initial_capital*100):.2f}%{RESET}")
    
    # Calcular máximo drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
    equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['peak']) * 100
    
    max_dd = equity_df['drawdown'].min()
    max_dd_pct = equity_df['drawdown_pct'].min()
    
    print(f"{BLUE_BOLD}📉 Max Drawdown     :{RESET} {RED_BOLD}${max_dd:,.2f} ({max_dd_pct:.2f}%){RESET}")
    print(f"{BLUE_BOLD}🔝 Peak Equity      :{RESET} ${equity_df['peak'].max():,.2f}")
    print(f"{BLUE_BOLD}📅 Start Date       :{RESET} {equity_df['date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"{BLUE_BOLD}📅 End Date         :{RESET} {equity_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"{BLUE_BOLD}{'=' * 70}{RESET}")
    print()
    
    # Crear gráfico
    if show_plot or save_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        fig.patch.set_facecolor('#d0d0d0')
        
        # Gráfico de equity
        ax1.plot(equity_df['date'], equity_df['equity'], 
                linewidth=2, color='#00008B', label='Equity')
        ax1.plot(equity_df['date'], equity_df['peak'], 
                linewidth=1, color='#06A77D', linestyle='--', 
                alpha=0.7, label='Peak Equity')
        ax1.axhline(y=initial_capital, color='gray', 
                   linestyle=':', alpha=0.5, label='Initial Capital')
        
        # Rellenar área positiva/negativa
        ax1.fill_between(equity_df['date'], initial_capital, equity_df['equity'],
                        where=(equity_df['equity'] >= initial_capital),
                        alpha=0.3, color='green', interpolate=True)
        ax1.fill_between(equity_df['date'], initial_capital, equity_df['equity'],
                        where=(equity_df['equity'] < initial_capital),
                        alpha=0.3, color='red', interpolate=True)
        
        ax1.set_title('Equity Curve', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico de drawdown
        ax2.fill_between(equity_df['date'], 0, equity_df['drawdown_pct'],
                        color='#D90429', alpha=0.6)
        ax2.plot(equity_df['date'], equity_df['drawdown_pct'],
                linewidth=1.5, color='#8B0000')
        
        ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim([min(equity_df['drawdown_pct'].min() * 1.1, -1), 1])
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"{GREEN_BOLD}✅ Plot saved as: {output_file}{RESET}\n")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Limpiar columnas auxiliares antes de retornar
    equity_df = equity_df[['date', 'equity', 'profit', 'cumulative_profit']]
    
    if return_data:
        return equity_df
    
    return None


if __name__ == '__main__':
    # Ejemplo de uso
    equity_data = equity_curve(
        excel_file='bot_trading_trades.xlsx',
        initial_capital=3671,
        show_plot=True,
        save_plot=True,
        output_file='equity_curve.png',
        return_data=True
    )
    
    if equity_data is not None:
        print("📋 Sample of equity data:")
        print(equity_data.head(10))
        print("\n...")
        print(equity_data.tail(5))