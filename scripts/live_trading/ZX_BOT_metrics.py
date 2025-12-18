import os
import pandas as pd

#  ANSI
RED_BOLD     = "\033[1;91m"
GREEN_BOLD   = "\033[1;92m"
MAGENTA_BOLD = "\033[1;85m"
BLUE         = "\033[0;94m"
YELLOW_BOLD  = "\033[0;93m"
RESET        = "\033[0m"
def remove_bold(ansi_color):
    # Cambia \033[1;XXm → \033[0;XXm
    return ansi_color.replace("\033[1;", "\033[0;")

def bot_metrics(
    excel_file=None,
    initial_capital=3671,
    show_table=True,
    return_data=False,
    color_code=None  
):
    if excel_file is None:
        print(f"{YELLOW_BOLD}❌ Error: excel_file parameter is required{RESET}")
        return None  # ✅ Indentado correctamente
    # Buscar archivo en bot_files si no existe en ruta directa
    if not os.path.exists(excel_file):
        bot_files_path = os.path.join("bot_files", os.path.basename(excel_file))
        if os.path.exists(bot_files_path):
            excel_file = bot_files_path
        else:
            print(f"{YELLOW_BOLD}⚠️  File not found: {excel_file}{RESET}")
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
    
    # Calcular duración en días
    df['OPEN_AT']  = pd.to_datetime(df['OPEN_AT'])
    df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
    df['DURATION'] = (df['CLOSE_AT'] - df['OPEN_AT']).dt.total_seconds() / 86400
    
    if show_table:
        print()
        print(f"{color_code}{'=' * 120}{RESET}")  # ⬅️ Era BLUE_BOLD
        print(f"{color_code}📈 STRATEGY ANALYSIS{RESET}")
        print(f"{color_code}{'=' * 120}{RESET}")
    
    # Lista para resultados por estrtegia
    results = []
    
    # Capital por estrategia
    num_strategies = df['STRATEGY'].nunique()
    capital_per_strategy = initial_capital / num_strategies
    
    # Análisis por estrategia
    for strategy in df['STRATEGY'].unique():
        df_strategy = df[df['STRATEGY'] == strategy]
        
        num_trades      = len(df_strategy)
        positive_trades = len(df_strategy[df_strategy['PROFIT'] > 0])
        pct_positive    = (positive_trades / num_trades * 100) if num_trades > 0 else 0
        total_profit    = df_strategy['PROFIT'].sum()
        profit_pct      = (total_profit / capital_per_strategy * 100) if capital_per_strategy > 0 else 0
        avg_duration    = round(df_strategy['DURATION'].mean(), 2)
        date_fo         = df_strategy['OPEN_AT'].min()
        
        total_reasons = len(df_strategy)
        tp_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('TP', na=False)])
        sl_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('SL', na=False)])
        oom_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('OUT_OF_MARGIN', na=False)])
        
        pct_tp = (tp_count / total_reasons * 100) if total_reasons > 0 else 0
        pct_sl = (sl_count / total_reasons * 100) if total_reasons > 0 else 0
        pct_oom = (oom_count / total_reasons * 100) if total_reasons > 0 else 0
        
        results.append({
            'Strategy': strategy,
            'date_fo': date_fo.strftime('%Y-%m-%d'),
            'Trades_num': num_trades,
            'Trades_pct': round(pct_positive, 2),
            'Total_profit': round(total_profit, 2),
            'Profit_pct': round(profit_pct, 2),
            'TP_pct': round(pct_tp, 2),
            'SL_pct': round(pct_sl, 2),
            'OOM_pct': round(pct_oom, 2),
            'Avg_days': avg_duration
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Strategy').reset_index(drop=True)
    
    # Anchos de columna
    col_widths = {
        'Strategy': 18,
        'date_fo': 10,
        'Trades_num': 11,
        'Trades_pct': 11,
        'Total_profit': 12,
        'Profit_pct': 11,
        'TP_pct': 6,
        'SL_pct': 6,
        'OOM_pct': 7,
        'Avg_days': 8
    }
    
    # Mostrar tabla con prints
    if show_table:
        # Print header (MAGENTA)
        header_parts = []
        for col in df_results.columns:
            width = col_widths.get(col, 10)
            if col in ['Strategy', 'date_fo']:
                header_parts.append(f'{MAGENTA_BOLD}{col:<{width}}{RESET}')
            else:
                header_parts.append(f'{MAGENTA_BOLD}{col:>{width}}{RESET}')
        print('  '.join(header_parts))
        
        # Print rows
        for _, row in df_results.iterrows():
            row_parts = []
            for col in df_results.columns:
                width = col_widths.get(col, 10)
                value = row[col]
                
                formatted = f"{value}"
                
                if col in ['Strategy', 'date_fo']:
                    cell = f"{formatted:<{width}}"
                else:
                    cell = f"{formatted:>{width}}"
                
                # Pintar en color SIN negrita la columna Strategy
                if col == 'Strategy':
                    normal_color = remove_bold(color_code)
                    cell = f"{normal_color}{cell}{RESET}"

                
                # Pintar Total_profit verde/rojo
                if col == 'Total_profit':
                    color = GREEN_BOLD if value >= 0 else RED_BOLD
                    cell = f"{color}{cell}{RESET}"
                
                row_parts.append(cell)
            
            print("  ".join(row_parts))
    
    # Resumen total
    num_trades_total      = len(df)
    positive_trades_total = len(df[df['PROFIT'] > 0])
    pct_positive_total    = (positive_trades_total / num_trades_total * 100) if num_trades_total > 0 else 0
    total_profit_general  = df['PROFIT'].sum()
    pct_profit            = (total_profit_general / initial_capital * 100) if initial_capital > 0 else 0
    avg_duration_total    = df['DURATION'].mean()
    
    if show_table:
        print()
        print(f"{color_code}{'=' * 120}{RESET}")  
        print(f"{color_code}📊 TOTAL SUMMARY{RESET}")  
        print(f"{color_code}{'=' * 120}{RESET}") 
        
        normal_color = remove_bold(color_code)
        
        print(f"{normal_color}📊 Trades_num   :{RESET} {num_trades_total}")
        print(f"{normal_color}🕜 Avg_duration :{RESET} {avg_duration_total:.1f} days")
        print(f"{normal_color}🎯 Trades_pct   :{RESET} {pct_positive_total:.2f} %")
        print(f"{normal_color}📈 Profit_pct   :{RESET} {pct_profit:.2f} %")
        
        color = GREEN_BOLD if total_profit_general >= 0 else RED_BOLD
        print(f"{'💵' if total_profit_general >= 0 else '⭕'} {color_code}TOTAL_profit :{RESET} {color}{total_profit_general:.2f} ${RESET}")  # ⬅️ Era BLUE_BOLD
        
        print(f"{color_code}{'=' * 120}{RESET}")  
        print()
    
    # Preparar datos de retorno
    summary = {
        'num_trades': num_trades_total,
        'positive_trades': positive_trades_total,
        'pct_positive': round(pct_positive_total, 2),
        'total_profit': round(total_profit_general, 2),
        'profit_pct': round(pct_profit, 2),
        'avg_duration_days': round(avg_duration_total, 2),
        'initial_capital': initial_capital
    }
    
    if return_data:
        return summary, df_results
    return summary


if __name__ == '__main__':
    # Ejemplo de uso directo
    bot_metrics(excel_file='bot_trading_trades.xlsx',initial_capital=3671,show_table=True)