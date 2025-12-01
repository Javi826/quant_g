import os
import pandas as pd


# Colores ANSI
RED_BOLD     = "\033[1;91m"
GREEN_BOLD   = "\033[1;92m"
MAGENTA_BOLD = "\033[1;94m"
BLUE_BOLD    = "\033[1;94m"
YELLOW_BOLD  = "\033[0;93m"
RESET        = "\033[0m"

# Configuration
INITIAL_CAPITAL = 2070
FILE_NAME       = os.path.join("bot_files", "bot_trading_trades.xlsx") 

# Read Excel file
df = pd.read_excel(FILE_NAME)

# Calculate duration in days
df['OPEN_AT']  = pd.to_datetime(df['OPEN_AT'])
df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
df['DURATION'] = (df['CLOSE_AT'] - df['OPEN_AT']).dt.total_seconds() / 86400  # in days

print("\n")
print(f"{BLUE_BOLD}{'=' * 115}{RESET}")
print(f"{BLUE_BOLD}📈 STRATEGY ANALYSIS{RESET}")
print(f"{BLUE_BOLD}{'=' * 115}{RESET}")

# List to store results per strategy
results = []

# Calculate capital per strategy
num_strategies = df['STRATEGY'].nunique()
capital_per_strategy = INITIAL_CAPITAL / num_strategies

# Analysis per strategy
for strategy in df['STRATEGY'].unique():
    df_strategy = df[df['STRATEGY'] == strategy]
    
    num_trades      = len(df_strategy)
    positive_trades = len(df_strategy[df_strategy['PROFIT'] > 0])
    pct_positive    = (positive_trades / num_trades * 100) if num_trades > 0 else 0
    total_profit    = df_strategy['PROFIT'].sum()
    profit_pct      = (total_profit / capital_per_strategy * 100) if capital_per_strategy > 0 else 0
    avg_duration    = round(df_strategy['DURATION'].mean(), 2)
    
    date_fo = df_strategy['OPEN_AT'].min()
    
    total_reasons = len(df_strategy)
    tp_count      = len(df_strategy[df_strategy['REASON_OUT'].str.contains('TP', na=False)])
    sl_count      = len(df_strategy[df_strategy['REASON_OUT'].str.contains('SL', na=False)])
    oom_count     = len(df_strategy[df_strategy['REASON_OUT'].str.contains('OUT_OF_MARGIN', na=False)])
    
    pct_tp  = (tp_count / total_reasons * 100) if total_reasons > 0 else 0
    pct_sl  = (sl_count / total_reasons * 100) if total_reasons > 0 else 0
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

col_widths = {
    'Strategy':15,
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

        # Pintar en amarillo la columna Strategy
        if col == 'Strategy':
            cell = f"{YELLOW_BOLD}{cell}{RESET}"

        # Pintar Total_profit verde/rojo
        if col == 'Total_profit':
            color = GREEN_BOLD if value >= 0 else RED_BOLD
            cell = f"{color}{cell}{RESET}"

        row_parts.append(cell)

    print("  ".join(row_parts))

print("\n" + f"{BLUE_BOLD}{'=' * 115}{RESET}")
print(f"{BLUE_BOLD}📊 TOTAL SUMMARY{RESET}")
print(f"{BLUE_BOLD}{'=' * 115}{RESET}")

num_trades_total      = len(df)
positive_trades_total = len(df[df['PROFIT'] > 0])
pct_positive_total    = (positive_trades_total / num_trades_total * 100) if num_trades_total > 0 else 0
total_profit_general  = df['PROFIT'].sum()
pct_profit            = (total_profit_general / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0
avg_duration_total    = df['DURATION'].mean()

print(f"{BLUE_BOLD}🧮 Trades_num   :{RESET} {num_trades_total}")
print(f"{BLUE_BOLD}⏱ Avg_duration :{RESET} {avg_duration_total:.1f} days")
print(f"{BLUE_BOLD}🎯 Trades_pct   :{RESET} {pct_positive_total:.2f} %")
print(f"{BLUE_BOLD}💱 Profit_pct   :{RESET} {pct_profit:.2f} %")
color = GREEN_BOLD if total_profit_general >= 0 else RED_BOLD
print(f"{'💵' if total_profit_general >= 0 else '⭕'} {BLUE_BOLD}TOTAL_profit :{RESET} {color}{total_profit_general:.2f} ${RESET}")

print(f"{BLUE_BOLD}{'=' * 115}{RESET}")
