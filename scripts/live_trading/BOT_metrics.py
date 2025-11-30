import os
import pandas as pd

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
print("=" * 80)
print("📈 STRATEGY ANALYSIS")
print("=" * 80)

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
    
    # Get the date of the first order (earliest OPEN_AT)
    date_fo = df_strategy['OPEN_AT'].min()
    
    # Count closing reasons
    total_reasons = len(df_strategy)
    tp_count      = len(df_strategy[df_strategy['REASON_OUT'].str.contains('TP', na=False)])
    sl_count      = len(df_strategy[df_strategy['REASON_OUT'].str.contains('SL', na=False)])
    oom_count     = len(df_strategy[df_strategy['REASON_OUT'].str.contains('OUT_OF_MARGIN', na=False)])
    
    pct_tp  = (tp_count / total_reasons * 100) if total_reasons > 0 else 0
    pct_sl  = (sl_count / total_reasons * 100) if total_reasons > 0 else 0
    pct_oom = (oom_count / total_reasons * 100) if total_reasons > 0 else 0
    
    # Add results to the list
    results.append({
        'Strategy': strategy,
        'date_fo': date_fo.strftime('%Y-%m-%d'),
        'Trades_num': num_trades,
        'Trades_pct': round(pct_positive, 2),
        'Total Profit': round(total_profit, 2),
        'Profit_pct': round(profit_pct, 2),
        'TP_pct': round(pct_tp, 2),
        'SL_pct': round(pct_sl, 2),
        'OOM_pct': round(pct_oom, 2),
        'Avg_days': avg_duration
    })

# Create DataFrame with results
df_results = pd.DataFrame(results)

# Custom table printing with left-aligned headers for Strategy and date_fo
col_widths = {
    'Strategy':15,
    'date_fo': 10,
    'Trades_num': 11,
    'Trades_pct': 11,
    'Total Profit': 12,
    'Profit_pct': 11,
    'TP_pct': 6,
    'SL_pct': 6,
    'OOM_pct': 7,
    'Avg_days': 8
}

# Print header
header_parts = []
for col in df_results.columns:
    width = col_widths.get(col, 10)
    if col in ['Strategy', 'date_fo']:
        header_parts.append(f'{col:<{width}}')
    else:
        header_parts.append(f'{col:>{width}}')
print('  '.join(header_parts))

# Print rows
for _, row in df_results.iterrows():
    row_parts = []
    for col in df_results.columns:
        width = col_widths.get(col, 10)
        value = row[col]
        if col in ['Strategy', 'date_fo']:
            row_parts.append(f'{value:<{width}}')
        else:
            row_parts.append(f'{value:>{width}}')
    print('  '.join(row_parts))

print("\n" + "=" * 80)
print("📊 TOTAL SUMMARY")
print("=" * 80)

# Total analysis
num_trades_total      = len(df)
positive_trades_total = len(df[df['PROFIT'] > 0])
pct_positive_total    = (positive_trades_total / num_trades_total * 100) if num_trades_total > 0 else 0
total_profit_general  = df['PROFIT'].sum()
pct_profit            = (total_profit_general / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0
avg_duration_total    = df['DURATION'].mean()

print(f"🧮 Trades_num   : {num_trades_total}")
print(f"⏱ Avg_duration : {avg_duration_total:.1f} days")
print(f"🎯 Trades_pct   : {pct_positive_total:.2f} %")
print(f"💱 Profit_pct   : {pct_profit:.2f} %")
print(f"{'💵' if total_profit_general >= 0 else '⭕'} TOTAL_profit : {total_profit_general:.2f} $")
print("=" * 80)