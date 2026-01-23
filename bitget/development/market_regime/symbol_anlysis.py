"""
symbol_analysis.py
Analyzes win rates and BTC correlations by symbol using out-of-sample 2025 data.
"""
import pandas as pd
import numpy as np
import glob
import os
from config import OUTPUT_FOLDER, OHLC_FOLDER, BTC_SYMBOL


def load_enriched_trades():
    """Load enriched trades from output folder."""
    pattern = os.path.join(OUTPUT_FOLDER, 'trades_enriched_*_OOS.xlsx')
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No enriched trades found in {OUTPUT_FOLDER}")
    
    print(f"Loading {len(files)} enriched trade files...\n")
    
    # Load and concatenate all files
    dfs = []
    for file in files:
        df_temp = pd.read_excel(file)
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter out NaN BTC metrics
    initial_count = len(df)
    df = df.dropna(subset=['hurst', 'efficiency_ratio', 'atr_pct', 'ma_20', 'ma_50', 'ma_200'])
    filtered_count = len(df)
    
    if filtered_count < initial_count:
        print(f"Filtered out {initial_count - filtered_count} trades with NaN BTC metrics")
        print(f"Remaining trades: {filtered_count}\n")
    
    return df


def calculate_win_rates(df):
    """Calculate win rate statistics by symbol."""
    results = []
    
    total_all_trades = len(df)
    total_all_profit = df['profit'].sum()
    
    for symbol in df['symbol'].unique():
        symbol_trades = df[df['symbol'] == symbol]
        
        total_trades = len(symbol_trades)
        wins = len(symbol_trades[symbol_trades['profit'] > 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_profit = symbol_trades['profit'].sum()
        
        pct_trades = (total_trades / total_all_trades * 100) if total_all_trades > 0 else 0
        pct_profit = (total_profit / total_all_profit * 100) if total_all_profit != 0 else 0
        
        results.append({
            'Symbol': symbol,
            'Trades': total_trades,
            '% Trades': pct_trades,
            'Wins': wins,
            'Total Profit': total_profit,
            '% Profit': pct_profit,
            'Win Rate %': win_rate
        })
    
    # Sort by win rate (ascending)
    results_df = pd.DataFrame(results).sort_values('Win Rate %')
    
    # Calculate average win rate
    avg_win_rate = results_df['Win Rate %'].mean()
    
    # Add indicator column
    results_df[''] = results_df['Win Rate %'].apply(
        lambda x: '✅' if x > avg_win_rate else '🟠'
    )
    
    # Add totals row
    totals_row = pd.DataFrame([{
        'Symbol': 'TOTAL/AVG',
        'Trades': results_df['Trades'].sum(),
        '% Trades': 100.0,
        'Wins': results_df['Wins'].sum(),
        'Total Profit': results_df['Total Profit'].sum(),
        '% Profit': 100.0,
        'Win Rate %': avg_win_rate,
        '': ''
    }])
    
    results_df = pd.concat([results_df, totals_row], ignore_index=True)
    
    return results_df


def load_ohlc(symbol, timeframe):
    """Load OHLC data for a symbol and timeframe."""
    filename = f"{symbol}_{timeframe}.parquet"
    filepath = os.path.join(OHLC_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_parquet(filepath)
    
    # Ensure datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    return df


def calculate_returns(ohlc_df):
    """Calculate daily returns from OHLC data."""
    if ohlc_df is None or len(ohlc_df) == 0:
        return None
    
    # Resample to daily and calculate returns
    daily = ohlc_df['close'].resample('1D').last()
    returns = daily.pct_change().dropna()
    
    return returns


def calculate_correlations(df):
    """Calculate correlations between each symbol and BTC."""
    symbols = df['symbol'].unique()
    timeframes = ['1H', '4H', '6Hutc']
    
    # Load BTC returns for each timeframe
    btc_returns = {}
    for tf in timeframes:
        btc_ohlc = load_ohlc(BTC_SYMBOL, tf)
        btc_returns[tf] = calculate_returns(btc_ohlc)
    
    results = []
    
    for symbol in symbols:
        if symbol == BTC_SYMBOL:
            continue  # Skip BTC itself
        
        corr_data = {'Symbol': symbol}
        correlations = []
        
        for tf in timeframes:
            # Load symbol returns
            symbol_ohlc = load_ohlc(symbol, tf)
            symbol_returns = calculate_returns(symbol_ohlc)
            
            # Calculate correlation
            if symbol_returns is not None and btc_returns[tf] is not None:
                # Align dates
                common_dates = symbol_returns.index.intersection(btc_returns[tf].index)
                
                if len(common_dates) > 30:  # Minimum 30 days for correlation
                    corr = symbol_returns.loc[common_dates].corr(btc_returns[tf].loc[common_dates])
                    corr_data[f'Corr {tf}'] = corr
                    correlations.append(corr)
                else:
                    corr_data[f'Corr {tf}'] = None
            else:
                corr_data[f'Corr {tf}'] = None
        
        # Calculate average correlation
        valid_corrs = [c for c in correlations if c is not None and not np.isnan(c)]
        corr_avg = np.mean(valid_corrs) if valid_corrs else None
        corr_data['Corr Avg'] = corr_avg
        
        results.append(corr_data)
    
    results_df = pd.DataFrame(results)
    
    # Sort by Corr Avg (ascending)
    results_df = results_df.sort_values('Corr Avg')
    
    # Format correlation columns BEFORE adding summary rows
    corr_cols = ['Corr 1H', 'Corr 4H', 'Corr 6Hutc', 'Corr Avg']
    for col in corr_cols:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    # Add indicator column based on original Corr Avg values (before formatting)
    # Need to get original values back temporarily
    corr_avg_numeric = pd.to_numeric(results_df['Corr Avg'], errors='coerce')
    results_df[''] = corr_avg_numeric.apply(
        lambda x: '✅' if pd.notna(x) and x > 0.7 else '🟠'
    )
    
    # Calculate % trades and % profit for correlated vs non-correlated
    total_trades = len(df)
    total_profit = df['profit'].sum()
    
    # Non-correlated symbols (≤0.7)
    non_corr_symbols = results_df[corr_avg_numeric <= 0.7]['Symbol'].tolist()
    non_corr_trades = len(df[df['symbol'].isin(non_corr_symbols)])
    non_corr_profit = df[df['symbol'].isin(non_corr_symbols)]['profit'].sum()
    
    pct_trades_non_corr = (non_corr_trades / total_trades * 100) if total_trades > 0 else 0
    pct_profit_non_corr = (non_corr_profit / total_profit * 100) if total_profit != 0 else 0
    
    # Correlated symbols (>0.7)
    corr_symbols = results_df[corr_avg_numeric > 0.7]['Symbol'].tolist()
    corr_trades = len(df[df['symbol'].isin(corr_symbols)])
    corr_profit = df[df['symbol'].isin(corr_symbols)]['profit'].sum()
    
    pct_trades_corr = (corr_trades / total_trades * 100) if total_trades > 0 else 0
    pct_profit_corr = (corr_profit / total_profit * 100) if total_profit != 0 else 0
    
    # Add summary rows
    summary_rows = pd.DataFrame([
        {
            'Symbol': '─' * 50,
            'Corr 1H': '',
            'Corr 4H': '',
            'Corr 6Hutc': '',
            'Corr Avg': '',
            '': ''
        },
        {
            'Symbol': f'NON-CORRELATED (≤0.7): {pct_trades_non_corr:.1f}% trades, {pct_profit_non_corr:.1f}% profit',
            'Corr 1H': '',
            'Corr 4H': '',
            'Corr 6Hutc': '',
            'Corr Avg': '',
            '': '🟠'
        },
        {
            'Symbol': f'CORRELATED (>0.7): {pct_trades_corr:.1f}% trades, {pct_profit_corr:.1f}% profit',
            'Corr 1H': '',
            'Corr 4H': '',
            'Corr 6Hutc': '',
            'Corr Avg': '',
            '': '✅'
        }
    ])
    
    results_df = pd.concat([results_df, summary_rows], ignore_index=True)
    
    return results_df


def main():
    print("=" * 80)
    print("SYMBOL ANALYSIS - OUT OF SAMPLE 2025")
    print("=" * 80)
    print()
    
    # Load trades
    df = load_enriched_trades()
    
    # Calculate win rates
    print("=" * 80)
    print("WIN RATE BY SYMBOL (sorted by win rate ascending)")
    print("=" * 80)
    win_rates = calculate_win_rates(df)
    print(win_rates.to_string(index=False))
    print()
    
    # Calculate correlations
    print("=" * 80)
    print("CORRELATION WITH BTC BY SYMBOL AND TIMEFRAME")
    print("=" * 80)
    correlations = calculate_correlations(df)
    
    print(correlations.to_string(index=False))
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()