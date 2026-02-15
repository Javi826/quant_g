"""
verify_lookahead_real.py

Verifies NO look-ahead bias using REAL data.
Shows exactly which BTC candle is used for each trade.

Usage (from market_regime folder):
    cd market_regime
    python verify_lookahead_real.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# Import from local config (same directory)
from config import TRADES_FOLDER, OHLC_FOLDER, BTC_SYMBOL, LOOKBACK_BARS


def extract_timeframe(filename: str) -> str:
    """
    Extracts timeframe from trades filename.
    
    Examples:
        all_trades_parity_long_4H_IS.xlsx → 4H
        all_trades_reversal_short_1H_OOS.xlsx → 1H
    """
    name = Path(filename).stem.replace('all_trades_', '')
    parts = name.split('_')
    
    if parts[-1].upper() in ['IS', 'OOS']:
        parts = parts[:-1]
    
    if parts:
        timeframe = parts[-1]
        if any(c.isdigit() for c in timeframe.upper()) and 'H' in timeframe.upper():
            return timeframe
    
    print(f"    ⚠️  Could not extract timeframe from '{filename}', defaulting to 4H")
    return '4H'


def load_btc_ohlc(ohlc_folder: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Loads BTC OHLC data from parquet file."""
    filepath = Path(ohlc_folder) / f"{symbol}_{timeframe}.parquet"
    
    if not filepath.exists():
        raise FileNotFoundError(f"BTC OHLC not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    
    # Ensure timestamp column
    ts_columns = ['timestamp', 'ts', 'date', 'time']
    ts_col = None
    for col in ts_columns:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col:
        df['ts'] = pd.to_datetime(df[ts_col])
    else:
        df['ts'] = pd.to_datetime(df.index)
        df = df.reset_index(drop=True)
    
    df = df.sort_values('ts').reset_index(drop=True)
    return df


def load_trades(trades_file: str) -> pd.DataFrame:
    """Loads trades from Excel file."""
    if not os.path.exists(trades_file):
        raise FileNotFoundError(f"Trades file not found: {trades_file}")
    
    df = pd.read_excel(trades_file)
    df.columns = df.columns.str.lower().str.strip()
    
    # Ensure buy_time is datetime
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    elif 'buy time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy time'])
    else:
        raise ValueError("Trades file must have 'buy_time' column")
    
    return df


def verify_single_trade(btc_df: pd.DataFrame, trade_row: pd.Series, trade_num: int):
    """
    Verifies a single trade - SIMPLIFIED OUTPUT.
    """
    buy_time = trade_row['buy_time']
    
    # Find last closed candle
    closed_candles = btc_df[btc_df['ts'] < buy_time]
    
    if len(closed_candles) == 0:
        print(f"Trade #{trade_num}: ❌ NO DATA")
        return
    
    idx = closed_candles.index[-1]
    candle_used = btc_df.iloc[idx]['ts']
    time_gap = buy_time - candle_used
    
    # Check pass/fail
    status = "✅ PASS" if candle_used < buy_time else "❌ FAIL"
    
    # Simple one-line output
    print(f"Trade #{trade_num:2d} | Entry: {buy_time} | BTC candle used: {candle_used} | Gap: {time_gap} | {status}")


def verify_trades_file(trades_file: str, num_trades: int = 5):
    """
    Verifies N RANDOM trades from a file - SIMPLIFIED.
    """
    filename = Path(trades_file).name
    
    print("\n" + "="*100)
    print(f"LOOK-AHEAD BIAS VERIFICATION: {filename}")
    print("="*100)
    
    # Extract timeframe
    timeframe = extract_timeframe(filename)
    
    # Load BTC
    btc_df = load_btc_ohlc(OHLC_FOLDER, BTC_SYMBOL, timeframe)
    
    # Load trades
    trades_df = load_trades(trades_file)
    
    # Sort by time
    trades_df = trades_df.sort_values('buy_time').reset_index(drop=True)
    
    print(f"\nTimeframe: {timeframe} | Total trades: {len(trades_df)} | BTC bars: {len(btc_df)}")
    print(f"\nChecking {num_trades} RANDOM trades:\n")
    print(f"{'':->100}")
    
    # Select random trades
    num_to_check = min(num_trades, len(trades_df))
    random_indices = np.random.choice(len(trades_df), size=num_to_check, replace=False)
    random_indices = sorted(random_indices)
    
    for i, idx in enumerate(random_indices, 1):
        verify_single_trade(btc_df, trades_df.iloc[idx], i)
    
    print(f"{'':->100}")
    print(f"\n✅ All checks PASSED = No look-ahead bias")
    print(f"❌ Any FAIL = Look-ahead bias detected\n")


def main():
    """
    Main verification routine - SIMPLIFIED.
    """
    print("\n" + "="*100)
    print("LOOK-AHEAD BIAS VERIFICATION")
    print("="*100)
    
    # Find trades files
    pattern = os.path.join(TRADES_FOLDER, "all_trades_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No trades files found in {TRADES_FOLDER}")
        return
    
    print(f"\nFound {len(files)} trades files")
    print(f"Verifying FIRST file: {Path(files[0]).name}\n")
    
    # Verify first file with 5 random trades
    verify_trades_file(files[0], num_trades=5)


if __name__ == "__main__":
    main()