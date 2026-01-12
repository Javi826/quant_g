"""
market_regime/trade_enricher.py

Enriches trades with BTC regime metrics at entry time.
Processes multiple files matching a glob pattern.
Auto-detects timeframe from filename and loads correct BTC OHLC.

Usage:
    python trade_enricher.py
    
    Or import:
    from market_regime.trade_enricher import enrich_all_trades
    results = enrich_all_trades()
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.config import (
    TRADES_FOLDER, TRADES_PATTERN, OHLC_FOLDER, OUTPUT_FOLDER,
    BTC_SYMBOL, LOOKBACK_BARS,
    HURST_WINDOW, ER_WINDOW, ATR_WINDOW, PE_WINDOW, PE_ORDER
)
from market_regime.regime_metrics import calc_all_metrics


# Cache for BTC dataframes by timeframe
_btc_cache = {}


def extract_timeframe(filename: str) -> str:
    """
    Extracts timeframe from trades filename.
    
    Examples:
        all_trades_parity_long_4H_IS.xlsx → 4H
        all_trades_reversal_short_1H_OOS.xlsx → 1H
        all_trades_parity_long_4H.xlsx → 4H
    """
    # Remove extension and prefix
    name = Path(filename).stem.replace('all_trades_', '')
    
    # Split by underscore
    parts = name.split('_')
    
    # Remove IS/OOS if present at the end
    if parts[-1].upper() in ['IS', 'OOS']:
        parts = parts[:-1]
    
    # Timeframe should be last part (e.g., '4H', '1H', '6H')
    if parts:
        timeframe = parts[-1].upper()
        # Validate it looks like a timeframe
        if any(c.isdigit() for c in timeframe) and timeframe.endswith('H'):
            return timeframe
    
    # Default fallback
    print(f"    ⚠️  Could not extract timeframe from '{filename}', defaulting to 4H")
    return '4H'


def load_btc_ohlc(ohlc_folder: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Loads BTC OHLC data from parquet file (with caching)."""
    cache_key = f"{symbol}_{timeframe}"
    
    if cache_key in _btc_cache:
        return _btc_cache[cache_key]
    
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
    
    _btc_cache[cache_key] = df
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


def get_metrics_at_time(btc_df: pd.DataFrame, buy_time: pd.Timestamp, lookback: int) -> dict:
    """
    Gets BTC metrics at a specific time.
    
    REQUIRES EXACT MATCH on timestamp.
    Uses lookback bars BEFORE buy_time for calculation.
    """
    # Find exact match
    exact_match = btc_df[btc_df['ts'] == buy_time]
    
    if len(exact_match) == 0:
        return None  # No match, will be handled by caller
    
    idx = exact_match.index[0]
    start_idx = max(0, idx - lookback + 1)
    
    if idx - start_idx < 20:
        return None  # Insufficient data
    
    subset = btc_df.iloc[start_idx:idx + 1]
    
    ohlc = {
        'open': subset['open'].values.astype(np.float64),
        'high': subset['high'].values.astype(np.float64),
        'low': subset['low'].values.astype(np.float64),
        'close': subset['close'].values.astype(np.float64)
    }
    
    return calc_all_metrics(
        ohlc,
        hurst_window=HURST_WINDOW,
        er_window=ER_WINDOW,
        atr_window=ATR_WINDOW,
        pe_window=PE_WINDOW,
        pe_order=PE_ORDER
    )


def enrich_single_file(
    trades_file: str,
    ohlc_folder: str,
    output_folder: str
) -> dict:
    """
    Enriches a single trades file with BTC regime metrics.
    Auto-detects timeframe from filename.
    
    Returns:
        Dict with summary info
    """
    filename = Path(trades_file).name
    strategy_name = Path(trades_file).stem.replace('all_trades_', '')
    
    # Extract timeframe from filename
    timeframe = extract_timeframe(filename)
    
    print(f"\n    Processing: {strategy_name}")
    print(f"        Timeframe: {timeframe}")
    
    # Load BTC for this timeframe
    btc_df = load_btc_ohlc(ohlc_folder, BTC_SYMBOL, timeframe)
    print(f"        BTC data: {len(btc_df)} bars ({btc_df['ts'].min()} → {btc_df['ts'].max()})")
    
    # Load trades
    trades_df = load_trades(trades_file)
    num_trades = len(trades_df)
    print(f"        Trades: {num_trades}")
    
    # Validate date ranges
    trades_min = trades_df['buy_time'].min()
    trades_max = trades_df['buy_time'].max()
    btc_min = btc_df['ts'].min()
    btc_max = btc_df['ts'].max()
    
    if trades_min < btc_min or trades_max > btc_max:
        print(f"        ⚠️  Date range warning:")
        print(f"            Trades: {trades_min} → {trades_max}")
        print(f"            BTC:    {btc_min} → {btc_max}")
    
    # Initialize columns
    metric_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
    for col in metric_cols:
        trades_df[col] = np.nan
    
    # Process each trade
    success = 0
    errors = 0
    
    for idx, row in trades_df.iterrows():
        buy_time = row['buy_time']
        metrics = get_metrics_at_time(btc_df, buy_time, LOOKBACK_BARS)
        
        if metrics is not None:
            for col in metric_cols:
                trades_df.at[idx, col] = metrics[col]
            success += 1
        else:
            errors += 1
    
    print(f"        Enriched: {success}/{num_trades} ({errors} errors)")
    
    # Add metadata
    trades_df['btc_symbol'] = BTC_SYMBOL
    trades_df['timeframe'] = timeframe
    trades_df['lookback_bars'] = LOOKBACK_BARS
    
    # Save
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_folder, f"trades_enriched_{strategy_name}.xlsx")
    trades_df.to_excel(output_file, index=False)
    print(f"        Saved: {output_file}")
    
    return {
        'strategy': strategy_name,
        'timeframe': timeframe,
        'num_trades': num_trades,
        'success': success,
        'errors': errors,
        'output_file': output_file
    }


def enrich_all_trades(
    trades_folder: str = None,
    trades_pattern: str = None,
    ohlc_folder: str = None,
    output_folder: str = None
) -> list:
    """
    Enriches all trades files matching pattern with BTC regime metrics.
    Auto-detects timeframe from each filename.
    
    Args:
        trades_folder: Folder with trades Excel files
        trades_pattern: Glob pattern to match files
        ohlc_folder: Path to folder with BTC OHLC parquet
        output_folder: Path to save enriched trades
    
    Returns:
        List of dicts with results per file
    """
    # Clear cache
    global _btc_cache
    _btc_cache = {}
    
    # Use defaults from config if not provided
    trades_folder = trades_folder or TRADES_FOLDER
    trades_pattern = trades_pattern or TRADES_PATTERN
    ohlc_folder = ohlc_folder or OHLC_FOLDER
    output_folder = output_folder or OUTPUT_FOLDER
    
    print("=" * 70)
    print("TRADE ENRICHER - Adding BTC regime metrics")
    print("=" * 70)
    
    # =================================================================
    # STEP 1: Find files to process
    # =================================================================
    print("\n[1] FINDING TRADES FILES")
    print("-" * 70)
    
    pattern_path = os.path.join(trades_folder, trades_pattern)
    files = sorted(glob(pattern_path))
    
    print(f"    Folder: {trades_folder}")
    print(f"    Pattern: {trades_pattern}")
    print(f"    Files found: {len(files)}")
    
    if not files:
        print(f"    ❌ No files found matching pattern")
        return []
    
    for f in files:
        tf = extract_timeframe(Path(f).name)
        print(f"        • {Path(f).name} [{tf}]")
    
    # =================================================================
    # STEP 2: Process each file
    # =================================================================
    print("\n[2] ENRICHING TRADES")
    print("-" * 70)
    
    results = []
    for f in files:
        result = enrich_single_file(f, ohlc_folder, output_folder)
        results.append(result)
    
    # =================================================================
    # STEP 3: Summary
    # =================================================================
    print("\n[3] SUMMARY")
    print("-" * 70)
    
    total_trades = sum(r['num_trades'] for r in results)
    total_success = sum(r['success'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    
    print(f"\n    {'STRATEGY':<35} {'TF':>4} {'TRADES':>8} {'OK':>8} {'ERRORS':>8}")
    print("    " + "-" * 70)
    
    for r in results:
        status = "✅" if r['errors'] == 0 else "⚠️"
        print(f"    {r['strategy']:<35} {r['timeframe']:>4} {r['num_trades']:>8} {r['success']:>8} {r['errors']:>8} {status}")
    
    print("    " + "-" * 70)
    print(f"    {'TOTAL':<35} {'':<4} {total_trades:>8} {total_success:>8} {total_errors:>8}")
    
    # Show BTC files loaded
    print(f"\n    BTC files loaded: {list(_btc_cache.keys())}")
    print(f"    Output folder: {output_folder}")
    
    print("\n" + "=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    enrich_all_trades()