"""
sentiment_layer/trade_enricher_sentiment.py

Enriches trades with Fear & Greed sentiment at entry time.
Processes multiple files matching a glob pattern.
Auto-detects timeframe from filename and loads correct sentiment data.

Usage:
    python trade_enricher_sentiment.py
    
    Or import:
    from sentiment_layer.trade_enricher_sentiment import enrich_all_trades
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

from sentiment_layer.config import (
    TRADES_FOLDER, TRADES_PATTERN, SENTIMENT_FOLDER, OUTPUT_FOLDER,
    SENTIMENT_THRESHOLDS
)


# Cache for sentiment dataframes by timeframe
_sentiment_cache = {}


def extract_timeframe(filename: str) -> str:
    """
    Extracts timeframe from trades filename.
    
    Examples:
        all_trades_parity_long_4H_IS.xlsx → 4H
        all_trades_reversal_short_1H_OOS.xlsx → 1H
        all_trades_parity_long_6Hutc_IS.xlsx → 6Hutc
        all_trades_parity_long_4H.xlsx → 4H
    """
    # Remove extension and prefix
    name = Path(filename).stem.replace('all_trades_', '')
    
    # Split by underscore
    parts = name.split('_')
    
    # Remove IS/OOS if present at the end
    if parts[-1].upper() in ['IS', 'OOS']:
        parts = parts[:-1]
    
    # Timeframe should be last part (e.g., '4H', '1H', '6Hutc')
    if parts:
        timeframe = parts[-1]
        # Validate it looks like a timeframe (contains digit and H)
        if any(c.isdigit() for c in timeframe.upper()) and 'H' in timeframe.upper():
            return timeframe
    
    # Default fallback
    print(f"    ⚠️  Could not extract timeframe from '{filename}', defaulting to 4H")
    return '4H'


def load_sentiment_data(sentiment_folder: str, timeframe: str) -> pd.DataFrame:
    """Loads Fear & Greed sentiment data from parquet file (with caching)."""
    cache_key = f"fear_greed_{timeframe}"
    
    if cache_key in _sentiment_cache:
        return _sentiment_cache[cache_key]
    
    filepath = Path(sentiment_folder) / f"fear_greed_{timeframe}.parquet"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Sentiment data not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    
    # Ensure timestamp column
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError(f"Sentiment file must have 'timestamp' column")
    
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Validate fear_greed_norm column exists
    if 'fear_greed_norm' not in df.columns:
        raise ValueError(f"Sentiment file must have 'fear_greed_norm' column")
    
    _sentiment_cache[cache_key] = df
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


def classify_sentiment(fear_greed_norm: float) -> str:
    """
    Classifies sentiment state based on fear_greed_norm value.
    
    Args:
        fear_greed_norm: Normalized fear & greed index (0-1)
    
    Returns:
        Sentiment state: 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
    """
    if np.isnan(fear_greed_norm):
        return 'unknown'
    
    if fear_greed_norm <= 0.25:
        return 'extreme_fear'
    elif fear_greed_norm <= 0.45:
        return 'fear'
    elif fear_greed_norm <= 0.55:
        return 'neutral'
    elif fear_greed_norm <= 0.75:
        return 'greed'
    else:
        return 'extreme_greed'


def get_sentiment_at_time(sentiment_df: pd.DataFrame, buy_time: pd.Timestamp) -> dict:
    """
    Gets sentiment metrics BEFORE buy_time using only CLOSED candles.
    
    CRITICAL FIX:
    - buy_time represents entry at candle OPEN (e.g., 08:00)
    - At 08:00, the 08:00-12:00 candle is opening (NOT closed yet)
    - We must use data up to the PREVIOUS candle (04:00-08:00)
    
    NO LOOKAHEAD BIAS.
    """
    
    # Use ONLY candles that closed BEFORE buy_time
    closed_candles = sentiment_df[sentiment_df['ts'] < buy_time]
    
    if len(closed_candles) == 0:
        return None  # No historical data available
    
    # Take LAST closed candle (most recent complete data)
    idx = closed_candles.index[-1]
    
    # Get sentiment value
    fear_greed_norm = float(sentiment_df.iloc[idx]['fear_greed_norm'])
    
    # Classify sentiment state
    sentiment_state = classify_sentiment(fear_greed_norm)
    
    return {
        'fear_greed_norm': fear_greed_norm,
        'sentiment_state': sentiment_state
    }


def enrich_single_file(
    trades_file: str,
    sentiment_folder: str,
    output_folder: str
) -> dict:
    """
    Enriches a single trades file with sentiment metrics.
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
    
    # Load sentiment data for this timeframe
    sentiment_df = load_sentiment_data(sentiment_folder, timeframe)
    print(f"        Sentiment data: {len(sentiment_df)} bars ({sentiment_df['ts'].min()} → {sentiment_df['ts'].max()})")
    
    # Load trades
    trades_df = load_trades(trades_file)
    num_trades = len(trades_df)
    print(f"        Trades: {num_trades}")
    
    # Ensure buy_time is timezone-aware (UTC) for comparison with sentiment data
    if trades_df['buy_time'].dt.tz is None:
        trades_df['buy_time'] = trades_df['buy_time'].dt.tz_localize('UTC')
    
    # Validate date ranges
    trades_min = trades_df['buy_time'].min()
    trades_max = trades_df['buy_time'].max()
    sentiment_min = sentiment_df['ts'].min()
    sentiment_max = sentiment_df['ts'].max()
    
    if trades_min < sentiment_min or trades_max > sentiment_max:
        print(f"        ⚠️  Date range warning:")
        print(f"            Trades:    {trades_min} → {trades_max}")
        print(f"            Sentiment: {sentiment_min} → {sentiment_max}")
    
    # Initialize columns
    trades_df['fear_greed_norm'] = np.nan
    trades_df['sentiment_state'] = 'unknown'
    
    # Process each trade
    success = 0
    errors = 0
    
    for idx, row in trades_df.iterrows():
        buy_time = row['buy_time']
        sentiment_data = get_sentiment_at_time(sentiment_df, buy_time)
        
        if sentiment_data is not None:
            trades_df.at[idx, 'fear_greed_norm'] = sentiment_data['fear_greed_norm']
            trades_df.at[idx, 'sentiment_state'] = sentiment_data['sentiment_state']
            success += 1
        else:
            errors += 1
    
    # DROP ROWS WITH NaN IN CRITICAL COLUMNS
    trades_before_drop = len(trades_df)
    
    # Drop rows where sentiment data is missing
    critical_cols = ['fear_greed_norm']
    trades_df = trades_df.dropna(subset=critical_cols).reset_index(drop=True)
    
    # Drop rows with unknown sentiment state
    trades_df = trades_df[trades_df['sentiment_state'] != 'unknown'].reset_index(drop=True)
    
    trades_after_drop = len(trades_df)
    dropped_rows = trades_before_drop - trades_after_drop
    
    if dropped_rows > 0:
        print(f"        ⚠️  Dropped {dropped_rows} rows with incomplete sentiment data")
    
    print(f"        Enriched: {success}/{num_trades} ({errors} errors)")
    print(f"        Final trades: {trades_after_drop}")
    
    # Add metadata
    trades_df['timeframe'] = timeframe
    
    # Convert buy_time back to timezone-naive for Excel compatibility
    trades_df['buy_time'] = trades_df['buy_time'].dt.tz_localize(None)
    
    # Save
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_folder, f"trades_sentiment_{strategy_name}.xlsx")
    trades_df.to_excel(output_file, index=False)
    print(f"        Saved: {output_file}")
    
    return {
        'strategy': strategy_name,
        'timeframe': timeframe,
        'num_trades': num_trades,
        'num_trades_final': trades_after_drop,
        'dropped': dropped_rows,
        'success': success,
        'errors': errors,
        'output_file': output_file
    }


def enrich_all_trades(
    trades_folder: str = None,
    trades_pattern: str = None,
    sentiment_folder: str = None,
    output_folder: str = None
) -> list:
    """
    Enriches all trades files matching pattern with sentiment metrics.
    Auto-detects timeframe from each filename.
    
    Args:
        trades_folder: Folder with trades Excel files
        trades_pattern: Glob pattern to match files
        sentiment_folder: Path to folder with sentiment parquet files
        output_folder: Path to save enriched trades
    
    Returns:
        List of dicts with results per file
    """
    # Clear cache
    global _sentiment_cache
    _sentiment_cache = {}
    
    # Use defaults from config if not provided
    trades_folder = trades_folder or TRADES_FOLDER
    trades_pattern = trades_pattern or TRADES_PATTERN
    sentiment_folder = sentiment_folder or SENTIMENT_FOLDER
    output_folder = output_folder or OUTPUT_FOLDER
    
    print("=" * 70)
    print("TRADE ENRICHER - Adding Sentiment metrics")
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
    print("\n[2] ENRICHING TRADES WITH SENTIMENT")
    print("-" * 70)
    
    results = []
    for f in files:
        result = enrich_single_file(f, sentiment_folder, output_folder)
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
    
    # Show sentiment files loaded
    print(f"\n    Sentiment files loaded: {list(_sentiment_cache.keys())}")
    print(f"    Output folder: {output_folder}")
    
    print("\n" + "=" * 70)
    print("SENTIMENT ENRICHMENT COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    enrich_all_trades()