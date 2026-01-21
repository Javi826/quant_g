"""
sentiment_layer/trade_enricher_combined.py

Enriches trades with BOTH:
1. BTC regime metrics (hurst, efficiency_ratio, atr_pct, etc.)
2. Sentiment metrics (fear_greed_norm, sentiment_state)

Generates combined states: regime_sentiment (e.g., trending_uptrend_greed)

Usage:
    python trade_enricher_combined.py
    
    Or import:
    from sentiment_layer.trade_enricher_combined import enrich_all_trades
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
    TRADES_FOLDER, TRADES_PATTERN, SENTIMENT_FOLDER, INITIAL_CAPITAL
)

# Import from market_regime
from market_regime.config import (
    OHLC_FOLDER, BTC_SYMBOL, LOOKBACK_BARS,
    HURST_WINDOW, ER_WINDOW, ATR_WINDOW, PE_WINDOW, PE_ORDER,
    FAMILIES, DIRECTION_METHOD, DIRECTION_MA_PERIOD,
    DIRECTION_MA_FAST, DIRECTION_MA_SLOW
)
from market_regime.regime_metrics import calc_all_metrics


# Output folder for combined enriched trades
OUTPUT_FOLDER_COMBINED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'sentiment_layer',
    'output_combined'
)


# Caches
_btc_cache = {}
_sentiment_cache = {}


def extract_timeframe(filename: str) -> str:
    """Extracts timeframe from trades filename."""
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
    """Loads BTC OHLC data from parquet file (with caching)."""
    cache_key = f"{symbol}_{timeframe}"
    
    if cache_key in _btc_cache:
        return _btc_cache[cache_key]
    
    filepath = Path(ohlc_folder) / f"{symbol}_{timeframe}.parquet"
    
    if not filepath.exists():
        raise FileNotFoundError(f"BTC OHLC not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    
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
    
    # Ensure timezone-aware for comparison with trades
    if df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize('UTC')
    
    _btc_cache[cache_key] = df
    return df


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
    
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError(f"Sentiment file must have 'timestamp' column")
    
    df = df.sort_values('ts').reset_index(drop=True)
    
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
    
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    elif 'buy time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy time'])
    else:
        raise ValueError("Trades file must have 'buy_time' column")
    
    return df


def classify_sentiment(fear_greed_norm: float) -> str:
    if np.isnan(fear_greed_norm):
        return 'unknown'
    
    if fear_greed_norm < 0.45:
        return 'fear'
    elif fear_greed_norm <= 0.55:
        return 'neutral'
    else:
        return 'greed'


def classify_family(row: pd.Series, families: dict) -> str:
    """Classifies a trade into a family based on its metrics."""
    for family_name, rules in families.items():
        if not rules:
            continue
        match = True
        for metric, (op, val) in rules.items():
            if metric not in row or pd.isna(row[metric]):
                match = False
                break
            if op == '>' and not (row[metric] > val):
                match = False
                break
            elif op == '<' and not (row[metric] < val):
                match = False
                break
        if match:
            return family_name
    for family_name, rules in families.items():
        if not rules:
            return family_name
    return 'unknown'


def get_btc_metrics_at_time(btc_df: pd.DataFrame, buy_time: pd.Timestamp, lookback: int) -> dict:
    """Gets BTC metrics BEFORE buy_time using only CLOSED candles."""
    closed_candles = btc_df[btc_df['ts'] < buy_time]
    
    if len(closed_candles) == 0:
        return None
    
    idx = closed_candles.index[-1]
    start_idx = max(0, idx - lookback + 1)
    
    if idx - start_idx < 20:
        return None
    
    subset = btc_df.iloc[start_idx:idx + 1]
    
    ohlc = {
        'open': subset['open'].values.astype(np.float64),
        'high': subset['high'].values.astype(np.float64),
        'low': subset['low'].values.astype(np.float64),
        'close': subset['close'].values.astype(np.float64)
    }
    
    metrics = calc_all_metrics(
        ohlc,
        hurst_window=HURST_WINDOW,
        er_window=ER_WINDOW,
        atr_window=ATR_WINDOW,
        pe_window=PE_WINDOW,
        pe_order=PE_ORDER
    )
    
    current_close = float(btc_df.iloc[idx]['close'])
    
    # MA_20
    if idx >= 19:
        ma_20_data = btc_df.iloc[idx - 19:idx + 1]['close'].values
        metrics['ma_20'] = float(np.mean(ma_20_data))
    else:
        metrics['ma_20'] = np.nan
    
    # MA_50
    if idx >= 49:
        ma_50_data = btc_df.iloc[idx - 49:idx + 1]['close'].values
        metrics['ma_50'] = float(np.mean(ma_50_data))
    else:
        metrics['ma_50'] = np.nan
    
    # MA_200
    if idx >= 199:
        ma_200_data = btc_df.iloc[idx - 199:idx + 1]['close'].values
        metrics['ma_200'] = float(np.mean(ma_200_data))
    else:
        metrics['ma_200'] = np.nan
    
    # Price vs MA ratios
    metrics['price_vs_ma_20'] = current_close / metrics['ma_20'] if not np.isnan(metrics['ma_20']) else np.nan
    metrics['price_vs_ma_50'] = current_close / metrics['ma_50'] if not np.isnan(metrics['ma_50']) else np.nan
    metrics['price_vs_ma_200'] = current_close / metrics['ma_200'] if not np.isnan(metrics['ma_200']) else np.nan
    
    # MA cross ratios
    if not np.isnan(metrics['ma_50']) and not np.isnan(metrics['ma_200']):
        metrics['ma_50_vs_ma_200'] = metrics['ma_50'] / metrics['ma_200']
    else:
        metrics['ma_50_vs_ma_200'] = np.nan
    
    if not np.isnan(metrics['ma_20']) and not np.isnan(metrics['ma_50']):
        metrics['ma_20_vs_ma_50'] = metrics['ma_20'] / metrics['ma_50']
    else:
        metrics['ma_20_vs_ma_50'] = np.nan
    
    return metrics


def get_sentiment_at_time(sentiment_df: pd.DataFrame, buy_time: pd.Timestamp) -> dict:
    """Gets sentiment metrics BEFORE buy_time using only CLOSED candles."""
    closed_candles = sentiment_df[sentiment_df['ts'] < buy_time]
    
    if len(closed_candles) == 0:
        return None
    
    idx = closed_candles.index[-1]
    fear_greed_norm = float(sentiment_df.iloc[idx]['fear_greed_norm'])
    sentiment_state = classify_sentiment(fear_greed_norm)
    
    return {
        'fear_greed_norm': fear_greed_norm,
        'sentiment_state': sentiment_state
    }


def determine_trend(row: pd.Series, direction_method: str) -> str:
    """Determines trend based on configured method."""
    if direction_method == 'price_vs_ma':
        price_vs_ma_col = f'price_vs_ma_{DIRECTION_MA_PERIOD}'
        if not pd.isna(row.get(price_vs_ma_col)):
            return 'uptrend' if row[price_vs_ma_col] > 1.0 else 'downtrend'
    
    elif direction_method == 'ma_cross':
        ma_cross_col = f'ma_{DIRECTION_MA_FAST}_vs_ma_{DIRECTION_MA_SLOW}'
        if not pd.isna(row.get(ma_cross_col)):
            return 'uptrend' if row[ma_cross_col] > 1.0 else 'downtrend'
    
    else:
        # Fallback
        if not pd.isna(row.get('price_vs_ma_50')):
            return 'uptrend' if row['price_vs_ma_50'] > 1.0 else 'downtrend'
    
    return 'unknown'


def enrich_single_file(
    trades_file: str,
    ohlc_folder: str,
    sentiment_folder: str,
    output_folder: str
) -> dict:
    """
    Enriches a single trades file with BOTH BTC regime and sentiment metrics.
    """
    filename = Path(trades_file).name
    strategy_name = Path(trades_file).stem.replace('all_trades_', '')
    
    timeframe = extract_timeframe(filename)
    
    print(f"\n    Processing: {strategy_name}")
    print(f"        Timeframe: {timeframe}")
    
    # Load BTC
    btc_df = load_btc_ohlc(ohlc_folder, BTC_SYMBOL, timeframe)
    print(f"        BTC data: {len(btc_df)} bars ({btc_df['ts'].min()} → {btc_df['ts'].max()})")
    
    # Load sentiment
    sentiment_df = load_sentiment_data(sentiment_folder, timeframe)
    print(f"        Sentiment data: {len(sentiment_df)} bars ({sentiment_df['ts'].min()} → {sentiment_df['ts'].max()})")
    
    # Load trades
    trades_df = load_trades(trades_file)
    num_trades = len(trades_df)
    print(f"        Trades: {num_trades}")
    
    # Ensure buy_time is timezone-aware
    if trades_df['buy_time'].dt.tz is None:
        trades_df['buy_time'] = trades_df['buy_time'].dt.tz_localize('UTC')
    
    # Initialize BTC metric columns
    btc_metric_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy',
                       'ma_20', 'ma_50', 'ma_200',
                       'price_vs_ma_20', 'price_vs_ma_50', 'price_vs_ma_200',
                       'ma_50_vs_ma_200', 'ma_20_vs_ma_50']
    for col in btc_metric_cols:
        trades_df[col] = np.nan
    
    # Initialize sentiment columns
    trades_df['fear_greed_norm'] = np.nan
    trades_df['sentiment_state'] = 'unknown'
    
    # Process each trade
    success_btc = 0
    success_sentiment = 0
    errors = 0
    
    for idx, row in trades_df.iterrows():
        buy_time = row['buy_time']
        
        # Get BTC metrics
        btc_metrics = get_btc_metrics_at_time(btc_df, buy_time, LOOKBACK_BARS)
        if btc_metrics is not None:
            for col in btc_metric_cols:
                trades_df.at[idx, col] = btc_metrics[col]
            success_btc += 1
        
        # Get sentiment metrics
        sentiment_data = get_sentiment_at_time(sentiment_df, buy_time)
        if sentiment_data is not None:
            trades_df.at[idx, 'fear_greed_norm'] = sentiment_data['fear_greed_norm']
            trades_df.at[idx, 'sentiment_state'] = sentiment_data['sentiment_state']
            success_sentiment += 1
        
        if btc_metrics is None or sentiment_data is None:
            errors += 1
    
    # Drop rows with incomplete data
    trades_before_drop = len(trades_df)
    
    critical_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy',
                     'ma_50', 'price_vs_ma_50', 'fear_greed_norm']
    
    trades_df = trades_df.dropna(subset=critical_cols).reset_index(drop=True)
    trades_df = trades_df[trades_df['sentiment_state'] != 'unknown'].reset_index(drop=True)
    
    trades_after_drop = len(trades_df)
    dropped_rows = trades_before_drop - trades_after_drop
    
    if dropped_rows > 0:
        print(f"        ⚠️  Dropped {dropped_rows} rows with incomplete data")
    
    print(f"        Enriched: BTC={success_btc}/{num_trades}, Sentiment={success_sentiment}/{num_trades}")
    print(f"        Final trades: {trades_after_drop}")
    
    # Classify family
    trades_df['family'] = trades_df.apply(lambda row: classify_family(row, FAMILIES), axis=1)
    
    # Determine trend
    trades_df['trend'] = trades_df.apply(lambda row: determine_trend(row, DIRECTION_METHOD), axis=1)
    
    # Create regime
    trades_df['regime'] = trades_df['family'] + '_' + trades_df['trend']
    
    # Create combined state
    trades_df['combined_state'] = trades_df['regime'] + '_' + trades_df['sentiment_state']
    
    # Add metadata
    trades_df['btc_symbol'] = BTC_SYMBOL
    trades_df['timeframe'] = timeframe
    trades_df['lookback_bars'] = LOOKBACK_BARS
    
    # Convert buy_time back to timezone-naive for Excel
    trades_df['buy_time'] = trades_df['buy_time'].dt.tz_localize(None)
    
    # Save
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_folder, f"trades_combined_{strategy_name}.xlsx")
    trades_df.to_excel(output_file, index=False)
    print(f"        Saved: {output_file}")
    
    return {
        'strategy': strategy_name,
        'timeframe': timeframe,
        'num_trades': num_trades,
        'num_trades_final': trades_after_drop,
        'dropped': dropped_rows,
        'success_btc': success_btc,
        'success_sentiment': success_sentiment,
        'errors': errors,
        'output_file': output_file
    }


def enrich_all_trades(
    trades_folder: str = None,
    trades_pattern: str = None,
    ohlc_folder: str = None,
    sentiment_folder: str = None,
    output_folder: str = None
) -> list:
    """
    Enriches all trades files with BOTH BTC regime and sentiment metrics.
    """
    # Clear caches
    global _btc_cache, _sentiment_cache
    _btc_cache = {}
    _sentiment_cache = {}
    
    # Use defaults
    trades_folder = trades_folder or TRADES_FOLDER
    trades_pattern = trades_pattern or TRADES_PATTERN
    ohlc_folder = ohlc_folder or OHLC_FOLDER
    sentiment_folder = sentiment_folder or SENTIMENT_FOLDER
    output_folder = output_folder or OUTPUT_FOLDER_COMBINED
    
    print("=" * 70)
    print("COMBINED TRADE ENRICHER - BTC Regime + Sentiment")
    print("=" * 70)
    
    # Find files
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
    
    # Process files
    print("\n[2] ENRICHING TRADES")
    print("-" * 70)
    
    results = []
    for f in files:
        result = enrich_single_file(f, ohlc_folder, sentiment_folder, output_folder)
        results.append(result)
    
    # Summary
    print("\n[3] SUMMARY")
    print("-" * 70)
    
    total_trades = sum(r['num_trades'] for r in results)
    total_success_btc = sum(r['success_btc'] for r in results)
    total_success_sentiment = sum(r['success_sentiment'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    
    print(f"\n    {'STRATEGY':<35} {'TF':>4} {'TRADES':>8} {'BTC_OK':>8} {'SENT_OK':>8} {'ERRORS':>8}")
    print("    " + "-" * 70)
    
    for r in results:
        status = "✅" if r['errors'] == 0 else "⚠️"
        print(f"    {r['strategy']:<35} {r['timeframe']:>4} {r['num_trades']:>8} {r['success_btc']:>8} {r['success_sentiment']:>8} {r['errors']:>8} {status}")
    
    print("    " + "-" * 70)
    print(f"    {'TOTAL':<35} {'':<4} {total_trades:>8} {total_success_btc:>8} {total_success_sentiment:>8} {total_errors:>8}")
    
    print(f"\n    BTC files loaded: {list(_btc_cache.keys())}")
    print(f"    Sentiment files loaded: {list(_sentiment_cache.keys())}")
    print(f"    Output folder: {output_folder}")
    
    print("\n" + "=" * 70)
    print("COMBINED ENRICHMENT COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    enrich_all_trades()