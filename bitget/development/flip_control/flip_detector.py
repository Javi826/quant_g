"""
flip_control/flip_detector.py

Detects regime flips in BTC based on MA50 crossovers.
Supports confirmation filters: bars and distance thresholds.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path


def load_btc_ohlc(ohlc_folder: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Loads BTC OHLC data from parquet file.
    
    Args:
        ohlc_folder: Path to folder with OHLC parquet files
        symbol: Symbol name (e.g., 'BTCUSDT')
        timeframe: Timeframe (e.g., '1H', '4H', '6Hutc')
    
    Returns:
        DataFrame with columns: ts, open, high, low, close
    """
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


def calculate_ma(close: pd.Series, period: int) -> pd.Series:
    """
    Calculates simple moving average.
    
    Args:
        close: Close prices series
        period: MA period
    
    Returns:
        Series with MA values
    """
    return close.rolling(window=period).mean()


def detect_flips(
    btc_df: pd.DataFrame,
    ma_period: int = 50,
    confirmation_bars: int = 0,
    distance_pct: float = 0.0
) -> List[Dict]:
    """
    Detects regime flips in BTC based on MA crossovers.
    
    Args:
        btc_df: BTC OHLC dataframe with columns: ts, close
        ma_period: Moving average period (default: 50)
        confirmation_bars: Number of consecutive bars needed (0 = immediate)
        distance_pct: Minimum distance % from MA to confirm (0.0 = disabled)
    
    Returns:
        List of flip events with format:
        [
            {
                'timestamp': pd.Timestamp,
                'flip_type': 'UP_TO_DOWN' or 'DOWN_TO_UP',
                'price': float,
                'ma_50': float,
                'distance_pct': float
            },
            ...
        ]
    """
    # Calculate MA50
    btc_df = btc_df.copy()
    btc_df['ma_50'] = calculate_ma(btc_df['close'], ma_period)
    
    # Drop rows without MA (first MA_PERIOD-1 rows)
    btc_df = btc_df.dropna(subset=['ma_50']).reset_index(drop=True)
    
    if len(btc_df) == 0:
        return []
    
    # Determine regime for each bar (1 = above MA, -1 = below MA)
    btc_df['regime'] = np.where(btc_df['close'] > btc_df['ma_50'], 1, -1)
    
    # Calculate distance from MA (%)
    btc_df['distance_pct'] = abs((btc_df['close'] - btc_df['ma_50']) / btc_df['ma_50'] * 100)
    
    # Detect regime changes
    btc_df['regime_change'] = btc_df['regime'].diff()
    
    flips = []
    
    for idx in btc_df[btc_df['regime_change'] != 0].index:
        if idx == 0:
            continue  # Skip first row (no previous regime)
        
        current_regime = btc_df.loc[idx, 'regime']
        previous_regime = btc_df.loc[idx - 1, 'regime']
        
        # Determine flip type
        if previous_regime == 1 and current_regime == -1:
            flip_type = 'UP_TO_DOWN'
        elif previous_regime == -1 and current_regime == 1:
            flip_type = 'DOWN_TO_UP'
        else:
            continue  # Should not happen
        
        # CONFIRMATION FILTER 1: Consecutive bars
        if confirmation_bars > 0:
            # Check if next N bars maintain the new regime
            end_idx = min(idx + confirmation_bars, len(btc_df))
            bars_in_new_regime = btc_df.loc[idx:end_idx - 1, 'regime']
            
            if len(bars_in_new_regime) < confirmation_bars:
                continue  # Not enough bars ahead
            
            if not all(bars_in_new_regime == current_regime):
                continue  # Not all bars confirm the flip
            
            # Use the last confirmation bar as flip timestamp
            flip_idx = end_idx - 1
        else:
            # Immediate flip
            flip_idx = idx
        
        # CONFIRMATION FILTER 2: Distance threshold
        if distance_pct > 0.0:
            flip_distance = btc_df.loc[flip_idx, 'distance_pct']
            if flip_distance < distance_pct:
                continue  # Not far enough from MA
        
        # Flip confirmed - record it
        flip_event = {
            'timestamp': btc_df.loc[flip_idx, 'ts'],
            'flip_type': flip_type,
            'price': float(btc_df.loc[flip_idx, 'close']),
            'ma_50': float(btc_df.loc[flip_idx, 'ma_50']),
            'distance_pct': float(btc_df.loc[flip_idx, 'distance_pct'])
        }
        
        flips.append(flip_event)
    
    return flips


def get_regime_at_time(btc_df: pd.DataFrame, timestamp: pd.Timestamp, ma_period: int = 50) -> str:
    """
    Returns regime (UP or DOWN) at a specific timestamp.
    
    Args:
        btc_df: BTC OHLC dataframe
        timestamp: Timestamp to check
        ma_period: MA period
    
    Returns:
        'UP' if price > MA, 'DOWN' if price < MA, 'UNKNOWN' if no data
    """
    # Get data up to (but not including) timestamp
    historical = btc_df[btc_df['ts'] < timestamp]
    
    if len(historical) < ma_period:
        return 'UNKNOWN'
    
    # Calculate MA at last available bar
    last_bar = historical.iloc[-1]
    ma = historical['close'].tail(ma_period).mean()
    
    if last_bar['close'] > ma:
        return 'UP'
    else:
        return 'DOWN'