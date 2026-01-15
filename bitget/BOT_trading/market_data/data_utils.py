"""
Market Data Utils - OHLCV data processing and symbol management.

This module provides utilities for:
- Loading and filtering trading symbols
- Fetching OHLCV data from API
- Normalizing DataFrames
- Converting DataFrames to numpy arrays
"""

import os
import logging
import numpy as np
import pandas as pd
from market_data.api_client import _call_history_candles, to_dataframe_from_api
from pandas.api.types import is_datetime64_any_dtype
from config.settings import API_LIMIT_DATA

logger = logging.getLogger('BOT_trading.market_data.data_utils')


def normalize_live_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV DataFrame for live trading.
    
    Ensures:
    - DatetimeIndex
    - Numeric columns (open, high, low, close, volume)
    
    Args:
        df: Raw OHLCV DataFrame
    
    Returns:
        Normalized DataFrame with DatetimeIndex
    """
    logger.debug(f"Normalizing OHLCV DataFrame with {len(df)} rows")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            df.index = pd.to_datetime(df.index)
    
    # Convert OHLCV columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume_base', 'volume_quote']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def df_to_arrays_live(df: pd.DataFrame) -> dict:
    """
    Convert OHLCV DataFrame to numpy arrays for signal detection.
    
    Args:
        df: Normalized OHLCV DataFrame
    
    Returns:
        Dictionary with numpy arrays:
            - ts: timestamps
            - open, high, low, close: OHLC prices
            - volume_quote: volume
    """
    logger.debug(f"Converting DataFrame to arrays ({len(df)} rows)")
    
    if not is_datetime64_any_dtype(df.index):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    
    arrays = {
        'ts': df.index.to_numpy(dtype='datetime64[ns]'),
        'open': df['open'].to_numpy(dtype=np.float64),
        'high': df['high'].to_numpy(dtype=np.float64),
        'low': df['low'].to_numpy(dtype=np.float64),
        'close': df['close'].to_numpy(dtype=np.float64),
        'volume_quote': (
            df['volume_quote'].to_numpy(dtype=np.float64)
            if 'volume_quote' in df
            else np.zeros(len(df))
        )
    }
    
    return arrays


def load_final_symbols(
    all_symbols: list,
    strategy: str = "_",
    timeframe: str = "4H"
) -> list:
    """
    Load filtered symbols for a specific strategy and timeframe.
    
    Reads from Excel file in symbols_live/ directory and filters
    the provided list of all symbols.
    
    Args:
        all_symbols: List of all available symbols
        strategy: Strategy name (e.g., 'reversal_long')
        timeframe: Timeframe (e.g., '4H', '1H', '2m')
    
    Returns:
        Sorted list of filtered symbols
    
    Raises:
        FileNotFoundError: If symbol file doesn't exist
        ValueError: If file is empty or invalid
    
    Example:
        >>> all_syms = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        >>> filtered = load_final_symbols(all_syms, 'reversal_long', '4H')
        >>> print(filtered)
        ['BTCUSDT', 'ETHUSDT']
    """
    folder = os.path.join(os.path.dirname(__file__), "..", "symbols_live")
    folder = os.path.abspath(folder)
    path_live = os.path.join(folder, f"symbols_live_{strategy}_{timeframe}.xlsx")
    
    # Check if file exists
    if not os.path.exists(path_live):
        error_msg = (
            f"Symbol file not found for strategy '{strategy}' timeframe '{timeframe}'\n"
            f"Expected file: {path_live}\n"
            f"Please create this file with the symbols for this strategy."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.debug(f"Loading symbols from: {path_live}")
        
        df_live = pd.read_excel(path_live)
        
        # Check if file is empty
        if df_live.empty:
            error_msg = f"Symbol file is empty: {path_live}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        live_symbols = set(df_live.iloc[:, 0].dropna().astype(str))
        
        # Check if any symbols found
        if not live_symbols:
            error_msg = f"No valid symbols found in: {path_live}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        final_symbols = set(all_symbols) & live_symbols
        
        # Warn if no intersection
        if not final_symbols:
            logger.warning(
                f"No symbols match between file and exchange for {strategy} {timeframe}. "
                f"File has {len(live_symbols)} symbols, but none are available on exchange."
            )
        
        logger.debug(f"Loaded {len(final_symbols)} symbols for {strategy} {timeframe}")
        return sorted(final_symbols)
        
    except (FileNotFoundError, ValueError):
        # Re-raise these to stop the bot
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error loading symbols for {strategy} {timeframe}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def fetch_ohlcv_data(symbols: list, timeframe: str) -> dict:
    """
    Fetch OHLCV data for multiple symbols.
    
    Downloads recent candle data from API and converts to DataFrame.
    
    Args:
        symbols: List of symbols to fetch
        timeframe: Timeframe (e.g., '2m', '5m', '1H', '4H')
    
    Returns:
        Dictionary mapping symbol to DataFrame
        {
            'BTCUSDT': DataFrame(...),
            'ETHUSDT': DataFrame(...),
            ...
        }
    
    Example:
        >>> ohlcv = fetch_ohlcv_data(['BTCUSDT', 'ETHUSDT'], '4H')
        >>> btc_df = ohlcv['BTCUSDT']
    """
    logger.debug(f"Fetching OHLCV data for {len(symbols)} symbols ({timeframe})")
    
    ohlcv_data = {}
    
    for sym in symbols:
        try:
            recent_candles = _call_history_candles(
                symbol=sym,
                granularity=timeframe,
                limit=API_LIMIT_DATA
            )
            df = to_dataframe_from_api(recent_candles)
            ohlcv_data[sym] = df
            
        except Exception as e:
            logger.error(f"Error-Failed to fetch OHLCV for {sym}: {e}")
            ohlcv_data[sym] = None
    
    logger.debug(f"Successfully fetched data for {len(ohlcv_data)} symbols")
    return ohlcv_data