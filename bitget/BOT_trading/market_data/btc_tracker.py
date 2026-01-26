"""
BTC Tracker - Capture BTC price and MA50 history for chart overlays.

This module handles:
- On-demand capture of BTC price snapshots
- Calculation of MA50
- Storage in PostgreSQL btc_history table
- Gap filling with last known price when API limit exceeded
"""

import logging
import psycopg2
from datetime import datetime, timezone, timedelta
import pandas as pd
from market_data.api_client import _call_history_candles, to_dataframe_from_api
from config.settings import POSTGRES_CONFIG

logger = logging.getLogger('BOT_trading.market_data.btc_tracker')


def capture_btc_snapshot(timeframe: str) -> bool:
    """
    Capture current BTC price and MA50 for given timeframe.
    Inserts into PostgreSQL if not already exists (ON CONFLICT DO NOTHING).
    
    Args:
        timeframe: Timeframe to capture ('1H', '4H', '6Hutc', '1Dutc')
    
    Returns:
        True if inserted, False if already exists or error
    """
    try:
        # Fetch recent candles (need 51 for MA50)
        candles = _call_history_candles(
            symbol='BTCUSDT',
            granularity=timeframe,
            limit=51
        )
        
        if not candles or len(candles) < 2:
            logger.warning(f"Not enough BTC candles for {timeframe}")
            return False
        
        df = to_dataframe_from_api(candles)
        
        # Ensure close is numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Get latest closed candle (second to last)
        latest = df.iloc[-2]
        timestamp = latest['timestamp']
        price = float(latest['close'])
        
        # Calculate MA50 if enough data
        ma50 = None
        if len(df) >= 51:
            ma50 = float(df['close'].iloc[-51:-1].mean())
        
        # Insert into PostgreSQL (skip if exists)
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO btc_history (timestamp, timeframe, price, ma50)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (timestamp, timeframe) DO NOTHING
        """, (timestamp, timeframe, price, ma50))
        
        inserted = cursor.rowcount > 0
        conn.commit()
        cursor.close()
        conn.close()
        
        if inserted:
            logger.debug(f"BTC snapshot captured: {timeframe} at {timestamp} = ${price:.2f}")
        
        return inserted
        
    except Exception as e:
        logger.error(f"Error capturing BTC snapshot ({timeframe}): {e}")
        return False


def fill_btc_gap(timeframe: str, last_timestamp: datetime, last_price: float) -> int:
    """
    Fill gap in BTC history with last known price (flat line).
    Used when gap exceeds API limit of 200 candles.
    
    Args:
        timeframe: Timeframe to fill ('1H', '4H', '6Hutc', '1Dutc')
        last_timestamp: Last known timestamp in database
        last_price: Last known price to use for filling
    
    Returns:
        Number of rows inserted
    """
    try:
        # Calculate time delta for timeframe
        tf_deltas = {
            '1H': timedelta(hours=1),
            '4H': timedelta(hours=4),
            '6Hutc': timedelta(hours=6),
            '1Dutc': timedelta(days=1)
        }
        
        delta = tf_deltas.get(timeframe)
        if not delta:
            logger.error(f"Unknown timeframe: {timeframe}")
            return 0
        
        # Generate timestamps from last_timestamp to now
        current = last_timestamp + delta
        now = datetime.now(timezone.utc)
        timestamps = []
        
        while current <= now:
            timestamps.append(current)
            current += delta
        
        if not timestamps:
            return 0
        
        # Batch insert with last known price (no MA50)
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        for ts in timestamps:
            cursor.execute("""
                INSERT INTO btc_history (timestamp, timeframe, price, ma50)
                VALUES (%s, %s, %s, NULL)
                ON CONFLICT (timestamp, timeframe) DO NOTHING
            """, (ts, timeframe, last_price))
        
        conn.commit()
        inserted = cursor.rowcount
        cursor.close()
        conn.close()
        
        logger.info(f"Filled BTC gap for {timeframe}: {inserted} rows with price ${last_price:.2f}")
        
        return inserted
        
    except Exception as e:
        logger.error(f"Error filling BTC gap ({timeframe}): {e}")
        return 0