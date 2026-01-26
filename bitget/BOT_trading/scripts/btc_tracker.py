"""
Load BTC History from Parquet Files

Reads historical BTC data from Parquet files and loads into PostgreSQL
with regime calculation.
"""

import os
import sys
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.settings import POSTGRES_CONFIG
from market_regime.regime_metrics import calc_all_metrics
from market_regime.regime_classifier import classify_regime
from config.settings import REGIME_HURST_WINDOW, REGIME_ER_WINDOW, REGIME_ATR_WINDOW
from config.settings import REGIME_PE_WINDOW, REGIME_PE_ORDER


def calculate_regime_for_candle(closes, highs, lows):
    """Calculate regime metrics for a single candle."""
    if len(closes) < 100:
        return 'unknown', None, None, None, None
    
    try:
        ohlc = {
            'open': np.array(closes),
            'high': np.array(highs),
            'low': np.array(lows),
            'close': np.array(closes)
        }
        
        metrics = calc_all_metrics(
            ohlc=ohlc,
            hurst_window=REGIME_HURST_WINDOW,
            er_window=REGIME_ER_WINDOW,
            atr_window=REGIME_ATR_WINDOW,
            pe_window=REGIME_PE_WINDOW,
            pe_order=REGIME_PE_ORDER
        )
        
        regime_family = classify_regime(metrics)
        
        return (
            regime_family,
            metrics.get('hurst'),
            metrics.get('efficiency_ratio'),
            metrics.get('atr_pct'),
            metrics.get('permutation_entropy')
        )
    except Exception as e:
        print(f"Error calculating regime: {e}")
        return 'unknown', None, None, None, None


def load_parquet_to_postgres(parquet_path, timeframe):
    """
    Load BTC data from Parquet file into PostgreSQL with regime calculation.
    
    Args:
        parquet_path: Path to Parquet file
        timeframe: Timeframe (1Dutc, 4H, 1H, 6Hutc)
    
    Returns:
        Number of records inserted
    """
    print(f"\n{'='*60}")
    print(f"Loading {timeframe} from {parquet_path}")
    print(f"{'='*60}")
    
    # Read Parquet
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from Parquet")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            print("ERROR: No timestamp column found")
            return 0
    
    # Sort by timestamp (oldest first)
    df = df.sort_index()
    
    # Extract arrays
    timestamps = df.index.to_pydatetime()
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    print(f"Date range: {timestamps[0]} to {timestamps[-1]}")
    
    # Prepare records
    records = []
    
    # Skip first 100 candles (insufficient lookback)
    start_idx = 100
    print(f"Processing {len(timestamps) - start_idx} candles (skipping first 100)...")
    
    for i in range(start_idx, len(timestamps)):
        ts = timestamps[i]
        price = float(closes[i])
        
        # Calculate MA50
        ma50_window = closes[max(0, i-49):i+1]
        ma50 = float(np.mean(ma50_window))
        
        # Calculate regime using lookback window
        lookback_closes = closes[max(0, i-150):i+1].tolist()
        lookback_highs = highs[max(0, i-150):i+1].tolist()
        lookback_lows = lows[max(0, i-150):i+1].tolist()
        
        regime_family, hurst, er, atr, pe = calculate_regime_for_candle(
            closes=lookback_closes,
            highs=lookback_highs,
            lows=lookback_lows
        )
        
        records.append((
            ts,
            timeframe,
            Decimal(str(price)),
            Decimal(str(ma50)),
            regime_family,
            hurst,
            er,
            atr,
            pe
        ))
        
        # Progress indicator
        if (i - start_idx + 1) % 100 == 0:
            print(f"  Processed {i - start_idx + 1}/{len(timestamps) - start_idx} candles...")
    
    # Remove duplicates (keep last occurrence)
    seen = {}
    unique_records = []
    for record in reversed(records):
        key = (record[0], record[1])  # (timestamp, timeframe)
        if key not in seen:
            seen[key] = True
            unique_records.append(record)
    unique_records.reverse()
    
    print(f"Removed {len(records) - len(unique_records)} duplicate timestamps")
    
    # Batch insert
    print(f"Inserting {len(unique_records)} records into PostgreSQL...")
    
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cursor = conn.cursor()
    
    execute_values(
        cursor,
        """
        INSERT INTO btc_history 
        (timestamp, timeframe, price, ma50, regime_family, hurst, efficiency_ratio, atr_pct, permutation_entropy)
        VALUES %s
        ON CONFLICT (timestamp, timeframe) 
        DO UPDATE SET
            price = EXCLUDED.price,
            ma50 = EXCLUDED.ma50,
            regime_family = EXCLUDED.regime_family,
            hurst = EXCLUDED.hurst,
            efficiency_ratio = EXCLUDED.efficiency_ratio,
            atr_pct = EXCLUDED.atr_pct,
            permutation_entropy = EXCLUDED.permutation_entropy
        """,
        unique_records
    )
    
    inserted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ {timeframe}: {inserted_count} records inserted/updated")
    return inserted_count


def main():
    """Load all timeframes from Parquet files."""
    
    # Define files and timeframes
    files = [
        ('BTCUSDT_1Dutc.parquet', '1Dutc'),
        ('BTCUSDT_4H.parquet', '4H'),
        ('BTCUSDT_1H.parquet', '1H'),
        ('BTCUSDT_6Hutc.parquet', '6Hutc')
    ]
    
    total_inserted = 0
    
    for filename, timeframe in files:
        parquet_path = os.path.join(os.path.dirname(__file__), filename)
        
        if not os.path.exists(parquet_path):
            print(f"⚠️  File not found: {parquet_path}")
            continue
        
        try:
            inserted = load_parquet_to_postgres(parquet_path, timeframe)
            total_inserted += inserted
        except Exception as e:
            print(f"❌ Error loading {timeframe}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETED - Total records: {total_inserted}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()