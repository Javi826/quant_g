"""
Load BTC History from Parquet - Simplified Version

Reads daily BTC close prices from Parquet and loads into PostgreSQL.
Only stores date and price (no regime, no MA50, no timeframes).
"""

import os
import sys
import pandas as pd
from decimal import Decimal
import psycopg2
from psycopg2.extras import execute_values

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.settings import POSTGRES_CONFIG


def load_parquet_to_postgres(parquet_path):
    """
    Load BTC daily prices from Parquet file into PostgreSQL.
    
    Args:
        parquet_path: Path to BTCUSDT_1Dutc.parquet file
    
    Returns:
        Number of records inserted
    """
    print(f"\n{'='*60}")
    print(f"Loading BTC history from {parquet_path}")
    print(f"{'='*60}")
    
    # Read Parquet
    df = pd.read_parquet(parquet_path)
    print(f"✅ Loaded {len(df)} rows from Parquet")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            print("❌ ERROR: No timestamp column found")
            return 0
    
    # Sort by timestamp (oldest first)
    df = df.sort_index()
    
    # Extract date and close price
    df['date'] = df.index.date
    df = df[['date', 'close']].copy()
    
    # Remove duplicates (keep last price if multiple entries per day)
    df = df.drop_duplicates(subset='date', keep='last')
    
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"📊 Total unique days: {len(df)}")
    
    # Prepare records for insertion
    records = [
        (row['date'], Decimal(str(row['close'])))
        for _, row in df.iterrows()
    ]
    
    # Insert into PostgreSQL
    print(f"💾 Inserting {len(records)} records into PostgreSQL...")
    
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cursor = conn.cursor()
    
    execute_values(
        cursor,
        """
        INSERT INTO btc_history (date, price)
        VALUES %s
        ON CONFLICT (date) 
        DO UPDATE SET price = EXCLUDED.price
        """,
        records
    )
    
    inserted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ Successfully inserted/updated {inserted_count} records")
    return inserted_count


def main():
    """Load BTC history from Parquet file."""
    
    parquet_path = os.path.join(
        os.path.dirname(__file__), 
        'BTCUSDT_1Dutc.parquet'
    )
    
    if not os.path.exists(parquet_path):
        print(f"❌ File not found: {parquet_path}")
        return
    
    try:
        inserted = load_parquet_to_postgres(parquet_path)
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETED - {inserted} records loaded")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()