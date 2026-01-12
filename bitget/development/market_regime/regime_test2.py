"""
verify_btc_coverage.py

Verifies that BTC OHLC data covers all trade timestamps.
Identifies missing timestamps that cause enrichment failures.

Usage:
    python verify_btc_coverage.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob


def check_btc_coverage(trades_folder='../brief_trades', ohlc_folder='../data/crypto_OOS', output_folder='output'):
    """Checks if BTC data covers all trade timestamps."""
    
    print("=" * 80)
    print("BTC COVERAGE VERIFICATION")
    print("=" * 80)
    
    # Find trade files
    trade_files = sorted(glob(f"{trades_folder}/all_trades_*.xlsx"))
    
    if not trade_files:
        print(f"\n❌ No trade files found in {trades_folder}")
        return
    
    print(f"\nFound {len(trade_files)} trade files")
    
    # Load BTC data by timeframe
    btc_data = {}
    for tf in ['1H', '4H', '6Hutc']:
        btc_file = f"{ohlc_folder}/BTCUSDT_{tf}.parquet"
        if Path(btc_file).exists():
            df = pd.read_parquet(btc_file)
            df.columns = df.columns.str.lower()
            
            # Find timestamp column
            ts_col = None
            for col in ['timestamp', 'ts', 'date', 'time']:
                if col in df.columns:
                    ts_col = col
                    break
            
            if ts_col:
                df['ts'] = pd.to_datetime(df[ts_col])
            else:
                df['ts'] = pd.to_datetime(df.index)
            
            btc_data[tf] = df
            print(f"  Loaded BTC {tf}: {len(df)} bars ({df['ts'].min()} → {df['ts'].max()})")
        else:
            print(f"  ⚠️  BTC {tf} not found: {btc_file}")
    
    # Analyze each trade file
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)
    
    print(f"\n{'STRATEGY':<35} {'TF':>4} {'TRADES':>8} {'MATCHED':>8} {'MISSING':>8} {'%_MISS':>8}")
    print("-" * 80)
    
    for trade_file in trade_files:
        name = Path(trade_file).stem.replace('all_trades_', '')
        
        # Extract timeframe
        if '1H' in name.upper():
            tf = '1H'
        elif '4H' in name.upper():
            tf = '4H'
        elif '6H' in name.upper():
            tf = '6H'
        else:
            tf = 'UNKNOWN'
        
        if tf not in btc_data:
            print(f"{name:<35} {tf:>4} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} ❌ NO BTC DATA")
            continue
        
        # Load trades
        df_trades = pd.read_excel(trade_file)
        df_trades.columns = df_trades.columns.str.lower().str.strip()
        
        if 'buy_time' in df_trades.columns:
            df_trades['buy_time'] = pd.to_datetime(df_trades['buy_time'])
        elif 'buy time' in df_trades.columns:
            df_trades['buy_time'] = pd.to_datetime(df_trades['buy time'])
        else:
            continue
        
        # Check matches
        btc_timestamps = set(btc_data[tf]['ts'])
        trade_timestamps = set(df_trades['buy_time'])
        
        matched = len(trade_timestamps & btc_timestamps)
        missing = len(trade_timestamps - btc_timestamps)
        total = len(trade_timestamps)
        pct_miss = (missing / total * 100) if total > 0 else 0
        
        status = "✅" if pct_miss == 0 else "⚠️" if pct_miss < 5 else "❌"
        print(f"{name:<35} {tf:>4} {total:>8} {matched:>8} {missing:>8} {pct_miss:>7.1f}% {status}")
        
        # Show sample of missing timestamps
        if missing > 0 and missing <= 10:
            missing_times = sorted(list(trade_timestamps - btc_timestamps))[:5]
            print(f"  Missing timestamps (sample): {missing_times}")
    
    # =================================================================
    # Detailed timestamp analysis
    # =================================================================
    print("\n" + "=" * 80)
    print("TIMESTAMP ALIGNMENT CHECK")
    print("=" * 80)
    
    for tf, btc_df in btc_data.items():
        print(f"\n{tf} TIMEFRAME:")
        
        # Check if timestamps are aligned to hour boundaries
        btc_times = btc_df['ts'].head(10)
        
        print("  Sample BTC timestamps:")
        for ts in btc_times:
            minute = ts.minute
            second = ts.second
            aligned = "✅" if minute == 0 and second == 0 else "⚠️"
            print(f"    {ts} {aligned}")
        
        # Check time gaps
        btc_df_sorted = btc_df.sort_values('ts')
        time_diffs = btc_df_sorted['ts'].diff()
        
        # Extract hours from timeframe string (handle '1H', '4H', '6Hutc')
        tf_hours = int(''.join(filter(str.isdigit, tf)))
        expected_diff = pd.Timedelta(hours=tf_hours)
        irregular = time_diffs[time_diffs != expected_diff].dropna()
        
        if len(irregular) > 0:
            print(f"\n  ⚠️  Found {len(irregular)} irregular time gaps!")
            print(f"  Expected: {expected_diff}")
            print(f"  Sample irregular gaps:")
            for i, (idx, gap) in enumerate(irregular.head(5).items()):
                print(f"    {btc_df_sorted.loc[idx, 'ts']}: gap = {gap}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. If missing timestamps are due to misalignment:")
    print("   → Ensure trade buy_time is aligned to BTC bar opens")
    print("   → Or use nearest-neighbor matching")
    
    print("\n2. If BTC data has gaps:")
    print("   → Fill missing bars with interpolation")
    print("   → Or download more complete BTC data")
    
    print("\n3. Check trade timestamp generation:")
    print("   → Ensure it uses same timezone as BTC data")
    print("   → Round to nearest hour boundary")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_btc_coverage()