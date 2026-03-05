#!/usr/bin/env python3
"""
Verify BTC 1D UTC data fetched by orchestrator matches broker reality.

This script replicates EXACTLY what production does:
1. Fetches BTC 1Dutc data from broker using fetch_ohlcv_data()
2. Shows last closed candle details
3. Calculates MA5
4. Shows LONG/SHORT filter decisions

Uses production functions as-is.
"""

import os
import sys
from datetime import datetime

# Add BOT_trading to path
BOT_ROOT = os.path.expanduser('~/projects/quant/quant_g/bitget/BOT_trading')
sys.path.insert(0, BOT_ROOT)

from market_data.data_utils import fetch_ohlcv_data


def main():
    print("=" * 100)
    print("BTC 1D UTC DATA VERIFICATION - Production vs Broker Reality")
    print("=" * 100)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Fetch BTC 1Dutc data (EXACTLY as orchestrator does)
    print("\n📡 Fetching BTC 1Dutc from broker (same as orchestrator)...")
    
    try:
        # EXACTLY as regime_classifier.fetch_btc_ohlcv() does
        ohlcv_data = fetch_ohlcv_data(['BTCUSDT'], '1Dutc')
        
        df = ohlcv_data.get('BTCUSDT')
        
        if df is None or df.empty:
            print("❌ No data returned for BTCUSDT 1Dutc")
            return
        
        print(f"✅ Fetched {len(df)} candles")
        
        # Show DataFrame structure
        print(f"\nDataFrame columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes.to_dict()}")
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show last 10 candles
    print("\n" + "=" * 100)
    print("LAST 10 CANDLES FROM BROKER")
    print("=" * 100)
    
    last_10 = df.tail(10)
    
    print("\nOHLC data:")
    for idx, row in last_10.iterrows():
        ts = row['timestamp']
        o = float(row['open'])
        h = float(row['high'])
        l = float(row['low'])
        c = float(row['close'])
        print(f"{ts} | O: ${o:,.2f} H: ${h:,.2f} L: ${l:,.2f} C: ${c:,.2f}")
    
    # Get last closed candle (EXACTLY as orchestrator does)
    last_closed = df.iloc[-1]
    
    print("\n" + "=" * 100)
    print("LAST CLOSED CANDLE (Used by Orchestrator)")
    print("=" * 100)
    print(f"Timestamp: {last_closed['timestamp']}")
    print(f"Open:      ${float(last_closed['open']):,.2f}")
    print(f"High:      ${float(last_closed['high']):,.2f}")
    print(f"Low:       ${float(last_closed['low']):,.2f}")
    print(f"Close:     ${float(last_closed['close']):,.2f}")
    
    # Calculate MA5 (EXACTLY as orchestrator does)
    if len(df) < 5:
        print("\n❌ Not enough data to calculate MA5")
        return
    
    # Convert close to float for calculation
    closes = df['close'].astype(float)
    ma5 = closes.tail(5).mean()
    
    print("\n" + "=" * 100)
    print("MA5 CALCULATION (Last 5 Closes)")
    print("=" * 100)
    
    last_5 = df.tail(5)
    for idx, row in last_5.iterrows():
        print(f"{row['timestamp']}: ${float(row['close']):,.2f}")
    
    print(f"\nMA5 = ${ma5:,.2f}")
    
    # Calculate thresholds (EXACTLY as orchestrator does)
    btc_close = float(df['close'].iloc[-1])
    long_threshold = ma5 * 1.02
    short_threshold = ma5 * 1.00
    
    print("\n" + "=" * 100)
    print("REGIME 0 FILTER DECISIONS")
    print("=" * 100)
    
    print(f"\nBTC Close:         ${btc_close:,.2f}")
    print(f"MA5:               ${ma5:,.2f}")
    print(f"LONG Threshold:    ${long_threshold:,.2f}  (MA5 * 1.02)")
    print(f"SHORT Threshold:   ${short_threshold:,.2f}  (MA5 * 1.00)")
    
    # LONG decision (EXACTLY as get_btc_1d_filter() does)
    long_allowed = btc_close > long_threshold
    long_status = "ALLOW" if long_allowed else "BLOCK"
    long_diff = btc_close - long_threshold
    
    print(f"\n{'='*50}")
    print(f"[REGIME 0 - 1D] LONG: BTC=${btc_close:.2f} vs MA5*1.02=${long_threshold:.2f} → {long_status}")
    print(f"  Difference: ${long_diff:+,.2f}")
    
    # SHORT decision (EXACTLY as get_btc_1d_filter() does)
    short_allowed = btc_close < short_threshold
    short_status = "ALLOW" if short_allowed else "BLOCK"
    short_diff = btc_close - short_threshold
    
    print(f"\n{'='*50}")
    print(f"[REGIME 0 - 1D] SHORT: BTC=${btc_close:.2f} vs MA5*1.00=${short_threshold:.2f} → {short_status}")
    print(f"  Difference: ${short_diff:+,.2f}")
    
    # Dead zone check
    in_dead_zone = not long_allowed and not short_allowed
    
    if in_dead_zone:
        print(f"\n{'='*50}")
        print("⚠️  DEAD ZONE ACTIVE")
        print(f"  BTC is between ${short_threshold:,.2f} and ${long_threshold::.2f}")
        print("  Both LONG and SHORT are BLOCKED")
    
    print("\n" + "=" * 100)
    print("VERIFICATION COMPLETE")
    print("=" * 100)
    print("\n✅ This output should MATCH orchestrator logs:")
    print("   [REGIME 0 - 1D] LONG: BTC=$... vs MA5*1.02=$... → ALLOW/BLOCK")
    print("   [REGIME 0 - 1D] SHORT: BTC=$... vs MA5*1.00=$... → ALLOW/BLOCK")


if __name__ == "__main__":
    main()