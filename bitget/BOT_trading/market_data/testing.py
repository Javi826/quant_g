"""
test_api_candles.py

Script autocontenido para verificar que la API de Bitget devuelve velas CERRADAS.
Ejecutar justo cuando cierre una vela (ej: 08:00:00 para 4H).

Usage:
    python test_api_candles.py
"""

import requests
import json
from datetime import datetime
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"  # o "USDC-FUTURES" según tu config
SYMBOL = "BTCUSDT"
TIMEFRAME = "4H"  # Prueba con el timeframe que uses


# =============================================================================
# API CALL
# =============================================================================
def fetch_candles(symbol: str, granularity: str, limit: int = 5):
    """
    Fetch candles from Bitget API.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        granularity: Timeframe (e.g., '4H', '1H', '15m')
        limit: Number of candles to fetch
    
    Returns:
        List of candles or empty list on error
    """
    url = f"{BASE_URL}/api/v2/mix/market/history-candles"
    
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": limit,
        "productType": PRODUCT_TYPE
    }
    
    try:
        print(f"\n{'='*80}")
        print(f"FETCHING CANDLES FROM BITGET API")
        print(f"{'='*80}")
        print(f"URL: {url}")
        print(f"Params: {json.dumps(params, indent=2)}")
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"\nAPI Response Code: {data.get('code', 'N/A')}")
        print(f"API Response Msg: {data.get('msg', 'N/A')}")
        
        if data.get('code') != '00000':
            print(f"\n❌ API Error: {data}")
            return []
        
        candles = data.get('data', [])
        
        if not candles:
            print("\n⚠️  No candles returned")
            return []
        
        print(f"\n✓ Successfully fetched {len(candles)} candles")
        return candles
        
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        return []


# =============================================================================
# ANALYSIS
# =============================================================================
def analyze_candles(candles: list, granularity: str):
    """
    Analyze fetched candles to check for lookahead bias.
    
    Args:
        candles: List of candles from API
        granularity: Timeframe used
    """
    if not candles:
        print("\n❌ No candles to analyze")
        return
    
    print(f"\n{'='*80}")
    print(f"CANDLE ANALYSIS")
    print(f"{'='*80}")
    
    # Current time
    now = datetime.now()
    now_ts = int(now.timestamp() * 1000)
    
    print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current timestamp (ms): {now_ts}")
    print(f"Timeframe: {granularity}")
    
    # Parse timeframe to get interval in milliseconds
    tf_map = {
        '1m': 60_000,
        '5m': 300_000,
        '15m': 900_000,
        '30m': 1_800_000,
        '1H': 3_600_000,
        '4H': 14_400_000,
        '6H': 21_600_000,
        '12H': 43_200_000,
        '1D': 86_400_000
    }
    
    interval_ms = tf_map.get(granularity, 14_400_000)  # Default to 4H
    
    print(f"\n{'─'*80}")
    print(f"LAST 5 CANDLES (from API)")
    print(f"{'─'*80}\n")
    
    print(f"{'IDX':<5} {'TIMESTAMP':<20} {'DATETIME':<22} {'OPEN':>12} {'HIGH':>12} {'LOW':>12} {'CLOSE':>12} {'STATUS':<15}")
    print(f"{'-'*120}")
    
    for i, candle in enumerate(candles[-5:]):
        # Candle format: [timestamp_ms, open, high, low, close, volume_base, volume_quote]
        ts_ms = int(candle[0])
        open_price = float(candle[1])
        high_price = float(candle[2])
        low_price = float(candle[3])
        close_price = float(candle[4])
        
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(ts_ms / 1000)
        dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate expected close time
        expected_close_ts = ts_ms + interval_ms
        
        # Determine status
        if now_ts >= expected_close_ts:
            status = "✓ CLOSED"
        else:
            status = "⚠️  OPEN (partial)"
        
        print(f"{i-4:<5} {ts_ms:<20} {dt_str:<22} {open_price:>12.2f} {high_price:>12.2f} {low_price:>12.2f} {close_price:>12.2f} {status:<15}")
    
    print(f"{'-'*120}")
    
    # Analysis: Check last candle
    print(f"\n{'='*80}")
    print(f"LOOKAHEAD BIAS CHECK")
    print(f"{'='*80}\n")
    
    last_candle = candles[-1]
    last_ts_ms = int(last_candle[0])
    last_close_ts = last_ts_ms + interval_ms
    
    last_dt = datetime.fromtimestamp(last_ts_ms / 1000)
    last_close_dt = datetime.fromtimestamp(last_close_ts / 1000)
    
    print(f"Last candle:")
    print(f"  Open time:  {last_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Close time: {last_close_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Close price: ${float(last_candle[4]):.2f}")
    
    print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if now_ts >= last_close_ts:
        print(f"\n✅ RESULT: Last candle has CLOSED")
        print(f"   → NO LOOKAHEAD BIAS")
        print(f"   → API returns completed candles only")
    else:
        time_until_close = (last_close_ts - now_ts) / 1000 / 60  # minutes
        print(f"\n⚠️  RESULT: Last candle is still OPEN")
        print(f"   → POTENTIAL LOOKAHEAD BIAS")
        print(f"   → Candle closes in {time_until_close:.1f} minutes")
        print(f"   → You are using partial/incomplete data")
    
    # Gap analysis
    print(f"\n{'─'*80}")
    print(f"GAP ANALYSIS (open of candle N should equal close of candle N-1)")
    print(f"{'─'*80}\n")
    
    for i in range(len(candles) - 4, len(candles)):
        if i > 0:
            prev_close = float(candles[i-1][4])
            curr_open = float(candles[i][1])
            gap = curr_open - prev_close
            gap_pct = (gap / prev_close * 100) if prev_close != 0 else 0
            
            status = "✓ No gap" if abs(gap_pct) < 0.01 else f"⚠️  Gap: {gap_pct:+.2f}%"
            
            print(f"Candle {i-1} → {i}: prev_close=${prev_close:.2f}, curr_open=${curr_open:.2f} | {status}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Main test function.
    
    Best run at exact candle close time (e.g., 08:00:00 for 4H).
    """
    print(f"\n{'#'*80}")
    print(f"# BITGET API CANDLE TEST - LOOKAHEAD BIAS VERIFICATION")
    print(f"{'#'*80}")
    
    # Fetch candles
    candles = fetch_candles(
        symbol=SYMBOL,
        granularity=TIMEFRAME,
        limit=10  # Last 10 candles
    )
    
    if candles:
        # Analyze
        analyze_candles(candles, TIMEFRAME)
        
        # Recommendation
        print(f"\n{'='*80}")
        print(f"RECOMMENDATION FOR ENRICHER")
        print(f"{'='*80}\n")
        
        last_candle = candles[-1]
        last_ts_ms = int(last_candle[0])
        now_ts = int(datetime.now().timestamp() * 1000)
        
        tf_map = {
            '1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
            '1H': 3_600_000, '4H': 14_400_000, '6H': 21_600_000,
            '12H': 43_200_000, '1D': 86_400_000
        }
        interval_ms = tf_map.get(TIMEFRAME, 14_400_000)
        
        if now_ts >= last_ts_ms + interval_ms:
            print("✓ API returns CLOSED candles")
            print("\nFor enricher, use:")
            print("  closed_candles = btc_df[btc_df['ts'] < buy_time]")
            print("\nThis ensures you only use candles that closed BEFORE entry time.")
        else:
            print("⚠️  API returns OPEN (partial) candles")
            print("\nWARNING: Using this data in backtesting will cause lookahead bias!")
    else:
        print("\n❌ Could not fetch candles from API")
    
    print(f"\n{'#'*80}\n")


if __name__ == "__main__":
    main()