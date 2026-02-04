#!/usr/bin/env python3
"""
Test script for sync_broker WebSocket state
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from market_data import get_ws_manager
from market_data.websocket_manager import init_websocket
import time

# Import credentials
from config.connect_pass import (
    BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00,
    BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1,
    BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01
)

def get_credentials(account):
    """Get API credentials for account"""
    if account == "00":
        return (BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00)
    elif account == "E1":
        return (BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1)
    elif account == "01":
        return (BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01)
    else:
        raise ValueError(f"Unknown account: {account}")

def test_websocket_positions(account='00'):
    print("="*60)
    print(f"WEBSOCKET POSITION TEST - Account {account}")
    print("="*60)
    
    # Get credentials
    api_key, api_secret, api_passphrase = get_credentials(account)
    
    # Initialize WebSocket
    print("\n1. Initializing WebSocket...")
    init_websocket(api_key, api_secret, api_passphrase)
    time.sleep(2)
    
    ws = get_ws_manager()
    
    # Check connection status
    print("\n2. Connection Status:")
    print(f"   Private WS connected: {ws.private_ws.sock.connected if ws.private_ws and ws.private_ws.sock else False}")
    print(f"   Authenticated: {ws.authenticated}")
    
    # Check positions BEFORE refresh
    print("\n3. Positions BEFORE refresh:")
    print(f"   Count: {len(ws.positions)}")
    print(f"   Keys: {list(ws.positions.keys())}")
    
    # Refresh positions
    print("\n4. Refreshing positions...")
    ws.refresh_positions()
    time.sleep(2.0)
    
    # Check positions AFTER refresh
    print("\n5. Positions AFTER refresh:")
    print(f"   Count: {len(ws.positions)}")
    print(f"   Keys: {list(ws.positions.keys())}")
    
    # Show detailed position data
    if ws.positions:
        print("\n6. Detailed position data (RAW ws.positions dict):")
        for symbol, pos_data in ws.positions.items():
            print(f"\n   {symbol}:")
            print(f"      instId: {pos_data.get('instId')}")
            print(f"      holdSide: {pos_data.get('holdSide')}")
            print(f"      total: {pos_data.get('total')}")
            print(f"      available: {pos_data.get('available')}")
    else:
        print("\n6. No positions in WebSocket")
    
    # Test get_positions_by_symbol() for hedge mode detection
    print("\n7. Testing get_positions_by_symbol() (hedge mode detection):")
    
    # Get unique symbols from ws.positions
    all_symbols = set()
    for pos_data in ws.positions.values():
        symbol = pos_data.get('instId')
        if symbol:
            all_symbols.add(symbol)
    
    print(f"   Found {len(all_symbols)} unique symbols in ws.positions")
    
    # Test each symbol with the new function
    total_long = 0
    total_short = 0
    symbols_with_both = []
    
    for symbol in sorted(all_symbols):
        positions = ws.get_positions_by_symbol(symbol)
        
        has_long = positions['long'] is not None
        has_short = positions['short'] is not None
        
        if has_long:
            total_long += 1
        if has_short:
            total_short += 1
        
        if has_long and has_short:
            symbols_with_both.append(symbol)
            print(f"\n   ⚠️  {symbol} has BOTH directions (hedge mode):")
            print(f"      LONG:  {float(positions['long'].get('total', 0))}")
            print(f"      SHORT: {float(positions['short'].get('total', 0))}")
        elif has_long:
            print(f"   ✓ {symbol} LONG: {float(positions['long'].get('total', 0))}")
        elif has_short:
            print(f"   ✓ {symbol} SHORT: {float(positions['short'].get('total', 0))}")
    
    print(f"\n8. Summary:")
    print(f"   Total LONG positions:  {total_long}")
    print(f"   Total SHORT positions: {total_short}")
    print(f"   Total positions:       {total_long + total_short}")
    print(f"   Symbols with BOTH:     {len(symbols_with_both)}")
    
    if symbols_with_both:
        print(f"\n   ⚠️  HEDGE MODE DETECTED on: {', '.join(symbols_with_both)}")
        print(f"   These were LOST in old ws.positions dict (only showed last direction)")
    else:
        print(f"\n   ✓ No hedge mode positions (each symbol has only one direction)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', default='00', help='Account number')
    args = parser.parse_args()
    
    test_websocket_positions(args.account)