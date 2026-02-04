#!/usr/bin/env python3
"""
Debug script to capture RAW WebSocket position messages
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import json
import time
import websocket
import hmac
import base64
import hashlib

# Import credentials
from config.connect_pass import (
    BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00
)

captured_messages = []

def on_message(ws, message):
    """Capture all messages"""
    try:
        if not message or message == "pong":
            return
        
        if message[0] not in ("{", "["):
            return
        
        data = json.loads(message)
        
        # Capture position-related messages
        arg = data.get('arg', {})
        channel = arg.get('channel')
        
        if channel == 'positions':
            action = data.get('action')
            data_list = data.get('data', [])
            
            print(f"\n{'='*60}")
            print(f"POSITIONS MESSAGE CAPTURED")
            print(f"{'='*60}")
            print(f"Action: {action}")
            print(f"Number of positions in message: {len(data_list)}")
            print(f"\nRAW MESSAGE:")
            print(json.dumps(data, indent=2))
            print(f"{'='*60}\n")
            
            captured_messages.append({
                'action': action,
                'count': len(data_list),
                'data': data
            })
    
    except Exception as e:
        print(f"Error processing message: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket closed: {close_status_code} - {close_msg}")

def on_open(ws):
    print("WebSocket connected - authenticating...")
    
    # Authenticate
    timestamp = str(int(time.time()))
    sign_str = timestamp + 'GET' + '/user/verify'
    signature = base64.b64encode(
        hmac.new(
            BITGET_API_SECRET_00.encode('utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).digest()
    ).decode('utf-8')
    
    auth_msg = {
        "op": "login",
        "args": [{
            "apiKey": BITGET_API_KEY_00,
            "passphrase": BITGET_API_PASS_00,
            "timestamp": timestamp,
            "sign": signature
        }]
    }
    
    ws.send(json.dumps(auth_msg))
    time.sleep(1)
    
    # Subscribe to positions
    print("Subscribing to positions channel...")
    sub_msg = {
        "op": "subscribe",
        "args": [{
            "instType": "USDT-FUTURES",
            "channel": "positions",
            "instId": "default"
        }]
    }
    
    ws.send(json.dumps(sub_msg))

def main():
    print("="*60)
    print("RAW WEBSOCKET POSITION SNAPSHOT CAPTURE")
    print("="*60)
    print("\nThis will capture the raw position messages from Bitget")
    print("Leave running for 10 seconds to capture snapshots...\n")
    
    ws = websocket.WebSocketApp(
        "wss://ws.bitget.com/v2/ws/private",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    
    # Run for 10 seconds
    import threading
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()
    
    print("Waiting 10 seconds to capture messages...")
    time.sleep(10)
    
    ws.close()
    time.sleep(1)
    
    # Summary
    print("\n" + "="*60)
    print("CAPTURE SUMMARY")
    print("="*60)
    print(f"Total messages captured: {len(captured_messages)}")
    
    total_positions = 0
    for msg in captured_messages:
        print(f"\nMessage {captured_messages.index(msg) + 1}:")
        print(f"  Action: {msg['action']}")
        print(f"  Positions: {msg['count']}")
        total_positions += msg['count']
    
    print(f"\nTotal positions across all messages: {total_positions}")
    
    # Check if we got all 19 positions
    if total_positions < 19:
        print(f"\n⚠️  WARNING: Only captured {total_positions} positions, but broker has 19!")
        print("This confirms the WebSocket snapshot is incomplete.")
    else:
        print(f"\n✓ Captured all 19 positions across {len(captured_messages)} message(s)")

if __name__ == "__main__":
    main()