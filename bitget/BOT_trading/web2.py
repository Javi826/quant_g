#!/usr/bin/env python3
"""
Test REAL del código actual de websocket_manager.py

Este test:
1. Importa el código REAL actual (con bug)
2. Mockea WebSocket para simular datos reales
3. Verifica el comportamiento ACTUAL
4. Documenta qué DEBERÍA pasar después del fix

Run: python test_websocket_real.py
"""

import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import real websocket manager
from market_data.websocket_manager import BitgetWSManager

# ==============================================================================
# MOCK DATA - REAL 19 POSITIONS FROM ACCOUNT 00
# ==============================================================================

REAL_WEBSOCKET_MESSAGE = {
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "positions",
        "instId": "default"
    },
    "data": [
        {"instId": "BTCUSDT", "holdSide": "long", "total": "0.0015", "averageOpenPrice": "102500"},
        {"instId": "BTCUSDT", "holdSide": "short", "total": "0.002", "averageOpenPrice": "102800"},
        {"instId": "BNBUSDT", "holdSide": "long", "total": "0.05", "averageOpenPrice": "685"},
        {"instId": "BNBUSDT", "holdSide": "short", "total": "0.15", "averageOpenPrice": "687"},
        {"instId": "XRPUSDT", "holdSide": "long", "total": "48", "averageOpenPrice": "3.10"},
        {"instId": "XRPUSDT", "holdSide": "short", "total": "26", "averageOpenPrice": "3.12"},
        {"instId": "DOGEUSDT", "holdSide": "long", "total": "368", "averageOpenPrice": "0.385"},
        {"instId": "DOGEUSDT", "holdSide": "short", "total": "1174", "averageOpenPrice": "0.387"},
        {"instId": "ETHUSDT", "holdSide": "long", "total": "0.05", "averageOpenPrice": "3450"},
        {"instId": "SOLUSDT", "holdSide": "long", "total": "1.2", "averageOpenPrice": "245"},
        {"instId": "ADAUSDT", "holdSide": "short", "total": "120", "averageOpenPrice": "1.05"},
        {"instId": "MATICUSDT", "holdSide": "long", "total": "85", "averageOpenPrice": "0.95"},
        {"instId": "DOTUSDT", "holdSide": "short", "total": "15", "averageOpenPrice": "8.50"},
        {"instId": "LINKUSDT", "holdSide": "long", "total": "3.5", "averageOpenPrice": "22.50"},
        {"instId": "AVAXUSDT", "holdSide": "short", "total": "8", "averageOpenPrice": "42.30"},
        {"instId": "UNIUSDT", "holdSide": "long", "total": "12", "averageOpenPrice": "14.20"},
        {"instId": "ATOMUSDT", "holdSide": "short", "total": "18", "averageOpenPrice": "9.80"},
        {"instId": "LTCUSDT", "holdSide": "long", "total": "0.8", "averageOpenPrice": "125"},
        {"instId": "NEARUSDT", "holdSide": "short", "total": "45", "averageOpenPrice": "5.60"},
    ]
}

# ==============================================================================
# TEST CLASS
# ==============================================================================

class TestWebSocketPositions:
    """Test real WebSocket positions behavior"""
    
    def __init__(self):
        self.ws_manager = None
        self.results = {
            'total_positions_expected': 19,
            'total_positions_actual': 0,
            'hedge_symbols': ['BTCUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT'],
            'tests_passed': [],
            'tests_failed': [],
            'warnings': []
        }
    
    def setup(self):
        """Initialize WebSocket manager with mocked connection"""
        print("=" * 80)
        print("SETUP: Initializing WebSocket Manager")
        print("=" * 80)
        
        # Create instance with mocked websocket
        self.ws_manager = BitgetWSManager(
            api_key="mock_key",
            api_secret="mock_secret",
            api_passphrase="mock_pass"
        )
        
        # Don't actually connect
        self.ws_manager.public_ws = None
        self.ws_manager.private_ws = None
        
        print("✅ WebSocket Manager initialized (mocked)")
        print()
    
    def simulate_websocket_message(self):
        """Simulate receiving positions snapshot from WebSocket"""
        print("=" * 80)
        print("TEST 1: Simulating WebSocket Position Snapshot")
        print("=" * 80)
        
        print(f"Simulating snapshot with {len(REAL_WEBSOCKET_MESSAGE['data'])} positions...")
        
        # Convert message to JSON string (as WebSocket would send it)
        message_json = json.dumps(REAL_WEBSOCKET_MESSAGE)
        
        # Call the REAL _on_private_message method
        self.ws_manager._on_private_message(None, message_json)
        
        self.results['total_positions_actual'] = len(self.ws_manager.positions)
        
        print(f"✅ Positions processed")
        print(f"   Total keys in dict: {len(self.ws_manager.positions)}")
        print(f"   Expected: {self.results['total_positions_expected']}")
        print()
        
        # Check if bug is present
        if self.results['total_positions_actual'] < self.results['total_positions_expected']:
            missing = self.results['total_positions_expected'] - self.results['total_positions_actual']
            self.results['warnings'].append(
                f"⚠️  BUG DETECTED: {missing} positions lost due to hedge mode overwrite"
            )
        else:
            self.results['tests_passed'].append("Position count correct")
    
    def test_dict_keys(self):
        """Test what keys are stored in positions dict"""
        print("=" * 80)
        print("TEST 2: Analyzing Dictionary Keys")
        print("=" * 80)
        
        keys = sorted(self.ws_manager.positions.keys())
        print(f"Keys stored ({len(keys)}):")
        for key in keys:
            pos = self.ws_manager.positions[key]
            side = pos.get('holdSide')
            size = pos.get('total')
            print(f"   {key}: {side} {size}")
        
        print()
    
    def test_hedge_mode_detection(self):
        """Test if hedge mode positions are detected correctly"""
        print("=" * 80)
        print("TEST 3: Hedge Mode Detection (get_positions_by_symbol)")
        print("=" * 80)
        
        for symbol in self.results['hedge_symbols']:
            result = self.ws_manager.get_positions_by_symbol(symbol)
            
            long_size = result['long'].get('total') if result['long'] else None
            short_size = result['short'].get('total') if result['short'] else None
            
            print(f"{symbol}:")
            print(f"   LONG:  {long_size}")
            print(f"   SHORT: {short_size}")
            
            # Check if both detected
            if long_size and short_size:
                self.results['tests_passed'].append(f"{symbol}: Both directions detected")
            elif not long_size and not short_size:
                self.results['tests_failed'].append(f"{symbol}: Neither direction detected")
            else:
                self.results['warnings'].append(
                    f"⚠️  {symbol}: Only {'LONG' if long_size else 'SHORT'} detected "
                    f"(other direction LOST)"
                )
        
        print()
    
    def test_values_iteration(self):
        """Test .values() iteration (used by get_positions_by_symbol internally)"""
        print("=" * 80)
        print("TEST 4: .values() Iteration")
        print("=" * 80)
        
        values_count = len(list(self.ws_manager.positions.values()))
        print(f"Positions via .values(): {values_count}")
        
        # Sum all sizes
        total_size = 0
        for pos in self.ws_manager.positions.values():
            total_size += float(pos.get('total', 0))
        
        print(f"Sum of all sizes: {total_size:.2f}")
        
        if values_count == self.results['total_positions_actual']:
            self.results['tests_passed'].append(".values() iteration works correctly")
        else:
            self.results['tests_failed'].append(".values() count mismatch")
        
        print()
    
    def test_items_iteration(self):
        """Test .items() iteration (used by refresh_positions)"""
        print("=" * 80)
        print("TEST 5: .items() Iteration (refresh_positions behavior)")
        print("=" * 80)
        
        # Simulate refresh_positions() logic
        old_positions = {k: v.get('total') for k, v in self.ws_manager.positions.items()}
        
        print(f"Extracted via .items(): {len(old_positions)} positions")
        print("Sample (first 5):")
        for k, size in sorted(old_positions.items())[:5]:
            print(f"   {k}: {size}")
        
        if len(old_positions) == self.results['total_positions_actual']:
            self.results['tests_passed'].append(".items() iteration works correctly")
        else:
            self.results['tests_failed'].append(".items() count mismatch")
        
        print()
    
    def test_len(self):
        """Test len() function"""
        print("=" * 80)
        print("TEST 6: len(positions)")
        print("=" * 80)
        
        length = len(self.ws_manager.positions)
        print(f"len(ws.positions) = {length}")
        print(f"Expected = {self.results['total_positions_expected']}")
        
        if length == self.results['total_positions_expected']:
            self.results['tests_passed'].append("len() returns correct count")
        else:
            self.results['warnings'].append(
                f"⚠️  len() returns {length}, expected {self.results['total_positions_expected']}"
            )
        
        print()
    
    def test_deprecated_get_position(self):
        """Test deprecated get_position() method"""
        print("=" * 80)
        print("TEST 7: get_position() (deprecated, single symbol lookup)")
        print("=" * 80)
        
        for symbol in self.results['hedge_symbols']:
            result = self.ws_manager.get_position(symbol)
            
            if result:
                side = result.get('holdSide')
                size = result.get('total')
                print(f"{symbol}: {side} {size}")
                self.results['warnings'].append(
                    f"⚠️  get_position('{symbol}') returns only {side} direction"
                )
            else:
                print(f"{symbol}: None")
        
        print()
    
    def print_summary(self):
        """Print test summary"""
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 STATISTICS:")
        print(f"   Positions expected: {self.results['total_positions_expected']}")
        print(f"   Positions actual:   {self.results['total_positions_actual']}")
        print(f"   Positions lost:     {self.results['total_positions_expected'] - self.results['total_positions_actual']}")
        
        print(f"\n✅ PASSED ({len(self.results['tests_passed'])}):")
        for test in self.results['tests_passed']:
            print(f"   - {test}")
        
        if self.results['tests_failed']:
            print(f"\n❌ FAILED ({len(self.results['tests_failed'])}):")
            for test in self.results['tests_failed']:
                print(f"   - {test}")
        
        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"   - {warning}")
        
        print("\n" + "=" * 80)
        print("CONCLUSION:")
        print("=" * 80)
        
        if self.results['total_positions_actual'] < self.results['total_positions_expected']:
            print("❌ BUG CONFIRMED: Hedge mode positions are being lost")
            print(f"   {self.results['total_positions_expected'] - self.results['total_positions_actual']} positions lost due to dict key overwrite")
            print("\n🔧 FIX REQUIRED:")
            print("   Change self.positions[symbol] to self.positions[f'{symbol}_{hold_side}']")
            print("   in websocket_manager.py line ~359")
        else:
            print("✅ All positions preserved correctly")
        
        print()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 80)
    print("REAL WEBSOCKET POSITIONS TEST")
    print("Testing ACTUAL code from websocket_manager.py")
    print("=" * 80 + "\n")
    
    test = TestWebSocketPositions()
    
    try:
        # Run tests
        test.setup()
        test.simulate_websocket_message()
        test.test_dict_keys()
        test.test_hedge_mode_detection()
        test.test_values_iteration()
        test.test_items_iteration()
        test.test_len()
        test.test_deprecated_get_position()
        
        # Summary
        test.print_summary()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR DURING TEST:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())