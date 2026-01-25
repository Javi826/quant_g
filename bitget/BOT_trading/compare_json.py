#!/usr/bin/env python3
"""
Compare PostgreSQL vs JSON state data

Usage:
    python compare_state.py
"""

import json
import psycopg2
import os

# PostgreSQL config
PG_CONFIG = {
    'dbname': 'bot_trading',
    'user': 'javi',
    'password': 'Laplaciano86-',
    'host': 'localhost',
    'port': 5432
}

# Accounts to compare
ACCOUNTS = ['00', 'E1', '01']

# JSON paths
JSON_PATHS = {
    '00': 'persistence/bot_files_00/bot_state_00.json',
    'E1': 'persistence/bot_files_E1/bot_state_E1.json',
    '01': 'persistence/bot_files_01/bot_state_01.json'
}


def count_positions(state_data):
    """Count total positions in state data"""
    positions = state_data.get('positions', {})
    total = 0
    for strategy_positions in positions.values():
        total += len(strategy_positions)
    return total


def count_strategies(state_data):
    """Count strategies with candle counters"""
    strategy_candles = state_data.get('strategy_candles', {})
    return len(strategy_candles)


def compare_position_field(field_name, pg_value, json_value, position_idx, symbol):
    """Compare single field with detailed output"""
    if pg_value == json_value:
        return True
    else:
        print(f"    ❌ Position {position_idx} ({symbol}) - {field_name}:")
        print(f"       PostgreSQL: {pg_value}")
        print(f"       JSON:       {json_value}")
        return False


def compare_account(account):
    """Compare single account state with field-by-field validation"""
    
    try:
        # Read PostgreSQL
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT state_data FROM bot_state WHERE account = %s",
            (account,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            print(f"\n❌ Account {account}:")
            print(f"  PostgreSQL: No data found")
            return False
        
        pg_state = result[0]  # JSONB data
        
        # Read JSON
        json_path = JSON_PATHS[account]
        if not os.path.exists(json_path):
            print(f"\n❌ Account {account}:")
            print(f"  JSON not found: {json_path}")
            return False
        
        with open(json_path, 'r') as f:
            json_state = json.load(f)
        
        print(f"\n{'=' * 80}")
        print(f"ACCOUNT {account} - DETAILED COMPARISON")
        print(f"{'=' * 80}")
        
        all_match = True
        
        # Compare strategy_candles
        pg_candles = pg_state.get('strategy_candles', {})
        json_candles = json_state.get('strategy_candles', {})
        
        print(f"\n📊 STRATEGY CANDLES:")
        
        all_strategies = set(list(pg_candles.keys()) + list(json_candles.keys()))
        
        for strategy_id in sorted(all_strategies):
            pg_count = pg_candles.get(strategy_id, None)
            json_count = json_candles.get(strategy_id, None)
            
            if pg_count == json_count:
                print(f"  ✅ {strategy_id}: {pg_count}")
            else:
                print(f"  ❌ {strategy_id}:")
                print(f"     PostgreSQL: {pg_count}")
                print(f"     JSON:       {json_count}")
                all_match = False
        
        # Compare positions
        pg_positions = pg_state.get('positions', {})
        json_positions = json_state.get('positions', {})
        
        print(f"\n📍 POSITIONS:")
        
        all_strategies_pos = set(list(pg_positions.keys()) + list(json_positions.keys()))
        
        for strategy_id in sorted(all_strategies_pos):
            pg_pos_list = pg_positions.get(strategy_id, [])
            json_pos_list = json_positions.get(strategy_id, [])
            
            print(f"\n  Strategy: {strategy_id}")
            print(f"  PostgreSQL: {len(pg_pos_list)} positions")
            print(f"  JSON:       {len(json_pos_list)} positions")
            
            if len(pg_pos_list) != len(json_pos_list):
                print(f"  ❌ Position count mismatch!")
                all_match = False
                continue
            
            # Compare each position field by field
            for idx, (pg_pos, json_pos) in enumerate(zip(pg_pos_list, json_pos_list), 1):
                symbol = pg_pos.get('symbol', '???')
                position_match = True
                
                # Compare all fields
                fields_to_compare = [
                    'symbol', 'size', 'entry_price', 'direction',
                    'tp', 'sl', 'order_id', 'opened_at', 'usdt_amount',
                    'regime_family', 'regime_multiplier',
                    'market_direction', 'direction_multiplier'
                ]
                
                for field in fields_to_compare:
                    pg_val = pg_pos.get(field)
                    json_val = json_pos.get(field)
                    
                    if not compare_position_field(field, pg_val, json_val, idx, symbol):
                        position_match = False
                        all_match = False
                
                if position_match:
                    print(f"    ✅ Position {idx} ({symbol}): All fields match")
        
        print(f"\n{'=' * 80}")
        
        if all_match:
            print(f"✅ Account {account}: PERFECT MATCH (all fields identical)")
        else:
            print(f"❌ Account {account}: MISMATCHES FOUND (see details above)")
        
        print(f"{'=' * 80}")
        
        return all_match
        
    except Exception as e:
        print(f"\n❌ Account {account}: ERROR")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("COMPARING POSTGRESQL vs JSON STATE")
    print("=" * 60)
    
    results = []
    for account in ACCOUNTS:
        results.append(compare_account(account))
    
    print("\n" + "=" * 60)
    
    if all(results):
        print("✅ ALL ACCOUNTS MATCH - DUAL-WRITE WORKING CORRECTLY")
    else:
        print("❌ SOME ACCOUNTS DON'T MATCH - CHECK LOGS")
    
    print("=" * 60)