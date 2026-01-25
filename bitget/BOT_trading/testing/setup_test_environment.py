"""
Setup Test Environment for BOT_trading

Creates isolated testing environment with:
- PostgreSQL schema 'testing' with tables bot_state and trades
- Directory structure for account 99
- Test configuration files

Run with:
    python3 testing/setup_test_environment.py

This ensures complete isolation from production data.
"""

import os
import sys
import psycopg2
from psycopg2 import sql

# Add BOT_trading to path
current_dir = os.path.dirname(os.path.abspath(__file__))
bot_root = os.path.dirname(current_dir)
sys.path.insert(0, bot_root)

from config.settings import POSTGRES_CONFIG

# =============================================================================
# CONFIGURATION
# =============================================================================

TEST_ACCOUNT = '99'
TEST_SCHEMA = 'testing'

# Paths for test account
TEST_BASE_DIR = os.path.join(bot_root, 'persistence', f'bot_files_{TEST_ACCOUNT}')
TEST_STATE_FILE = os.path.join(bot_root, 'persistence', f'bot_state_{TEST_ACCOUNT}.json')
TEST_TRADES_FILE = os.path.join(TEST_BASE_DIR, f'bot_trades_{TEST_ACCOUNT}.xlsx')
TEST_LOG_FILE = os.path.join(TEST_BASE_DIR, f'BOT_orchestrator_{TEST_ACCOUNT}.log')


# =============================================================================
# SQL SCHEMAS
# =============================================================================

CREATE_SCHEMA_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA};
"""

CREATE_BOT_STATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TEST_SCHEMA}.bot_state (
    account TEXT PRIMARY KEY,
    state_data JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_TRADES_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TEST_SCHEMA}.trades (
    id SERIAL PRIMARY KEY,
    account TEXT NOT NULL,
    open_at TIMESTAMP NOT NULL,
    close_at TIMESTAMP NOT NULL,
    duration_days NUMERIC(10, 4),
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    usdt_amount NUMERIC(12, 2),
    size NUMERIC(16, 6),
    price_entry NUMERIC(16, 6),
    price_close NUMERIC(16, 6),
    profit NUMERIC(12, 2),
    fee NUMERIC(12, 4),
    profit_pct NUMERIC(8, 1),
    reason_out TEXT,
    regime_family TEXT,
    regime_multiplier NUMERIC(4, 1),
    market_direction TEXT,
    direction_multiplier NUMERIC(4, 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_TRADES_INDEXES_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_trades_test_account ON {TEST_SCHEMA}.trades(account);
CREATE INDEX IF NOT EXISTS idx_trades_test_strategy ON {TEST_SCHEMA}.trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_test_symbol ON {TEST_SCHEMA}.trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_test_open_at ON {TEST_SCHEMA}.trades(open_at);
CREATE INDEX IF NOT EXISTS idx_trades_test_close_at ON {TEST_SCHEMA}.trades(close_at);
"""


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def create_postgresql_schema():
    """Create PostgreSQL testing schema and tables."""
    print("\n" + "="*70)
    print("POSTGRESQL SETUP")
    print("="*70)
    
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        # Create schema
        print(f"Creating schema '{TEST_SCHEMA}'...")
        cursor.execute(CREATE_SCHEMA_SQL)
        print(f"✅ Schema '{TEST_SCHEMA}' created")
        
        # Create bot_state table
        print(f"Creating table '{TEST_SCHEMA}.bot_state'...")
        cursor.execute(CREATE_BOT_STATE_TABLE_SQL)
        print(f"✅ Table '{TEST_SCHEMA}.bot_state' created")
        
        # Create trades table
        print(f"Creating table '{TEST_SCHEMA}.trades'...")
        cursor.execute(CREATE_TRADES_TABLE_SQL)
        print(f"✅ Table '{TEST_SCHEMA}.trades' created")
        
        # Create indexes
        print(f"Creating indexes on '{TEST_SCHEMA}.trades'...")
        cursor.execute(CREATE_TRADES_INDEXES_SQL)
        print(f"✅ Indexes created")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n✅ PostgreSQL testing schema setup completed")
        return True
        
    except Exception as e:
        print(f"\n❌ PostgreSQL setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_directory_structure():
    """Create directory structure for test account."""
    print("\n" + "="*70)
    print("DIRECTORY SETUP")
    print("="*70)
    
    try:
        # Create base directory
        os.makedirs(TEST_BASE_DIR, exist_ok=True)
        print(f"✅ Created directory: {TEST_BASE_DIR}")
        
        # Create persistence directory
        persistence_dir = os.path.join(bot_root, 'persistence')
        os.makedirs(persistence_dir, exist_ok=True)
        print(f"✅ Created directory: {persistence_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Directory setup failed: {e}")
        return False


def create_initial_state_file():
    """Create empty initial state file."""
    print("\n" + "="*70)
    print("STATE FILE SETUP")
    print("="*70)
    
    try:
        import json
        
        initial_state = {
            'open_positions': {},
            'strategy_candles': {}
        }
        
        with open(TEST_STATE_FILE, 'w') as f:
            json.dump(initial_state, f, indent=2)
        
        print(f"✅ Created state file: {TEST_STATE_FILE}")
        return True
        
    except Exception as e:
        print(f"\n❌ State file creation failed: {e}")
        return False


def create_initial_trades_file():
    """Create empty trades Excel file."""
    print("\n" + "="*70)
    print("TRADES FILE SETUP")
    print("="*70)
    
    try:
        import pandas as pd
        
        # Create empty DataFrame with correct columns
        columns = [
            'OPEN_AT', 'CLOSE_AT', 'DURATION_DAYS', 'STRATEGY', 'SYMBOL',
            'DIRECTION', 'USDT_AMOUNT', 'SIZE', 'PRICE_ENTRY', 'PRICE_CLOSE',
            'PROFIT', 'FEE', 'PROFIT_PCT', 'REASON_OUT', 'REGIME_FAMILY',
            'REGIME_MULTIPLIER', 'MARKET_DIRECTION', 'DIRECTION_MULTIPLIER'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_excel(TEST_TRADES_FILE, index=False, engine='openpyxl')
        
        print(f"✅ Created trades file: {TEST_TRADES_FILE}")
        return True
        
    except Exception as e:
        print(f"\n❌ Trades file creation failed: {e}")
        return False


def verify_setup():
    """Verify all components are correctly set up."""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    all_ok = True
    
    # Check PostgreSQL
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        # Check schema exists
        cursor.execute(f"""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = '{TEST_SCHEMA}'
        """)
        if cursor.fetchone():
            print(f"✅ Schema '{TEST_SCHEMA}' exists")
        else:
            print(f"❌ Schema '{TEST_SCHEMA}' not found")
            all_ok = False
        
        # Check bot_state table
        cursor.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{TEST_SCHEMA}' 
            AND table_name = 'bot_state'
        """)
        if cursor.fetchone():
            print(f"✅ Table '{TEST_SCHEMA}.bot_state' exists")
        else:
            print(f"❌ Table '{TEST_SCHEMA}.bot_state' not found")
            all_ok = False
        
        # Check trades table
        cursor.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{TEST_SCHEMA}' 
            AND table_name = 'trades'
        """)
        if cursor.fetchone():
            print(f"✅ Table '{TEST_SCHEMA}.trades' exists")
        else:
            print(f"❌ Table '{TEST_SCHEMA}.trades' not found")
            all_ok = False
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ PostgreSQL verification failed: {e}")
        all_ok = False
    
    # Check directories
    if os.path.exists(TEST_BASE_DIR):
        print(f"✅ Directory exists: {TEST_BASE_DIR}")
    else:
        print(f"❌ Directory not found: {TEST_BASE_DIR}")
        all_ok = False
    
    # Check state file
    if os.path.exists(TEST_STATE_FILE):
        print(f"✅ State file exists: {TEST_STATE_FILE}")
    else:
        print(f"❌ State file not found: {TEST_STATE_FILE}")
        all_ok = False
    
    # Check trades file
    if os.path.exists(TEST_TRADES_FILE):
        print(f"✅ Trades file exists: {TEST_TRADES_FILE}")
    else:
        print(f"❌ Trades file not found: {TEST_TRADES_FILE}")
        all_ok = False
    
    return all_ok


def print_summary():
    """Print setup summary."""
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    print(f"\nTest Account: {TEST_ACCOUNT}")
    print(f"PostgreSQL Schema: {TEST_SCHEMA}")
    print(f"\nPostgreSQL Tables:")
    print(f"  - {TEST_SCHEMA}.bot_state")
    print(f"  - {TEST_SCHEMA}.trades")
    print(f"\nFiles:")
    print(f"  - State: {TEST_STATE_FILE}")
    print(f"  - Trades: {TEST_TRADES_FILE}")
    print(f"  - Logs: {TEST_LOG_FILE}")
    print("\n" + "="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete setup."""
    print("\n" + "="*70)
    print("BOT_TRADING TEST ENVIRONMENT SETUP")
    print("="*70)
    print(f"\nThis will create isolated testing environment for account {TEST_ACCOUNT}")
    print(f"PostgreSQL schema: {TEST_SCHEMA}")
    print(f"Base directory: {TEST_BASE_DIR}")
    
    # Confirm
    response = input("\nProceed with setup? (yes/no): ")
    if response.lower() != 'yes':
        print("\n❌ Setup cancelled")
        return
    
    # Run setup steps
    success = True
    success &= create_postgresql_schema()
    success &= create_directory_structure()
    success &= create_initial_state_file()
    success &= create_initial_trades_file()
    
    # Verify
    if success:
        verified = verify_setup()
        if verified:
            print("\n" + "="*70)
            print("✅ SETUP COMPLETED SUCCESSFULLY")
            print("="*70)
            print_summary()
            print("\nYou can now run tests with account 99")
            print("All data will be isolated in testing schema")
        else:
            print("\n" + "="*70)
            print("⚠️  SETUP COMPLETED WITH WARNINGS")
            print("="*70)
            print("Some verification checks failed. Review output above.")
    else:
        print("\n" + "="*70)
        print("❌ SETUP FAILED")
        print("="*70)
        print("Review errors above and try again")


if __name__ == "__main__":
    main()