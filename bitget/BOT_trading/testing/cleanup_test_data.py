"""
Cleanup Test Environment for BOT_trading

Cleans up testing data after tests complete:
- Truncates testing.bot_state table
- Truncates testing.trades table
- Optionally drops entire testing schema
- Optionally removes test files

Run with:
    python3 testing/cleanup_test_data.py
    python3 testing/cleanup_test_data.py --drop-schema
    python3 testing/cleanup_test_data.py --remove-files
"""

import os
import sys
import argparse
import psycopg2

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

# Paths
TEST_BASE_DIR = os.path.join(bot_root, 'persistence', f'bot_files_{TEST_ACCOUNT}')
TEST_STATE_FILE = os.path.join(bot_root, 'persistence', f'bot_state_{TEST_ACCOUNT}.json')


# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

def truncate_tables():
    """Truncate all testing tables (keep schema/structure)."""
    print("\n" + "="*70)
    print("TRUNCATING TABLES")
    print("="*70)
    
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        # Truncate bot_state
        print(f"Truncating {TEST_SCHEMA}.bot_state...")
        cursor.execute(f"TRUNCATE TABLE {TEST_SCHEMA}.bot_state;")
        print(f"✅ {TEST_SCHEMA}.bot_state truncated")
        
        # Truncate trades (restart sequence)
        print(f"Truncating {TEST_SCHEMA}.trades...")
        cursor.execute(f"TRUNCATE TABLE {TEST_SCHEMA}.trades RESTART IDENTITY;")
        print(f"✅ {TEST_SCHEMA}.trades truncated")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n✅ Tables truncated successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Truncate failed: {e}")
        return False


def drop_schema():
    """Drop entire testing schema (destructive)."""
    print("\n" + "="*70)
    print("DROPPING SCHEMA")
    print("="*70)
    print(f"⚠️  WARNING: This will permanently delete schema '{TEST_SCHEMA}'")
    
    response = input("Are you sure? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Schema drop cancelled")
        return False
    
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        print(f"Dropping schema {TEST_SCHEMA}...")
        cursor.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")
        print(f"✅ Schema '{TEST_SCHEMA}' dropped")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n✅ Schema dropped successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Schema drop failed: {e}")
        return False


def remove_files():
    """Remove test files and directories."""
    print("\n" + "="*70)
    print("REMOVING FILES")
    print("="*70)
    print(f"⚠️  WARNING: This will delete test files for account {TEST_ACCOUNT}")
    
    response = input("Are you sure? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ File removal cancelled")
        return False
    
    try:
        import shutil
        
        # Remove base directory
        if os.path.exists(TEST_BASE_DIR):
            shutil.rmtree(TEST_BASE_DIR)
            print(f"✅ Removed directory: {TEST_BASE_DIR}")
        
        # Remove state file
        if os.path.exists(TEST_STATE_FILE):
            os.remove(TEST_STATE_FILE)
            print(f"✅ Removed file: {TEST_STATE_FILE}")
        
        print("\n✅ Files removed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ File removal failed: {e}")
        return False


def verify_cleanup():
    """Verify cleanup was successful."""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        # Check if schema exists
        cursor.execute(f"""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = '{TEST_SCHEMA}'
        """)
        schema_exists = cursor.fetchone() is not None
        
        if schema_exists:
            # Count records in bot_state
            cursor.execute(f"SELECT COUNT(*) FROM {TEST_SCHEMA}.bot_state;")
            bot_state_count = cursor.fetchone()[0]
            print(f"Records in {TEST_SCHEMA}.bot_state: {bot_state_count}")
            
            # Count records in trades
            cursor.execute(f"SELECT COUNT(*) FROM {TEST_SCHEMA}.trades;")
            trades_count = cursor.fetchone()[0]
            print(f"Records in {TEST_SCHEMA}.trades: {trades_count}")
            
            if bot_state_count == 0 and trades_count == 0:
                print("✅ All tables empty")
        else:
            print(f"✅ Schema '{TEST_SCHEMA}' does not exist (dropped)")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"⚠️  Verification error: {e}")
    
    # Check files
    if not os.path.exists(TEST_BASE_DIR):
        print(f"✅ Directory removed: {TEST_BASE_DIR}")
    else:
        print(f"⚠️  Directory still exists: {TEST_BASE_DIR}")
    
    if not os.path.exists(TEST_STATE_FILE):
        print(f"✅ State file removed: {TEST_STATE_FILE}")
    else:
        print(f"⚠️  State file still exists: {TEST_STATE_FILE}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run cleanup with command line options."""
    parser = argparse.ArgumentParser(description='Cleanup BOT_trading test environment')
    parser.add_argument('--drop-schema', action='store_true',
                       help='Drop entire testing schema (destructive)')
    parser.add_argument('--remove-files', action='store_true',
                       help='Remove test files and directories')
    parser.add_argument('--all', action='store_true',
                       help='Truncate tables, drop schema, and remove files')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BOT_TRADING TEST ENVIRONMENT CLEANUP")
    print("="*70)
    
    # If --all, enable everything
    if args.all:
        args.drop_schema = True
        args.remove_files = True
    
    # Default action: truncate tables only
    if not args.drop_schema and not args.remove_files:
        print("\nDefault cleanup: Truncating tables (keeping schema)")
        print("Use --drop-schema to remove schema")
        print("Use --remove-files to remove test files")
        print("Use --all for complete cleanup")
        truncate_tables()
    else:
        # Execute requested cleanup
        if not args.drop_schema:
            truncate_tables()
        
        if args.drop_schema:
            drop_schema()
        
        if args.remove_files:
            remove_files()
    
    # Verify
    verify_cleanup()
    
    print("\n" + "="*70)
    print("CLEANUP COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()