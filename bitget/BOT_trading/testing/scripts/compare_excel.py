#!/usr/bin/env python3
"""
Compare PostgreSQL vs Excel trades data

Usage:
    python compare_trades.py
"""

import pandas as pd
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
from pathlib import Path
# Accounts to compare
ACCOUNTS = ['00', 'E1', '01']

# Excel paths
# Excel paths
_BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_PATHS = {
    '00': _BASE_DIR / 'persistence/bot_files_00/bot_trades_00.xlsx',
    'E1': _BASE_DIR / 'persistence/bot_files_E1/bot_trades_E1.xlsx',
    '01': _BASE_DIR / 'persistence/bot_files_01/bot_trades_01.xlsx'
}
def compare_account(account):
    """Compare single account"""
    
    try:
        # Read PostgreSQL
        conn = psycopg2.connect(**PG_CONFIG)
        df_pg = pd.read_sql(
            f"SELECT * FROM trades WHERE account = '{account}'", 
            conn
        )
        conn.close()
        
        # Read Excel
        excel_path = EXCEL_PATHS[account]
        if not os.path.exists(excel_path):
            print(f"  ✗ Excel not found: {excel_path}")
            return False
        
        df_excel = pd.read_excel(excel_path)
        
        # Compare
        pg_count = len(df_pg)
        excel_count = len(df_excel)
        pg_profit = df_pg['profit'].sum()
        excel_profit = df_excel['PROFIT'].sum()
        
        count_match = (pg_count == excel_count)
        profit_match = (abs(pg_profit - excel_profit) < 0.01)
        match = count_match and profit_match
        
        status = "✅" if match else "❌"
        
        print(f"\n{status} Account {account}:")
        print(f"  PostgreSQL: {pg_count} trades, ${pg_profit:.2f} profit")
        print(f"  Excel:      {excel_count} trades, ${excel_profit:.2f} profit")
        
        if not match:
            if not count_match:
                print(f"  ❌ Trade count mismatch!")
            if not profit_match:
                print(f"  ❌ Profit mismatch!")
        
        return match
        
    except Exception as e:
        print(f"\n❌ Account {account}: ERROR")
        print(f"  {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("COMPARING POSTGRESQL vs EXCEL")
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

if __name__ == "__main__":
    print("=" * 60)
    print("COMPARING POSTGRESQL vs EXCEL")
    print("=" * 60)
    
    for account in ACCOUNTS:
        compare_account(account)
    
    print("\n" + "=" * 60)