"""
sync_excel_to_postgres.py
Production-grade synchronization tool for Excel → PostgreSQL trades.

FIXED: Removed order_price_open, order_ts_open, exec_ts_open columns
       (only keeps CLOSE execution tracking)

Compares by CLOSE_AT + STRATEGY + SYMBOL + PROFIT (exact match)
to find truly missing trades (not timestamp precision issues).
"""

import pandas as pd
import psycopg2
from datetime import datetime
from typing import Tuple, List, Dict
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import settings
try:
    from config.settings import POSTGRES_CONFIG
except ImportError:
    logger.error("Failed to import POSTGRES_CONFIG from config.settings")
    POSTGRES_CONFIG = None


class TradeSync:
    """Handles synchronization of trades from Excel to PostgreSQL."""
    
    def __init__(self, account: str, dry_run: bool = False):
        """
        Initialize trade synchronization.
        
        Args:
            account: Account number ('00', 'E1', '01')
            dry_run: If True, only simulate without writing to database
        """
        self.account = account
        self.dry_run = dry_run
        self.excel_file = os.path.join(
            project_root,
            f'persistence/bot_files_{account}/bot_trades_{account}.xlsx'
        )
        self.conn = None
        
    def load_excel_trades(self) -> pd.DataFrame:
        """Load trades from Excel file."""
        try:
            if not os.path.exists(self.excel_file):
                raise FileNotFoundError(f"Excel file not found: {self.excel_file}")
            
            df = pd.read_excel(self.excel_file)
            logger.info(f"✅ Excel loaded: {len(df)} trades from {self.excel_file}")
            
            # Validate required columns
            required_cols = ['OPEN_AT', 'CLOSE_AT', 'STRATEGY', 'SYMBOL', 
                           'DIRECTION', 'PROFIT', 'REASON_OUT']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load Excel: {e}")
            raise
    
    def load_postgres_trades(self) -> pd.DataFrame:
        """
        Load all trades from PostgreSQL for this account.
        
        Returns:
            DataFrame with PostgreSQL trades
        """
        try:
            self.conn = psycopg2.connect(**POSTGRES_CONFIG)
            
            query = f"SELECT * FROM trades WHERE account = '{self.account}'"
            df_pg = pd.read_sql(query, self.conn)
            
            logger.info(f"✅ PostgreSQL loaded: {len(df_pg)} trades")
            
            return df_pg
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            if self.conn:
                self.conn.close()
            raise
    
    def find_missing_trades(self, df_excel: pd.DataFrame, 
                           df_pg: pd.DataFrame) -> List[Dict]:
        """
        Find trades that exist in Excel but not in PostgreSQL.
        
        Uses robust comparison: CLOSE_AT (date only) + STRATEGY + SYMBOL + PROFIT
        to avoid timestamp precision issues.
        
        Args:
            df_excel: DataFrame with Excel trades
            df_pg: DataFrame with PostgreSQL trades
            
        Returns:
            List of missing trade dictionaries
        """
        # Convert timestamps to date strings for robust comparison
        df_excel['CLOSE_DATE'] = pd.to_datetime(df_excel['CLOSE_AT']).dt.strftime('%Y-%m-%d %H:%M')
        df_pg['close_date'] = pd.to_datetime(df_pg['close_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Round profits for comparison
        df_excel['PROFIT_ROUNDED'] = df_excel['PROFIT'].round(2)
        df_pg['profit_rounded'] = df_pg['profit'].round(2)
        
        # Create unique keys for comparison
        excel_keys = set(
            df_excel.apply(
                lambda row: (
                    row['CLOSE_DATE'],
                    row['STRATEGY'],
                    row['SYMBOL'],
                    row['PROFIT_ROUNDED']
                ),
                axis=1
            )
        )
        
        pg_keys = set(
            df_pg.apply(
                lambda row: (
                    row['close_date'],
                    row['strategy'],
                    row['symbol'],
                    row['profit_rounded']
                ),
                axis=1
            )
        )
        
        # Find missing keys
        missing_keys = excel_keys - pg_keys
        
        # Get full trade data for missing keys
        missing_trades = []
        for idx, row in df_excel.iterrows():
            key = (
                row['CLOSE_DATE'],
                row['STRATEGY'],
                row['SYMBOL'],
                row['PROFIT_ROUNDED']
            )
            if key in missing_keys:
                missing_trades.append(row.to_dict())
        
        logger.info(f"⚠️  Found {len(missing_trades)} missing trades in PostgreSQL")
        return missing_trades
    
    def validate_trade(self, trade: Dict) -> bool:
        """
        Validate trade data before insertion.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if pd.isna(trade['OPEN_AT']) or pd.isna(trade['CLOSE_AT']):
                logger.warning(f"❌ Invalid dates: {trade.get('SYMBOL')}")
                return False
            
            if pd.isna(trade['PROFIT']):
                logger.warning(f"❌ Invalid profit: {trade.get('SYMBOL')}")
                return False
            
            # Validate numeric fields
            float(trade['PROFIT'])
            float(trade['PRICE_ENTRY'])
            float(trade['PRICE_CLOSE'])
            
            return True
            
        except Exception as e:
            logger.warning(f"❌ Validation failed for {trade.get('SYMBOL')}: {e}")
            return False
    
    def insert_trade(self, cursor, trade: Dict) -> bool:
        """
        Insert single trade into PostgreSQL.
        
        FIXED: Removed order_price_open, order_ts_open, exec_ts_open
               (only keeps CLOSE execution tracking)
        
        Args:
            cursor: Database cursor
            trade: Trade dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor.execute("""
                INSERT INTO trades (
                    account, open_at, close_at, duration_days, strategy, symbol, direction,
                    usdt_amount, size, price_entry, price_close, profit, fee,
                    profit_pct, reason_out, regime_family, regime_multiplier,
                    market_direction, direction_multiplier,
                    order_price_close, order_ts_close, exec_ts_close
                ) VALUES (
                    %(account)s, %(open_at)s, %(close_at)s, %(duration_days)s, %(strategy)s,
                    %(symbol)s, %(direction)s, %(usdt_amount)s, %(size)s,
                    %(price_entry)s, %(price_close)s, %(profit)s, %(fee)s,
                    %(profit_pct)s, %(reason_out)s, %(regime_family)s,
                    %(regime_multiplier)s, %(market_direction)s, %(direction_multiplier)s,
                    %(order_price_close)s, %(order_ts_close)s, %(exec_ts_close)s
                )
            """, {
                'account': self.account,
                'open_at': pd.to_datetime(trade['OPEN_AT']),
                'close_at': pd.to_datetime(trade['CLOSE_AT']),
                'duration_days': float(trade['DURATION_DAYS']),
                'strategy': trade['STRATEGY'],
                'symbol': trade['SYMBOL'],
                'direction': trade['DIRECTION'],
                'usdt_amount': float(trade['USDT_AMOUNT']),
                'size': float(trade['SIZE']) if pd.notna(trade.get('SIZE')) else None,
                'price_entry': float(trade['PRICE_ENTRY']),
                'price_close': float(trade['PRICE_CLOSE']),
                'profit': float(trade['PROFIT']),
                'fee': float(trade['FEE']) if pd.notna(trade.get('FEE')) else 0,
                'profit_pct': float(trade['PROFIT_PCT']),
                'reason_out': trade['REASON_OUT'],
                'regime_family': trade.get('REGIME_FAMILY', 'unknown'),
                'regime_multiplier': float(trade.get('REGIME_MULTIPLIER', 1.0)),
                'market_direction': trade.get('MARKET_DIRECTION', 'unknown'),
                'direction_multiplier': float(trade.get('DIRECTION_MULTIPLIER', 1.0)),
                # CLOSE execution tracking (kept)
                'order_price_close': float(trade['ORDER_PRICE_CLOSE']) if pd.notna(trade.get('ORDER_PRICE_CLOSE')) else None,
                'order_ts_close': float(trade['ORDER_TS_CLOSE']) if pd.notna(trade.get('ORDER_TS_CLOSE')) else None,
                'exec_ts_close': float(trade['EXEC_TS_CLOSE']) if pd.notna(trade.get('EXEC_TS_CLOSE')) else None
            })
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert {trade['SYMBOL']}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def sync(self, auto_confirm: bool = False) -> Tuple[int, int]:
        """
        Execute synchronization process.
        
        Args:
            auto_confirm: If True, skip confirmation prompt
            
        Returns:
            Tuple of (inserted_count, failed_count)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"SYNC ACCOUNT {self.account} {'[DRY-RUN]' if self.dry_run else '[LIVE]'}")
        logger.info(f"{'='*70}\n")
        
        try:
            # Load data
            df_excel = self.load_excel_trades()
            df_pg = self.load_postgres_trades()
            
            # Find missing trades
            missing_trades = self.find_missing_trades(df_excel, df_pg)
            
            if not missing_trades:
                logger.info("✅ No missing trades - already synchronized")
                return 0, 0
            
            # Show preview
            logger.info("\n⚠️  Missing trades preview:")
            for i, trade in enumerate(missing_trades[:10], 1):
                logger.info(f"   {i}. {trade['CLOSE_AT']} | {trade['SYMBOL']:8} | "
                          f"{trade['STRATEGY']:15} | ${trade['PROFIT']:7.2f}")
            
            if len(missing_trades) > 10:
                logger.info(f"   ... and {len(missing_trades) - 10} more")
            
            # Calculate totals
            total_profit = sum(t['PROFIT'] for t in missing_trades)
            logger.info(f"\n📊 Total missing: {len(missing_trades)} trades, "
                       f"${total_profit:.2f} profit")
            
            # Confirmation
            if not auto_confirm and not self.dry_run:
                response = input(f"\n➡️  Insert {len(missing_trades)} trades? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("❌ Cancelled by user")
                    return 0, 0
            
            # Insert trades
            if self.dry_run:
                logger.info("\n🔍 DRY-RUN: Would insert trades (no actual write)")
                return len(missing_trades), 0
            
            cursor = self.conn.cursor()
            inserted = 0
            failed = 0
            
            for trade in missing_trades:
                if not self.validate_trade(trade):
                    failed += 1
                    continue
                
                if self.insert_trade(cursor, trade):
                    inserted += 1
                else:
                    failed += 1
            
            # Commit transaction
            self.conn.commit()
            cursor.close()
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ Inserted: {inserted} trades")
            if failed > 0:
                logger.warning(f"⚠️  Failed: {failed} trades")
            logger.info(f"{'='*70}\n")
            
            return inserted, failed
            
        except Exception as e:
            logger.error(f"❌ Sync failed: {e}")
            if self.conn:
                self.conn.rollback()
                logger.info("🔄 Transaction rolled back")
            raise
        
        finally:
            if self.conn:
                self.conn.close()


def main():
    """Main entry point for interactive or CLI usage."""
    
    # Parse arguments
    if len(sys.argv) > 1:
        account = sys.argv[1]
        dry_run = '--dry-run' in sys.argv
        auto_confirm = '--yes' in sys.argv
    else:
        # Interactive mode
        print("\n" + "="*70)
        print("TRADE SYNCHRONIZATION TOOL")
        print("="*70)
        print("\nAvailable accounts: 00, E1, 01")
        account = input("Enter account number: ").strip()
        
        dry_run_input = input("Dry-run only? (yes/no) [yes]: ").strip().lower()
        dry_run = dry_run_input != 'no'
        
        auto_confirm = False
    
    # Validate account
    if account not in ['00', 'E1', '01']:
        logger.error(f"❌ Invalid account: {account}")
        return
    
    # Execute sync
    try:
        syncer = TradeSync(account, dry_run=dry_run)
        inserted, failed = syncer.sync(auto_confirm=auto_confirm)
        
        if not dry_run:
            logger.info(f"\n✅ Sync completed: {inserted} inserted, {failed} failed")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
    