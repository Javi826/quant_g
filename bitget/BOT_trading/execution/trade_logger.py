"""
Trade Logger - Handles trade logging to Excel.

This module is responsible for:
- Logging closed positions to Excel file
- Calculating profit and fees
- Managing trade history
- Tracking market regime at position open

This module is intentionally separated from order_manager and position_tracker
to avoid circular dependencies. It does not import from any execution submodules.
"""

import os
import pandas as pd
import traceback
from datetime import datetime
from decimal import Decimal
from typing import Optional
from config.settings import POSTGRES_CONFIG
import logging
logger = logging.getLogger('BOT_trading.execution.trade_logger')


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Excel configuration
TRADES_LOG_PATH = None

# PostgreSQL configuration (dual-write)
POSTGRES_ENABLED = True

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def configure_log_path(trades_log_path: str) -> None:
    """
    Configure path for trade logging Excel file.
    
    Args:
        trades_log_path: Path to Excel file for trade logging
    """
    global TRADES_LOG_PATH
    TRADES_LOG_PATH = trades_log_path


# =============================================================================
# POSTGRESQL HOOK (NEW - NON-BLOCKING)
# =============================================================================

def _write_to_postgresql(opened_at_dt, closed_at, delta_days, strategy_id, symbol, 
                        direction, usdt_amount, size_val, entry_price, close_price,
                        profit, fee, profit_pct, reason, regime_family, regime_multiplier,
                        market_direction, direction_multiplier,
                        order_price_close, order_ts_close, exec_ts_close):
    """
    Write trade to PostgreSQL (dual-write hook).
    
    This function is called after all calculations are done.
    Errors are logged but don't affect Excel writing.
    """
    if not POSTGRES_ENABLED:
        return
    
    try:
        import psycopg2
        from psycopg2 import sql
        
        # Extract account from TRADES_LOG_PATH
        account = 'unknown'
        if TRADES_LOG_PATH:
            filename = os.path.basename(TRADES_LOG_PATH)
            if filename.startswith('bot_trades_') and filename.endswith('.xlsx'):
                account = filename.replace('bot_trades_', '').replace('.xlsx', '')
        
        # Connect and insert
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        
        insert_query = sql.SQL("""
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
        """)
        
        cursor.execute(insert_query, {
            'account': account,
            'open_at': opened_at_dt,
            'close_at': closed_at,
            'duration_days': round(delta_days, 4),
            'strategy': strategy_id,
            'symbol': symbol,
            'direction': direction.upper(),
            'usdt_amount': round(usdt_amount, 2),
            'size': round(size_val, 6) if size_val else None,
            'price_entry': round(entry_price, 6),
            'price_close': round(close_price, 6),
            'profit': round(profit, 2),
            'fee': round(fee, 4),
            'profit_pct': round(profit_pct, 1),
            'reason_out': reason,
            'regime_family': regime_family,
            'regime_multiplier': round(regime_multiplier, 1),
            'market_direction': market_direction,
            'direction_multiplier': round(direction_multiplier, 1),
            'order_price_close': order_price_close,
            'order_ts_close': order_ts_close,
            'exec_ts_close': exec_ts_close
        })
        
        conn.commit()
        cursor.close()
        conn.close()
        

    except Exception as e:
        logger.error(f"Error-PostgreSQL write FAILED: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# ORIGINAL CODE - UNCHANGED
# =============================================================================

def log_closed_position(opened_at,
                       strategy_id: str,
                       symbol: str,
                       direction: str,
                       usdt_amount: float,
                       entry_price: Decimal,
                       close_price: Decimal,
                       reason: str,
                       size: Decimal,
                       profit_from_api: Optional[Decimal] = None,
                       fee_from_api: Optional[Decimal] = None,
                       bot_state=None,
                       regime_family: Optional[str] = None,
                       regime_multiplier: Optional[float] = None,
                       market_direction: Optional[str] = None,            
                       direction_multiplier: Optional[float] = None,
                        order_price_close: Optional[float] = None,
                        order_ts_close: Optional[float] = None,
                        exec_ts_close: Optional[float] = None
                       ) -> None:  
    """
    Log a closed position to Excel file.
    
    This function:
    1. Calculates profit and fees
    2. Formats trade data
    3. Appends to Excel file
    4. Updates bot state
    
    Args:
        opened_at: Position open timestamp (datetime or string)
        strategy_id: Strategy ID that opened the position
        symbol: Trading symbol (e.g., 'BTCUSDT')
        direction: Position direction ('long' or 'short')
        usdt_amount: USDT amount invested
        entry_price: Entry price
        close_price: Close price
        reason: Close reason ('TP', 'SL', 'TIMEOUT', 'OUT_OF_MARGIN', etc.)
        size: Position size
        profit_from_api: Profit from API (if available)
        fee_from_api: Fee from API (if available)
        bot_state: Bot state object for profit tracking
        regime_family: Market regime family when position opened ('volatile', 'ranging', 'trending')
        regime_multiplier: Multiplier applied based on regime (0, 0.5, 1.0, 1.5, etc.)
    """
    if TRADES_LOG_PATH is None:
        logger.warning("WAR-TRADES_LOG_PATH not configured. Trade not logged.")
        return
    
    try:
        # Convert to float
        entry_price = float(entry_price)
        close_price = float(close_price)
        usdt_amount = float(usdt_amount)

        # Extract size value
        size_val = None
        if size is not None:
            try:
                size_val = float(size)
            except Exception:
                size_val = None

        # Calculate USDT amount if not provided
        if usdt_amount == 0 and size_val is not None:
            usdt_amount = size_val * entry_price

        # Calculate profit
        if profit_from_api is not None:
            # Use API profit
            profit_gross = float(profit_from_api)
            fee = float(fee_from_api) if fee_from_api is not None else 0
            fee = 2 * fee  # Double fee (open + close)
            profit = profit_gross - fee
            
            if usdt_amount > 0:
                profit_pct = (profit / usdt_amount) * 100
            else:
                profit_pct = 0
        else:
            # Calculate from prices
            if size_val is not None:
                if direction.lower() == 'long':
                    profit     = (close_price - entry_price) * size_val
                    profit_pct = ((close_price - entry_price) / entry_price) * 100
                else:
                    profit     = (entry_price - close_price) * size_val
                    profit_pct = ((entry_price - close_price) / entry_price) * 100
            else:
                if direction.lower() == 'long':
                    profit     = (close_price - entry_price) * (usdt_amount / entry_price)
                    profit_pct = ((close_price - entry_price) / entry_price) * 100
                else:
                    profit     = (entry_price - close_price) * (usdt_amount / entry_price)
                    profit_pct = ((entry_price - close_price) / entry_price) * 100
            
            fee = 0

        closed_at = datetime.now()

        # Convert opened_at to datetime if string
        if isinstance(opened_at, str):
            opened_at_dt = datetime.strptime(opened_at, '%Y-%m-%d %H:%M:%S')
        else:
            opened_at_dt = opened_at

        # Remove timezone info
        if opened_at_dt.tzinfo is not None:
            opened_at_dt = opened_at_dt.replace(tzinfo=None)
            
        if closed_at.tzinfo is not None:
            closed_at = closed_at.replace(tzinfo=None)

        # Calculate duration
        delta_days = (closed_at - opened_at_dt).total_seconds() / (3600 * 24)

        # Build record with regime tracking
        new_record = {
            'OPEN_AT': opened_at_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'CLOSE_AT': closed_at.strftime('%Y-%m-%d %H:%M:%S'),
            'DURATION_DAYS': round(delta_days, 4),
            'STRATEGY': strategy_id,
            'SYMBOL': symbol,
            'DIRECTION': direction.upper(),
            'USDT_AMOUNT': round(usdt_amount, 2),
            'SIZE': round(size_val, 6) if size_val else 0,
            'PRICE_ENTRY': round(entry_price, 6),
            'PRICE_CLOSE': round(close_price, 6),
            'PROFIT': round(profit, 2),
            'FEE': round(fee, 4) if profit_from_api is not None else 0,
            'PROFIT_PCT': round(profit_pct, 1),
            'REASON_OUT': reason,
            'REGIME_FAMILY': regime_family if regime_family else 'unknown',
            'REGIME_MULTIPLIER': round(regime_multiplier, 1) if regime_multiplier is not None else 1.0,
            'MARKET_DIRECTION': market_direction if market_direction else 'unknown',
            'DIRECTION_MULTIPLIER': round(direction_multiplier, 1) if direction_multiplier is not None else 1.0,
            'ORDER_PRICE_CLOSE': order_price_close,
            'ORDER_TS_CLOSE': order_ts_close,
            'EXEC_TS_CLOSE': exec_ts_close
        }

        # Append to Excel
        if os.path.exists(TRADES_LOG_PATH):
            df = pd.read_excel(TRADES_LOG_PATH)
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        else:
            df = pd.DataFrame([new_record])

        df.to_excel(TRADES_LOG_PATH, index=False, engine='openpyxl')
        
        # Update bot state
        if bot_state is not None:
            bot_state.closed_total_profit += profit

        # =====================================================================
        # POSTGRESQL HOOK (NEW - added before logging)
        # =====================================================================
        _write_to_postgresql(
            opened_at_dt, closed_at, delta_days, strategy_id, symbol,
            direction, usdt_amount, size_val, entry_price, close_price,
            profit, fee, profit_pct, reason,
            regime_family if regime_family else 'unknown',
            regime_multiplier if regime_multiplier is not None else 1.0,
            market_direction if market_direction else 'unknown',
            direction_multiplier if direction_multiplier is not None else 1.0,
            order_price_close, order_ts_close, exec_ts_close  
        )

        # Enhanced logging with regime info
        regime_info = ""
        if regime_family:
            regime_info = f" | Regime: {regime_family.upper()} ({regime_multiplier}x)"
        
        logger.info(f"Logged: {symbol} | Profit: {profit:.2f} $ ({profit_pct:+.2f}%){regime_info}")

    except Exception as e:
        logger.error(f"Error-logging to Excel: {e}")
        traceback.print_exc()