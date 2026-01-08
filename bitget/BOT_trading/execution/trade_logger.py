"""
Trade Logger - Handles trade logging to Excel.

This module is responsible for:
- Logging closed positions to Excel file
- Calculating profit and fees
- Managing trade history

This module is intentionally separated from order_manager and position_tracker
to avoid circular dependencies. It does not import from any execution submodules.
"""

import os
import pandas as pd
import traceback
from datetime import datetime
from decimal import Decimal
from typing import Optional

import logging
logger = logging.getLogger('BOT_trading.execution.trade_logger')


# Global configuration
TRADES_LOG_PATH = None


def configure_log_path(trades_log_path: str) -> None:
    """
    Configure path for trade logging Excel file.
    
    Args:
        trades_log_path: Path to Excel file for trade logging
    """
    global TRADES_LOG_PATH
    TRADES_LOG_PATH = trades_log_path


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
                       bot_state=None) -> None:
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

        # Build record
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
            'REASON_OUT': reason
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

        logger.info(f"Logged: {symbol} | Profit: {profit:.2f} $ ({profit_pct:+.2f}%)")

    except Exception as e:
        logger.error(f"Error-logging to Excel: {e}")
        traceback.print_exc()
