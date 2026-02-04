"""
State Manager - Handles bot state persistence and broker synchronization.

This module is responsible for:
- Loading bot state from JSON file
- Saving bot state to JSON file (PRIMARY + dual-write to PostgreSQL)
- Synchronizing local state with broker positions

This module imports from trade_logger for logging removed positions,
but trade_logger does not import back, avoiding circular dependencies.
"""

import os
import json
import copy
import traceback
from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple

from market_data import get_ws_manager
from execution.trade_logger import log_closed_position
from config.settings import POSTGRES_CONFIG
from alerts.telegram_notifier import send_sync_alert
from execution.trade_logger import log_sync_discrepancy

import logging
logger = logging.getLogger('BOT_trading.execution.state_manager')


# ==========================================================================
# POSTGRESQL CONFIGURATION (DUAL-WRITE)
# ==========================================================================
POSTGRES_ENABLED = True
# ==========================================================================
# STATE PERSISTENCE
# ==========================================================================
class BotState:
    def __init__(self):
        self.closed_total_profit = 0.0
        
def load_state(account_number: str, state_file: str) -> Tuple[Dict, Dict]:
    """
    Load bot state from PostgreSQL (primary) with JSON fallback.
    
    Args:
        account_number: Account identifier ('00', 'E1', '01')
        state_file: Path to JSON file (for fallback only)
    
    Returns:
        Tuple of (OPEN_POSITIONS, STRATEGY_CANDLES)
    """
    
    OPEN_POSITIONS = {}
    STRATEGY_CANDLES = {}
    
    logger.info(f"Loading state for account {account_number}...")
    
    # ======================================================================
    # PRIMARY: Try PostgreSQL first
    # ======================================================================
    if POSTGRES_ENABLED:
        try:
            import psycopg2
            
            conn = psycopg2.connect(**POSTGRES_CONFIG)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT state_data FROM bot_state WHERE account = %s",
                (account_number,)
            )
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                # Parse JSON from PostgreSQL
                data = result[0]
                
                STRATEGY_CANDLES = data.get('strategy_candles', {})
                positions_data = data.get('positions', {})
                
                # Reconstruct positions with Decimal types
                for strat_id, positions in positions_data.items():
                    OPEN_POSITIONS[strat_id] = []
                    for pos in positions:
                        OPEN_POSITIONS[strat_id].append({
                            'symbol': pos.get('symbol'),
                            'size': Decimal(pos.get('size')),
                            'entry_price': Decimal(pos.get('entry_price')),
                            'direction': pos.get('direction'),
                            'tp': Decimal(pos.get('tp')),
                            'sl': Decimal(pos.get('sl')),
                            'order_id': pos.get('order_id'),
                            'opened_at': datetime.fromisoformat(pos.get('opened_at')),
                            'usdt_amount': float(pos.get('usdt_amount', 0)),
                            'regime_family': pos.get('regime_family', 'unknown'),
                            'regime_multiplier': float(pos.get('regime_multiplier', 1.0)),
                            'market_direction': pos.get('market_direction', 'unknown'),
                            'direction_multiplier': float(pos.get('direction_multiplier', 1.0))
                        })
                
                total_positions = sum(len(p) for p in OPEN_POSITIONS.values())
                
                logger.info(f"State loaded from PostgreSQL: {total_positions} positions")
                logger.info(f"{'-' * 48}")
                
                # Display summary
                for strat_id, positions in OPEN_POSITIONS.items():
                    if positions:
                        candles = STRATEGY_CANDLES.get(strat_id, 0)
                        logger.info(f"{strat_id:<24}: {len(positions):>2} positions | Candles: {candles:>2}")
                
                return OPEN_POSITIONS, STRATEGY_CANDLES
            else:
                logger.warning(f"No state found in PostgreSQL for account {account_number}")
                # Fall through to JSON fallback
                
        except Exception as e:
            logger.error(f"Error-PostgreSQL load failed: {e}")
            logger.info("Falling back to JSON...")
            # Fall through to JSON fallback
    
    # ======================================================================
    # FALLBACK: Load from JSON
    # ======================================================================
    logger.info(f"Loading from JSON fallback: {state_file}")
    
    if not os.path.exists(state_file):
        logger.info(f"No previous state file found")
        return OPEN_POSITIONS, STRATEGY_CANDLES
    
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
        
        STRATEGY_CANDLES = data.get('strategy_candles', {})
        positions_data = data.get('positions', {})
        
        # Reconstruct positions with Decimal types
        for strat_id, positions in positions_data.items():
            OPEN_POSITIONS[strat_id] = []
            for pos in positions:
                OPEN_POSITIONS[strat_id].append({
                    'symbol': pos.get('symbol'),
                    'size': Decimal(pos.get('size')),
                    'entry_price': Decimal(pos.get('entry_price')),
                    'direction': pos.get('direction'),
                    'tp': Decimal(pos.get('tp')),
                    'sl': Decimal(pos.get('sl')),
                    'order_id': pos.get('order_id'),
                    'opened_at': datetime.fromisoformat(pos.get('opened_at')),
                    'usdt_amount': float(pos.get('usdt_amount', 0)),
                    'regime_family': pos.get('regime_family', 'unknown'),
                    'regime_multiplier': float(pos.get('regime_multiplier', 1.0)),
                    'market_direction': pos.get('market_direction', 'unknown'),
                    'direction_multiplier': float(pos.get('direction_multiplier', 1.0))
                })
        
        total_positions = sum(len(p) for p in OPEN_POSITIONS.values())
        
        logger.info(f"✓ State loaded from JSON: {total_positions} positions")
        logger.info(f"{'-' * 48}")
        
        # Display summary
        for strat_id, positions in OPEN_POSITIONS.items():
            if positions:
                candles = STRATEGY_CANDLES.get(strat_id, 0)
                logger.info(f"{strat_id:<24}: {len(positions):>2} positions | Candles: {candles:>2}")
        
        return OPEN_POSITIONS, STRATEGY_CANDLES
        
    except Exception as e:
        logger.error(f"✗ Error loading state from JSON: {e}")
        traceback.print_exc()
        return OPEN_POSITIONS, STRATEGY_CANDLES


def save_state_local(
    open_positions: Dict,
    strategy_candles: Dict,
    account_number: str,
    state_file: str
) -> None:

    try:
        # Deep copy to avoid modifying original
        positions_copy = copy.deepcopy(open_positions)
        strategy_candles_copy = copy.deepcopy(strategy_candles)

        # Convert Decimal and datetime to serializable types
        serializable_positions = {}
        for strat_id, positions in positions_copy.items():
            serializable_positions[strat_id] = []
            for pos in positions:
                serializable_positions[strat_id].append({
                    'symbol': pos['symbol'],
                    'size': str(pos['size']),
                    'entry_price': str(pos['entry_price']),
                    'direction': pos['direction'],
                    'tp': str(pos['tp']),
                    'sl': str(pos['sl']),
                    'order_id': pos['order_id'],
                    'opened_at': pos['opened_at'].isoformat(),
                    'usdt_amount': float(pos.get('usdt_amount', 0)),
                    'regime_family': pos.get('regime_family', 'unknown'),
                    'regime_multiplier': float(pos.get('regime_multiplier', 1.0)),
                    'market_direction': pos.get('market_direction', 'unknown'),
                    'direction_multiplier': float(pos.get('direction_multiplier', 1.0)) 
                })

        state_data = {
            'positions': serializable_positions,
            'strategy_candles': strategy_candles_copy
        }

        # ======================================================================
        # DUAL-WRITE: JSON (PRIMARY) + PostgreSQL (BACKUP)
        # ======================================================================
        
        json_success = False
        postgres_success = False
        
        # Write to JSON (primary)
        try:
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            json_success = True
        except Exception as e:
            logger.error(f"✗ JSON state save failed: {e}")
        
        # Write to PostgreSQL (dual-write backup)
        if POSTGRES_ENABLED:
            try:
                import psycopg2
                from psycopg2 import sql
                
                # Use account_number parameter directly (no extraction needed)
                account = account_number
                
                # Connect and upsert
                conn = psycopg2.connect(**POSTGRES_CONFIG)
                cursor = conn.cursor()
                
                upsert_query = sql.SQL("""
                    INSERT INTO bot_state (account, state_data, updated_at)
                    VALUES (%(account)s, %(state_data)s, CURRENT_TIMESTAMP)
                    ON CONFLICT (account)
                    DO UPDATE SET
                        state_data = EXCLUDED.state_data,
                        updated_at = CURRENT_TIMESTAMP
                """)
                
                cursor.execute(upsert_query, {
                    'account': account,
                    'state_data': json.dumps(state_data)
                })
                
                conn.commit()
                cursor.close()
                conn.close()
                
                postgres_success = True
            except Exception as e:
                logger.error(f"Error-PostgreSQL state save failed: {e}")
        
        # Log dual-write status (only if verbose)
        status_indicators = []
        if postgres_success:
            status_indicators.append("PG✓")
        if json_success:
            status_indicators.append("JSON✓")
        
        if status_indicators:
            total_positions = sum(len(p) for p in open_positions.values())
            logger.debug(f"[{' '.join(status_indicators)}] State saved: {total_positions} positions")
            
    except Exception as e:
        logger.error(f"Error-saving state: {e}")
        traceback.print_exc()


# ==========================================================================
# BROKER SYNCHRONIZATION (READ-ONLY MONITORING)
# ==========================================================================
def sync_broker(open_positions: Dict, 
                strategy_candles: Dict,
                account_number: str,
                state_file: str) -> None:
    """
    Monitor and alert on discrepancies between local state and broker.
    READ-ONLY: Does not modify state or log trades.
    
    Args:
        open_positions: Current open positions dict
        strategy_candles: Candle counters dict
        account_number: Account identifier
        state_file: State file path (unused, kept for signature compatibility)
    """
    from alerts.telegram_notifier import send_sync_alert
    
    total_issues = 0
    
    if not get_ws_manager():
        raise RuntimeError("WebSocket manager not init.")
    
    # Refresh WebSocket position data
    get_ws_manager().refresh_positions()
    
    # Check each strategy's positions
    for strat_id, positions in open_positions.items():
        for pos in positions:
            try:
                symbol = pos['symbol']
                direction = pos['direction']
                local_size = pos['size']
                
                # Get position from WebSocket
                ws_position = get_ws_manager().get_position(symbol)
                
                # Check if position exists
                position_exists = False
                broker_size = 0.0
                
                if ws_position:
                    total_size = float(ws_position.get('total', 0))
                    broker_size = total_size
                    position_exists = (total_size > 0)
                    
                    # Debug info if position is closed
                    if not position_exists:
                        logger.info(f"{symbol}: total={total_size} (position closed)")
                
                # Alert if position doesn't exist in broker
                if not position_exists:
                    logger.warning(
                        f"[SYNC] Position NOT FOUND in broker: {symbol} {direction} "
                        f"| Strategy: {strat_id} | Local size: {local_size}"
                    )
                    
                    send_sync_alert(
                        account=account_number,
                        symbol=symbol,
                        issue_type="NOT_IN_BROKER",
                        local_size=local_size,
                        broker_size=0.0,
                        strategies=[strat_id]
                    )
                    
                    total_issues += 1
                
                # Alert if sizes don't match (position exists but different size)
                elif abs(broker_size - local_size) > 0.0001:
                    logger.warning(
                        f"[SYNC] SIZE MISMATCH: {symbol} {direction} "
                        f"| Strategy: {strat_id} | Local: {local_size} | Broker: {broker_size}"
                    )
                    
                    send_sync_alert(
                        account=account_number,
                        symbol=symbol,
                        issue_type="SIZE_MISMATCH",
                        local_size=local_size,
                        broker_size=broker_size,
                        strategies=[strat_id]
                    )
                    
                    total_issues += 1
                
            except Exception as e:
                logger.error(f"[SYNC] Error checking {pos['symbol']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary log
    if total_issues > 0:
        logger.warning(f"[SYNC] Broker sync completed: {total_issues} issue(s) detected - MANUAL REVIEW REQUIRED")
    else:
        logger.info(f"[SYNC] Broker sync completed: All positions match")