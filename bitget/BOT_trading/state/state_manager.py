"""
State Manager - Handles bot state persistence and broker synchronization.

This module is responsible for:
- Loading bot state from JSON file
- Saving bot state to JSON file
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

import logging
logger = logging.getLogger('BOT_trading.execution.state_manager')


# ==========================================================================
# STATE PERSISTENCE
# ==========================================================================
def load_state(state_file: str) -> Tuple[Dict, Dict]:
    """
    Load bot state from JSON file.
    
    This function reads the JSON state file and reconstructs:
    - Open positions with proper Decimal types
    - Strategy candle counters
    
    Args:
        state_file: Path to state JSON file
    
    Returns:
        Tuple of (open_positions, strategy_candles) dictionaries
    
    Example:
        >>> positions, candles = load_state('bot_state_E1.json')
        >>> print(f"Loaded {sum(len(p) for p in positions.values())} positions")
    """
    
    OPEN_POSITIONS = {}
    STRATEGY_CANDLES = {}
    
    logger.info(f"Loading BOT state...")
    
    if not os.path.exists(state_file):
        logger.info(f"No previous state file found")
        logger.info(f"{'=' * 20}\n")
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
                    'regime_multiplier': float(pos.get('regime_multiplier', 1.0))
                })
        
        total_positions = sum(len(p) for p in OPEN_POSITIONS.values())
        
        logger.info(f"Total positions recovered: {total_positions}")
        logger.info(f"{'-' * 48}")
        
        # Display summary
        for strat_id, positions in OPEN_POSITIONS.items():
            if positions:
                candles = STRATEGY_CANDLES.get(strat_id, 0)
                logger.info(f"{strat_id:<24}: {len(positions):>2} positions | Candles: {candles:>2}")
        
        logger.info(f"State loaded successfully")
        
        return OPEN_POSITIONS, STRATEGY_CANDLES
        
    except Exception as e:
        logger.error(f"Error-loading state: {e}")
        traceback.print_exc()
        logger.info(f"{'=' * 120}\n")
        return OPEN_POSITIONS, STRATEGY_CANDLES


def save_state_local(open_positions: Dict, 
                     strategy_candles: Dict, 
                     state_file: str) -> None:
    """
    Save bot state to JSON file.
    
    This function serializes the current bot state including:
    - All open positions (converting Decimal to string)
    - Candle counters for each strategy
    
    Args:
        open_positions: Dictionary of open positions by strategy
        strategy_candles: Dictionary of candle counters by strategy
        state_file: Path to state JSON file
    
    Example:
        >>> save_state_local(positions, candles, 'bot_state_E1.json')
    """
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
                    'regime_multiplier': float(pos.get('regime_multiplier', 1.0))
                })

        state_data = {
            'positions': serializable_positions,
            'strategy_candles': strategy_candles_copy
        }

        # Write to file
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error-saving state: {e}")
        traceback.print_exc()


# ==========================================================================
# BROKER SYNCHRONIZATION
# ==========================================================================
def sync_broker(open_positions: Dict, 
                strategy_candles: Dict, 
                state_file: str) -> None:
    """
    Synchronize local positions with broker via WebSocket.
    
    This function:
    1. Refreshes WebSocket position data
    2. Checks each local position against broker
    3. Removes positions that no longer exist in broker
    4. Logs removed positions as NOT_FOUND
    
    This is critical for recovering from:
    - Manual closes in broker interface
    - Liquidations
    - Stop losses triggered by broker
    
    Args:
        open_positions: Dictionary of open positions
        strategy_candles: Dictionary of candle counters
        state_file: Path to state file
    
    Example:
        >>> sync_broker(positions, candles, 'bot_state_E1.json')
        Position BTCUSDT doesn't exist in broker - treating as SL
        Sync with broker completed: 1 position(s) removed
    """
    total_removed = 0
    
    if not get_ws_manager():
        raise RuntimeError("WebSocket manager not init.")
    
    # Refresh WebSocket position data
    get_ws_manager().refresh_positions()
    
    # Check each strategy's positions
    for strat_id, positions in list(open_positions.items()):
        positions_to_remove = []
        
        for i, pos in enumerate(positions):
            try:
                symbol = pos['symbol']
                
                # Get position from WebSocket
                ws_position = get_ws_manager().get_position(symbol)
                
                # Check if position exists
                position_exists = False
                if ws_position:
                    total_size = float(ws_position.get('total', 0))
                    position_exists = (total_size > 0)
                    
                    # Debug info if position is closed
                    if not position_exists:
                        logger.info(f"{symbol}: total={total_size} (position closed)")
                
                # If position doesn't exist, mark for removal
                if not position_exists:
                    logger.info(f"Position {symbol} doesn't exist in broker - treating as SL")
                    
                    sl_price = pos['sl']
                    
                    position_data = {
                        'opened_at': pos['opened_at'],
                        'strategy_id': strat_id,
                        'usdt_amount': pos.get('usdt_amount', 0),
                        'entry_price': pos['entry_price']
                    }
                    
                    log_closed_position(
                        opened_at=position_data['opened_at'],
                        strategy_id=position_data['strategy_id'],
                        symbol=symbol,
                        direction=pos['direction'],
                        usdt_amount=position_data['usdt_amount'],
                        entry_price=position_data['entry_price'],
                        close_price=sl_price,
                        reason="NOT_FOUND",
                        size=pos['size'],
                        profit_from_api=None,
                        fee_from_api=None
                    )
                    
                    positions_to_remove.append(i)
                    total_removed += 1
                
            except Exception as e:
                logger.error(f"Error checking {pos['symbol']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Remove positions that don't exist
        for i in reversed(positions_to_remove):
            if i < len(open_positions[strat_id]):
                open_positions[strat_id].pop(i)
        
        # Reset candle counter if no positions left
        if not open_positions[strat_id]:
            if strategy_candles.get(strat_id, 0) > 0:
                strategy_candles[strat_id] = 0
    
    # Save state if changes were made
    if total_removed > 0:
        save_state_local(open_positions, strategy_candles, state_file)
        logger.info(f"Sync with broker completed: {total_removed} position(s) removed")
    else:
        logger.info(f"Sync with broker completed.")
