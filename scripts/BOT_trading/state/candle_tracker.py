"""
Candle Tracker - Manages candle counting and position timeouts.

This module handles:
- Incrementing candle counters for strategies
- Resetting candle counters
- Checking for position timeouts based on candle count

This module imports from order_manager for closing timed-out positions,
but order_manager does not import back, avoiding circular dependencies.
"""
from typing import Dict

from execution.order_manager import close_position
from state.state_manager import save_state_local

import logging
logger = logging.getLogger('BOT_trading.execution.candle_tracker')


# ==========================================================================
# CANDLE COUNTING
# ==========================================================================
def increment_strategy_candles(strat_id: str,
                               strategy_candles: Dict,
                               open_positions: Dict,
                               state_file: str) -> None:
    """
    Increment candle counter for a strategy.
    
    This is called when a new candle closes for the strategy's timeframe.
    The counter is used to track how long positions have been open.
    
    Args:
        strat_id: Strategy ID
        strategy_candles: Dictionary of candle counters
        open_positions: Dictionary of open positions
        state_file: Path to state file
    
    Example:
        >>> increment_strategy_candles('01_double_top_long_4H', candles, positions, 'bot_state.json')
        # Increments counter from 5 to 6
    """
    if strat_id not in strategy_candles:
        strategy_candles[strat_id] = 0
    
    strategy_candles[strat_id] += 1
    save_state_local(open_positions, strategy_candles, state_file)


def reset_strategy_candles(strat_id: str,
                           strategy_candles: Dict,
                           open_positions: Dict,
                           state_file: str) -> None:
    """
    Reset candle counter for a strategy to zero.
    
    This is called when all positions are closed or when
    starting fresh tracking.
    
    Args:
        strat_id: Strategy ID
        strategy_candles: Dictionary of candle counters
        open_positions: Dictionary of open positions
        state_file: Path to state file
    
    Example:
        >>> reset_strategy_candles('01_double_top_long_4H', candles, positions, 'bot_state.json')
        # Resets counter to 0
    """
    strategy_candles[strat_id] = 0
    save_state_local(open_positions, strategy_candles, state_file)


# ==========================================================================
# TIMEOUT CHECKING
# ==========================================================================
def check_candles_timeout_for_strategy(strat_id: str,
                                       sell_after_ncandles: int,
                                       open_positions: Dict,
                                       strategy_candles: Dict,
                                       state_file: str,
                                       send_request_func,
                                       bot_state=None) -> None:
    """
    Close all positions of a strategy if candle timeout is reached.
    
    This function checks if the elapsed candle count has exceeded
    the configured limit, and closes all positions if so. This is
    a safety mechanism to prevent positions from staying open
    indefinitely.
    
    Args:
        strat_id: Strategy ID
        sell_after_ncandles: Maximum candles before closing
        open_positions: Dictionary of open positions
        strategy_candles: Dictionary of candle counters
        state_file: Path to state file
        send_request_func: Function to send REST requests
        bot_state: Bot state for profit tracking
    
    Example:
        >>> check_candles_timeout_for_strategy(
        ...     '01_double_top_long_4H', 
        ...     10,  # Max 10 candles (40 hours for 4H timeframe)
        ...     positions, 
        ...     candles, 
        ...     'bot_state.json',
        ...     send_request
        ... )
        TIMEOUT REACHED for 01_double_top_long_4H
        Candles: 10/10
        Closing 2 positions...
    """
    candles_elapsed = strategy_candles.get(strat_id, 0)
    
    # Check if timeout not reached
    if candles_elapsed < sell_after_ncandles:
        return

    # Check if strategy has positions
    if strat_id not in open_positions or not open_positions[strat_id]:
        return

    positions = open_positions[strat_id][:]

    if not positions:
        return

    logger.info(f"TIMEOUT REACHED for {strat_id}")
    logger.info(f"Candles: {candles_elapsed}/{sell_after_ncandles}")
    logger.info(f"Closing {len(positions)} positions...")

    # Close all positions
    all_closed = True
    for pos in positions:
        position_data = {
            'opened_at': pos['opened_at'],
            'strategy_id': strat_id,
            'usdt_amount': pos.get('usdt_amount', 0),
            'entry_price': pos['entry_price']
        }
        
        if not close_position(
            pos['symbol'], 
            pos['size'], 
            pos['direction'],
            send_request_func, 
            reason="TIMEOUT", 
            position_data=position_data,
            bot_state=bot_state
        ):
            all_closed = False

    # Reset state if all closed successfully
    if all_closed:
        open_positions[strat_id] = []
        strategy_candles[strat_id] = 0
        save_state_local(open_positions, strategy_candles, state_file)
