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
                               account_number: str,  # ← AÑADIR
                               state_file: str) -> None:

    if strat_id not in strategy_candles:
        strategy_candles[strat_id] = 0
    
    strategy_candles[strat_id] += 1
    save_state_local(open_positions, strategy_candles, account_number, state_file)


def reset_strategy_candles(strat_id: str,
                           strategy_candles: Dict,
                           open_positions: Dict,
                           account_number: str,  # ← AÑADIR
                           state_file: str) -> None:

    strategy_candles[strat_id] = 0
    save_state_local(open_positions, strategy_candles, account_number, state_file)


# ==========================================================================
# TIMEOUT CHECKING
# ==========================================================================
def check_candles_timeout_for_strategy(strat_id: str,
                                       sell_after_ncandles: int,
                                       open_positions: Dict,
                                       strategy_candles: Dict,
                                       account_number: str,  # ← AÑADIR
                                       state_file: str,
                                       send_request_func,
                                       bot_state=None) -> None:

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
        save_state_local(open_positions, strategy_candles, account_number, state_file)
