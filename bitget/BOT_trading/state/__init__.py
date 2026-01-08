"""
State module - Handles bot state persistence and synchronization.

This module manages:
- Loading and saving bot state to JSON
- Synchronizing with broker
- Candle counting for strategies
- Timeout checking
"""

from .state_manager import (
    load_state,
    save_state_local,
    sync_broker
)

from .candle_tracker import (
    increment_strategy_candles,
    reset_strategy_candles,
    check_candles_timeout_for_strategy
)

__all__ = [
    # State management
    'load_state',
    'save_state_local',
    'sync_broker',
    
    # Candle tracking
    'increment_strategy_candles',
    'reset_strategy_candles',
    'check_candles_timeout_for_strategy',
]
