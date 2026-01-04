"""
Strategy Registry - Maps strategy names to implementation functions.

This module provides:
- STRATEGY_FUNCTIONS: Dict mapping strategy names to their functions
- IMPLEMENTED_STRATEGIES: Set of all implemented strategy names

Single source of truth for available strategies.
"""

import sys
import os
import logging

logger = logging.getLogger('BOT_trading.strategies.registry')

# Add scripts directory to path for signal modules
# Get scripts dir dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))  
bot_dir     = os.path.dirname(current_dir)                    
scripts_dir = os.path.dirname(bot_dir)   
                  # scripts/
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import all strategy implementations from external modules
from Z_add_signals_double_top import double_top_long
from Z_add_signals_reversal import reversal_long, reversal_short
from Z_add_signals_parity import parity_long, parity_short
from Z_add_signals_orderblocks import orderblocks_long, orderblocks_short

# Strategy name → implementation function
STRATEGY_FUNCTIONS = {
    'double_top_long_4H': double_top_long,
    'reversal_long_4H': reversal_long,
    'parity_long_4H': parity_long,
    'reversal_short_4H': reversal_short,
    'parity_short_4H': parity_short,
    'reversal_long_1H': reversal_long,
    'reversal_short_1H': reversal_short,
    'reversal_long_6Hutc': reversal_long,
    'reversal_short_6Hutc': reversal_short,
    'parity_long_1H': parity_long,
    'parity_short_1H': parity_short,
    'parity_long_6Hutc': parity_long,
    'orderblocks_short_4H': orderblocks_short,
    'orderblocks_long_4H': orderblocks_long,
}

# Derive set of implemented strategies from registry
IMPLEMENTED_STRATEGIES = set(STRATEGY_FUNCTIONS.keys())


def get_strategy_function(strategy_name: str):
    """
    Get strategy function by name.
    
    Args:
        strategy_name: Name of the strategy
    
    Returns:
        Strategy function
    
    Raises:
        KeyError: If strategy not found in registry
    """
    if strategy_name not in STRATEGY_FUNCTIONS:
        available = ', '.join(sorted(STRATEGY_FUNCTIONS.keys()))
        raise KeyError(
            f"Strategy '{strategy_name}' not found in registry. "
            f"Available strategies: {available}"
        )
    
    return STRATEGY_FUNCTIONS[strategy_name]