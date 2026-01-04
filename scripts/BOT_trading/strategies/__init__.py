"""
Strategies Module

Provides strategy processing, signal detection, and configuration loading.
"""

from .strategy_processor import StrategyProcessor
from .registry import STRATEGY_FUNCTIONS, IMPLEMENTED_STRATEGIES, get_strategy_function
from .strategy_loader import (
    load_strategies,
    load_strategies_from_yaml,
    filter_strategies_by_ids,
    apply_set_active_argument,
    get_all_strategy_ids
)

__all__ = [
    # Core processing
    'StrategyProcessor',
    
    # Strategy registry
    'STRATEGY_FUNCTIONS',
    'IMPLEMENTED_STRATEGIES',
    'get_strategy_function',
    
    # Strategy loading (YAML)
    'load_strategies',
    'load_strategies_from_yaml',
    'filter_strategies_by_ids',
    'apply_set_active_argument',
    'get_all_strategy_ids'
]
