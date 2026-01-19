"""
Strategies Module
Provides strategy processing, signal detection, and configuration loading.
This module exports the main components needed for strategy execution:
- StrategyProcessor: Main class for processing strategies
- IMPLEMENTED_STRATEGIES: Set of all implemented strategies
- Strategy loading functions from YAML
"""
from .strategy_processor import StrategyProcessor
from .strategy_registry import IMPLEMENTED_STRATEGIES, detect_signals_for_strategy, get_implemented_strategies
from .strategy_loader import (
    load_strategies,
    load_strategies_from_yaml,
    apply_set_active_argument,
    get_all_strategy_ids
)

__all__ = [
    # Core processing
    'StrategyProcessor',
    
    # Strategy registry (from strategy_registry.py)
    'IMPLEMENTED_STRATEGIES',
    'detect_signals_for_strategy',
    'get_implemented_strategies',
    
    # Strategy loading (YAML)
    'load_strategies',
    'load_strategies_from_yaml',
    'apply_set_active_argument',
    'get_all_strategy_ids'
]