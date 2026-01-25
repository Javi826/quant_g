"""
Strategies Module
Provides strategy processing, signal detection, and configuration loading.
This module exports the main components needed for strategy execution:
- StrategyProcessor: Main class for processing strategies
- IMPLEMENTED_STRATEGIES: Set of all implemented strategies
- Strategy loading functions from Python config
"""
from .strategy_processor import StrategyProcessor
from .strategy_registry import IMPLEMENTED_STRATEGIES
from .strategy_registry import detect_signals_for_strategy
from .strategy_registry import get_implemented_strategies
from .strategy_loader import load_strategies
from .strategy_loader import apply_set_active_argument
from .strategy_loader import get_all_strategy_ids

__all__ = [
    'StrategyProcessor',
    'IMPLEMENTED_STRATEGIES',
    'detect_signals_for_strategy',
    'get_implemented_strategies',
    'load_strategies',
    'apply_set_active_argument',
    'get_all_strategy_ids'
]