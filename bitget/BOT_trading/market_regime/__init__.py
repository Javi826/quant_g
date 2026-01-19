"""
Market regime classification and position sizing.

This module provides:
- Regime classification: Detect market state (trending/ranging/volatile)
- Direction detection: Detect trend direction (uptrend/dwtrend)
- Position sizing: Adjust order amounts based on regime/direction alignment
"""

from .regime_classifier import (
    get_current_regime,
    get_current_direction,
    get_regime_multiplier,
    get_regime_info,
)

from .position_sizer import PositionSizer

__all__ = [
    'get_current_regime',
    'get_current_direction',
    'get_regime_multiplier',
    'get_regime_info',
    'PositionSizer',
]