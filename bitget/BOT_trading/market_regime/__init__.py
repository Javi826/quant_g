"""
Market regime classification and position sizing.

This module provides:
- Regime classification: Detect market state (trending/ranging/volatile)
- Direction detection: Detect trend direction (uptrend/dwtrend)
- Position sizing: Adjust order amounts based on regime/direction alignment
- Daily filter: Global BTC 1D filter for trade direction validation
"""

from .regime_classifier import (
    get_current_regime,
    get_btc_1d_direction,
    get_regime_info,
)
from .position_sizer import PositionSizer

__all__ = [
    'get_current_regime',
    'get_btc_1d_direction',
    'get_regime_info',
    'PositionSizer',
]