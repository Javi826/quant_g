"""
market_regime module - Regime-based position sizing for BOT_trading

This module provides:
- Regime metrics calculation (Hurst, ER, ATR%, Permutation Entropy)
- Market regime classification (trending, ranging, volatile)
- Position sizing multipliers based on current regime

Usage:
    from market_regime.regime_classifier import get_regime_multiplier
    
    multiplier = get_regime_multiplier('BTCUSDT', '4H')
    adjusted_size = base_size * multiplier
"""

from market_regime.regime_classifier import (
    get_regime_multiplier,
    get_current_regime,
    get_regime_metrics
)

__all__ = [
    'get_regime_multiplier',
    'get_current_regime',
    'get_regime_metrics'
]
