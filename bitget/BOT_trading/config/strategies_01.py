"""
Trading Strategies Configuration - Account 01

This file defines all trading strategies used by the bot.
Each strategy must have all required parameters defined.

IMPORTANT:
- Strategy IDs must match those in IMPLEMENTED_STRATEGIES
- All strategies must be listed here even if inactive
- Parameter validation happens at bot startup
"""

STRATEGIES = [
    {
        'id': '99_reversal_long_2m',
        'name': 'reversal_long_2m',
        'timeframe': '2m',
        'active': True,
        'direction': 'long',
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 1.0,
        'direction_mode': 'general',
        'sell_after_ncandles': 2,
        'order_amount': 20,
        'lookback': 7,
        'tolerance': 40,
        'ma_period': 25,
        'tp_pct': 2,
        'sl_pct': 10
    },
    {
        'id': '07_reversal_short_1H',
        'name': 'reversal_short_1H',
        'timeframe': '1H',
        'active': True,
        'direction': 'short',
        'regime_trending': 0,
        'regime_ranging': 1.5,
        'regime_volatile': 1.0,
        'direction_mode': 'short_only',
        'sell_after_ncandles': 50,
        'order_amount': 80,
        'lookback': 5,
        'tolerance': 30,
        'ma_period': 50,
        'tp_pct': 1.9,
        'sl_pct': 5
    },
]