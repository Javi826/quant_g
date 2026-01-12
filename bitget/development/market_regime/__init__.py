"""
market_regime - Regime-based position sizing module

Usage:
    1. Edit config.py with your paths and settings
    2. Run trade_enricher.py to add BTC metrics to trades
    3. Run position_sizer.py to apply sizing and see results
"""

from .config import (
    TRADES_FOLDER, TRADES_PATTERN, OHLC_FOLDER, OUTPUT_FOLDER,
    BTC_SYMBOL, LOOKBACK_BARS,
    FAMILIES, FAMILY_SIZING, INITIAL_CAPITAL
)
from .regime_metrics import calc_all_metrics
from .trade_enricher import enrich_all_trades
from .position_sizer import apply_sizing

__all__ = [
    'enrich_all_trades',
    'apply_sizing',
    'calc_all_metrics',
    'FAMILIES',
    'FAMILY_SIZING'
]