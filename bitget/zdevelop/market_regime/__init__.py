"""
market_regime - Market regime analysis module

Usage:
    1. Edit config.py with your paths and settings
    2. Run trade_enricher.py to add BTC metrics to trades
    3. Run regime_performance.py or regime_analyzer.py for analysis
"""

from .config import (
    TRADES_FOLDER, TRADES_PATTERN, OHLC_FOLDER, OUTPUT_FOLDER,
    BTC_SYMBOL, LOOKBACK_BARS,
    FAMILIES, INITIAL_CAPITAL
)
from .regime_metrics import calc_all_metrics
from .trade_enricher import enrich_all_trades

__all__ = [
    'enrich_all_trades',
    'calc_all_metrics',
    'FAMILIES',
    'INITIAL_CAPITAL'
]