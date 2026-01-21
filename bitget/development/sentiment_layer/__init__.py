"""
sentiment_layer - Sentiment-based position sizing module
Usage:
    1. Edit config.py with your paths and settings
    2. Run trade_enricher_sentiment.py to add sentiment metrics to trades
    3. Run position_sizer.py to apply sizing and see results
"""
from .config import (
    TRADES_FOLDER, TRADES_PATTERN, SENTIMENT_FOLDER, OUTPUT_FOLDER,
    SENTIMENT_THRESHOLDS, SENTIMENT_SIZING, INITIAL_CAPITAL
)
from .trade_enricher import enrich_all_trades

__all__ = [
    'enrich_all_trades',
    'SENTIMENT_THRESHOLDS',
    'SENTIMENT_SIZING'
]