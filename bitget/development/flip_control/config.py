"""
flip_control/config.py
Configuration for flip detection and partial closing simulation.
"""
import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: enriched trades from market_regime
ENRICHED_TRADES_FOLDER = os.path.join(BASE_DIR, 'market_regime', 'output_2025')
ENRICHED_TRADES_PATTERN = 'trades_enriched_*.xlsx'

# Input: BTC OHLC for flip detection (same as market_regime)
BTC_OHLC_FOLDER = os.path.join(BASE_DIR, 'data', 'crypto_OOS_2025')

# Input: High-resolution OHLC for precise flip prices (15m data)
OHLC_FOLDER_15M = os.path.join(BASE_DIR, 'data', 'crypto_2026_BTC')

# =============================================================================
# BTC SETTINGS
# =============================================================================
BTC_SYMBOL = 'BTCUSDT'
MA_PERIOD = 50  # Moving average for flip detection

# =============================================================================
# FLIP DETECTION PARAMETERS
# =============================================================================
# Confirmation bars: 0=immediate flip, 1-2=wait N consecutive bars
FLIP_CONFIRMATION_BARS = 0

# Distance threshold: 0.0=disabled, 1.0=need 1% away from MA50
# Example: 1.0 means price must be >1% above/below MA50 to confirm flip
FLIP_DISTANCE_PCT = 0.0

# =============================================================================
# PARTIAL CLOSING PARAMETERS
# =============================================================================
# Percentage to close when flip detected: 0.0-1.0
# 0.0 = no closing (test mode, should match original)
# 0.5 = close 50% at flip, keep 50% until real exit
# 1.0 = close 100% at flip (full liquidation)
PARTIAL_CLOSE_PCT = 1.0

# =============================================================================
# CAPITAL (for equity curve calculation)
# =============================================================================
INITIAL_CAPITAL = 800

# =============================================================================
# DATE RANGE FILTER (optional, same as market_regime)
# =============================================================================
DATE_RANGE_FILTER = None
# Example: DATE_RANGE_FILTER = ('2025-01-01', '2025-12-31')