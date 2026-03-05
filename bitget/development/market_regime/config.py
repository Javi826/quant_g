"""
market_regime/config.py
Centralized configuration for market regime analysis.
"""
import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: folder with trades files
TRADES_FOLDER = os.path.join(BASE_DIR, 'market_regime', 'brief_trades_2026')

# Input: pattern to match trades files (glob pattern)
TRADES_PATTERN = 'all_trades_*.xlsx'

# Input: OHLC folder with BTC parquet
OHLC_FOLDER = os.path.join(BASE_DIR, 'data', 'crypto_OOS_2025')

# Output: folder for enriched trades
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'market_regime', 'output_2026')

# =============================================================================
# BTC SETTINGS
# =============================================================================
BTC_SYMBOL    = 'BTCUSDT'
LOOKBACK_BARS = 100

# =============================================================================
# METRIC WINDOWS
# =============================================================================
HURST_WINDOW = 100
ER_WINDOW    = 14
ATR_WINDOW   = 14
PE_WINDOW    = 50
PE_ORDER     = 3

# =============================================================================
# FAMILY CLASSIFICATION THRESHOLDS
# =============================================================================
FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging': {}  # Default: everything else
}

# =============================================================================
# DIRECTION DETECTION
# =============================================================================
# Method for detecting market direction: 'price_vs_ma' or 'ma_cross'
DIRECTION_METHOD = 'price_vs_ma'

# For 'price_vs_ma' method (price vs single MA):
DIRECTION_MA_PERIOD = 50

# For 'ma_cross' method (MA fast vs MA slow):
DIRECTION_MA_FAST = 50
DIRECTION_MA_SLOW = 200

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================
INITIAL_CAPITAL = 800

# Optional: Filter trades by date range (format: ('YYYY-MM-DD', 'YYYY-MM-DD'))
DATE_RANGE_FILTER = None