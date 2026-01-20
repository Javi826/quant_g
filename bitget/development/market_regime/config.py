"""
market_regime/config.py
Centralized configuration for regime-based position sizing.
"""
import os
# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Input: folder with trades files
TRADES_FOLDER = os.path.join(BASE_DIR, 'brief_trades_2025')
# Input: pattern to match trades files (glob pattern)
TRADES_PATTERN = 'all_trades_*.xlsx'
# Input: OHLC folder with BTC parquet
OHLC_FOLDER = os.path.join(BASE_DIR, 'data', 'crypto_OOS_2025')
# Output: folder for enriched trades
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'market_regime', 'output_2025')
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
# Order matters: first match wins. 'ranging' should be last (default).
# =============================================================================
# FAMILIES = {
#    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
#     'volatile': {'atr_pct': ('>', 2.0)},
#     'ranging': {},  # Default: everything else
# }
# =============================================================================

FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging': {}
}
# =============================================================================
# POSITION SIZING MULTIPLIERS
# =============================================================================
FAMILY_SIZING = {
    'trending': 1.0,
    'volatile': 1.0,
    'ranging': 1.0,
}

# =============================================================================
# DIRECTION SIZING MULTIPLIERS (based on BTC price vs MA)
# =============================================================================
# Applied based on BTC trend and trade direction:
#   - uptrend: price > selected MA
#   - downtrend: price <= selected MA
# Set all to 1.0 to disable direction filtering (backward compatible)

# Choose which MA to use for trend detection: 'ma_20', 'ma_50', or 'ma_200'
DIRECTION_MA_REFERENCE = 'ma_50'  # Options: 'ma_20', 'ma_50', 'ma_200'

DIRECTION_SIZING = {
    'long': {
        'uptrend': 1.0,      # Multiplier for long trades when price > MA
        'downtrend': 0     # Multiplier for long trades when price <= MA
    },
    'short': {
        'uptrend': 0,      # Multiplier for short trades when price > MA
        'downtrend': 1.0     # Multiplier for short trades when price <= MA
    }
}

# =============================================================================
# DIRECTION DETECTION METHOD
# =============================================================================
# Method for detecting market direction:
#   'price_vs_ma': Compare current price vs single MA
#   'ma_cross': Compare two moving averages (e.g., MA50 vs MA200)
DIRECTION_METHOD = 'price_vs_ma'  # Options: 'price_vs_ma', 'ma_cross'

# For 'price_vs_ma' method:
DIRECTION_MA_PERIOD = 50  # Compare price vs this MA period

# For 'ma_cross' method:  
DIRECTION_MA_FAST = 50   # Fast MA period
DIRECTION_MA_SLOW = 200  # Slow MA period

# =============================================================================
# CAPITAL
# =============================================================================
INITIAL_CAPITAL = 800

# =============================================================================
# DATE RANGE FILTER (optional)
# =============================================================================
# Filter trades by date range for split testing
# Format: tuple of (start_date, end_date) as strings 'YYYY-MM-DD'
# Examples:
DATE_RANGE_FILTER = ('2025-01-01', '2025-06-30')  # H1
#DATE_RANGE_FILTER = ('2025-07-01', '2025-12-31')  # H2
#DATE_RANGE_FILTER = None