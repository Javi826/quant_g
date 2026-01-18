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
# =============================================================================
# FAMILIES = {
#     'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
#     'volatile': {'atr_pct': ('>', 1.5), 'permutation_entropy': ('>', 0.1)},
#     'ranging': {}
# }
# =============================================================================
FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 1.5), 'permutation_entropy': ('>', 0.1)},
    'ranging': {}
}
# =============================================================================
# POSITION SIZING MULTIPLIERS
# =============================================================================
FAMILY_SIZING = {
    'trending': 1.8,
    'volatile': 0,
    'ranging': 1.0,
}

# =============================================================================
# DIRECTION SIZING MULTIPLIERS (based on BTC MA50 vs MA200)
# =============================================================================
# Applied based on BTC trend and trade direction:
#   - uptrend: MA50 > MA200
#   - downtrend: MA50 <= MA200
# Set all to 1.0 to disable direction filtering (backward compatible)
DIRECTION_SIZING = {
    'long': {
        'uptrend': 1.2,      # Multiplier for long trades when MA50 > MA200
        'downtrend': 0     # Multiplier for long trades when MA50 <= MA200
    },
    'short': {
        'uptrend': 0,      # Multiplier for short trades when MA50 > MA200
        'downtrend': 1.2     # Multiplier for short trades when MA50 <= MA200
    }
}

# =============================================================================
# CAPITAL
# =============================================================================
INITIAL_CAPITAL = 800