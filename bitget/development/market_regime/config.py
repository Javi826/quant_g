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
TRADES_FOLDER = os.path.join(BASE_DIR,'market_regime', 'brief_trades_2025')

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
FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging': {}  # Default: everything else
}

# =============================================================================
# GLOBAL MULTIPLIERS (applied on top of individual strategy configs)
# =============================================================================
# These multipliers are applied to ALL strategies as a global boost/reduction
# Example: If strategy has regime_trending=1.0 and GLOBAL trending=1.5,
#          final multiplier for trending trades = 1.0 * 1.5 = 1.5
#
# Use cases:
# - Boost aggressive strategies: Set trending=1.5, volatile=1.5
# - Reduce risk globally: Set all to 0.5
# - Favor longs vs shorts: Set long=1.5, short=0.8

GLOBAL_REGIME_MULTIPLIERS = {
    'trending': 1.5,   # Multiplier applied to all trending regime trades
    'ranging': 1.5,    # Multiplier applied to all ranging regime trades
    'volatile': 1.5    # Multiplier applied to all volatile regime trades
}

GLOBAL_DIRECTION_MULTIPLIERS = {
    'long': 1.5,    # Multiplier applied to all long direction strategies
    'short': 1.5    # Multiplier applied to all short direction strategies
}

# =============================================================================
# STRATEGY CONFIGURATIONS (from analysis FULL 2025)
# =============================================================================
# Individual strategy configs for regime and direction filtering
# Based on statistical analysis of OOS data

STRATEGY_CONFIGS = {
    # STRATEGY 01: Double Top Long (4H)
    'double_top_long_4H': {
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 1.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 02: Reversal Long (4H)
    'reversal_long_4H': {
        'regime_trending': 0.0,
        'regime_ranging': 1.0,
        'regime_volatile': 0.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 03: Parity Long (4H)
    'parity_long_4H': {
        'regime_trending': 1.0,
        'regime_ranging': 0.0,
        'regime_volatile': 0.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 04: Reversal Short (4H)
    'reversal_short_4H': {
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 05: Parity Short (4H)
    'parity_short_4H': {
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 1.0,
        'direction_mode': 'short_only',
        'active': True
    },
    
    # STRATEGY 06: Reversal Long (1H)
    'reversal_long_1H': {
        'regime_trending': 1.0,
        'regime_ranging': 0.0,
        'regime_volatile': 0.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 07: Reversal Short (1H)
    'reversal_short_1H': {
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 08: Reversal Long (6Hutc)
    'reversal_long_6Hutc': {
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 0.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 09: Reversal Short (6Hutc)
    'reversal_short_6Hutc': {
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 10: Parity Long (1H)
    'parity_long_1H': {
        'regime_trending': 1.0,
        'regime_ranging': 0.0,
        'regime_volatile': 0.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 11: Parity Short (1H)
    'parity_short_1H': {
        'regime_trending': 0.0,
        'regime_ranging': 1.0,
        'regime_volatile': 0.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 12: Parity Long (6Hutc)
    'parity_long_6Hutc': {
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 0.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 13: Order Blocks Short (4H)
    'orderblocks_short_4H': {
        'regime_trending': 0.0,
        'regime_ranging': 1.0,
        'regime_volatile': 0.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 16: Ranging Short (6Hutc)
    'ranging_short_6Hutc': {
        'regime_trending': 1.0,
        'regime_ranging': 1.0,
        'regime_volatile': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # INACTIVE STRATEGIES
    # STRATEGY 14: Order Blocks Long (4H) - DESACTIVADA (profit negativo)
    'orderblocks_long_4H': {
        'regime_trending': 1.0,
        'regime_ranging': 0.0,
        'regime_volatile': 0.0,
        'direction_mode': 'long_only',
        'active': False
    },
    
    # STRATEGY 15: Ranging Long (4H) - DESACTIVADA (profit muy bajo)
    'ranging_long_4H': {
        'regime_trending': 0.0,
        'regime_ranging': 0.0,
        'regime_volatile': 1.0,
        'direction_mode': 'general',
        'active': True
    }
}

# =============================================================================
# LEGACY CONFIGS (mantener por compatibilidad si no existe strategy config)
# =============================================================================
FAMILY_SIZING = {
    'trending': 1.0,
    'volatile': 1.0,
    'ranging': 1.0,
}

# =============================================================================
# DIRECTION DETECTION
# =============================================================================
# Choose which MA to use for trend detection: 'ma_20', 'ma_50', or 'ma_200'
DIRECTION_MA_REFERENCE = 'ma_50'

# Method for detecting market direction
DIRECTION_METHOD = 'price_vs_ma'

# For 'price_vs_ma' method:
DIRECTION_MA_PERIOD = 50

# For 'ma_cross' method:  
DIRECTION_MA_FAST = 50
DIRECTION_MA_SLOW = 200

# =============================================================================
# CAPITAL
# =============================================================================
INITIAL_CAPITAL = 800

# =============================================================================
# DATE RANGE FILTER (optional)
# =============================================================================
DATE_RANGE_FILTER = None