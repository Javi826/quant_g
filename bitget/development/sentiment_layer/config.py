"""
sentiment_layer/config.py
Centralized configuration for sentiment-based position sizing.
"""
import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: folder with trades files
TRADES_FOLDER = os.path.join(BASE_DIR, 'sentiment_layer', 'brief_trades_2025')

# Input: pattern to match trades files (glob pattern)
TRADES_PATTERN = 'all_trades_*.xlsx'

# Input: Sentiment folder with fear_greed parquet files
SENTIMENT_FOLDER = os.path.join(BASE_DIR, 'sentiment_layer', 'data_sentiment')

# Output: folder for enriched trades
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'sentiment_layer', 'output_2025')

# =============================================================================
# SENTIMENT SETTINGS
# =============================================================================
LOOKBACK_BARS = 100  # For future sentiment metrics (e.g., rolling averages)

# =============================================================================
# SENTIMENT CLASSIFICATION THRESHOLDS
# =============================================================================
SENTIMENT_THRESHOLDS = {
    'fear': ('<', 0.45),
    'neutral': ('>=', 0.45, '<=', 0.55),
    'greed': ('>', 0.55)
}

# =============================================================================
# SENTIMENT STATE CLASSIFICATION
# =============================================================================
SENTIMENT_STATES = {
    'fear': {'fear_greed_norm': ('<', 0.45)},
    'neutral': {'fear_greed_norm': ('>=', 0.45, '<=', 0.55)},
    'greed': {'fear_greed_norm': ('>', 0.55)}
}

# =============================================================================
# GLOBAL MULTIPLIERS (applied on top of individual strategy configs)
# =============================================================================
GLOBAL_SENTIMENT_MULTIPLIERS = {
    'fear': 1.0,
    'neutral': 1.0,
    'greed': 1.0
}

GLOBAL_DIRECTION_MULTIPLIERS = {
    'long': 1.0,
    'short': 1.0
}

# =============================================================================
# STRATEGY CONFIGURATIONS
# =============================================================================
# Individual strategy configs for sentiment filtering
# To be populated based on statistical analysis

STRATEGY_CONFIGS = {
    # STRATEGY 01: Double Top Long (4H)
    'double_top_long_4H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 02: Reversal Long (4H)
    'reversal_long_4H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 03: Parity Long (4H)
    'parity_long_4H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 04: Reversal Short (4H)
    'reversal_short_4H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 05: Parity Short (4H)
    'parity_short_4H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'short_only',
        'active': True
    },
    
    # STRATEGY 06: Reversal Long (1H)
    'reversal_long_1H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 07: Reversal Short (1H)
    'reversal_short_1H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 08: Reversal Long (6Hutc)
    'reversal_long_6Hutc': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 09: Reversal Short (6Hutc)
    'reversal_short_6Hutc': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 10: Parity Long (1H)
    'parity_long_1H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 11: Parity Short (1H)
    'parity_short_1H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 12: Parity Long (6Hutc)
    'parity_long_6Hutc': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'long_only',
        'active': True
    },
    
    # STRATEGY 13: Order Blocks Short (4H)
    'orderblocks_short_4H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # STRATEGY 16: Ranging Short (6Hutc)
    'ranging_short_6Hutc': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    },
    
    # INACTIVE STRATEGIES
    # STRATEGY 14: Order Blocks Long (4H)
    'orderblocks_long_4H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'long_only',
        'active': False
    },
    
    # STRATEGY 15: Ranging Long (4H)
    'ranging_long_4H': {
        'sentiment_fear': 1.0,
        'sentiment_neutral': 1.0,
        'sentiment_greed': 1.0,
        'direction_mode': 'general',
        'active': True
    }
}

# =============================================================================
# LEGACY CONFIGS (mantener por compatibilidad)
# =============================================================================
SENTIMENT_SIZING = {
    'fear': 1.0,
    'neutral': 1.0,
    'greed': 1.0
}

# =============================================================================
# CAPITAL
# =============================================================================
INITIAL_CAPITAL = 800

# =============================================================================
# DATE RANGE FILTER (optional)
# =============================================================================
DATE_RANGE_FILTER = None