"""
market_regime/config.py
Configuration for regime-based position sizing in BOT_trading.
"""

# =============================================================================
# BTC SETTINGS
# =============================================================================
BTC_SYMBOL = 'BTCUSDT'
LOOKBACK_BARS = 200  # Bars to fetch for metric calculation

# =============================================================================
# METRIC WINDOWS
# =============================================================================
HURST_WINDOW = 100
ER_WINDOW = 14
ATR_WINDOW = 14
PE_WINDOW = 50
PE_ORDER = 3

# =============================================================================
# FAMILY CLASSIFICATION THRESHOLDS
# =============================================================================
# Order matters: first match wins. 'ranging' is default (empty rules).
# =============================================================================

FAMILIES = {
    'trending': {
        'hurst': ('>', 0.55),
        'efficiency_ratio': ('>', 0.4)
    },
    'volatile': {
        'atr_pct': ('>', 2.0),
        'permutation_entropy': ('>', 0.8)
    },
    'ranging': {}  # Default: catches everything else
}

# =============================================================================
# POSITION SIZING MULTIPLIERS
# =============================================================================
# Multipliers applied to base order_amount based on regime family
# =============================================================================

FAMILY_SIZING = {
    'trending': 1.5,   # 50% larger positions in trending markets
    'volatile': 0.5,   # 50% smaller positions in volatile markets  
    'ranging': 1.0,    # Normal size in ranging markets
}

# =============================================================================
# LOGGING
# =============================================================================
LOG_REGIME_DECISIONS = True  # Log every regime classification
