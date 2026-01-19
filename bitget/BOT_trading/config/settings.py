"""
Bot Configuration Settings

Centralizes all bot configuration including exchange settings,
validation limits, paths, and account-specific settings.
"""

from zoneinfo import ZoneInfo
import os

# ==========================================================================
# ACCOUNT-SPECIFIC SETTINGS
# ==========================================================================

ACCOUNTS = {
    "00": {
        "initial_capital": 3671,
        "dashboard_port": 5000,
        "description": "Main Account"
    },
    "E1": {
        "initial_capital": 1761,
        "dashboard_port": 5001,
        "description": "Elite Account"
    },
    "01": {
        "initial_capital": 117,
        "dashboard_port": 5099,
        "description": "Testing Account"
    }
}

# ==========================================================================
# MARKET REGIME SETTINGS
# ==========================================================================

# BTC settings for regime calculation
REGIME_REFERENCE_SYMBOL = 'BTCUSDT'

# Metric calculation windows
REGIME_HURST_WINDOW = 100
REGIME_ER_WINDOW    = 14
REGIME_ATR_WINDOW   = 14
REGIME_PE_WINDOW    = 50
REGIME_PE_ORDER     = 3

REGIME_FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.5)},
    'ranging': {}
}
REGIME_MATRIX = {
    'trending': {
        'trending': 1.5,   
        'ranging': 1.0,    
        'volatile': 1.0    
    },
    'ranging': {
        'trending': 1.0,   
        'ranging': 1.5,    
        'volatile': 1.0    
    },
    'volatile': {
        'trending': 0.0,  
        'ranging': 0.0,    
        'volatile': 0.0    
    }
}

REGIME_GENERAL = {
    'trending': 1.0,   
    'ranging': 1.0,
    'volatile': 1.0,    
}

# ==========================================================================
# DIRECTION SIZING (BTC TREND FILTER)
# ==========================================================================
DIRECTION_MA_REFERENCE = 'ma_50'

# =============================================================================
# DIRECTION_MATRIX = {
#     'uptrend': {
#         'uptrend': 1.0,
#         'dwtrend': 0
#     },
#     'dwtrend': {
#         'uptrend': 0,
#         'dwtrend': 1.0
#     }
# }
# =============================================================================

DIRECTION_GENERAL = {
    'uptrend': 1.0,
    'dwtrend': 1.0
}

DIRECTION_MATRIX = {
    'long_only': {      # ← KEY cambiada (era 'uptrend')
        'uptrend': 1.0, # ← VALUE sin cambiar (condición real)
        'dwtrend': 0
    },
    'short_only': {     # ← KEY cambiada (era 'dwtrend')
        'uptrend': 0,
        'dwtrend': 1.0  # ← VALUE sin cambiar (condición real)
    }
}

# ==========================================================================
# VALIDATION SETTINGS
# ==========================================================================

# Order amount limits (USDT)
MIN_ORDER_AMOUNT = 40
MAX_ORDER_AMOUNT = 100

# TP/SL limits (%)
MIN_TP_PCT = 1.5
MAX_TP_PCT = 10
MIN_SL_PCT = 1.5
MAX_SL_PCT = 15

# Candles timeout limits
MIN_CANDLES = 49
MAX_CANDLES = 51

# Valid timeframes
VALID_TIMEFRAMES = ['1H', '4H', '6Hutc']

# ==========================================================================
# EXCHANGE SETTINGS
# ==========================================================================
# Bitget API
BASE_URL        = "https://api.bitget.com"
PRODUCT_TYPE    = "USDT-FUTURES"
MARGIN_MODE     = "crossed"  
MARGIN_COIN     = "USDT"
API_LIMIT_DATA  = 180  # Limit for live trading candle fetch

# ==========================================================================
# GENERAL BOT SETTINGS
# ==========================================================================
HOUR_ZONE             = ZoneInfo('UTC')
CHECK_INTERVAL        = 10  
USE_HARDCODED_SIGNALS = False  
PERSISTENCE_DIR       = "persistence"

# ==========================================================================
# API - WEBSOCKET SETTINGS
# ==========================================================================

WS_PUBLIC_URL   = "wss://ws.bitget.com/v2/ws/public"
WS_PRIVATE_URL  = "wss://ws.bitget.com/v2/ws/private"
API_TIMEOUT     = 10  #seconds
API_MAX_RETRIES = 3

# ==========================================================================
# logger SETTINGS
# ==========================================================================
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL    = "INFO"
LOG_MAX_BYTES     = 10 * 1024 * 1024  
LOG_BACKUP_COUNT  = 5
LOG_NAMESPACE     = "BOT_trading"

