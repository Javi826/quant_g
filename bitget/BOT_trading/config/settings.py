"""
config/settings.pyBot Configuration Settings

Centralizes all bot configuration including exchange settings,
validation limits, paths, and account-specific settings.
"""

from zoneinfo import ZoneInfo
DEMO_MODE_ACCOUNTS = ['01']
# ==========================================================================
# ACCOUNT-SPECIFIC SETTINGS
# ==========================================================================

ACCOUNTS = {
    "00": {
        "initial_capital": 6000,
        "dashboard_port": 5000,
        "description": "Main Account"
    },
    "E1": {
        "initial_capital": 24000,
        "dashboard_port": 5001,
        "description": "Elite Account"
    },
    "01": {
        "initial_capital": 12000,
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
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging': {}
}

REGIME_GENERAL = {
    'trending': 1.0,   
    'ranging': 1.0,
    'volatile': 1.0,    
}

GLOBAL_SYSTEM_REGIME_TH1 = 1.00  # SHORT threshold (BTC < MA5 * TH1)
GLOBAL_SYSTEM_REGIME_TH2 = 1.02  # LONG threshold (BTC > MA5 * TH2)

# ==========================================================================
# DIRECTION SIZING (BTC TREND FILTER)
# ==========================================================================

DIRECTION_MATRIX = {
    'long_only': {     
        'uptrend': 1.0, 
        'dwtrend': 0
    },
    'short_only': {     
        'uptrend': 0,
        'dwtrend': 1.0  
    }
}

DIRECTION_GENERAL = {
    'uptrend': 1.0,
    'dwtrend': 1.0
}

# ==========================================================================
# RISK CONTROL SETTINGS
# ==========================================================================
RISK_LIMITS = {
    'max_gross_exposure_pct': 50.0,  
    'max_net_exposure_pct': 20.0     
}
LEVERAGE = 10

# =============================================================================
# QUALITY CONTROL PARAMETERS
# =============================================================================

# Drift detection
DRIFT_WINDOW_SIZE    = 100
DRIFT_CHECK_INTERVAL = 30

# Execution quality
EXECUTION_WINDOW_SIZE = 20
SLIPPAGE_WARNING_PCT  = 0.2
SLIPPAGE_CRITICAL_PCT = 0.3
LATENCY_WARNING_SEC   = 0.5
LATENCY_CRITICAL_SEC  = 1.0

DRIFT_BINOMIAL_WINDOW      = 100
DRIFT_BINOMIAL_DEFAULT_P50 = 0.55 


# ==========================================================================
# STRATEGY VALIDATION CONFIGURATION
# ==========================================================================

# Common parameters required for ALL strategies
COMMON_REQUIRED_PARAMS = [
    'id', 'name', 'timeframe', 'active', 'sell_after_ncandles', 
    'order_amount', 'tp_pct', 'sl_pct', 'direction', 
    'regime_trending', 'regime_ranging', 'regime_volatile', 'direction_mode'
]

# Strategy-specific required parameters by strategy type
STRATEGY_TYPE_REQUIRED_PARAMS = {
    'double_top_long': ['lookback', 'tolerance', 'trend_th'],
    'reversal_long': ['lookback', 'tolerance', 'ma_period'],
    'reversal_short': ['lookback', 'tolerance', 'ma_period'],
    'parity_long': ['lookback', 'tolerance', 'ma_period'],
    'parity_short': ['lookback', 'tolerance', 'ma_period'],
    'orderblocks_long': ['lookback', 'tolerance', 'impulse'],
    'orderblocks_short': ['lookback', 'tolerance', 'impulse'],
    'ranging_long': ['lookback', 'tolerance', 'range'],
    'ranging_short': ['lookback', 'tolerance', 'range'],
    'flag_long': ['lookback', 'impulse', 'flag', 'ma_period'],
    'flag_short': ['lookback', 'impulse', 'flag', 'ma_period'],
}
# Order amount limits (USDT)
MIN_ORDER_AMOUNT = 35
MAX_ORDER_AMOUNT = 180

# TP/SL limits (%)
MIN_TP_PCT = 1.5
MAX_TP_PCT = 10
MIN_SL_PCT = 5
MAX_SL_PCT = 15

# Candles timeout limits
MIN_CANDLES = 50
MAX_CANDLES = 100

# Valid timeframes
VALID_TIMEFRAMES = ['1H', '4H', '6Hutc']

# ==========================================================================
# POSTGRESQL CONFIGURATION
# ==========================================================================
import socket

# Environment detection
HOSTNAME      = socket.gethostname()
VPS_HOSTNAMES = ['srv1326826', 'hstgr.cloud']
IS_VPS        = any(h in HOSTNAME for h in VPS_HOSTNAMES)

POSTGRES_CONFIG = {
    'dbname': 'bot_trading',
    'user': 'javi',
    'password': 'Laplaciano86-',
    'host': 'localhost',
    'port': 5432,
    'connect_timeout': 3
}

# VPS PostgreSQL connection (for split-brain check from LOCAL)
VPS_CHECK_CONFIG = {
    'host': '100.123.10.95',        # Tailscale IP
    'user': 'javi',
    'password': 'Laplaciano86-',
    'dbname': 'bot_trading',
    'timeout': 5                     # seconds
}

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
CHECK_INTERVAL        = 5  
USE_HARDCODED_SIGNALS = False  
PERSISTENCE_DIR       = "persistence"

# ==========================================================================
# API - WEBSOCKET SETTINGS
# ==========================================================================

WS_PUBLIC_URL   = "wss://ws.bitget.com/v2/ws/public"
WS_PRIVATE_URL  = "wss://ws.bitget.com/v2/ws/private"
API_TIMEOUT     = 10  
API_MAX_RETRIES = 3

# ==========================================================================
# logger SETTINGS
# ==========================================================================
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL    = "INFO"
LOG_MAX_BYTES     = 10 * 1024 * 1024  
LOG_BACKUP_COUNT  = 5
LOG_NAMESPACE     = "BOT_trading"

