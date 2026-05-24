#BOT_trading/config/settings.py
"""
Centralizes all bot configuration including exchange settings,
validation limits, paths, and account-specific settings.
"""
import sys, os
import socket
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from zoneinfo import ZoneInfo

# ==========================================================================
# ACCOUNT-SPECIFIC SETTINGS
# ==========================================================================

ACCOUNTS = {
    "E1": {
        "initial_capital": 12000,
        "dashboard_port": 5001,
        "description": "Elite Account",
        "type": "production",
        "regime01_enabled": False,
        "risk_control_enabled": True,
        "postgresql_enabled": True,
        "regime01_ma_period": 2,
        "regime01_long_th":   1.00,
        "regime01_short_th":  1.00,
        "regime_reference_symbol": "BTCUSDT",
    },
    "00": {
        "initial_capital": 3200,
        "dashboard_port": 5000,
        "description": "Main Account",
        "type": "demo",
        "regime01_enabled": True,
        "risk_control_enabled": True,
        "postgresql_enabled": False,
        "regime01_ma_period": 2,
        "regime01_long_th":   1.00,
        "regime01_short_th":  1.00,
        "regime_reference_symbol": "QQQUSDT"
    },
    "01": {
        "initial_capital": 12000,
        "dashboard_port": 5099,
        "description": "Testing Account",
        "type": "demo",
        "regime01_enabled": False,
        "risk_control_enabled": False,
        "postgresql_enabled": False,
        "regime01_ma_period": 5,
        "regime01_long_th":   1.00,
        "regime01_short_th":  1.00,
    }
}

COMMISSION_PCT = 0.1

# ==========================================================================
# MARKET REGIME SETTINGS
# ==========================================================================

from shared_batchs.regime.regime_module import REGIME_FAMILIES, ER_WINDOW as REGIME_ER_WINDOW, ATR_WINDOW as REGIME_ATR_WINDOW
REGIME_GENERAL = {
    'trending': 1.0,
    'ranging':  1.0,
    'volatile': 1.0,
}

# ==========================================================================
# RISK CONTROL SETTINGS
# ==========================================================================
RISK_LIMITS = {
    'max_gross_exposure_pct': 10.0,
    'max_net_exposure_pct':   10.0,
}
LEVERAGE = 10

# =============================================================================
# QUALITY CONTROL PARAMETERS
# =============================================================================

# Drift detection
DRIFT_BINOMIAL_WINDOW  = 100
DRIFT_CHECK_INTERVAL   = 15
DRIFT_BINOMIAL_DEFAULT = 0.55

# Execution quality
EXECUTION_WINDOW_SIZE  = 20
SLIPPAGE_WARNING_PCT   = 0.2
SLIPPAGE_CRITICAL_PCT  = 0.3
LATENCY_WARNING_SEC    = 0.5
LATENCY_CRITICAL_SEC   = 1.0

# ==========================================================================
# STRATEGY VALIDATION CONFIGURATION
# ==========================================================================

# Common parameters required for ALL strategies
COMMON_REQUIRED_PARAMS = [
    'id', 'name', 'timeframe', 'active', 'sell_after_ncandles',
    'order_amount', 'tp_pct', 'sl_pct', 'direction',
    'regime_trending_uptrend', 'regime_trending_dwtrend',
    'regime_ranging_uptrend',  'regime_ranging_dwtrend',
    'regime_volatile_uptrend', 'regime_volatile_dwtrend',
]

# Strategy-specific required parameters by strategy type
STRATEGY_TYPE_REQUIRED_PARAMS = {
    'reversal_long':     ['lookback', 'tolerance', 'ma_period'],
    'reversal_short':    ['lookback', 'tolerance', 'ma_period'],
    'parity_long':       ['lookback', 'tolerance', 'ma_period'],
    'parity_short':      ['lookback', 'tolerance', 'ma_period'],
    'orderblocks_long':  ['lookback', 'tolerance', 'impulse'],
    'orderblocks_short': ['lookback', 'tolerance', 'impulse'],
    'flag_long':         ['lookback', 'impulse', 'flag', 'ma_period'],
    'flag_short':        ['lookback', 'impulse', 'flag', 'ma_period'],
}

# Order amount limits (USDT)
MIN_ORDER_AMOUNT = 80
MAX_ORDER_AMOUNT = 240

# TP/SL limits (%)
MIN_TP_PCT = 2
MAX_TP_PCT = 9
MIN_SL_PCT = 2
MAX_SL_PCT = 9

# Candles timeout limits
MIN_CANDLES = 50
MAX_CANDLES = 100

# Valid timeframes
VALID_TIMEFRAMES = ['15m','30m','1H','4H','6Hutc']

# ==========================================================================
# POSTGRESQL CONFIGURATION
# ==========================================================================

# Environment detection
HOSTNAME      = socket.gethostname()
VPS_HOSTNAMES = ['srv1326826', 'hstgr.cloud']
IS_VPS        = any(h in HOSTNAME for h in VPS_HOSTNAMES)

POSTGRES_CONFIG = {
    'dbname':          'bot_trading',
    'user':            'javi',
    'password':        'Laplaciano86-',
    'host':            'localhost',
    'port':            5432,
    'connect_timeout': 3,
}

# VPS PostgreSQL connection (for split-brain check from LOCAL)
VPS_CHECK_CONFIG = {
    'host':     '100.123.10.95',
    'user':     'javi',
    'password': 'Laplaciano86-',
    'dbname':   'bot_trading',
    'timeout':  5,
}

# ==========================================================================
# EXCHANGE SETTINGS
# ==========================================================================
MARGIN_MODE     = "crossed"
MARGIN_COIN     = "USDT"
API_LIMIT_DATA  = 180

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
from shared.shared_trading_data.config_trading_data import BASE_URL, PRODUCT_TYPE
# ==========================================================================
# LOGGER SETTINGS
# ==========================================================================
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL    = "INFO"
LOG_MAX_BYTES     = 10 * 1024 * 1024
LOG_BACKUP_COUNT  = 5
LOG_NAMESPACE     = "BOT_trading"