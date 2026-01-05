"""
Bot Configuration Settings

Centralizes all bot configuration including exchange settings,
validation limits, paths, and account-specific settings.
"""

from zoneinfo import ZoneInfo
import os

# ==========================================================================
# EXCHANGE SETTINGS
# ==========================================================================

# Bitget API
BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_MODE  = "crossed"  
MARGIN_COIN  = "USDT"

# API request settings
API_TIMEOUT     = 10  # seconds
API_MAX_RETRIES = 3

# ==========================================================================
# GENERAL BOT SETTINGS
# ==========================================================================

# Timezone
HOUR_ZONE = ZoneInfo('UTC')

# Check intervals (seconds)
CHECK_INTERVAL = 10  

# Signal detection mode
USE_HARDCODED_SIGNALS = False  

DISPLAY_MODE ="summary"

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
# STRATEGY ASSIGNMENT PER ACCOUNT
# ==========================================================================

# Maps account numbers to strategy IDs
# Strategy IDs reference strategies defined in config/strategies.yaml
ACCOUNT_STRATEGIES = {
    "00": [
        # All 14 strategies
        '01_double_top_long_4H',
        '02_reversal_long_4H',
        '03_parity_long_4H',
        '04_reversal_short_4H',
        '05_parity_short_4H',
        '06_reversal_long_1H',
        '07_reversal_short_1H',
        '08_reversal_long_6Hutc',
        '09_reversal_short_6Hutc',
        '10_parity_long_1H',
        '11_parity_short_1H',
        '12_parity_long_6Hutc',
        '13_orderblocks_short_4H',
        '14_orderblocks_long_4H'
    ],
    "E1": [
        # All except STRAT_L (12_parity_long_6Hutc) and STRAT_N (14_orderblocks_long_4H)
        '01_double_top_long_4H',
        '02_reversal_long_4H',
        '03_parity_long_4H',
        '04_reversal_short_4H',
        '05_parity_short_4H',
        '06_reversal_long_1H',
        '07_reversal_short_1H',
        '08_reversal_long_6Hutc',
        '09_reversal_short_6Hutc',
        '10_parity_long_1H',
        '11_parity_short_1H',
        '13_orderblocks_short_4H'
    ],
    "01": [
        # Only testing strategies
        '01_double_top_long_2m',
        '02_reversal_long_5m'
    ]
}

# ==========================================================================
# PATHS CONFIGURATION
# ==========================================================================

# Base directory for bot files (relative to live_trading2/)
PERSISTENCE_DIR = "persistence"

def get_account_paths(account_number: str) -> dict:
    """
    Get all file paths for a specific account.
    
    Args:
        account_number: Account number (e.g., "01", "E1")
    
    Returns:
        Dictionary with all paths for the account
    """
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        PERSISTENCE_DIR,
        f'bot_files_{account_number}'
    )
    
    return {
        'base_dir': base_dir,
        'state_file': os.path.join(base_dir, f'bot_state_{account_number}.json'),
        'trades_file': os.path.join(base_dir, f'bot_trades_{account_number}.xlsx'),
        'log_file': os.path.join(base_dir, f'BOT_orchestator_{account_number}.log')
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
MAX_SL_PCT = 10

# Candles timeout limits
MIN_CANDLES = 49
MAX_CANDLES = 51

# Valid timeframes
VALID_TIMEFRAMES = ['1H', '4H', '6Hutc']

# ==========================================================================
# STRATEGY CONFIGURATION & VALIDATION
# ==========================================================================

# Timeframe suffixes for strategy name parsing
TIMEFRAME_SUFFIXES = ['_4H', '_1H', '_6Hutc', '_12H', '_8H', '_30m']

# Strategy type required parameters
# When adding a new strategy function, add its required params here
STRATEGY_TYPE_REQUIRED_PARAMS = {
    'double_top_long': ['lookback', 'tolerance', 'trend_th'],
    'reversal_long': ['lookback', 'tolerance', 'ma_period'],
    'reversal_short': ['lookback', 'tolerance', 'ma_period'],
    'parity_long': ['lookback', 'tolerance', 'ma_period'],
    'parity_short': ['lookback', 'tolerance', 'ma_period'],
    'orderblocks_long': ['lookback', 'tolerance', 'impulse'],
    'orderblocks_short': ['lookback', 'tolerance', 'impulse'],
}

# Common parameters required for ALL strategies
COMMON_REQUIRED_PARAMS = ['id', 'name', 'timeframe', 'active', 'sell_after_ncandles', 'order_amount', 'tp_pct', 'sl_pct', 'direction']

# ==========================================================================
# WEBSOCKET SETTINGS
# ==========================================================================

WS_PUBLIC_URL  = "wss://ws.bitget.com/v2/ws/public"
WS_PRIVATE_URL = "wss://ws.bitget.com/v2/ws/private"

# ==========================================================================
# logger SETTINGS
# ==========================================================================

# Log levels
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL    = "DEBUG"

# Log rotation
LOG_MAX_BYTES    = 10 * 1024 * 1024  
LOG_BACKUP_COUNT = 5

# logger namespace
LOG_NAMESPACE = "BOT_trading"

# ==========================================================================
# RISK MANAGEMENT (Future)
# ==========================================================================

# Maximum open positions per strategy
MAX_POSITIONS_PER_STRATEGY = 5

# Maximum total open positions
MAX_TOTAL_POSITIONS = 20

# Maximum daily loss (% of capital)
MAX_DAILY_LOSS_PCT = 5.0

# Circuit breaker - pause trading if hit
ENABLE_CIRCUIT_BREAKER = False

# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def get_account_config(account_number: str) -> dict:
    """
    Get complete configuration for an account.
    
    Args:
        account_number: Account number
    
    Returns:
        Dictionary with account configuration
    
    Raises:
        ValueError: If account number is invalid
    """
    if account_number not in ACCOUNTS:
        available = ', '.join(ACCOUNTS.keys())
        raise ValueError(
            f"Invalid account number: {account_number}. "
            f"Available: {available}"
        )
    
    config = ACCOUNTS[account_number].copy()
    config['account_number'] = account_number
    config['paths'] = get_account_paths(account_number)
    
    return config


def get_account_strategies(account_number: str) -> list:
    """
    Get list of strategy IDs assigned to an account.
    
    Args:
        account_number: Account number (e.g., "01", "E1")
    
    Returns:
        List of strategy IDs
    
    Raises:
        ValueError: If account number is invalid
    """
    if account_number not in ACCOUNT_STRATEGIES:
        available = ', '.join(ACCOUNT_STRATEGIES.keys())
        raise ValueError(
            f"Invalid account number: {account_number}. "
            f"Available: {available}"
        )
    
    return ACCOUNT_STRATEGIES[account_number]


def validate_settings():
    """
    Validate that settings.py is correctly configured.
    
    Raises:
        ValueError: If settings are invalid
    """
    # Validate account ports are unique
    ports = [acc['dashboard_port'] for acc in ACCOUNTS.values()]
    if len(ports) != len(set(ports)):
        raise ValueError("Dashboard ports must be unique across accounts")
    
    # Validate timeframes
    if not VALID_TIMEFRAMES:
        raise ValueError("VALID_TIMEFRAMES cannot be empty")
    
    # Validate limits
    if MIN_ORDER_AMOUNT >= MAX_ORDER_AMOUNT:
        raise ValueError("MIN_ORDER_AMOUNT must be less than MAX_ORDER_AMOUNT")
    
    if MIN_TP_PCT >= MAX_TP_PCT:
        raise ValueError("MIN_TP_PCT must be less than MAX_TP_PCT")
    
    # Validate URLs
    if not BASE_URL.startswith("https://"):
        raise ValueError("BASE_URL must use HTTPS")
    
    # Validate account strategies mapping
    for account_num in ACCOUNTS.keys():
        if account_num not in ACCOUNT_STRATEGIES:
            raise ValueError(f"Account {account_num} missing in ACCOUNT_STRATEGIES")


# Validate on import
validate_settings()
