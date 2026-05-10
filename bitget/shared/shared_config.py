#shared/shared_config.py
# =============================================================================
# MARKET REGIME SETTINGS
# =============================================================================

REGIME_REFERENCE_SYMBOL = 'BTCUSDT'
SYMBOL_SUFFIX           = "USDT"
VOLUME_COL              = "volume_quote"

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
# =============================================================================
# API SOCKETS SETTINGS
# =============================================================================
WS_PUBLIC_URL   = "wss://ws.bitget.com/v2/ws/public"
WS_PRIVATE_URL  = "wss://ws.bitget.com/v2/ws/private"
BASE_URL        = "https://api.bitget.com"
PRODUCT_TYPE    = "USDT-FUTURES"
API_TIMEOUT     = 10
API_MAX_RETRIES = 3


# =============================================================================
# REGIME0 SETTINGS (BTC MA filter)
# =============================================================================
REGIME0_MA_PERIOD = 2
REGIME0_LONG_TH   = 1.00
REGIME0_SHORT_TH  = 1.00