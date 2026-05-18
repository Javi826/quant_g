#shared/shared_batchs/regime/regime_config.py

# =============================================================================
# REGIME FILTER SETTINGS
# =============================================================================
REGIME_ENABLED         = True       
REGIME_REFERENCE       = 'BTCUSDT'   
#REGIME_REFERENCE       = 'QQQUSDT'
FORCE_DIRECTION_FILTER = True
REGIME_MIN_TRADES      = 10
REGIME_LOOKBACK_BARS   = 50
REGIME_FAMILY_SOURCE   = 'strategy'    # 'strategy' | 'macro'

# =============================================================================
# REGIME0 SETTINGS (REF MA filter)
# =============================================================================
REGIME0_MA_PERIOD = 2
REGIME0_LONG_TH   = 1.00
REGIME0_SHORT_TH  = 1.00