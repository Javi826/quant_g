# shared/shared_trading_batch/config_trading_batch.py

# =============================================================================
# MARKET REGIME SETTINGS
# =============================================================================
REGIME_ER_WINDOW    = 14
REGIME_ATR_WINDOW   = 14


REGIME_FAMILIES = {
    'trending': {'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0)},
    'ranging':  {}
}
