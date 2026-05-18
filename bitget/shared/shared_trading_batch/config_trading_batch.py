# shared/shared_trading_batch/config_trading_batch.py

# =============================================================================
# MARKET REGIME SETTINGS
# =============================================================================
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
# REGIME_FAMILIES = {
#     'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.6)},
#     'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
#     'ranging': {}
# }
# =============================================================================
