#shared/shared_trading_batch/config_trading_batch.py
# =============================================================================
# CONFIGURATION  — mirror from regime_GE.py after calibration
# =============================================================================

INDICATORS: dict[str, dict] = {
    "atr_norm": {
        "windows":    [10],
        "thresholds": [0.04],
        "enabled":    True,
    },
    "er": {
        "windows":    [40],
        "thresholds": [0.8],
        "enabled":    True,
    },
    "hurst": {
        "windows":    [30],
        "thresholds": [0.5],
        "enabled":    False,
    },
}


COMBINE_MODE          = "OR"
ANALYSIS_MODE         = "SYMBOL"
REGIME_TIMEFRAME_MODE = "DAILY"    # "DAILY" | "STRATEGY"

# =============================================================================
# ANALYSIS_MODE         = "SYMBOL"   # "BTC" | "SYMBOL"
# REGIME_TIMEFRAME_MODE = "DAILY"    # "DAILY" | "STRATEGY"
# COMBINE_MODE         = ["OR"]    
# 
# INDICATORS: dict[str, dict] = {
#     "atr_norm": {
#         "windows":    [30],
#         "thresholds": [0.02],
#         "enabled":    True,
#     },
#     "er": {
#         "windows":    [40],
#         "thresholds": [0.6],
#         "enabled":    True,
#     },
#     "hurst": {
#         "windows":    [30],
#         "thresholds": [0.5],
#         "enabled":    False,
#     },
# }
# =============================================================================
