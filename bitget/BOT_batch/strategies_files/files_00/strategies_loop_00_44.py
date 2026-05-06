"""
strategies_loop.py — Batch loop configuration.
Edit param_grid, n_symbols and order_amount before each run.
This file is NOT updated by the batch automatically.
"""

STRATEGIES_LOOP = [
    {
        "id": "06_reversal_long_1H",
        "n_symbols": 8,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK": [6,7,8,9],
            "TOLERANCE": [10,20,25],
            "MA_PERIOD": [10,25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
# =============================================================================
#     {
#         "id": "06_reversal_long_1H",
#         "n_symbols": 8,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [100],
#             "LOOKBACK": [6,7,8,9],
#             "TOLERANCE": [10,20,25],
#             "MA_PERIOD": [10,25],
#             "TP_PCT": [2,3,4,5,6,7,8,9],
#             "SL_PCT": [2,3,4,5,6,7,8,9],
#         },
#     },
# =============================================================================

]