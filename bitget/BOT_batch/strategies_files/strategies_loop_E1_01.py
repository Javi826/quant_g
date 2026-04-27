"""
strategies_loop_E1.py — Batch loop configuration.
Edit param_grid, n_symbols and order_amount before each run.
This file is NOT updated by the batch automatically.
"""

STRATEGIES_LOOP = [
    {
        "id": "07_reversal_short_1H",
        "n_symbols": 8,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK": [4],
            "TOLERANCE": [15],
            "MA_PERIOD": [10],
            "TP_PCT": [2],
            "SL_PCT": [3],
        },
    },
    {
        "id": "10_parity_long_1H",
        "n_symbols": 8,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK": [150],
            "TOLERANCE": [10],
            "MA_PERIOD": [10],
            "TP_PCT": [2],
            "SL_PCT": [3],
        },
    },
]
