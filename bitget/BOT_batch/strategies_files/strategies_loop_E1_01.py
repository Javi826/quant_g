"""
strategies_loop_E1.py — Batch loop configuration.
Edit param_grid, n_symbols and order_amount before each run.
This file is NOT updated by the batch automatically.
"""

STRATEGIES_LOOP = [
    {
        "id": "06_reversal_long_1H",
        "n_symbols": 3,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK": [6],
            "TOLERANCE": [25],
            "MA_PERIOD": [10],
            "TP_PCT": [2],
            "SL_PCT": [3],
        },
    },
    {
        "id": "07_reversal_short_1H",
        "n_symbols": 3,
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
        "n_symbols": 3,
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
    {
        "id": "20_flag_short_1H",
        "n_symbols": 3,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK": [40],
            "MA_PERIOD": [25],
            "IMPULSE": [2],
            "FLAG": [30],
            "TP_PCT": [2],
            "SL_PCT": [3],
        },
    },
    {
        "id": "21_parity_short_4H",
        "n_symbols": 3,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [50],
            "LOOKBACK": [100],
            "TOLERANCE": [40],
            "MA_PERIOD": [10],
            "TP_PCT": [3],
            "SL_PCT": [3],
        },
    },
    {
        "id": "23_flag_long_1H",
        "n_symbols": 3,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [50],
            "LOOKBACK": [60],
            "MA_PERIOD": [10],
            "IMPULSE": [2],
            "FLAG": [20],
            "TP_PCT": [2],
            "SL_PCT": [3],
        },
    },
]
