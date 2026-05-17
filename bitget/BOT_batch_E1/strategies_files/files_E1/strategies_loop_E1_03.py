"""
strategies_loop_00_01_rwa.py — Batch loop configuration for RWA.
Edit param_grid, n_symbols and order_amount before each run.
This file is NOT updated by the batch automatically.
"""

STRATEGIES_LOOP = [
    # =========================================================================
    # REVERSAL LONG
    # =========================================================================
    {
        "id": "30_reversal_long_30m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [6,8,10],
            "TOLERANCE": [10,20,30],
            "MA_PERIOD": [10,25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    {
        "id": "31_reversal_long_15m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [6,8,10],
            "TOLERANCE": [10,20,30],
            "MA_PERIOD": [10,25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    # =========================================================================
    # REVERSAL SHORT
    # =========================================================================
    {
        "id": "32_reversal_short_30m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [6,8,10],
            "TOLERANCE": [10,20,30],
            "MA_PERIOD": [10,25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    {
        "id": "33_reversal_short_15m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [6,8,10],
            "TOLERANCE": [10,20,30],
            "MA_PERIOD": [10,25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    # =========================================================================
    # PARITY LONG
    # =========================================================================
    {
        "id": "34_parity_long_30m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [50,100,150],
            "TOLERANCE": [10,20,30],
            "MA_PERIOD": [10,25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    {
        "id": "35_parity_long_15m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [50,100,150],
            "TOLERANCE": [10,20,30],
            "MA_PERIOD": [10,25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    # =========================================================================
    # PARITY SHORT
    # =========================================================================
    {
        "id": "36_parity_short_30m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [50,100,150],
            "TOLERANCE": [10,20,30],
            "MA_PERIOD": [10,25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    {
        "id": "37_parity_short_15m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [50, 100,150],
            "TOLERANCE": [10,20,30],
            "MA_PERIOD": [10, 25],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    # =========================================================================
    # FLAG LONG
    # =========================================================================
    {
        "id": "38_flag_long_30m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [20,40,60],
            "MA_PERIOD": [10,25],
            "IMPULSE": [2,4,6],
            "FLAG": [10,20,30],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    {
        "id": "39_flag_long_15m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [20,40,60],
            "MA_PERIOD": [10,25],
            "IMPULSE": [2,4,6],
            "FLAG": [10,20,30],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    # =========================================================================
    # FLAG SHORT
    # =========================================================================
    {
        "id": "40_flag_short_30m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [20,40,60],
            "MA_PERIOD": [10,25],
            "IMPULSE": [2, 4, 6],
            "FLAG": [10,20,30],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    {
        "id": "41_flag_short_15m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [20,40,60],
            "MA_PERIOD": [10, 25],
            "IMPULSE": [2,4,6],
            "FLAG": [10,20,30],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    # =========================================================================
    # ORDERBLOCKS LONG
    # =========================================================================
    {
        "id": "42_orderblocks_long_30m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [30,50,75],
            "TOLERANCE": [10,20,30],
            "IMPULSE": [0.02,0.05,0.08],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    {
        "id": "43_orderblocks_long_15m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [30,50,75],
            "TOLERANCE": [10,20,30],
            "IMPULSE": [0.02,0.05,0.08],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    # =========================================================================
    # ORDERBLOCKS SHORT
    # =========================================================================
    {
        "id": "44_orderblocks_short_30m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [30,50,75],
            "TOLERANCE": [10,20,30],
            "IMPULSE": [0.02,0.05,0.08],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
    {
        "id": "45_orderblocks_short_15m",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK": [30,50,75],
            "TOLERANCE": [10,20,30],
            "IMPULSE": [0.02, 0.05, 0.08],
            "TP_PCT": [2,3,4],
            "SL_PCT": [2,3],
        },
    },
]