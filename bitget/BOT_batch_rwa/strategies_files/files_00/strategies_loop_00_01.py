"""
strategies_loop_00_01_rwa.py — Batch loop configuration for RWA.
Fixed params from validated batch. Single combo per strategy.
This file is NOT updated by the batch automatically.
"""

STRATEGIES_LOOP = [
    # =========================================================================
    # REVERSAL LONG
    # =========================================================================
    {
        "id": "30_reversal_long_1H",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [9],
            "TOLERANCE":  [10],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "31_reversal_long_30m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [8],
            "TOLERANCE":  [10],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "32_reversal_long_15m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [9],
            "TOLERANCE":  [10],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    # =========================================================================
    # REVERSAL SHORT
    # =========================================================================
    {
        "id": "33_reversal_short_1H",
        "n_symbols": 10,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [6],
            "TOLERANCE":  [15],
            "MA_PERIOD":  [25],
            "TP_PCT":     [3.0],
            "SL_PCT":     [2.0],
        },
    },
    {
        "id": "34_reversal_short_30m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [6],
            "TOLERANCE":  [5],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.0],
            "SL_PCT":     [2.0],
        },
    },
    {
        "id": "35_reversal_short_15m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [6],
            "TOLERANCE":  [5],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.0],
            "SL_PCT":     [2.5],
        },
    },
    # =========================================================================
    # PARITY LONG
    # =========================================================================
    {
        "id": "36_parity_long_1H",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [150],
            "TOLERANCE":  [6],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "37_parity_long_30m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [50],
            "TOLERANCE":  [4],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "38_parity_long_15m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [50],
            "TOLERANCE":  [4],
            "MA_PERIOD":  [25],
            "TP_PCT":     [3.0],
            "SL_PCT":     [3.0],
        },
    },
    # =========================================================================
    # PARITY SHORT
    # =========================================================================
    {
        "id": "39_parity_short_1H",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [100],
            "TOLERANCE":  [4],
            "MA_PERIOD":  [10],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "40_parity_short_30m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [50],
            "TOLERANCE":  [4],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "41_parity_short_15m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [50],
            "TOLERANCE":  [4],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2.5],
            "SL_PCT":     [3.0],
        },
    },
    # =========================================================================
    # FLAG LONG
    # =========================================================================
    {
        "id": "42_flag_long_1H",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [10],
            "MA_PERIOD":  [50],
            "IMPULSE":    [5],
            "FLAG":       [10],
            "TP_PCT":     [3.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "43_flag_long_30m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [10],
            "MA_PERIOD":  [50],
            "IMPULSE":    [5],
            "FLAG":       [20],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "44_flag_long_15m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [20],
            "MA_PERIOD":  [50],
            "IMPULSE":    [5],
            "FLAG":       [30],
            "TP_PCT":     [2.0],
            "SL_PCT":     [2.5],
        },
    },
    # =========================================================================
    # FLAG SHORT
    # =========================================================================
    {
        "id": "45_flag_short_1H",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [10],
            "MA_PERIOD":  [50],
            "IMPULSE":    [5],
            "FLAG":       [40],
            "TP_PCT":     [2.5],
            "SL_PCT":     [2.5],
        },
    },
    {
        "id": "46_flag_short_30m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [10],
            "MA_PERIOD":  [50],
            "IMPULSE":    [5],
            "FLAG":       [20],
            "TP_PCT":     [2.0],
            "SL_PCT":     [2.0],
        },
    },
    {
        "id": "47_flag_short_15m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [100],
            "LOOKBACK":   [10],
            "MA_PERIOD":  [25],
            "IMPULSE":    [5],
            "FLAG":       [10],
            "TP_PCT":     [3.0],
            "SL_PCT":     [3.0],
        },
    },
    # =========================================================================
    # ORDERBLOCKS LONG
    # =========================================================================
    {
        "id": "48_orderblocks_long_1H",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [75],
            "TOLERANCE":  [25],
            "IMPULSE":    [0.05],
            "TP_PCT":     [3.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "49_orderblocks_long_30m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [30],
            "TOLERANCE":  [25],
            "IMPULSE":    [0.08],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "50_orderblocks_long_15m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [30],
            "TOLERANCE":  [25],
            "IMPULSE":    [0.08],
            "TP_PCT":     [2.5],
            "SL_PCT":     [3.0],
        },
    },
    # =========================================================================
    # ORDERBLOCKS SHORT
    # =========================================================================
    {
        "id": "51_orderblocks_short_1H",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [75],
            "TOLERANCE":  [45],
            "IMPULSE":    [0.02],
            "TP_PCT":     [2.0],
            "SL_PCT":     [2.5],
        },
    },
    {
        "id": "52_orderblocks_short_30m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [30],
            "TOLERANCE":  [25],
            "IMPULSE":    [0.05],
            "TP_PCT":     [2.0],
            "SL_PCT":     [3.0],
        },
    },
    {
        "id": "53_orderblocks_short_15m",
        "n_symbols": 11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [75],
            "LOOKBACK":   [30],
            "TOLERANCE":  [25],
            "IMPULSE":    [0.08],
            "TP_PCT":     [2.0],
            "SL_PCT":     [2.0],
        },
    },
]