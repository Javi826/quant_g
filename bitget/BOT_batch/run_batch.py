import logging
import time
import main_batch
from main_batch import run_batch, run_portfolio_analysis

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO   # Change to logging.DEBUG for full verbosity
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# =============================================================================
# PLOT & PROGRESS CONFIGURATION
# =============================================================================
SHOW_PLOTS = False   # Set to False to suppress all plots
main_batch.SHOW_PROGRESS = (LOG_LEVEL <= logging.DEBUG)

if not SHOW_PLOTS:
    import matplotlib
    matplotlib.use("Agg")

# =============================================================================
# STRATEGIES
# =============================================================================
STRATEGIES = [
    {
        "strategy_id":  "02_reversal_long_4H",
        "signal":       "reversal_long",
        "side":         "long",
        "timeframe":    "4H",
        "n_symbols":    31,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [4],
            "MA_PERIOD":  [50],
            "TOLERANCE":  [20],
            "TP_PCT":     [3],
            "SL_PCT":     [10],
        },
    },
    {
        "strategy_id":  "03_parity_long_4H",
        "signal":       "parity_long",
        "side":         "long",
        "timeframe":    "4H",
        "n_symbols":    9,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [150],
            "MA_PERIOD":  [50],
            "TOLERANCE":  [40],
            "TP_PCT":     [3],
            "SL_PCT":     [10],
        },
    },
    {
        "strategy_id":  "04_reversal_short_4H",
        "signal":       "reversal_short",
        "side":         "short",
        "timeframe":    "4H",
        "n_symbols":    31,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [4],
            "MA_PERIOD":  [50],
            "TOLERANCE":  [25],
            "TP_PCT":     [3],
            "SL_PCT":     [9],
        },
    },
    {
        "strategy_id":  "06_reversal_long_1H",
        "signal":       "reversal_long",
        "side":         "long",
        "timeframe":    "1H",
        "n_symbols":    8,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [7],
            "MA_PERIOD":  [25],
            "TOLERANCE":  [40],
            "TP_PCT":     [2],
            "SL_PCT":     [10],
        },
    },
    {
        "strategy_id":  "07_reversal_short_1H",
        "signal":       "reversal_short",
        "side":         "short",
        "timeframe":    "1H",
        "n_symbols":    8,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [5],
            "MA_PERIOD":  [50],
            "TOLERANCE":  [30],
            "TP_PCT":     [2],
            "SL_PCT":     [5],
        },
    },
    {
        "strategy_id":  "08_reversal_long_6Hutc",
        "signal":       "reversal_long",
        "side":         "long",
        "timeframe":    "6Hutc",
        "n_symbols":    8,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [3],
            "MA_PERIOD":  [50],
            "TOLERANCE":  [20],
            "TP_PCT":     [4],
            "SL_PCT":     [10],
        },
    },
    {
        "strategy_id":  "09_reversal_short_6Hutc",
        "signal":       "reversal_short",
        "side":         "short",
        "timeframe":    "6Hutc",
        "n_symbols":    31,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [6],
            "MA_PERIOD":  [25],
            "TOLERANCE":  [30],
            "TP_PCT":     [4],
            "SL_PCT":     [7.5],
        },
    },
    {
        "strategy_id":  "10_parity_long_1H",
        "signal":       "parity_long",
        "side":         "long",
        "timeframe":    "1H",
        "n_symbols":    6,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [150],
            "MA_PERIOD":  [25],
            "TOLERANCE":  [15],
            "TP_PCT":     [2],
            "SL_PCT":     [10],
        },
    },
    {
        "strategy_id":  "11_parity_short_1H",
        "signal":       "parity_short",
        "side":         "short",
        "timeframe":    "1H",
        "n_symbols":    6,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [150],
            "MA_PERIOD":  [50],
            "TOLERANCE":  [20],
            "TP_PCT":     [2],
            "SL_PCT":     [7.5],
        },
    },
    {
        "strategy_id":  "12_parity_long_6Hutc",
        "signal":       "parity_long",
        "side":         "long",
        "timeframe":    "6Hutc",
        "n_symbols":    15,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [50],
            "MA_PERIOD":  [25],
            "TOLERANCE":  [40],
            "TP_PCT":     [3.5],
            "SL_PCT":     [10],
        },
    },
    {
        "strategy_id":  "13_orderblocks_short_4H",
        "signal":       "orderblocks_short",
        "side":         "short",
        "timeframe":    "4H",
        "n_symbols":    31,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [50],
            "IMPULSE":    [0.01],
            "TOLERANCE":  [35],
            "TP_PCT":     [4],
            "SL_PCT":     [11],
        },
    },
    {
        "strategy_id":  "16_ranging_short_6Hutc",
        "signal":       "ranging_short",
        "side":         "short",
        "timeframe":    "6Hutc",
        "n_symbols":    32,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [10],
            "MA_PERIOD":  [25],
            "RANGE_STR":  [25],
            "TOLERANCE":  [5],
            "TP_PCT":     [4],
            "SL_PCT":     [6],
        },
    },
    {
        "strategy_id":  "17_flag_long_4H",
        "signal":       "flag_long",
        "side":         "long",
        "timeframe":    "4H",
        "n_symbols":    9,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [13],
            "FLAG":       [40],
            "IMPULSE":    [5],
            "MA_PERIOD":  [50],
            "TP_PCT":     [4],
            "SL_PCT":     [10],
        },
    },
    {
        "strategy_id":  "19_flag_short_4H",
        "signal":       "flag_short",
        "side":         "short",
        "timeframe":    "4H",
        "n_symbols":    23,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [10],
            "FLAG":       [50],
            "IMPULSE":    [3],
            "MA_PERIOD":  [50],
            "TP_PCT":     [3],
            "SL_PCT":     [9],
        },
    },
    {
        "strategy_id":  "20_flag_short_1H",
        "signal":       "flag_short",
        "side":         "short",
        "timeframe":    "1H",
        "n_symbols":    11,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [20],
            "FLAG":       [60],
            "IMPULSE":    [3],
            "MA_PERIOD":  [25],
            "TP_PCT":     [2],
            "SL_PCT":     [8],
        },
    },
]
# =============================================================================
# STRATEGIES = [
#     {
#         "strategy_id":  "02_reversal_long_4H",
#         "signal":       "reversal_long",
#         "side":         "long",
#         "timeframe":    "4H",
#         "n_symbols":    31,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [3, 4, 5],
#             "MA_PERIOD":  [50],
#             "TOLERANCE":  [15, 20, 25],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "03_parity_long_4H",
#         "signal":       "parity_long",
#         "side":         "long",
#         "timeframe":    "4H",
#         "n_symbols":    9,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [100, 150, 200],
#             "MA_PERIOD":  [50],
#             "TOLERANCE":  [30, 40, 50],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "04_reversal_short_4H",
#         "signal":       "reversal_short",
#         "side":         "short",
#         "timeframe":    "4H",
#         "n_symbols":    31,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [3, 4, 5],
#             "MA_PERIOD":  [50],
#             "TOLERANCE":  [20, 25, 30],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "06_reversal_long_1H",
#         "signal":       "reversal_long",
#         "side":         "long",
#         "timeframe":    "1H",
#         "n_symbols":    8,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [6, 7, 8],
#             "MA_PERIOD":  [25, 50],
#             "TOLERANCE":  [35, 40, 45],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "07_reversal_short_1H",
#         "signal":       "reversal_short",
#         "side":         "short",
#         "timeframe":    "1H",
#         "n_symbols":    8,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [4, 5, 6],
#             "MA_PERIOD":  [25, 50],
#             "TOLERANCE":  [25, 30, 35],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "08_reversal_long_6Hutc",
#         "signal":       "reversal_long",
#         "side":         "long",
#         "timeframe":    "6Hutc",
#         "n_symbols":    8,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [2, 3, 4],
#             "MA_PERIOD":  [25, 50],
#             "TOLERANCE":  [15, 20, 25],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "09_reversal_short_6Hutc",
#         "signal":       "reversal_short",
#         "side":         "short",
#         "timeframe":    "6Hutc",
#         "n_symbols":    31,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [5, 6, 7],
#             "MA_PERIOD":  [25, 50],
#             "TOLERANCE":  [25, 30, 35],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "10_parity_long_1H",
#         "signal":       "parity_long",
#         "side":         "long",
#         "timeframe":    "1H",
#         "n_symbols":    6,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [100, 150, 200],
#             "MA_PERIOD":  [25, 50],
#             "TOLERANCE":  [10, 15, 20],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "11_parity_short_1H",
#         "signal":       "parity_short",
#         "side":         "short",
#         "timeframe":    "1H",
#         "n_symbols":    6,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [100, 150, 200],
#             "MA_PERIOD":  [25, 50],
#             "TOLERANCE":  [15, 20, 25],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "12_parity_long_6Hutc",
#         "signal":       "parity_long",
#         "side":         "long",
#         "timeframe":    "6Hutc",
#         "n_symbols":    15,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [25, 50, 75],
#             "MA_PERIOD":  [25, 50],
#             "TOLERANCE":  [30, 40, 50],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "13_orderblocks_short_4H",
#         "signal":       "orderblocks_short",
#         "side":         "short",
#         "timeframe":    "4H",
#         "n_symbols":    31,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [40,50,60],
#             "IMPULSE":    [0.008, 0.01, 0.012],
#             "TOLERANCE":  [30, 35, 40],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "16_ranging_short_6Hutc",
#         "signal":       "ranging_short",
#         "side":         "short",
#         "timeframe":    "6Hutc",
#         "n_symbols":    32,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [8, 10, 12],
#             "RANGE_STR":   [20, 25, 30],
#             "TOLERANCE":  [4, 5, 6],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "17_flag_long_4H",
#         "signal":       "flag_long",
#         "side":         "long",
#         "timeframe":    "4H",
#         "n_symbols":    9,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [10, 15, 20],
#             "FLAG":       [40, 50, 60],
#             "IMPULSE":    [2, 3, 4],
#             "MA_PERIOD":  [50],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "19_flag_short_4H",
#         "signal":       "flag_short",
#         "side":         "short",
#         "timeframe":    "4H",
#         "n_symbols":    23,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [8, 10, 12],
#             "FLAG":       [40, 50, 60],
#             "IMPULSE":    [2, 3, 4],
#             "MA_PERIOD":  [50],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
#     {
#         "strategy_id":  "20_flag_short_1H",
#         "signal":       "flag_short",
#         "side":         "short",
#         "timeframe":    "1H",
#         "n_symbols":    11,
#         "order_amount": 80,
#         "param_grid": {
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [15, 20, 25],
#             "FLAG":       [50, 60, 70],
#             "IMPULSE":    [2, 3, 4],
#             "MA_PERIOD":  [25, 50],
#             "TP_PCT":     [2, 3, 4, 5],
#             "SL_PCT":     [6, 7, 8, 9, 10],
#         },
#     },
# ]
# =============================================================================

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    _start = time.time()
    _logger = logging.getLogger("BOT_trading.batch.main_batch")

    for strategy in STRATEGIES:
        _logger.info(f"\n\033[94m{'='*100}\033[0m")
        _logger.info(f"\033[94m  Running: {strategy['strategy_id']}\033[0m")
        _logger.info(f"\033[94m{'='*100}\033[0m")
        run_batch(strategy)

    run_portfolio_analysis()

    _elapsed = int(time.time() - _start)
    _logger.info(f"\n🏁 TOTAL — {_elapsed//3600} h {(_elapsed%3600)//60} min {_elapsed%60} s")