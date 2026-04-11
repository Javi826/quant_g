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
        "strategy_id":  "03_parity_long_4H",
        "signal":       "parity_long",
        "side":         "long",
        "timeframe":    "4H",
        "n_symbols":    9,
        "order_amount": 80,
        "param_grid": {
            "SELL_AFTER": [0],
            "LOOKBACK":   [100, 150],
            "TOLERANCE":  [20, 30, 40],
            "MA_PERIOD":  [25,50],
            "TP_PCT":     [2, 3, 4],
            "SL_PCT":     [8, 9, 10],
# =============================================================================
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [150],
#             "TOLERANCE":  [40],
#             "MA_PERIOD":  [50],
#             "TP_PCT":     [3],
#             "SL_PCT":     [10],
# =============================================================================
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
# =============================================================================
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [100, 150],
#             "TOLERANCE":  [15, 30, 45],
#             "MA_PERIOD":  [25, 50],
#             "TP_PCT":     [2, 3],
#             "SL_PCT":     [7, 8, 9],
# =============================================================================
            "SELL_AFTER": [0],
            "LOOKBACK":   [150],
            "TOLERANCE":  [20],
            "MA_PERIOD":  [50],
            "TP_PCT":     [2],
            "SL_PCT":     [7.5],
        },
    },
]

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