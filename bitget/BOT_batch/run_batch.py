#BOT_batch/run_batch
from main_batch import run_batch, run_portfolio_analysis


STRATEGIES = [
    {
        "strategy_id":  "03_parity_long_4H",
        "signal":       "parity_long",
        "side":         "long",
        "timeframe":    "4H",
        "n_symbols":    9,
        "order_amount": 80,
        "param_grid": {
# =============================================================================
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [100, 150],
#             "TOLERANCE":  [15, 30, 45],
#             "MA_PERIOD":  [25],
#             "TP_PCT":     [2, 3, 4],
#             "SL_PCT":     [8, 9, 10],
# =============================================================================
            "SELL_AFTER": [0],
            "LOOKBACK":   [150],
            "TOLERANCE":  [40],
            "MA_PERIOD":  [50],
            "TP_PCT":     [3],
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
import logging
import time

LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", force=True)
if __name__ == "__main__":
    _start = time.time()
    for strategy in STRATEGIES:
        ...
        run_batch(strategy)
    run_portfolio_analysis()
    _elapsed = int(time.time() - _start)
    logging.getLogger("BOT_trading.batch.main_batch").info(
        f"🏁 TOTAL — {_elapsed//3600} h {(_elapsed%3600)//60} min {_elapsed%60} s"
    )