#BOT_batch/main_rule_mining.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch_regime")))

import time
import logging
import numpy as np

# LOGGING CONFIGURATION
#------------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logger = logging.getLogger("BOT_batch.main_rule_mining")

from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.backtesters.ZX_compute_BT import MIN_PRICE
from shared_batch_regime.config_paths import DATA_FOLDER_IS, DATA_FOLDER_OOS1
from shared_batchs.rule_mining.rule_runner import run_rule_mining
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH

# =============================================================================
# RUN CONFIGURATION
# =============================================================================

DTYPE   = np.float32
N_JOBS  = -1

TIMEFRAME = "6Hutc"
N_SYMBOLS = 10

ORDER_AMOUNT = 100

WFO_NET_GAIN_TH = 30
WFO_DD_TH       = 25

RULE_MAX_DEPTH = MAX_DEPTH

SHOW_PLOTS             = True
RUN_CORRELATION        = True
RUN_BEST_WFO_PORTFOLIO = True
CORRELATION_DD_THRESHOLD = 0.75

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [2,4,6,8,10],
    "SL_PCT":     [2,4,6,8,10],
}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'=' * 115}")
    logger.info(f"  RULE MINING START")
    logger.info(f"{'=' * 115}")
    logger.info(f"  Timeframe      : {TIMEFRAME}")
    logger.info(f"  Max depth      : {RULE_MAX_DEPTH}")
    logger.info(f"  Param grid     : {PARAM_GRID}")
    logger.info(f"{'=' * 115}\n")

    symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos1 = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        data_folder_oos   = DATA_FOLDER_OOS1,
        timeframe         = TIMEFRAME,
        n_symbols         = N_SYMBOLS,
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )

    run_rule_mining(
        ohlcv_data             = ohlcv_is,
        timeframe              = TIMEFRAME,
        param_grid             = PARAM_GRID,
        order_amount           = ORDER_AMOUNT,
        net_gain_th            = WFO_NET_GAIN_TH,
        dd_th                  = WFO_DD_TH,
        dtype                  = DTYPE,
        data_folder            = DATA_FOLDER_IS,
        n_jobs                 = N_JOBS,
        n_symbols              = N_SYMBOLS,
        max_depth              = RULE_MAX_DEPTH,
        show_plots             = SHOW_PLOTS,
        correlation_threshold  = CORRELATION_DD_THRESHOLD,
        run_correlation        = RUN_CORRELATION,
        run_best_portfolio     = RUN_BEST_WFO_PORTFOLIO,
    )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")