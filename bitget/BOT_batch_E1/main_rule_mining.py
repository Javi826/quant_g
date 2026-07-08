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
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", stream=sys.stdout, force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("BOT_batch.runs.run_correlation").setLevel(logging.INFO)
logger = logging.getLogger("BOT_batch.main_rule_mining")

from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.backtesters.ZX_compute_BT import MIN_PRICE
from shared_batch_regime.config_paths import DATA_FOLDER_IS, DATA_FOLDER_OOS1
from shared_batchs.pipeline.wfo import WFO_WINDOW_CONFIG
from shared_batchs.rule_mining.rule_runner import run_rule_mining, finalize_rule_mining
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH

# =============================================================================
# RUN CONFIGURATION
# =============================================================================

DTYPE         = np.float32
RULES_N_JOBS  = 32
INNER_N_JOBS  = 1

TIMEFRAMES = ["6Hutc"]
N_SYMBOLS  = 10

ORDER_AMOUNT = 100

WFO_NET_GAIN_TH = 35
WFO_DD_TH       = 20

RULE_MAX_DEPTH = MAX_DEPTH

SHOW_PLOTS             = True
RUN_CORRELATION        = True
RUN_BEST_WFO_PORTFOLIO = True
RUN_DEPLOY             = True
SAVE_TRADES            = False

CORRELATION_DD_THRESHOLD = 0.70

STRATEGIES_E1_FOLDER = os.path.join(os.path.dirname(__file__), "strategies_E1")
SYMBOLS_LIVE_FOLDER   = os.path.join(STRATEGIES_E1_FOLDER, "symbols_live")
BRIEF_TRADES_FOLDER   = os.path.join(STRATEGIES_E1_FOLDER, "brief_trades")
DEPLOY_OUTPUT_PATH    = os.path.join(STRATEGIES_E1_FOLDER, "rules_files", "rules_batch.py")

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
    logger.info(f"  TIMEFRAMES      : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS       : {N_SYMBOLS}")
    logger.debug(f"  MAX DEPTH      : {RULE_MAX_DEPTH}")
    logger.info(f"  PARAM GRID      : {PARAM_GRID}")
    _windows_str = "  |  ".join(
        f"{tf}: train={WFO_WINDOW_CONFIG.get(tf, {}).get('train_months')}m test={WFO_WINDOW_CONFIG.get(tf, {}).get('test_months')}m"
        for tf in TIMEFRAMES
    )
    logger.info(f"  WFO WINDOWS     : {_windows_str}")
    logger.info(
        f"  RUN CORRELATION : {'🟢' if RUN_CORRELATION else '⚪'}  "
        f"RUN BEST PORTFOLIO: {'🟢' if RUN_BEST_WFO_PORTFOLIO else '⚪'}  "
        f"RUN DEPLOY: {'🟢' if RUN_DEPLOY else '⚪'}"
    )
    logger.info(f"{'=' * 115}\n")

    all_raw_results         = []
    ohlcv_data_by_timeframe = {}

    for timeframe in TIMEFRAMES:
        tf_start = time.time()

        symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos1 = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            data_folder_oos   = DATA_FOLDER_OOS1,
            timeframe         = timeframe,
            n_symbols         = N_SYMBOLS,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
        ohlcv_data_by_timeframe[timeframe] = ohlcv_is

        raw_results = run_rule_mining(
            ohlcv_data           = ohlcv_is,
            timeframe            = timeframe,
            param_grid           = PARAM_GRID,
            order_amount         = ORDER_AMOUNT,
            net_gain_th          = WFO_NET_GAIN_TH,
            dd_th                = WFO_DD_TH,
            dtype                = DTYPE,
            rules_n_jobs         = RULES_N_JOBS,
            inner_n_jobs         = INNER_N_JOBS,
            n_symbols            = N_SYMBOLS,
            max_depth            = RULE_MAX_DEPTH,
            log_level            = LOG_LEVEL,
            save_trades          = SAVE_TRADES,
            brief_trades_folder  = BRIEF_TRADES_FOLDER,
        )
        all_raw_results.extend(raw_results)

        tf_elapsed = int(time.time() - tf_start)
        logger.info(f"\n🏁 {timeframe} DONE — {tf_elapsed // 3600} h {(tf_elapsed % 3600) // 60} min {tf_elapsed % 60} s")

    finalize_rule_mining(
        all_raw_results          = all_raw_results,
        ohlcv_data_by_timeframe  = ohlcv_data_by_timeframe,
        param_grid               = PARAM_GRID,
        order_amount             = ORDER_AMOUNT,
        dtype                    = DTYPE,
        data_folder              = DATA_FOLDER_IS,
        inner_n_jobs             = INNER_N_JOBS,
        n_symbols                = N_SYMBOLS,
        show_plots               = SHOW_PLOTS,
        correlation_threshold    = CORRELATION_DD_THRESHOLD,
        run_correlation          = RUN_CORRELATION,
        run_best_portfolio       = RUN_BEST_WFO_PORTFOLIO,
        run_deploy               = RUN_DEPLOY,
        symbols_live_folder      = SYMBOLS_LIVE_FOLDER,
        deploy_output_path       = DEPLOY_OUTPUT_PATH,
    )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")