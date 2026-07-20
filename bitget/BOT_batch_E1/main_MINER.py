#BOT_batch/main_MINER.py
import os
import sys
import time
import logging
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch_regime")))

# LOGGING CONFIGURATION
#------------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_rule_mining")
RULE_RUNNER_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.rule_mining.runner").setLevel(RULE_RUNNER_LOG_LEVEL)
DSR_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.dsr").setLevel(DSR_LOG_LEVEL)
MULTIVERSE_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.multiverse").setLevel(MULTIVERSE_LOG_LEVEL)
DEPLOY_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.runs.run_deploy").setLevel(DEPLOY_LOG_LEVEL)

logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.backtesters.ZX_compute_BT import MIN_PRICE
from shared_batch_regime.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_runner import run_rule_mining, finalize_rule_mining
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.pipeline.wfo import WFO_WINDOW_CONFIG, EMA_ALPHA

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION
# =============================================================================
DTYPE        = np.float32
RULES_N_JOBS = -1
INNER_N_JOBS = 1

# =============================================================================
# MISC OUTPUT / DEBUG OPTIONS
# =============================================================================
SHOW_PLOTS  = True
SAVE_TRADES = False

TIMEFRAMES   = ["1H","4H","6Hutc","12Hutc"]
#TIMEFRAMES   = ["6Hutc","12Hutc"]
#TIMEFRAMES   = ["12Hutc"]
N_SYMBOLS    = 10
ORDER_AMOUNT = 100

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [2,4,6,8,10],
    "SL_PCT":     [2,4,6,8,10],
}

# =============================================================================
# WFO — Walk-Forward Optimization approval thresholds (Stage 1)
# =============================================================================
WFO_NET_GAIN_TH = 45
WFO_DD_TH       = 20
WFO_R2_TH       = 0.6
WFO_WFR_TH      = 0.5

# =============================================================================
# RUNS — portfolio construction and output stages
# =============================================================================

RUN_CORRELATION   = True
CORRELATION_DD_TH = 0.7
RUN_PORTFOLIO     = True
RUN_DEPLOY        = False

# =============================================================================
# PIPELINES — sequential validation filters (executed in this order)
# =============================================================================

PIPELINE_DSR         = True
DSR_TH               = 0.8
PIPELINE_MONTECARLO  = True
MONTECARLO_RUIN_TH   = 10
PIPELINE_MULTIVERSE  = False
MULTIVERSE_PVALUE_TH = 0.05

STRATEGIES_E1_FOLDER = os.path.join(os.path.dirname(__file__), "strategies_E1")
SYMBOLS_LIVE_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "symbols_live")
BRIEF_TRADES_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "brief_trades")
DEPLOY_OUTPUT_PATH   = os.path.join(STRATEGIES_E1_FOLDER, "rules_files", "rules_batch.py")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  RULE MINING START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES  : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS   : {N_SYMBOLS}")
    logger.info(f"  VALIDATION  : NET_GAIN_TH={WFO_NET_GAIN_TH}  DD_TH={WFO_DD_TH}  R2_TH={WFO_R2_TH}  WFR_TH={WFO_WFR_TH}")
    logger.debug(f"  MAX DEPTH  : {RULE_MAX_DEPTH}")
    logger.info(f"  PARAM GRID  : {PARAM_GRID}")
    _windows_str = "  |  ".join(
        f"{tf}: train={WFO_WINDOW_CONFIG.get(tf, {}).get('train_months')}m test={WFO_WINDOW_CONFIG.get(tf, {}).get('test_months')}m"
        for tf in TIMEFRAMES
    )
    logger.info(f"  WFO WINDOWS : {_windows_str}")
    logger.info(f"  EMA_ALPHA   : {EMA_ALPHA}")
    logger.info(
        f"  PIPELINES   : DSR: {'🟢' if PIPELINE_DSR else '⚪'} (DSR_TH={DSR_TH})  "
        f"MONTECARLO: {'🟢' if PIPELINE_MONTECARLO else '⚪'} (RUIN_TH={MONTECARLO_RUIN_TH})  "
        f"MULTIVERSE: {'🟢' if PIPELINE_MULTIVERSE else '⚪'} (PCT_TH={MULTIVERSE_PVALUE_TH})"
    )
    logger.info(
        f"  RUNS        : CORRELATION: {'🟢' if RUN_CORRELATION else '⚪'}  "
        f"BEST PORTFOLIO: {'🟢' if RUN_PORTFOLIO else '⚪'}  "
        f"DEPLOY: {'🟢' if RUN_DEPLOY else '⚪'}"
    )
    logger.info(f"{'─' * 115}\n")

    all_raw_results         = []
    ohlcv_data_by_timeframe = {}

    for timeframe in TIMEFRAMES:
        tf_start = time.time()

        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
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
            r2_th                = WFO_R2_TH,
            wfr_th               = WFO_WFR_TH,
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
        net_gain_th              = WFO_NET_GAIN_TH,
        dd_th                    = WFO_DD_TH,
        r2_th                    = WFO_R2_TH,
        wfr_th                   = WFO_WFR_TH,
        dtype                    = DTYPE,
        data_folder              = DATA_FOLDER_IS,
        inner_n_jobs             = INNER_N_JOBS,
        n_symbols                = N_SYMBOLS,
        show_plots               = SHOW_PLOTS,
        # ---- RUNS ----
        run_correlation          = RUN_CORRELATION,
        correlation_threshold    = CORRELATION_DD_TH,
        run_best_portfolio       = RUN_PORTFOLIO,
        run_deploy               = RUN_DEPLOY,
        symbols_live_folder      = SYMBOLS_LIVE_FOLDER,
        deploy_output_path       = DEPLOY_OUTPUT_PATH,
        # ---- PIPELINES ----
        run_dsr                  = PIPELINE_DSR,
        dsr_th                   = DSR_TH,
        pipeline_montecarlo      = PIPELINE_MONTECARLO,
        montecarlo_ruin_th       = MONTECARLO_RUIN_TH,
        pipeline_multiverse      = PIPELINE_MULTIVERSE,
        multiverse_p_value_th    = MULTIVERSE_PVALUE_TH,
    )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")