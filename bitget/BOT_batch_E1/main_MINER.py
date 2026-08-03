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

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.DEBUG, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_rule_mining")
logger.setLevel(LOG_LEVEL)
#UNIVERSE0
#------------------------------------------------------------------------------
UNIVERSE_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.universe").setLevel(UNIVERSE_LOG_LEVEL)
#RULE_MINNING
#------------------------------------------------------------------------------
RULE_RUNNER_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.rule_mining.runner").setLevel(RULE_RUNNER_LOG_LEVEL)
RULE_GENERATOR_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.rule_mining.generator").setLevel(RULE_GENERATOR_LOG_LEVEL)
#DSR
#------------------------------------------------------------------------------
DSR_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.dsr").setLevel(DSR_LOG_LEVEL)
#WFO
#------------------------------------------------------------------------------
WFO_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.wfo").setLevel(WFO_LOG_LEVEL)
logging.getLogger("BOT_batch.engines.wfo_WF").setLevel(WFO_LOG_LEVEL)
#CORRELATION
#------------------------------------------------------------------------------
CORRELATION_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.correlation").setLevel(CORRELATION_LOG_LEVEL)
#MONTECARLO
#------------------------------------------------------------------------------
MONTECARLO_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.montecarlo").setLevel(MONTECARLO_LOG_LEVEL)
#MULTIVERSE
#------------------------------------------------------------------------------
MULTIVERSE_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.multiverse").setLevel(MULTIVERSE_LOG_LEVEL)
#BESTPORFTOLIO
#------------------------------------------------------------------------------
RUN_PORTFOLIO_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.runs.run_best_wfo_portfolio").setLevel(RUN_PORTFOLIO_LOG_LEVEL)
#DEPLOY
#------------------------------------------------------------------------------
DEPLOY_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.rule_mining.deploy").setLevel(DEPLOY_LOG_LEVEL)
logging.getLogger("BOT_batch.runs.run_deploy").setLevel(DEPLOY_LOG_LEVEL)
#REPORTING
#------------------------------------------------------------------------------
REPORTING_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.utils.reporting").setLevel(REPORTING_LOG_LEVEL)

logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
from shared_batchs.symbols.universe import filter_symbols, select_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.pipeline.wfo import WFO_WINDOW_CONFIG, EMA_ALPHA
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from shared_batchs.rule_mining.rule_runner import run_rule_mining_pipeline

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION
# =============================================================================
DTYPE        = np.float32
RULES_N_JOBS = -1
INNER_N_JOBS = 1

# =============================================================================
# RUNS + OUTPUTS — portfolio construction and output stages
# =============================================================================
SHOW_PLOTS    = True
SAVE_TRADES   = False
RUN_PORTFOLIO = True
RUN_DEPLOY    = False
#------------------------------------------------------------------------------

TIMEFRAMES = ["1H","4H","6Hutc","12Hutc"]
#TIMEFRAMES = ["15m","30m"]
N_SYMBOLS  = 10

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [2,4,6,8,10],
    "SL_PCT":     [2,4,6,8,10],
}

# =============================================================================
# PIPELINES — sequential validation filters
# =============================================================================

PIPELINE_DSR         = True
DSR_TH               = 0.8

WFO_NET_GAIN_TH      = 30
WFO_DD_TH            = 15
WFO_R2_TH            = 0.8
WFO_WFR_TH           = 0.6

PIPELINE_CORRELATION = True
CORRELATION_DD_TH    = 0.55
PIPELINE_MONTECARLO  = True
MONTECARLO_RUIN_TH   = 10
PIPELINE_MULTIVERSE  = True
MULTIVERSE_PVALUE_TH = 0.05

STRATEGIES_E1_FOLDER = os.path.join(os.path.dirname(__file__), "strategies_E1")
SYMBOLS_LIVE_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "symbols_live")
BRIEF_TRADES_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "brief_trades")
DEPLOY_OUTPUT_PATH   = os.path.join(STRATEGIES_E1_FOLDER, "rules_files", "rules_batch.py")

run_config = {"DSR_TH":DSR_TH,"WFO_NET_GAIN_TH":WFO_NET_GAIN_TH,"WFO_DD_TH":WFO_DD_TH,"WFO_R2_TH":WFO_R2_TH,"WFO_WFR_TH":WFO_WFR_TH,"CORRELATION_DD_TH":CORRELATION_DD_TH,"MONTECARLO_RUIN_TH": MONTECARLO_RUIN_TH,"MULTIVERSE_PVALUE_TH": MULTIVERSE_PVALUE_TH}
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
    logger.debug(f"  MAX DEPTH  : {RULE_MAX_DEPTH}")
    logger.info(f"  PARAM GRID  : {PARAM_GRID}")
    _windows_str = "  |  ".join(
        f"{tf}: train={WFO_WINDOW_CONFIG.get(tf, {}).get('train_months')}m test={WFO_WINDOW_CONFIG.get(tf, {}).get('test_months')}m"
        for tf in TIMEFRAMES
    )
    logger.info(f"  WFO WINDOWS : {_windows_str}")
    logger.info(f"  EMA_ALPHA   : {EMA_ALPHA}")
    logger.info(
        f"  PIPELINES   : DSR: {'🟢' if PIPELINE_DSR else '⚪'}  "
        f"WFO: 🟢  "
        f"CORRELATION: {'🟢' if PIPELINE_CORRELATION else '⚪'}  "
        f"MONTECARLO: {'🟢' if PIPELINE_MONTECARLO else '⚪'}  "
        f"MULTIVERSE: {'🟢' if PIPELINE_MULTIVERSE else '⚪'}"
    )
    logger.info(f"  DSR         : DSR_TH={DSR_TH}")
    logger.info(f"  WFO         : NET_GAIN_TH={WFO_NET_GAIN_TH}  DD_TH={WFO_DD_TH}  R2_TH={WFO_R2_TH}  WFR_TH={WFO_WFR_TH}")
    logger.info(f"  CORRELATION : DD_TH={CORRELATION_DD_TH}")
    logger.info(f"  MONTECARLO  : RUIN_TH={MONTECARLO_RUIN_TH}")
    logger.info(f"  MULTIVERSE  : PVALUE_TH={MULTIVERSE_PVALUE_TH}")
    logger.info(
        f"  RUNS        : BEST PORTFOLIO: {'🟢' if RUN_PORTFOLIO else '⚪'}  "
        f"DEPLOY: {'🟢' if RUN_DEPLOY else '⚪'}"
    )
    logger.info(f"{'─' * 115}\n")

    # -------------------------------------------------------------------
    # DATA LOADING — cheap, sequential across timeframes. Rule mining
    # -------------------------------------------------------------------
    ohlcv_data_by_timeframe = {}
    ohlcv_arr_by_timeframe  = {}

    for timeframe in TIMEFRAMES:
        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
        ohlcv_data_by_timeframe[timeframe] = ohlcv_is
        ohlcv_arr_by_timeframe[timeframe]  = prepare_ohlcv_arrays(ohlcv_is)
    # -------------------------------------------------------------------
    # RULE MINING — Phase A: DSR for every timeframe, then a combined
    # -------------------------------------------------------------------
    validated_wfo_test = run_rule_mining_pipeline(
        ohlcv_data_by_timeframe = ohlcv_data_by_timeframe,
        ohlcv_arr_by_timeframe  = ohlcv_arr_by_timeframe,
        timeframes              = TIMEFRAMES,
        param_grid              = PARAM_GRID,
        order_amount            = ORDER_AMOUNT,
        net_gain_th             = WFO_NET_GAIN_TH,
        dd_th                   = WFO_DD_TH,
        r2_th                   = WFO_R2_TH,
        wfr_th                  = WFO_WFR_TH,
        dtype                   = DTYPE,
        dsr_th                  = DSR_TH,
        data_folder             = DATA_FOLDER_IS,
        run_dsr                 = PIPELINE_DSR,
        rules_n_jobs            = RULES_N_JOBS,
        inner_n_jobs            = INNER_N_JOBS,
        n_symbols               = N_SYMBOLS,
        max_depth               = RULE_MAX_DEPTH,
        log_level               = WFO_LOG_LEVEL,
        save_trades             = SAVE_TRADES,
        brief_trades_folder     = BRIEF_TRADES_FOLDER,
        show_plots              = SHOW_PLOTS,
        # ---- PIPELINES ----
        pipeline_correlation    = PIPELINE_CORRELATION,
        correlation_threshold   = CORRELATION_DD_TH,
        pipeline_montecarlo     = PIPELINE_MONTECARLO,
        montecarlo_ruin_th      = MONTECARLO_RUIN_TH,
        pipeline_multiverse     = PIPELINE_MULTIVERSE,
        multiverse_p_value_th   = MULTIVERSE_PVALUE_TH,
        # ---- RUNS ----
        run_best_portfolio      = RUN_PORTFOLIO,
        run_deploy              = RUN_DEPLOY,
        symbols_live_folder     = SYMBOLS_LIVE_FOLDER,
        deploy_output_path      = DEPLOY_OUTPUT_PATH,
        run_config = run_config,
    )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
    #profile_pipeline.print_summary()  # DIAGNOSTIC ONLY — remove after profiling