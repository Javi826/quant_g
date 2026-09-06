#BOT_batch_E1/main_pipeline.py (crypto)
import os
import sys
import time
import logging
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
MODULE_LOG_LEVELS = {
    "BOT_batch.pipeline.universe":          logging.INFO,
    "BOT_batch.rule_mining.generator":      logging.INFO,
    "BOT_batch.pipeline.signal_cleaning":   logging.INFO,
    "BOT_batch.pipeline.backtest_runner":   logging.INFO,
    "BOT_batch.pipeline.stepM":             logging.INFO,
    "BOT_batch.pipeline.wfo":               logging.INFO,
    "BOT_batch.engines.wfo_WF":             logging.INFO,
    "BOT_batch.pipeline.correlation":       logging.INFO,
    "BOT_batch.pipeline.montecarlo":        logging.INFO,
    "BOT_batch.pipeline.multiverse":        logging.INFO,
    "BOT_batch.runs.run_best_wfo_portfolio":logging.INFO,
    "BOT_batch.rule_mining.writter":        logging.INFO,
    "BOT_batch.runs.run_deploy":            logging.INFO,
    "BOT_batch.utils.reporting":            logging.INFO,
}
for module_name, level in MODULE_LOG_LEVELS.items():
    logging.getLogger(module_name).setLevel(level)
for noisy_logger in ("joblib", "matplotlib", "numba"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
 #-----------------------------------------------------------------------------
   
from shared_batchs.symbols.universe import build_universe, MIN_START_DATE_BY_DATASET
from shared_batchs.setup.config_paths import DATA_FOLDER_BY_DATASET
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.pipeline.wfo import WFO_WINDOW_CONFIG, EMA_ALPHA, WFO_NET_GAIN_TH, WFO_DD_TH, WFO_R2_TH, WFO_WFR_TH
from shared_batchs.pipeline.correlation import CORRELATION_DD_TH
from shared_batchs.pipeline.montecarlo import MONTECARLO_RUIN_TH
from shared_batchs.pipeline.multiverse import MULTIVERSE_PVALUE_TH
from shared_batchs.pipeline.stepM import STEPM_K_ESIME_TF
from shared_batchs.pipeline.signal_cleaning import JACCARD_SIMILARITY_TH
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import  ORDER_AMOUNT
from shared_batchs.rule_mining.rule_runner import run_rule_mining_pipeline

# =============================================================================
# RUNS + OUTPUTS — portfolio construction and output stages
# =============================================================================
SHOW_PLOTS    = True
SAVE_TRADES   = False
RUN_PORTFOLIO = True
RUN_DEPLOY    = True
#------------------------------------------------------------------------------
SPLIT_MODE = False

DATASET_MINING, DATASET_VALIDATION = ("IS", "OOS") if SPLIT_MODE else ("MERGED", "MERGED")
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

TIMEFRAMES = ["1H","4H","6Hutc","12Hutc"]
#TIMEFRAMES = ["12Hutc"]

SYMBOLS_BY_TIMEFRAME = {
    "1H":     ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "4H":     ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "6Hutc":  ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
    "12Hutc": ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"],
}

PARAM_GRID = {
    "SELL_AFTER": [0],
    "TP_PCT":     [6,8,10],
    "SL_PCT":     [6,8],
}

# =============================================================================
# PIPELINES — sequential validation filters
# =============================================================================
PIPELINE_WFO         = True
PIPELINE_CORRELATION = True
PIPELINE_MONTECARLO  = True
PIPELINE_MULTIVERSE  = True

STRATEGIES_E1_FOLDER = os.path.join(os.path.dirname(__file__), "strategies_E1")
SYMBOLS_LIVE_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "symbols_live")
BRIEF_TRADES_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "brief_trades")
DEPLOY_OUTPUT_PATH   = os.path.join(STRATEGIES_E1_FOLDER, "rules_files", "rules_batch.py")

# =============================================================================
# RUN CONFIG — single source of truth: printed at startup AND persisted
# =============================================================================
run_config = {"SPLIT_MODE": SPLIT_MODE, "DATASET_MINING": DATASET_MINING, "DATASET_VALIDATION": DATASET_VALIDATION, "TIMEFRAMES": TIMEFRAMES, "SYMBOLS_BY_TIMEFRAME": SYMBOLS_BY_TIMEFRAME, "PARAM_GRID": PARAM_GRID, "WFO_WINDOW_CONFIG": {tf: WFO_WINDOW_CONFIG.get(tf, {}) for tf in TIMEFRAMES}, "EMA_ALPHA": EMA_ALPHA, "PIPELINE_WFO": PIPELINE_WFO, "PIPELINE_CORRELATION": PIPELINE_CORRELATION, "PIPELINE_MONTECARLO": PIPELINE_MONTECARLO, "PIPELINE_MULTIVERSE": PIPELINE_MULTIVERSE,
              "WFO_NET_GAIN_TH": WFO_NET_GAIN_TH, "WFO_DD_TH": WFO_DD_TH, "WFO_R2_TH": WFO_R2_TH, "WFO_WFR_TH": WFO_WFR_TH, "CORRELATION_DD_TH": CORRELATION_DD_TH, "MONTECARLO_RUIN_TH": MONTECARLO_RUIN_TH, "MULTIVERSE_PVALUE_TH": MULTIVERSE_PVALUE_TH, "STEPM_K_ESIME": {tf: STEPM_K_ESIME_TF[tf] for tf in TIMEFRAMES}, "JACCARD_SIMILARITY_TH": JACCARD_SIMILARITY_TH}
# =============================================================================
# LOGGING HELPERS — render the startup banner from run_config
# =============================================================================
def _format_wfo_windows(wfo_window_config: dict) -> str:
    return "  |  ".join(
        f"{tf}: train={cfg.get('train_months')}m test={cfg.get('test_months')}m"
        for tf, cfg in wfo_window_config.items()
    )
def _pipeline_icon(enabled: bool) -> str:
    return "🟢" if enabled else "⚪"

def log_run_config() -> None:
    logger.info(f"\n{'─' * 115}")
    logger.info(f"  RULE MINING START")
    logger.info(f"{'─' * 115}")
    if SPLIT_MODE:
        logger.info(
            f"  DATASET     : SPLIT ── mining: {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET_MINING])} "
            f"({MIN_START_DATE_BY_DATASET[DATASET_MINING]}) | "
            f"validation: {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET_VALIDATION])} "
            f"({MIN_START_DATE_BY_DATASET[DATASET_VALIDATION]})"
        )
    else:
        logger.info(
            f"  DATASET     : {DATASET_MINING} ── {os.path.basename(DATA_FOLDER_BY_DATASET[DATASET_MINING])} "
            f"({MIN_START_DATE_BY_DATASET[DATASET_MINING]})"
        )
    logger.info(f"  SYMBOLS     :")
    for tf, symbols in SYMBOLS_BY_TIMEFRAME.items():
        logger.info(f"    {tf:<10}({len(symbols)}) {symbols}")
    logger.info(f"  TIMEFRAMES  : {TIMEFRAMES}")
    logger.debug(f"  MAX DEPTH  : {RULE_MAX_DEPTH}")
    logger.info(f"  PARAM GRID  : {PARAM_GRID}")
    logger.info(f"  WFO WINDOWS : {_format_wfo_windows({tf: WFO_WINDOW_CONFIG.get(tf, {}) for tf in TIMEFRAMES})} | EMA_ALPHA: {EMA_ALPHA}")
    logger.info(
        f"  PIPELINES   : WFO: {_pipeline_icon(PIPELINE_WFO)}  "
        f"CORRELATION: {_pipeline_icon(PIPELINE_CORRELATION)}  "
        f"MONTECARLO: {_pipeline_icon(PIPELINE_MONTECARLO)}  "
        f"MULTIVERSE: {_pipeline_icon(PIPELINE_MULTIVERSE)}"
    )
    logger.info(
        f"  PIPES       : JACCARD_TH={JACCARD_SIMILARITY_TH} | "
     #   f"STEPM_K_MODE={STEPM_K_MODE} | "
        f"K_ESIME={run_config['STEPM_K_ESIME']} | "
        f"NET_GAIN_TH={WFO_NET_GAIN_TH} DD_TH={WFO_DD_TH} R2_TH={WFO_R2_TH} WFR_TH={WFO_WFR_TH} | "
        f"CORR_TH={CORRELATION_DD_TH} | "
        f"MC_RUIN_TH={MONTECARLO_RUIN_TH} | "
        f"MV_PVALUE_TH={MULTIVERSE_PVALUE_TH}"
    )
    logger.info(
        f"  RUNS        : BEST PORTFOLIO: {_pipeline_icon(RUN_PORTFOLIO)}  "
        f"DEPLOY: {_pipeline_icon(RUN_DEPLOY)}"
    )
    logger.info(f"{'─' * 115}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()
    try:
        missing_tf = [tf for tf in TIMEFRAMES if tf not in SYMBOLS_BY_TIMEFRAME]
        if missing_tf:
            raise ValueError(f"SYMBOLS_BY_TIMEFRAME is missing entries for timeframes: {missing_tf}")

        log_run_config()

        # -------------------------------------------------------------------
        # DATA LOADING — one call validates and loads every symbol, every
        # timeframe, up front (fails fast if any symbol is bad).
        # -------------------------------------------------------------------
        ohlcv_data_mining_by_timeframe = build_universe(
            DATA_FOLDER_BY_DATASET[DATASET_MINING], SYMBOLS_BY_TIMEFRAME,
            dataset=DATASET_MINING,
        )
        ohlcv_arr_mining_by_timeframe  = {
            timeframe: prepare_ohlcv_arrays(ohlcv_df)
            for timeframe, ohlcv_df in ohlcv_data_mining_by_timeframe.items()
        }

        if SPLIT_MODE:
            ohlcv_data_validation_by_timeframe = build_universe(
                DATA_FOLDER_BY_DATASET[DATASET_VALIDATION], SYMBOLS_BY_TIMEFRAME,
                dataset=DATASET_VALIDATION,
            )
            ohlcv_arr_validation_by_timeframe = {
                timeframe: prepare_ohlcv_arrays(ohlcv_df)
                for timeframe, ohlcv_df in ohlcv_data_validation_by_timeframe.items()
            }
        else:
            # Single-source mode: both roles share the same already-loaded dataset.
            ohlcv_data_validation_by_timeframe = ohlcv_data_mining_by_timeframe
            ohlcv_arr_validation_by_timeframe  = ohlcv_arr_mining_by_timeframe
        # -------------------------------------------------------------------
        # RULE MINING — Phase A: DSR for every timeframe, then a combined
        # -------------------------------------------------------------------
        validated_wfo_test, all_mbias_results = run_rule_mining_pipeline(
            ohlcv_data_mining_by_timeframe     = ohlcv_data_mining_by_timeframe,
            ohlcv_arr_mining_by_timeframe      = ohlcv_arr_mining_by_timeframe,
            ohlcv_data_validation_by_timeframe = ohlcv_data_validation_by_timeframe,
            ohlcv_arr_validation_by_timeframe  = ohlcv_arr_validation_by_timeframe,
            timeframes                         = TIMEFRAMES,
            param_grid                         = PARAM_GRID,
            order_amount                       = ORDER_AMOUNT,
            data_folder                        = DATA_FOLDER_BY_DATASET[DATASET_VALIDATION],
            max_depth                          = RULE_MAX_DEPTH,
            log_level                          = MODULE_LOG_LEVELS["BOT_batch.pipeline.wfo"],
            save_trades                        = SAVE_TRADES,
            brief_trades_folder                = BRIEF_TRADES_FOLDER,
            show_plots                         = SHOW_PLOTS,
            symbols_live_folder                = SYMBOLS_LIVE_FOLDER,
            deploy_output_path                 = DEPLOY_OUTPUT_PATH,
            run_config                         = run_config,
            pipeline_wfo                       = PIPELINE_WFO,
            pipeline_correlation               = PIPELINE_CORRELATION,
            pipeline_montecarlo                = PIPELINE_MONTECARLO,
            pipeline_multiverse                = PIPELINE_MULTIVERSE,
            run_best_portfolio                 = RUN_PORTFOLIO,
            run_deploy                         = RUN_DEPLOY,
        )

        elapsed = int(time.time() - start)
        logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    except KeyboardInterrupt:
        elapsed = int(time.time() - start)
        logger.info(f"\n⛔  INTERRUPTED BY USER — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
        sys.exit(0)