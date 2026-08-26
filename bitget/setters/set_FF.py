#BOT_batch/set_FF.py
import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_comp")
logger.setLevel(LOG_LEVEL)

DSR_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.backtest_runner").setLevel(DSR_LOG_LEVEL)

FF_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.FF_test").setLevel(FF_LOG_LEVEL)
#------------------------------------------------------------------------------
REPORTING_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.utils.reporting").setLevel(REPORTING_LOG_LEVEL)

logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from shared_batchs.symbols.universe import filter_symbols, select_universe, select_top_n_by_volume
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.rule_mining.rule_runner import _build_rule_dicts
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from shared_batchs.pipeline import backtest_runner as backtest_module
from FF_test import pipe_FF_test
from shared_batchs.pipeline.signal_cleaning import pipe_signal_cleaning_jaccard
# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION
# =============================================================================
N_JOBS = -1  # -1 = use all available cores, for both the backtest search and the StepM bootstrap

TIMEFRAMES = ["1H","4H", "6Hutc", "12Hutc"]
N_SYMBOLS  = 2

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [6,8,10],
    "SL_PCT":     [6,8],
}
# =============================================================================
# SIGNAL CLEANING — dedupe near-identical rules before the brute backtest
# =============================================================================
SIGNAL_CLEANING_TEST = True

# =============================================================================
# FF BOOTSTRAP — standalone diagnostic
# =============================================================================
FF_TEST            = True
FF_SAMPLE_REPLICAS = 5     # raw null replicas kept for the histogram overlay
FF_SHOW_PLOTS      = True

# =============================================================================
# BACKTEST UNIVERSE — build the raw universe once, hand it to the FF pipe.
# =============================================================================
def _run_backtest_universe(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    timeframe: str,
    n_jobs: int = -1,
    apply_signal_cleaning: bool = False,
) -> tuple:
    """Run the brute-force backtest once; shared by the method comparison
    and any standalone diagnostic (e.g. FF) that needs the same matrix."""
    if apply_signal_cleaning:
        rules = pipe_signal_cleaning_jaccard(
            rules     = rules,
            ohlcv_arr = ohlcv_arr,
            timeframe = timeframe,
        )

    original_n_jobs = backtest_module.BACKTEST_N_JOBS
    backtest_module.BACKTEST_N_JOBS = n_jobs
    try:
        return backtest_module.pipe_backtesting(
            rules        = rules,
            ohlcv_arr    = ohlcv_arr,
            param_grid   = param_grid,
            order_amount = order_amount,
            timeframe    = timeframe,
        )
    finally:
        backtest_module.BACKTEST_N_JOBS = original_n_jobs


# =============================================================================
# FF BOOTSTRAP PLOTS — visual read of the same numbers pipe_FF_test logs
# =============================================================================
def _plot_ff_bootstrap(result: dict, timeframe: str) -> None:
    if result is None:
        logger.warning(f"FF PLOT ── {timeframe} ── no result available, skipping")
        return

    fig, (ax_hist, ax_pct) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"FF Bootstrap ── {timeframe}")

    # ---- Left: real cross-section vs a sample of null replicas ------------
    real_tstat = result["real_tstat"]
    ax_hist.hist(real_tstat, bins=100, density=True, alpha=0.6, label="Real (observed)", color="tab:blue")

    sim_sample = result.get("sim_tstat_sample")
    if sim_sample is not None:
        sim_flat = sim_sample[np.isfinite(sim_sample)]
        ax_hist.hist(sim_flat, bins=100, density=True, alpha=0.5, label="Null (bootstrap replicas)", color="tab:orange")

    ax_hist.set_xlabel("t(α)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Cross-sectional distribution")
    ax_hist.legend()

    # ---- Right: Sim vs Act percentile curve --------------------------------
    percentiles      = result["percentiles"]
    real_percentiles = result["real_percentiles"]
    sim_percentiles  = result["sim_percentiles"]
    x_pos            = np.arange(len(percentiles))

    ax_pct.plot(x_pos, real_percentiles, marker="o", label="Act (real)", color="tab:blue")
    ax_pct.plot(x_pos, sim_percentiles, marker="o", label="Sim (null)", color="tab:orange")
    ax_pct.set_xticks(x_pos)
    ax_pct.set_xticklabels([f"{p:g}" for p in percentiles])
    ax_pct.set_xlabel("percentile")
    ax_pct.set_ylabel("t(α)")
    ax_pct.set_title("Sim vs Act by percentile")
    ax_pct.legend()

    fig.tight_layout()
    plt.show()
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  FF BOOTSTRAP — FULL BRUTE UNIVERSE DIAGNOSTIC")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES     : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS      : {N_SYMBOLS}")
    logger.debug(f"  MAX_DEPTH      : {RULE_MAX_DEPTH}")
    logger.info(f"  PARAM_GRID     : {PARAM_GRID}")
    logger.info(f"  SIGNAL_CLEANING: {SIGNAL_CLEANING_TEST}")
    logger.info(f"  FF_TEST        : {FF_TEST}")
    logger.info(f"{'─' * 115}\n")

    # -------------------------------------------------------------------
    # DATA LOADING — cheap, sequential across timeframes.
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
        ohlcv_is = select_top_n_by_volume(ohlcv_is, N_SYMBOLS)
        ohlcv_data_by_timeframe[timeframe] = ohlcv_is
        ohlcv_arr_by_timeframe[timeframe]  = prepare_ohlcv_arrays(ohlcv_is)

    # -------------------------------------------------------------------
    # FF BOOTSTRAP — one diagnostic per timeframe, on the full brute
    # -------------------------------------------------------------------
    for timeframe in TIMEFRAMES:

        rules_for_timeframe = _build_rule_dicts(
            ohlcv_data_by_timeframe[timeframe], timeframe, RULE_MAX_DEPTH,
        )

        raw_results, n_combos, matrix_arr, col_names = _run_backtest_universe(
            rules                  = rules_for_timeframe,
            ohlcv_arr              = ohlcv_arr_by_timeframe[timeframe],
            param_grid              = PARAM_GRID,
            order_amount           = ORDER_AMOUNT,
            timeframe               = timeframe,
            n_jobs                  = N_JOBS,
            apply_signal_cleaning   = SIGNAL_CLEANING_TEST,
        )

        if FF_TEST:
            ff_result = pipe_FF_test(
                matrix_arr        = matrix_arr,
                col_names         = col_names,
                timeframe         = timeframe,
                n_sample_replicas = FF_SAMPLE_REPLICAS,
            )
            if FF_SHOW_PLOTS:
                _plot_ff_bootstrap(ff_result, timeframe)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")