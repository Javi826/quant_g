#experiments/multiverse_exp_FF.py
import os
import sys
import time
import random
import logging
import itertools
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.experiment.FF_combos")
logger.setLevel(logging.INFO)
DEBUG_METRICS = False
_metrics_level = logging.DEBUG if DEBUG_METRICS else logging.INFO
logging.getLogger("BOT_batch.pipeline.FF_test").setLevel(_metrics_level)
logging.getLogger("BOT_batch.pipeline.signal_cleaning").setLevel(_metrics_level)

# Silence per-combo backtest chatter — too noisy across hundreds of runs.
logging.getLogger("BOT_batch.pipeline.backtest_runner").setLevel(logging.WARNING)
logging.getLogger("joblib").setLevel(logging.WARNING)

import shared_batchs.symbols.universe as universe_module
from shared_batchs.symbols.universe import filter_symbols, select_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.rule_mining.rule_runner import _build_rule_dicts
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from FF_test import pipe_FF_test
from univers_exp_FF import _run_backtest_universe

# =============================================================================
# GENERAL
# =============================================================================
RANDOM_SEED       = 42
N_JOBS            = -1
SIGNAL_CLEANING   = True

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
SYMBOL_POOL = [
    "BTCUSDT","ETHUSDT", 
    "ADAUSDT","AVAXUSDT", 
    "BCHUSDT","BNBUSDT",
    "DOGEUSDT","LINKUSDT",
    "NEARUSDT","SOLUSDT", 
    "XLMUSDT","XRPUSDT",
] 

TIMEFRAMES   = ["1H"]
COMBO_SIZES  = [1,2,5,10]

# Sample size per combo size. None = exhaustive (used automatically for N=1).
N_SAMPLES_PER_SIZE = {
    1: None,
    2: 20,
    5: 40,
   10: 20,
}

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [6,8,10],
    "SL_PCT":     [6,8],
}

RANK_PERCENTILES = [95,98,99,99.9,99.99]
TOP_N_DISPLAY    = 20

# =============================================================================
# COMBO GENERATION
# =============================================================================
def _generate_combos(pool: list, size: int, n_samples: int | None, seed: int) -> list:
    all_combos = list(itertools.combinations(pool, size))
    if n_samples is None or n_samples >= len(all_combos):
        return all_combos
    rng = random.Random(seed)
    return rng.sample(all_combos, n_samples)

# =============================================================================
# POOL DATA LOADING — once per timeframe, reused across every combo
# =============================================================================
def _load_pool_data(timeframe: str) -> tuple:
    original_flag = universe_module.ENABLE_INCLUDE_FILTER
    original_list = universe_module.INCLUDED_SYMBOLS
    universe_module.ENABLE_INCLUDE_FILTER = True
    universe_module.INCLUDED_SYMBOLS = SYMBOL_POOL
    try:
        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
    finally:
        universe_module.ENABLE_INCLUDE_FILTER = original_flag
        universe_module.INCLUDED_SYMBOLS = original_list

    ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)
    return ohlcv_is, ohlcv_arr

# =============================================================================
# RANKING SCORE — mean of %<Real at the chosen percentiles
# =============================================================================
def _ranking_score(ff_result: dict, rank_percentiles: list) -> float:
    percentiles      = ff_result["percentiles"]
    pct_below_actual = ff_result["pct_below_actual"]
    idx = [int(np.where(np.isclose(percentiles, p))[0][0]) for p in rank_percentiles]
    return float(np.mean(pct_below_actual[idx]))

# =============================================================================
# SINGLE COMBO RUN
# =============================================================================
def _run_combo(combo: tuple, ohlcv_is_pool: dict, ohlcv_arr_pool: dict, timeframe: str) -> dict | None:
    ohlcv_is_combo  = {sym: ohlcv_is_pool[sym] for sym in combo}
    ohlcv_arr_combo = {sym: ohlcv_arr_pool[sym] for sym in combo}

    rules = _build_rule_dicts(ohlcv_is_combo, timeframe, RULE_MAX_DEPTH)

    try:
        _, _, matrix_arr, col_names = _run_backtest_universe(
            rules                  = rules,
            ohlcv_arr              = ohlcv_arr_combo,
            param_grid             = PARAM_GRID,
            order_amount           = ORDER_AMOUNT,
            timeframe              = timeframe,
            n_jobs                 = N_JOBS,
            apply_signal_cleaning  = SIGNAL_CLEANING,
        )
        ff_result = pipe_FF_test(
            matrix_arr = matrix_arr,
            col_names  = col_names,
            timeframe  = timeframe,
        )
    except ValueError as exc:
        logger.warning(f"SKIP ── {timeframe} ── {combo} ── {exc}")
        return None

    if ff_result is None:
        return None

    return {
        "timeframe": timeframe,
        "n_symbols": len(combo),
        "symbols":   "+".join(combo),
        "score":     _ranking_score(ff_result, RANK_PERCENTILES),
    }

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 100}")
    logger.info("  FF BOOTSTRAP — SYMBOL COMBINATION EXPERIMENT")
    logger.info(f"{'─' * 100}")
    logger.info(f"  SYMBOL_POOL        : {SYMBOL_POOL}")
    logger.info(f"  COMBO_SIZES        : {COMBO_SIZES}")
    logger.info(f"  N_SAMPLES_PER_SIZE : {N_SAMPLES_PER_SIZE}")
    logger.info(f"  RANK_PERCENTILES   : {RANK_PERCENTILES}")
    logger.info(f"{'─' * 100}\n")

    all_rows = []

    for timeframe in TIMEFRAMES:
        ohlcv_is_pool, ohlcv_arr_pool = _load_pool_data(timeframe)

        for size in COMBO_SIZES:
            combos = _generate_combos(SYMBOL_POOL, size, N_SAMPLES_PER_SIZE.get(size), RANDOM_SEED)
            logger.info(f"\n{'=' * 100}")
            logger.info(f"  {timeframe.upper()} ── N={size} ── RUNNING {len(combos)} COMBO(S)")
            logger.info(f"{'=' * 100}\n")

            for combo in combos:
                logger.info(f"{'-' * 100}")
                logger.info(f"Testing: {' + '.join(combo)}")
                logger.info(f"{'-' * 100}")
                row = _run_combo(combo, ohlcv_is_pool, ohlcv_arr_pool, timeframe)
                if row is not None:
                    all_rows.append(row)

    results_df = pd.DataFrame(all_rows)

    for timeframe in TIMEFRAMES:
        subset = results_df[results_df["timeframe"] == timeframe].sort_values("score", ascending=False)
        logger.info(f"\n{'=' * 100}")
        logger.info(f"  TOP {TOP_N_DISPLAY} ── {timeframe.upper()} ── RANKED BY MEAN %<REAL AT {RANK_PERCENTILES}")
        logger.info(f"{'=' * 100}")
        if DEBUG_METRICS:
            logger.info(subset.head(TOP_N_DISPLAY).to_string(index=False))
        else:
            logger.info(subset.head(TOP_N_DISPLAY)[["symbols", "n_symbols", "score"]].to_string(index=False))

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")