#BOT_batch/MAIN_validator.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batchs")))

import time
import logging
import importlib.util
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", stream=sys.stdout, force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger("BOT_batch.main_rule_validation")

from shared_batchs.symbols.universe import filter_symbols, select_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, INITIAL_BALANCE
from shared_batchs.pipeline.wfo import _run_wfo_for_rule
from shared_batchs.rule_mining.rule_runner import _build_rule_id, _print_ranking
from signals.signal_builder import build_signal_fn

# =============================================================================
# RUN CONFIGURATION
# =============================================================================
DTYPE        = np.float32
RULES_N_JOBS = -1
INNER_N_JOBS = 1
N_SYMBOLS    = 10   # must match len(symbols_to_include) in universe.py (MY_SYMBOLS=True)

WFO_NET_GAIN_TH = -10
WFO_DD_TH       = 50
WFO_R2_TH       = -1.0
WFO_WFR_TH      = 0.0   # 0.0 = disabled (no rule discarded on WFR)

SAVE_TRADES          = True
STRATEGIES_E1_FOLDER = os.path.join(os.path.dirname(__file__), "strategies_E1")
BRIEF_TRADES_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "brief_trades")
RULES_BATCH_PATH     = os.path.join(STRATEGIES_E1_FOLDER, "rules_files", "rules_batch_topV.py")

# =============================================================================
# LOAD PRODUCTION STRATEGIES
# =============================================================================

def _load_strategies(rules_batch_path: str) -> list:
    spec   = importlib.util.spec_from_file_location("rules_batch", rules_batch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STRATEGIES


def _build_validation_rule(i: int, timeframe: str, entry: dict) -> dict:
    """Builds a rule dict carrying its own fixed production params (no grid search)."""
    side = entry["direction"]
    return {
        "rule_id":         _build_rule_id(i, timeframe, {"side": side, "label": entry["id"]}),
        "timeframe":       timeframe,
        "side":            side,
        "specs":           entry["specs"],
        "label":           entry["id"],
        "signal_fn":       build_signal_fn(entry["specs"], side),
        "param_names":     ["SELL_AFTER", "TP_PCT", "SL_PCT"],
        "lists_for_grid":  [[entry["sell_after_ncandles"]], [entry["tp_pct"]], [entry["sl_pct"]]],
        "order_amount":    entry["order_amount"],
    }

# =============================================================================
# WFO — one rule at a time, each with its own fixed (non-grid) params
# =============================================================================

def _run_validation_wfo(rules: list, ohlcv_arr: dict, timeframe: str) -> list:
    total = len(rules)

    with tqdm_joblib(tqdm(desc=f"VALIDATION WFO {timeframe}", total=total, dynamic_ncols=True)):
        results = Parallel(n_jobs=RULES_N_JOBS)(
            delayed(_run_wfo_for_rule)(
                i, total, rule, ohlcv_arr, rule["param_names"], rule["lists_for_grid"],
                rule["order_amount"], timeframe, WFO_NET_GAIN_TH, WFO_DD_TH, WFO_R2_TH, WFO_WFR_TH,
                DTYPE, INNER_N_JOBS, False, N_SYMBOLS, LOG_LEVEL, SAVE_TRADES, BRIEF_TRADES_FOLDER,
            )
            for i, rule in enumerate(rules)
        )

    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    strategies = _load_strategies(RULES_BATCH_PATH)
    timeframes = sorted({entry["timeframe"] for entry in strategies})

    logger.info(f"\n{'=' * 115}")
    logger.info(f"  RULE VALIDATION START — production rules vs current IS data")
    logger.info(f"{'=' * 115}")
    logger.info(f"  RULES FILE  : {RULES_BATCH_PATH}")
    logger.info(f"  STRATEGIES  : {len(strategies)}")
    logger.info(f"  TIMEFRAMES  : {timeframes}")
    logger.info(f"{'=' * 115}\n")

    all_raw_results = []

    for timeframe in timeframes:
        tf_start = time.time()

        tf_strategies = [s for s in strategies if s["timeframe"] == timeframe]

        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
        ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)

        rules = [
            _build_validation_rule(i, timeframe, entry)
            for i, entry in enumerate(tf_strategies)
        ]

        tf_results = _run_validation_wfo(rules, ohlcv_arr, timeframe)
        all_raw_results.extend(tf_results)

        tf_elapsed = int(time.time() - tf_start)
        logger.info(f"\n🏁 {timeframe} DONE — {tf_elapsed // 3600} h {(tf_elapsed % 3600) // 60} min {tf_elapsed % 60} s")

    all_ids = [r["rule_id"] for r in all_raw_results]
    _print_ranking(all_raw_results, all_ids, "PRODUCTION VALIDATION")

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")