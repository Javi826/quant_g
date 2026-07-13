#BOT_batch/main_rule_validation.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch_regime")))

import time
import logging
import importlib.util
import numpy as np

# LOGGING CONFIGURATION
#------------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", stream=sys.stdout, force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger("BOT_batch.main_rule_validation")

from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.backtesters.ZX_compute_BT import MIN_PRICE
from shared_batch_regime.config_paths import DATA_FOLDER_IS, DATA_FOLDER_OOS1
from shared_batchs.rule_mining.rule_runner import _run_single_rule, _print_ranking
from shared_batchs.runs.run_correlation import decorrelate_by_profit
from shared_batchs.runs.run_best_wfo_portfolio import find_best_portfolio_combination_wfo
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE
from signals.signal_builder import build_signal_fn

# =============================================================================
# RUN CONFIGURATION
# =============================================================================

DTYPE        = np.float32
RULES_N_JOBS = 32
INNER_N_JOBS = 1

N_SYMBOLS = 10

WFO_NET_GAIN_TH  = -10
WFO_DD_TH        = 50
WFO_R2_TH        = -1.0
WFO_STABILITY_TH = 1.0   # 1.0 = disabled (no rule discarded)

SAVE_TRADES = True

RUN_CORRELATION           = True
RUN_BEST_WFO_PORTFOLIO    = True
CORRELATION_DD_THRESHOLD  = 0.55
SHOW_PLOTS                = True

STRATEGIES_E1_FOLDER = os.path.join(os.path.dirname(__file__), "strategies_E1")
BRIEF_TRADES_FOLDER  = os.path.join(STRATEGIES_E1_FOLDER, "brief_trades")
RULES_BATCH_PATH      = os.path.join(STRATEGIES_E1_FOLDER, "rules_files", "rules_batch.py")

# =============================================================================
# LOAD PRODUCTION STRATEGIES
# =============================================================================

def _load_strategies(rules_batch_path: str) -> list:
    spec   = importlib.util.spec_from_file_location("rules_batch", rules_batch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STRATEGIES


def _strategy_to_rule(entry: dict) -> dict:
    specs = entry["specs"]
    side  = entry["direction"]
    return {
        "side":      side,
        "specs":     specs,
        "label":     entry["id"],
        "signal_fn": build_signal_fn(specs, side),
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    strategies = _load_strategies(RULES_BATCH_PATH)
    timeframes = sorted({entry["timeframe"] for entry in strategies})

    logger.info(f"\n{'=' * 115}")
    logger.info(f"  RULE VALIDATION START — production vs batch")
    logger.info(f"{'=' * 115}")
    logger.info(f"  RULES FILE      : {RULES_BATCH_PATH}")
    logger.info(f"  STRATEGIES      : {len(strategies)}")
    logger.info(f"  TIMEFRAMES      : {timeframes}")
    logger.info(f"{'=' * 115}\n")

    all_raw_results = []

    for timeframe in timeframes:
        tf_start = time.time()

        tf_strategies = [s for s in strategies if s["timeframe"] == timeframe]

        symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos1 = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            data_folder_oos   = DATA_FOLDER_OOS1,
            timeframe         = timeframe,
            n_symbols         = N_SYMBOLS,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )

        tf_results = []
        for i, entry in enumerate(tf_strategies):
            rule = _strategy_to_rule(entry)

            param_names    = ["SELL_AFTER", "TP_PCT", "SL_PCT"]
            lists_for_grid = [[entry["sell_after_ncandles"]], [entry["tp_pct"]], [entry["sl_pct"]]]

            result = _run_single_rule(
                i, len(tf_strategies), rule, ohlcv_is, param_names, lists_for_grid,
                entry["order_amount"], timeframe, WFO_NET_GAIN_TH, WFO_DD_TH, WFO_R2_TH, WFO_STABILITY_TH,
                DTYPE, INNER_N_JOBS, False, N_SYMBOLS, LOG_LEVEL, SAVE_TRADES, BRIEF_TRADES_FOLDER,
            )
            tf_results.append(result)

        all_raw_results.extend(tf_results)

        tf_elapsed = int(time.time() - tf_start)
        logger.info(f"\n🏁 {timeframe} DONE — {tf_elapsed // 3600} h {(tf_elapsed % 3600) // 60} min {tf_elapsed % 60} s")

    all_ids = [r["rule_id"] for r in all_raw_results]
    _print_ranking(all_raw_results, all_ids, "PRODUCTION VALIDATION")

    validated_wfo_test = [
        (r["rule_id"], r["wfo_test_trades"])
        for r in all_raw_results
        if r["wfo_test_trades"] is not None and not r["wfo_test_trades"].empty
    ]

    if RUN_CORRELATION and validated_wfo_test:
        logger.info(f"\n{'─' * 115}\n  CORRELATION ANALYSIS — PRODUCTION VALIDATION (threshold={CORRELATION_DD_THRESHOLD})\n{'─' * 115}")
        validated_wfo_test = decorrelate_by_profit(
            strategy_trades_wfo_test = validated_wfo_test,
            initial_balance          = INITIAL_BALANCE,
            threshold                = CORRELATION_DD_THRESHOLD,
        )
        _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "POST-CORRELATION")

    if RUN_BEST_WFO_PORTFOLIO and validated_wfo_test:
        find_best_portfolio_combination_wfo(
            validated_wfo_trades = validated_wfo_test,
            initial_balance      = INITIAL_BALANCE,
            show_plots           = SHOW_PLOTS,
        )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")