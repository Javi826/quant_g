#develop/market_regime/main_regime.py
import gc
import os
import sys
import time
import logging
import numpy as np
from joblib import Parallel, delayed

for _key in list(sys.modules.keys()):
    if any(_key.startswith(_mod) for _mod in ("shared_batchs", "shared_batch_regime", "shared_trading_batch_regime", "shared", "bitget")):
        del sys.modules[_key]

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from itertools import product as _product
from shared_batch_regime.regime_core import BINS, REGIME_TIMEFRAME
from shared_batch_regime.regime_core import pct_improvement, combo_label
from regime_reporting import print_combo_strategy_table, print_combo_summary
from regime_reporting import print_ranking, print_classification_summary
from BOT_regime.regime_engine import load_strategies_config, load_ohlcv_is, build_indicator_cache
from BOT_regime.regime_engine import run_baseline_wfo, run_combo_from_baseline, compute_metrics_per_bin
from BOT_regime.regime_engine import classify_strategy_integro, classify_strategy_split, save_bins
from shared_batch_regime.config_paths import DATA_FOLDER_IS

LOG_LEVEL = logging.INFO
logging.basicConfig(format="%(message)s", level=LOG_LEVEL, force=True)
logger = logging.getLogger(__name__)
logging.getLogger("regime_reporting").setLevel(logging.INFO)

N_JOBS = -1
DTYPE  = np.float32

STRATEGIES_SET_NAME = "E1"
BINS_OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "..", f"BOT_batch_{STRATEGIES_SET_NAME}", "strategies_files", f"regime_bins_{STRATEGIES_SET_NAME}.py")

# =============================================================================
# REGIME CONFIGURATION
# =============================================================================
AUTO_SAVE_BINS  = True
OPTIMIZE_METRIC = "Calmar"      # any numeric key from compute_metrics (e.g. "Net_Gain_pct", "Calmar")
RANKING_MODE    = "weighted_delta"    # "weighted_delta" | "combo_delta"

# Bin classification strategy:
#   "integro" — bin must beat baseline on the full-period aggregate metrics
#   "split"   — bin must beat baseline in every valid REGIME_N_SPLITS time partition
CLASSIFICATION_MODE = "split"   # "integro" | "split"
REGIME_N_SPLITS      = 2        # only used when CLASSIFICATION_MODE == "split"

# =============================================================================
# INDICATOR GRID
# =============================================================================
# =============================================================================
# INDICATOR_GRID: dict = {
#     "ma_window": [4,12],
# }
# =============================================================================
INDICATOR_GRID: dict = {
    "ma_window":     [4,16,32],
    "atr_period":    [7,14,21],
    "atr_threshold": [0.02,0.04,0.06],
}
INDICATOR_CFGS: list[dict] = [
    dict(zip(INDICATOR_GRID.keys(), values))
    for values in _product(*INDICATOR_GRID.values())
]
SELECTED_STRATEGIES = [
    # 15m
# =============================================================================
#     "01_reversal_long_15m",
#     "02_reversal_short_15m",
#     "11_parity_long_15m",
#     "12_parity_short_15m",
#     "21_flag_long_15m",
#     "22_flag_short_15m",
#     "31_orderblocks_long_15m",
#     "32_orderblocks_short_15m",
#     # 30m
#     "03_reversal_long_30m",
#     "04_reversal_short_30m",
#     "13_parity_long_30m",
#     "14_parity_short_30m",
#     "23_flag_long_30m",
#     "24_flag_short_30m",
#     "33_orderblocks_long_30m",
#     "34_orderblocks_short_30m",
#     # 1H
#     "05_reversal_long_1H",
#     "06_reversal_short_1H",
#     "15_parity_long_1H",
#     "16_parity_short_1H",
#     "25_flag_long_1H",
#     "26_flag_short_1H",
#     "35_orderblocks_long_1H",
#     "36_orderblocks_short_1H",
# =============================================================================
    # 4H
    "07_reversal_long_4H",
    "08_reversal_short_4H",
    "17_parity_long_4H",
    "18_parity_short_4H",
    "27_flag_long_4H",
    "28_flag_short_4H",
    "37_orderblocks_long_4H",
    "38_orderblocks_short_4H",
    # 6H UTC
    "09_reversal_long_6Hutc",
    "10_reversal_short_6Hutc",
    "19_parity_long_6Hutc",
    "20_parity_short_6Hutc",
    "29_flag_long_6Hutc",
    "30_flag_short_6Hutc",
    "39_orderblocks_long_6Hutc",
    "40_orderblocks_short_6Hutc",
]
# =============================================================================
# COMBINED METRIC ACROSS STRATEGIES
# =============================================================================

def _combined_metric(strategy_metrics: dict, optimize_metric: str) -> tuple[float, float]:
    """Sum classified-bin profit and average DD across all strategies for one combo."""
    profits, dds = [], []
    for sid, data in strategy_metrics.items():
        m   = data["metrics"]
        cls = data.get("classification", [])
        if cls:
            for b in cls:
                profits.append(m[f"{b}_{optimize_metric}"] / len(cls))
                dds.append(m[f"{b}_Max_DD_pct"])
        else:
            profits.append(m[f"b_{optimize_metric}"])
            dds.append(m["b_Max_DD_pct"])
    return sum(profits), (sum(dds) / len(dds) if dds else 0.0)

# =============================================================================
# PROCESS SINGLE COMBO (parallelizable unit)
# =============================================================================

def _process_combo(
    combo_idx:             int,
    strategies:            list[dict],
    ohlcv_by_sid:          dict,
    baseline_by_sid:       dict,
    baseline_metrics:      dict,
    wfo_results_by_sid:    dict,
    indicator_cache_by_tf: dict,
    indicator_cfg:         dict,
) -> dict:
    label = combo_label(indicator_cfg)
    strategy_metrics: dict = {}
    for strategy in strategies:
        sid = strategy["id"]
        bin_trades = run_combo_from_baseline(
            strategy        = strategy,
            ohlcv_is        = ohlcv_by_sid[sid],
            df_results      = wfo_results_by_sid[sid],
            indicator_cache = indicator_cache_by_tf[strategy["timeframe"]],
            indicator_cfg   = indicator_cfg,
            dtype           = DTYPE,
            order_amount    = strategy["order_amount"],
        )
        metrics = compute_metrics_per_bin(bin_trades, baseline_by_sid[sid])

        if CLASSIFICATION_MODE == "split":
            classification = classify_strategy_split(
                baseline_trades = baseline_by_sid[sid],
                bin_trades      = bin_trades,
                combo_metrics   = metrics,
                n_splits        = REGIME_N_SPLITS,
                optimize_metric = OPTIMIZE_METRIC,
            )
        else:
            classification = classify_strategy_integro(metrics, OPTIMIZE_METRIC)

        strategy_metrics[sid] = {
            "metrics":        metrics,
            "classification": classification,
            "is_long":        strategy["is_long"],
        }
        print_combo_strategy_table({sid: strategy_metrics[sid]}, label, combo_idx=combo_idx, n_combos=len(INDICATOR_CFGS))
    all_cls    = [b for data in strategy_metrics.values() for b in data["classification"]]
    bin_counts = {b: all_cls.count(b) for b in BINS}
    n_neutral  = sum(1 for data in strategy_metrics.values() if not data["classification"])
    comb_p, comb_d = _combined_metric(strategy_metrics, OPTIMIZE_METRIC)
    base_p         = sum(baseline_metrics[sid][f"b_{OPTIMIZE_METRIC}"] for sid in strategy_metrics)
    base_dds       = [baseline_metrics[sid]["b_Max_DD_pct"] for sid in strategy_metrics]
    base_d         = sum(base_dds) / len(base_dds) if base_dds else 0.0
    print_combo_summary(bin_counts, n_neutral, comb_p, comb_d, base_p, base_d, label, combo_idx=combo_idx, n_combos=len(INDICATOR_CFGS))
    return {
        "combo_idx":        combo_idx,
        "indicator_cfg":    indicator_cfg,
        "label":            label,
        "strategy_metrics": strategy_metrics,
        "bin_counts":       bin_counts,
        "n_neutral":        n_neutral,
        "combined_profit":  comb_p,
        "combined_dd":      comb_d,
        "baseline_profit":  base_p,
        "baseline_dd":      base_d,
        "weighted_delta":   pct_improvement(comb_p, base_p),
    }

# =============================================================================
# MAIN RUN
# =============================================================================

def run() -> None:
    _t0 = time.time()
    gc.collect()

    logger.info(f"\n{'='*120}")
    logger.info(f"  REGIME CALIBRATION (WFO) — {len(INDICATOR_CFGS)} combinations")
    logger.info(f"  INDICATOR_CFGS ({REGIME_TIMEFRAME}): {len(INDICATOR_CFGS)} combos — GRID: {INDICATOR_GRID}")
    logger.info(f"  BINS: {' | '.join(BINS)}")
    logger.info(f"  OPTIMIZE_METRIC={OPTIMIZE_METRIC} | RANKING_MODE={RANKING_MODE}")
    logger.info(f"  CLASSIFICATION_MODE={CLASSIFICATION_MODE}" + (f" | REGIME_N_SPLITS={REGIME_N_SPLITS}" if CLASSIFICATION_MODE == "split" else ""))
    logger.info(f"{'='*120}")

    strategies = load_strategies_config(STRATEGIES_SET_NAME)
    if SELECTED_STRATEGIES:
        strategies = [s for s in strategies if s["id"] in SELECTED_STRATEGIES]
    if not strategies:
        logger.info("  No strategies found — aborting.")
        return

    # -------------------------------------------------------------------------
    # Load IS universe + run baseline WFO (no regime) once per strategy
    # -------------------------------------------------------------------------
    ohlcv_by_sid:     dict = {}
    baseline_by_sid:  dict = {}
    baseline_metrics: dict = {}

    wfo_results_by_sid: dict = {}

    for strategy in strategies:
        sid = strategy["id"]
        logger.info(f"  Baseline WFO — {sid}")
        ohlcv_by_sid[sid]    = load_ohlcv_is(strategy)
        baseline_by_sid[sid], wfo_results_by_sid[sid] = run_baseline_wfo(strategy, ohlcv_by_sid[sid], dtype=DTYPE, n_jobs=N_JOBS)
        baseline_metrics[sid] = compute_metrics_per_bin({}, baseline_by_sid[sid])

    # -------------------------------------------------------------------------
    # Precompute regime indicator cache once per timeframe x combo
    # -------------------------------------------------------------------------
    timeframe_by_sid = {s["id"]: s["timeframe"] for s in strategies}

    symbols_by_tf: dict[str, dict] = {}
    for sid, tf in timeframe_by_sid.items():
        symbols_by_tf.setdefault(tf, {}).update(ohlcv_by_sid[sid])

    indicator_caches: list[dict[str, dict]] = [
        {tf: build_indicator_cache(symbols, cfg, DATA_FOLDER_IS) for tf, symbols in symbols_by_tf.items()}
        for cfg in INDICATOR_CFGS
    ]

    # -------------------------------------------------------------------------
    # Run all combos in parallel
    # -------------------------------------------------------------------------
    ranking: list[dict] = Parallel(n_jobs=N_JOBS)(
        delayed(_process_combo)(
            combo_idx              = combo_idx,
            strategies              = strategies,
            ohlcv_by_sid            = ohlcv_by_sid,
            baseline_by_sid         = baseline_by_sid,
            baseline_metrics        = baseline_metrics,
            wfo_results_by_sid      = wfo_results_by_sid,
            indicator_cache_by_tf   = indicator_caches[combo_idx - 1],
            indicator_cfg           = cfg,
        )
        for combo_idx, cfg in enumerate(INDICATOR_CFGS, 1)
    )

    if RANKING_MODE == "combo_delta":
        ranking.sort(key=lambda x: pct_improvement(x["combined_profit"], x["baseline_profit"]), reverse=True)
    else:
        ranking.sort(key=lambda x: x["weighted_delta"], reverse=True)

    print_ranking(ranking)

    # =========================================================================
    # TOP1 CLASSIFICATION & BINS
    # =========================================================================
    top1 = ranking[0]
    logger.info(f"\n  TOP1 COMBO — {top1['label']}")
    print_classification_summary(top1["strategy_metrics"])

    if AUTO_SAVE_BINS:
        save_bins(
            strategy_results    = top1["strategy_metrics"],
            indicator_cfg       = top1["indicator_cfg"],
            output_path         = BINS_OUTPUT_PATH,
            strategies_set_name = STRATEGIES_SET_NAME,
            all_strategies      = strategies,
            optimize_metric     = OPTIMIZE_METRIC,
        )
    else:
        logger.info("\n  ⚠️  AUTO_SAVE_BINS=False — bins not saved. Set to True to persist.")

    elapsed = int(time.time() - _t0)
    print(f"\n  Completed in {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s\n")
    del ohlcv_by_sid, baseline_by_sid, ranking
    gc.collect()

if __name__ == "__main__":
    run()
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)
    gc.collect()