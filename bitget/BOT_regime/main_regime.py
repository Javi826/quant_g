#develop/market_regime/main_regime.py
import gc
import os
import sys
import time
import logging
from joblib import Parallel, delayed

for _key in list(sys.modules.keys()):
    if any(_key.startswith(_mod) for _mod in ("shared_batchs", "shared_batch_regime", "shared_trading_batch_regime", "shared", "bitget")):
        del sys.modules[_key]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batch_regime.regime_core import BINS, REGIME_TIMEFRAME
from shared_batch_regime.regime_core import pct_improvement, combo_label
from regime_reporting import print_combo_period_table, print_combo_summary
from regime_reporting import print_ranking, print_classification_summary
from BOT_regime.regime_engine import EVAL_KEYS
from BOT_regime.regime_engine import classify_strategy, combined_metrics
from BOT_regime.regime_engine import load_strategies_config, precompute_baselines
from BOT_regime.regime_engine import build_indicator_cache, run_filtered_combo
from BOT_regime.regime_engine import save_bins, _metric_value, _METRIC_MAP, _DD_KEY_MAP

LOG_LEVEL = logging.INFO
logging.basicConfig(format="%(message)s", level=LOG_LEVEL, force=True)
logger = logging.getLogger(__name__)
logging.getLogger("shared_batch_regime.regime_core").setLevel(logging.INFO)

N_JOBS = -1
PERIOD_WEIGHTS = {
    "OOS1": 0.50,
    "OOS2": 0.25,
    "OOS3": 0.25,
}

STRATEGIES_SET_NAME = "00"
BINS_OUTPUT_PATH    = os.path.join(os.path.dirname(__file__), "..", f"BOT_batch_{STRATEGIES_SET_NAME}","strategies_files", f"regime_bins_{STRATEGIES_SET_NAME}.py",)

# =============================================================================
# REGIME CONFIGURATION
# =============================================================================
AUTO_SAVE_BINS  = True
OPTIMIZE_METRIC = "calmar"            # "profit" | "win_rate" | "calmar"
RANKING_MODE    = "weighted_delta"    # "weighted_delta" | "combo_delta"

FILTER_NEGATIVE_BASELINE: bool = False

# =============================================================================
# INDICATOR GRID
# =============================================================================
INDICATOR_CFGS: list[dict] = [
    {"ma_window": 2},
    {"ma_window": 3},
    {"ma_window": 4},
]

# =============================================================================
# INDICATOR_CFGS: list[dict] = [
#     {"ma_window": 2, "atr_period": 7, "atr_threshold": 0.02},
#     {"ma_window": 3, "atr_period": 14, "atr_threshold": 0.04},
#     {"ma_window": 4, "atr_period": 21, "atr_threshold": 0.06},
# ]
# =============================================================================
# =============================================================================
# COMBINED METRIC FOR A SINGLE PERIOD
# =============================================================================

def _combined_metric_for_period(
    results:         dict,
    period_key:      str,
    optimize_metric: str = "profit",
) -> tuple[float, float]:
    comb = base = 0.0
    for sid, data in results.items():
        if sid == "is_long" or period_key not in data or not isinstance(data[period_key], dict):
            continue
        d   = data[period_key]
        cls = data.get("classification", "neutral")

        base += _metric_value(d, "b_prof", "b_dd", optimize_metric)
        if cls in BINS:
            val_key = _METRIC_MAP[cls][optimize_metric]
            dd_key  = _DD_KEY_MAP[cls]
            comb   += _metric_value(d, val_key, dd_key, optimize_metric)
        else:
            comb += _metric_value(d, "b_prof", "b_dd", optimize_metric)

    return comb, base

# =============================================================================
# PROCESS SINGLE COMBO  (parallelizable unit)
# =============================================================================

def _process_combo(
    combo_idx:       int,
    baselines:       dict,
    strategies:      list[dict],
    indicator_cache: dict,
    indicator_cfg:   dict,
    baseline_profit: float,
    baseline_dd:     float,
) -> dict:
    label   = combo_label(indicator_cfg)
    results = run_filtered_combo(baselines, strategies, indicator_cache, indicator_cfg)

    for sid in results:
        if sid != 'is_long':
            results[sid]['classification'] = classify_strategy(
                results, sid,
                optimize_metric = OPTIMIZE_METRIC,
            )

    period_summaries: dict[str, dict] = {}
    for pk in EVAL_KEYS:
        period_summaries[pk] = print_combo_period_table(results, strategies, pk, label)

    all_cls    = [data.get('classification', 'neutral') for sid, data in results.items() if sid != 'is_long']
    bin_counts = {b: all_cls.count(b) for b in BINS}
    n_neutral  = all_cls.count('neutral')

    comb_p, comb_d = combined_metrics(results)

    weighted_delta = sum(
        pct_improvement(*_combined_metric_for_period(results, pk, OPTIMIZE_METRIC)) * PERIOD_WEIGHTS.get(pk, 0)
        for pk in period_summaries
    ) / sum(PERIOD_WEIGHTS.get(pk, 0) for pk in period_summaries)

    return {
        'combo_idx':        combo_idx,
        'combined_profit':  comb_p,
        'combined_dd':      comb_d,
        'weighted_delta':   weighted_delta,
        'baseline_profit':  baseline_profit,
        'baseline_dd':      baseline_dd,
        'bin_counts':       bin_counts,
        'n_neutral':        n_neutral,
        'period_summaries': period_summaries,
        'label':            label,
        'results':          results,
        'indicator_cfg':    indicator_cfg,
    }


# =============================================================================
# MAIN RUN
# =============================================================================

def run() -> None:
    _t0 = time.time()
    gc.collect()

    logger.info(f"\n{'='*120}")
    logger.info(f"  REGIME CALIBRATION — {len(INDICATOR_CFGS)} combinations")
    logger.info(f"  INDICATOR_CFGS ({REGIME_TIMEFRAME}): {INDICATOR_CFGS}")
    logger.info(f"  BINS: {' | '.join(BINS)}")
    logger.info(f"  OPTIMIZE_METRIC={OPTIMIZE_METRIC} | RANKING_MODE={RANKING_MODE}")
    logger.info(f"  Lookahead fix: D-1 daily candle")
    logger.info(f"  Periods: {' + '.join(EVAL_KEYS)}")
    logger.info(f"{'='*120}")

    strategies_all = load_strategies_config(STRATEGIES_SET_NAME)
    if not strategies_all:
        logger.info("  No strategies found — aborting.")
        return

    baselines, strategies_filtered = precompute_baselines(strategies_all, STRATEGIES_SET_NAME, FILTER_NEGATIVE_BASELINE)
    if not strategies_filtered:
        logger.info("  No strategies passed the baseline filter — aborting.")
        return

    base_profits = [
        baselines[s['id']][pk]['metrics']['profit']
        for s in strategies_filtered for pk in EVAL_KEYS
        if pk in baselines.get(s['id'], {})
    ]
    base_dds = [
        baselines[s['id']][pk]['metrics']['max_dd']
        for s in strategies_filtered for pk in EVAL_KEYS
        if pk in baselines.get(s['id'], {})
    ]
    baseline_profit = sum(base_profits)
    baseline_dd     = sum(base_dds) / len(base_dds) if base_dds else 0.0

    combos: list[tuple[dict, dict]] = [
        build_indicator_cache(baselines, strategies_filtered, indicator_cfg=cfg)
        for cfg in INDICATOR_CFGS
    ]

    ranking: list[dict] = Parallel(n_jobs=N_JOBS)(
        delayed(_process_combo)(
            combo_idx       = combo_idx,
            baselines       = baselines,
            strategies      = strategies_filtered,
            indicator_cache = cache,
            indicator_cfg   = indicator_cfg,
            baseline_profit = baseline_profit,
            baseline_dd     = baseline_dd,
        )
        for combo_idx, (cache, indicator_cfg) in enumerate(combos, 1)
    )

    for row in sorted(ranking, key=lambda x: x['combo_idx']):
        logger.debug(f"\n  COMBO {row['combo_idx']}/{len(INDICATOR_CFGS)}")
        print_combo_summary(
            row['period_summaries'],
            row['bin_counts'],
            row['n_neutral'],
            row['combined_profit'], row['combined_dd'],
            row['baseline_profit'], row['baseline_dd'],
            row['label'],
        )

    if RANKING_MODE == "combo_delta":
        ranking.sort(key=lambda x: pct_improvement(x['combined_profit'], x['baseline_profit']), reverse=True)
    else:
        ranking.sort(key=lambda x: x['weighted_delta'], reverse=True)

    print_ranking(ranking)

    # =========================================================================
    # TOP1 CLASSIFICATION & BINS
    # =========================================================================
    top1         = ranking[0]
    top1_results = top1['results']

    logger.info(f"\n  TOP1 COMBO — {top1['label']}")
    excluded_ids = [s['id'] for s in strategies_all if s['id'] not in top1_results]
    print_classification_summary(top1_results, excluded_ids)

    if AUTO_SAVE_BINS:
        save_bins(
            strategy_results    = top1_results,
            indicator_cfg       = top1['indicator_cfg'],
            output_path         = BINS_OUTPUT_PATH,
            strategies_set_name = STRATEGIES_SET_NAME,
            all_strategies      = strategies_all,
            optimize_metric     = OPTIMIZE_METRIC,
        )
    else:
        logger.info("\n  ⚠️  AUTO_SAVE_BINS=False — bins not saved. Set to True to persist.")

    elapsed = int(time.time() - _t0)
    print(f"\n  Completed in {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s\n")
    del baselines, combos, ranking
    gc.collect()

if __name__ == "__main__":
    run()
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)
    gc.collect()