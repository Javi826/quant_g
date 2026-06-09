#develop/market_regime/regime_calibration.py
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

from shared_batch_regime.regime_core import EVAL_KEYS, BINS, REGIME_TIMEFRAME
from shared_batch_regime.regime_core import pct_improvement, combo_label, classify_strategy
from shared_batch_regime.regime_core import load_strategies_config, precompute_baselines
from shared_batch_regime.regime_core import build_indicator_cache, combined_metrics, _calmar
from shared_batch_regime.regime_core import save_bins, run_filtered_combo

from shared_batch_regime.regime_reporting import print_combo_period_table, print_combo_summary
from shared_batch_regime.regime_reporting import print_ranking, print_classification_summary

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
OPTIMIZE_METRIC = "calmar"   # "profit" | "win_rate" | "calmar"
RANKING_MODE    = "weighted_delta"  # "weighted_delta" | "combo_delta"

# =============================================================================
# INDICATOR GRID
# =============================================================================
MA_WINDOWS: list[int] = [2,3,4]

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

        if optimize_metric == "calmar":
            base += _calmar(d["b_prof"], d["b_dd"])
            if cls in BINS:
                comb += _calmar(d[f"{cls}_prof"], d[f"{cls}_dd"])
            else:
                comb += _calmar(d["b_prof"], d["b_dd"])
        else:
            base += d["b_prof"]
            if cls in BINS:
                comb += d[f"{cls}_prof"]
            else:
                comb += d["b_prof"]

    return comb, base

# =============================================================================
# PROCESS SINGLE COMBO  (parallelizable unit)
# =============================================================================

def _process_combo(
    combo_idx:       int,
    ma_window:       int,
    total_combos:    int,
    baselines:       dict,
    strategies:      list[dict],
    indicator_cache: dict,
    baseline_profit: float,
    baseline_dd:     float,
) -> dict:
    label   = combo_label(ma_window)
    results = run_filtered_combo(
        baselines, strategies, indicator_cache, ma_window
    )

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
    avg_up         = sum(ps['avg_up_pct'] for ps in period_summaries.values()) / max(len(period_summaries), 1)

    weighted_delta = sum(
        pct_improvement(*_combined_metric_for_period(results, pk, OPTIMIZE_METRIC)) * PERIOD_WEIGHTS.get(pk, 0)
        for pk in period_summaries
    ) / sum(PERIOD_WEIGHTS.get(pk, 0) for pk in period_summaries)

    return {
        'ma_window':        ma_window,
        'combo_idx':        combo_idx,
        'combined_profit':  comb_p,
        'combined_dd':      comb_d,
        'weighted_delta':   weighted_delta,
        'baseline_profit':  baseline_profit,
        'baseline_dd':      baseline_dd,
        'avg_up_pct':       avg_up,
        'bin_counts':       bin_counts,
        'n_neutral':        n_neutral,
        'period_summaries': period_summaries,
        'label':            label,
    }

# =============================================================================
# MAIN RUN
# =============================================================================

def run() -> None:
    _t0 = time.time()
    gc.collect()

    total_combos = len(MA_WINDOWS)

    logger.info(f"\n{'='*120}")
    logger.info(f"  REGIME CALIBRATION — {total_combos} combinations")
    logger.info(f"  MA windows ({REGIME_TIMEFRAME}): {MA_WINDOWS}")
    logger.info(f"  BINS: {' | '.join(BINS)}")
    logger.info(f"  OPTIMIZE_METRIC={OPTIMIZE_METRIC} | RANKING_MODE={RANKING_MODE}")
    logger.info(f"  Lookahead fix: D-1 daily candle")
    logger.info(f"  Periods: {' + '.join(EVAL_KEYS)}")
    logger.info(f"{'='*120}")

    strategies_all = load_strategies_config(STRATEGIES_SET_NAME)
    if not strategies_all:
        logger.info("  No strategies found — aborting.")
        return

    baselines, strategies_filtered = precompute_baselines(strategies_all, STRATEGIES_SET_NAME)
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

    cache_map: dict[int, dict] = {}
    for ma_w in MA_WINDOWS:
        if ma_w not in cache_map:
            cache_map[ma_w] = build_indicator_cache(
                baselines, strategies_filtered,
                ma_window = ma_w,
            )

    ranking: list[dict] = Parallel(n_jobs=N_JOBS)(
        delayed(_process_combo)(
            combo_idx       = combo_idx,
            ma_window       = ma_w,
            total_combos    = total_combos,
            baselines       = baselines,
            strategies      = strategies_filtered,
            indicator_cache = cache_map[ma_w],
            baseline_profit = baseline_profit,
            baseline_dd     = baseline_dd,
        )
        for combo_idx, ma_w in enumerate(MA_WINDOWS, 1)
    )

    for row in sorted(ranking, key=lambda x: x['combo_idx']):
        logger.debug(f"\n  COMBO {row['combo_idx']}/{total_combos}")
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
    top1 = ranking[0]
    logger.info(f"\n  TOP1 COMBO — {top1['label']}")

    top1_results = run_filtered_combo(
        baselines, strategies_filtered,
        cache_map[top1['ma_window']],
        top1['ma_window'],
    )
    for sid in top1_results:
        if sid != 'is_long':
            top1_results[sid]['classification'] = classify_strategy(
                top1_results, sid,
                optimize_metric = OPTIMIZE_METRIC,
            )

    print_classification_summary(top1_results)

    if AUTO_SAVE_BINS:
        save_bins(
            strategy_results    = top1_results,
            ma_window           = top1['ma_window'],
            output_path         = BINS_OUTPUT_PATH,
            strategies_set_name = STRATEGIES_SET_NAME,
            all_strategies      = strategies_all,
            optimize_metric     = OPTIMIZE_METRIC,
        )
    else:
        logger.info("\n  ⚠️  AUTO_SAVE_BINS=False — bins not saved. Set to True to persist.")

    elapsed = int(time.time() - _t0)
    print(f"\n  Completed in {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s\n")
    del baselines, cache_map, ranking
    gc.collect()

if __name__ == "__main__":
    run()
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)
    gc.collect()