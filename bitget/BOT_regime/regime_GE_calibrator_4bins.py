#develop/market_regime/regime_GE_calibration_4bins.py
import gc
import os
import sys
import time
import itertools
import logging
from joblib import Parallel, delayed

for _key in list(sys.modules.keys()):
    if any(_key.startswith(_mod) for _mod in ("shared_batchs", "shared_batch_regime", "shared_trading_batch_regime", "shared", "bitget")):
        del sys.modules[_key]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batch_regime.regime_GE_core_4bins import (
    EVAL_KEYS, BINS, pct_improvement,
    combo_label, load_strategies_config,
    precompute_baselines,
    build_indicator_cache, get_cache_key,
    classify_strategy, combined_metrics,
    lookup_indicators_batch, _METRIC_MAP, _DD_KEY_MAP, _calmar, _mix_score,
    print_classification_summary, save_bins,
    run_backtest, print_ranking,
    print_combo_period_table, print_combo_summary,
    run_filtered_combo,
)
import numpy as np

LOG_LEVEL = logging.INFO
logging.basicConfig(format="%(message)s", level=LOG_LEVEL, force=True)
logger = logging.getLogger(__name__)
logging.getLogger("shared_batch_regime.regime_GE_core_4bins").setLevel(logging.INFO)

N_JOBS = -1
PERIOD_WEIGHTS = {
    "OOS1": 0.50,
    "OOS2": 0.25,
    "OOS3": 0.25,
}

STRATEGIES_SET_NAME = "E1"
BINS_OUTPUT_PATH    = os.path.join(
    os.path.dirname(__file__), "..", f"BOT_batch_{STRATEGIES_SET_NAME}",
    "strategies_files", f"regime_bins_4bins_{STRATEGIES_SET_NAME}.py",
)

# =============================================================================
# REGIME CONFIGURATION
# =============================================================================
AUTO_SAVE_BINS        = False
ANALYSIS_MODE         = "SYMBOL"  # "BTC" | "SYMBOL"
REGIME_TIMEFRAME_MODE = "DAILY"   # "DAILY" | "STRATEGY"

MIX_WEIGHT_PROFIT         = 0.5
MIX_WEIGHT_DD             = 0.5
OPTIMIZE_METRIC           = "calmar"  # "profit" | "win_rate" | "calmar" | "mix"
CLASSIFICATION_MODE       = "strict"  # "strict" | "oos1_weighted"
CLASSIFY_SECONDARY_METRIC = "r2"      # None | "r2" | "profit" | "calmar"

MIN_CLASSIFIED_PCT = 0.0
RANKING_MODE       = "combo_delta"  # "weighted_delta" | "n_classified" | "combo_delta"

# =============================================================================
# INDICATOR GRID
# ER  → direction  (trending vs ranging)
# ATR → volatility (highvol vs lowvol)
# =============================================================================
ER_WINDOWS:      list[int]   = [10]
ER_THRESHOLDS:   list[float] = [0.5, 0.6, 0.7, 0.8]
 
ATR_WINDOWS:     list[int]   = [10]
ATR_THRESHOLDS:  list[float] = [0.02, 0.03, 0.04, 0.05]

ORDER_AMOUNT               = 80
LONG_KEYWORD               = "long"
DEBUG_TF_FILTER: list[str] = []

# =============================================================================
# GRID BUILDER
# =============================================================================

def _build_grid() -> list[tuple[int, float, int, float]]:
    """
    Build the full parameter grid from ER and ATR windows/thresholds.
    Returns list of (er_window, er_threshold, atr_window, atr_threshold).
    """
    return list(itertools.product(ER_WINDOWS, ER_THRESHOLDS, ATR_WINDOWS, ATR_THRESHOLDS))


# =============================================================================
# COMBINED METRIC FOR A SINGLE PERIOD
# =============================================================================

def _combined_metric_for_period(results: dict, period_key: str, optimize_metric: str = "profit") -> tuple[float, float]:
    """
    Aggregate combined vs baseline metric across all strategies for one period.
    Each strategy uses its best-profit bin; neutral falls back to baseline.
    """
    comb = base = 0.0
    for sid, data in results.items():
        if sid == 'is_long' or period_key not in data or not isinstance(data[period_key], dict):
            continue
        d    = data[period_key]
        bins = data.get('classification', ['neutral'])

        if optimize_metric == "calmar":
            base += _calmar(d['b_prof'], d['b_dd'])
            if bins == ["neutral"]:
                comb += _calmar(d['b_prof'], d['b_dd'])
            else:
                best = max(bins, key=lambda b: _calmar(d[f"{b}_prof"], d[f"{b}_dd"]))
                comb += _calmar(d[f"{best}_prof"], d[f"{best}_dd"])

        elif optimize_metric == "mix":
            base += _mix_score(d['b_prof'], d['b_dd'])
            if bins == ["neutral"]:
                comb += _mix_score(d['b_prof'], d['b_dd'])
            else:
                best = max(bins, key=lambda b: _mix_score(d[f"{b}_prof"], d[f"{b}_dd"]))
                comb += _mix_score(d[f"{best}_prof"], d[f"{best}_dd"])

        else:
            base += d['b_prof']
            if bins == ["neutral"]:
                comb += d['b_prof']
            else:
                best = max(bins, key=lambda b: d[f"{b}_prof"])
                comb += d[f"{best}_prof"]

    return comb, base


# =============================================================================
# PROCESS SINGLE COMBO  (parallelizable unit)
# =============================================================================

def _process_combo(
    combo_idx:       int,
    er_window:       int,
    er_threshold:    float,
    atr_window:      int,
    atr_threshold:   float,
    total_combos:    int,
    baselines:       dict,
    strategies:      list[dict],
    indicator_cache: dict,
    baseline_profit: float,
    baseline_dd:     float,
) -> dict:
    label   = combo_label(er_window, er_threshold, atr_window, atr_threshold)
    results = run_filtered_combo(
        baselines, strategies, indicator_cache,
        er_threshold, atr_threshold,
        ANALYSIS_MODE, REGIME_TIMEFRAME_MODE,
    )

    for sid in results:
        if sid != 'is_long':
            results[sid]['classification'] = classify_strategy(
                results, sid,
                optimize_metric     = OPTIMIZE_METRIC,
                classification_mode = CLASSIFICATION_MODE,
                secondary_metric    = CLASSIFY_SECONDARY_METRIC,
            )

    period_summaries: dict[str, dict] = {}
    for pk in EVAL_KEYS:
        period_summaries[pk] = print_combo_period_table(results, strategies, pk, label)

    all_bins = [b for sid, data in results.items() if sid != 'is_long' for b in data.get('classification', ['neutral'])]
    bin_counts = {b: all_bins.count(b) for b in BINS}
    n_neutral  = sum(1 for sid, data in results.items() if sid != 'is_long' and data.get('classification') == ['neutral'])

    comb_p, comb_d = combined_metrics(results)
    avg_trend      = sum(ps['avg_trend_pct'] for ps in period_summaries.values()) / max(len(period_summaries), 1)

    weighted_delta = sum(
        pct_improvement(*_combined_metric_for_period(results, pk, OPTIMIZE_METRIC)) * PERIOD_WEIGHTS.get(pk, 0)
        for pk in period_summaries
    ) / sum(PERIOD_WEIGHTS.get(pk, 0) for pk in period_summaries)

    return {
        'er_window':       er_window,
        'er_threshold':    er_threshold,
        'atr_window':      atr_window,
        'atr_threshold':   atr_threshold,
        'combo_idx':       combo_idx,
        'combined_profit': comb_p,
        'combined_dd':     comb_d,
        'weighted_delta':  weighted_delta,
        'baseline_profit': baseline_profit,
        'baseline_dd':     baseline_dd,
        'avg_trend_pct':   avg_trend,
        'bin_counts':      bin_counts,
        'n_neutral':       n_neutral,
        'period_summaries': period_summaries,
        'label':           label,
    }


# =============================================================================
# MAIN RUN
# =============================================================================

def run() -> None:
    _t0 = time.time()
    gc.collect()

    grid         = _build_grid()
    total_combos = len(grid)

    logger.info(f"\n{'='*120}")
    logger.info(f"  REGIME CALIBRATION [4-BIN MODE] — {total_combos} combinations")
    logger.info(f"  ER (direction):   windows={ER_WINDOWS}  thresholds={ER_THRESHOLDS}")
    logger.info(f"  ATR_NORM (volatility): windows={ATR_WINDOWS}  thresholds={ATR_THRESHOLDS}")
    logger.info(f"  BINS: {' | '.join(BINS)}")
    logger.info(f"  ANALYSIS_MODE={ANALYSIS_MODE} | REGIME_TIMEFRAME_MODE={REGIME_TIMEFRAME_MODE} | OPTIMIZE_METRIC={OPTIMIZE_METRIC} | CLASSIFICATION_MODE={CLASSIFICATION_MODE}")
    logger.info(f"  Lookahead fix: normalize()-1day")
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

    # Precalculate indicator caches — one per unique (er_window, atr_window) pair
    cache_map: dict[tuple[int, int], dict] = {}
    for er_w, er_th, atr_w, atr_th in grid:
        win_key = (er_w, atr_w)
        if win_key not in cache_map:
            cache_map[win_key] = build_indicator_cache(
                baselines, strategies_filtered,
                er_window  = er_w,
                atr_window = atr_w,
                analysis_mode         = ANALYSIS_MODE,
                regime_timeframe_mode = REGIME_TIMEFRAME_MODE,
            )

    ranking: list[dict] = Parallel(n_jobs=N_JOBS)(
        delayed(_process_combo)(
            combo_idx       = combo_idx,
            er_window       = er_w,
            er_threshold    = er_th,
            atr_window      = atr_w,
            atr_threshold   = atr_th,
            total_combos    = total_combos,
            baselines       = baselines,
            strategies      = strategies_filtered,
            indicator_cache = cache_map[(er_w, atr_w)],
            baseline_profit = baseline_profit,
            baseline_dd     = baseline_dd,
        )
        for combo_idx, (er_w, er_th, atr_w, atr_th) in enumerate(grid, 1)
    )

    for row in sorted(ranking, key=lambda x: x['combo_idx']):
        logger.info(f"\n  COMBO {row['combo_idx']}/{total_combos}")
        print_combo_summary(
            row['period_summaries'],
            row['bin_counts'],
            row['n_neutral'],
            row['combined_profit'], row['combined_dd'],
            row['baseline_profit'], row['baseline_dd'],
            row['label'],
        )

    n_total        = len(strategies_filtered)
    min_classified = int(n_total * MIN_CLASSIFIED_PCT)

    ranking_filtered = [
        row for row in ranking
        if sum(row['bin_counts'].get(b, 0) for b in BINS) >= min_classified
    ]

    if not ranking_filtered:
        logger.info(f"\n  ⚠️  No combos passed MIN_CLASSIFIED_PCT={MIN_CLASSIFIED_PCT} — showing full ranking.")
        ranking_filtered = ranking

    if RANKING_MODE == "n_classified":
        ranking_filtered.sort(key=lambda x: sum(x['bin_counts'].values()), reverse=True)
    elif RANKING_MODE == "combo_delta":
        ranking_filtered.sort(key=lambda x: pct_improvement(x['combined_profit'], x['baseline_profit']), reverse=True)
    else:
        ranking_filtered.sort(key=lambda x: x['weighted_delta'], reverse=True)

    print_ranking(ranking_filtered)

    # =========================================================================
    # TOP1 CLASSIFICATION & BINS
    # =========================================================================
    top1 = ranking_filtered[0]
    logger.info(f"\n  TOP1 COMBO — {top1['label']}")

    top1_results = run_filtered_combo(
        baselines, strategies_filtered,
        cache_map[(top1['er_window'], top1['atr_window'])],
        top1['er_threshold'], top1['atr_threshold'],
        ANALYSIS_MODE, REGIME_TIMEFRAME_MODE,
    )
    for sid in top1_results:
        if sid != 'is_long':
            top1_results[sid]['classification'] = classify_strategy(
                top1_results, sid,
                optimize_metric     = OPTIMIZE_METRIC,
                classification_mode = CLASSIFICATION_MODE,
                secondary_metric    = CLASSIFY_SECONDARY_METRIC,
            )

    print_classification_summary(top1_results)

    if AUTO_SAVE_BINS:
        save_bins(
            strategy_results      = top1_results,
            er_window             = top1['er_window'],
            er_threshold          = top1['er_threshold'],
            atr_window            = top1['atr_window'],
            atr_threshold         = top1['atr_threshold'],
            output_path           = BINS_OUTPUT_PATH,
            strategies_set_name   = STRATEGIES_SET_NAME,
            all_strategies        = strategies_all,
            optimize_metric       = OPTIMIZE_METRIC,
            classification_mode   = CLASSIFICATION_MODE,
            secondary_metric      = CLASSIFY_SECONDARY_METRIC,
            analysis_mode         = ANALYSIS_MODE,
            regime_timeframe_mode = REGIME_TIMEFRAME_MODE,
        )
    else:
        logger.info("\n  ⚠️  AUTO_SAVE_BINS=False — bins not saved. Set to True to persist.")

    elapsed = int(time.time() - _t0)
    print(f"\n  Completed in {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s\n")
    del baselines, cache_map, ranking, ranking_filtered
    gc.collect()


if __name__ == "__main__":
    run()
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)
    gc.collect()