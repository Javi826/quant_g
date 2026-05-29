#develop/market_regime/regime_GE_calibration.py
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

from shared_batch_regime.regime_GE_core import EVAL_KEYS, pct_improvement
from shared_batch_regime.regime_GE_core import is_trending
from shared_batch_regime.regime_GE_core import combo_label, load_strategies_config
from shared_batch_regime.regime_GE_core import precompute_baselines
from shared_batch_regime.regime_GE_core import build_indicator_cache, get_cache_key, classify_strategy, combined_metrics, lookup_indicators_batch, _METRIC_MAP, _calmar, _mix_score
from shared_batch_regime.regime_GE_core import print_consistency_table, print_classification_summary, save_bins
from shared_batch_regime.regime_GE_core import run_backtest, print_ranking, print_combo_period_table, print_combo_summary
import numpy as np

LOG_LEVEL = logging.INFO
logging.basicConfig(format="%(message)s", level=LOG_LEVEL, force=True)
logger = logging.getLogger(__name__)

N_JOBS = -1
PERIOD_WEIGHTS = {
    "OOS1": 0.50,
    "OOS2": 0.25,
    "OOS3": 0.25,
}

STRATEGIES_SET_NAME  = "E1"
BINS_OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), f"regime_BINS_{STRATEGIES_SET_NAME}.py")

# =============================================================================
# REGIME CONFIGURATION
# =============================================================================
COMBINE_MODES         = ["OR"]
ANALYSIS_MODE         = "SYMBOL" # "BTC" | "SYMBOL"
REGIME_TIMEFRAME_MODE = "DAILY"  # "DAILY" | "STRATEGY"

MIX_WEIGHT_PROFIT     = 0.8
MIX_WEIGHT_DD         = 0.2
OPTIMIZE_METRIC       = "calmar" # "profit" | "win_rate" | "calmar" | "mix"
CLASSIFICATION_MODE   = "oos1_weighted" # "strict" | "oos1_weighted"

AUTO_SAVE_BINS        = True            # True = auto-save top1 bins after calibration

INDICATORS: dict[str, dict] = {
    "atr_norm": {
        "windows":    [10, 30, 50],
        "thresholds": [0.02, 0.04, 0.06],
        "enabled":    True,
    },
    "er": {
        "windows":    [10, 30, 50],
        "thresholds": [0.2, 0.4, 0.6],
        "enabled":    True,
    },
    "hurst": {
        "windows":    [30, 50],
        "thresholds": [0.5, 0.6, 0.8],
        "enabled":    False,
    },
}

ORDER_AMOUNT         = 80
LONG_KEYWORD         = "long"
DEBUG_TF_FILTER: list[str] = []


# =============================================================================
# GRID BUILDER
# =============================================================================

def _build_grid() -> tuple[list[str], list[tuple]]:
    """
    Build the parameter grid from enabled INDICATORS.
    Returns (active_keys, grid) where active_keys is the ordered list of
    enabled indicator names and grid is the list of all parameter combos.
    """
    active_keys = [k for k, v in INDICATORS.items() if v.get("enabled", True)]
    indicator_axes = [
        [(w, th) for w in INDICATORS[k]["windows"] for th in INDICATORS[k]["thresholds"]]
        for k in active_keys
    ]
    combos = list(itertools.product(*indicator_axes, COMBINE_MODES))
    return active_keys, combos


def _unpack_combo(active_keys: list[str], combo: tuple) -> tuple[dict[str, int], dict[str, float], str]:
    """Unpack a raw combo tuple into (windows, thresholds, mode) dicts."""
    *indicator_pairs, mode = combo
    windows    = {k: indicator_pairs[i][0] for i, k in enumerate(active_keys)}
    thresholds = {k: indicator_pairs[i][1] for i, k in enumerate(active_keys)}
    return windows, thresholds, mode


# =============================================================================
# FILTERED BACKTEST FOR A SINGLE COMBO
# =============================================================================

def _run_filtered_combo(
    baselines:       dict,
    strategies:      list[dict],
    indicator_cache: dict,
    thresholds:      dict[str, float],
    mode:            str,
) -> dict:
    results: dict = {}

    for strategy in strategies:
        sid = strategy['id']
        if sid not in baselines:
            continue

        results[sid] = {'is_long': strategy['is_long']}

        for period_key in EVAL_KEYS:
            if period_key not in baselines[sid]:
                continue

            cached       = baselines[sid][period_key]
            m_base       = cached['metrics']
            n_trending   = n_ranging = 0
            trending_arr = {}
            ranging_arr  = {}

            for sym, arr in cached['ohlcv_arrays'].items():
                signals     = cached['signal_cache'][sym]
                signal_idxs = np.nonzero(signals)[0]
                filt_t      = signals.copy()
                filt_r      = signals.copy()

                sym_cache = indicator_cache.get(get_cache_key(sym, strategy, ANALYSIS_MODE, REGIME_TIMEFRAME_MODE))
                if sym_cache is None:
                    filt_r[:] = 0
                    n_ranging += int(signals.sum())
                    trending_arr[sym] = {**arr, 'signal': filt_t}
                    ranging_arr[sym]  = {**arr, 'signal': filt_r}
                    continue
                ts_arr, values_arr = sym_cache

                if signal_idxs.size > 0:
                    tf    = strategy['timeframe'] if REGIME_TIMEFRAME_MODE == "STRATEGY" else None
                    batch = lookup_indicators_batch(
                        ts_arr, values_arr, arr['ts'][signal_idxs], timeframe=tf,
                    )
                    for i, idx in enumerate(signal_idxs):
                        indicator_values = {k: float(v[i]) if not np.isnan(v[i]) else None for k, v in batch.items()}
                        trending         = is_trending(indicator_values, thresholds, mode)
                        if trending:
                            filt_r[idx] = 0
                            n_trending  += 1
                        else:
                            filt_t[idx] = 0
                            n_ranging   += 1

                trending_arr[sym] = {**arr, 'signal': filt_t}
                ranging_arr[sym]  = {**arr, 'signal': filt_r}

            m_t          = run_backtest(trending_arr, strategy['best_params'])
            m_r          = run_backtest(ranging_arr,  strategy['best_params'])
            total        = n_trending + n_ranging
            trending_pct = n_trending / max(total, 1) * 100

            results[sid][period_key] = {
                'b_prof':            m_base['profit'],   'trending_prof': m_t['profit'],   'ranging_prof':  m_r['profit'],
                'b_dd':              m_base['max_dd'],   'trending_dd':   m_t['max_dd'],   'ranging_dd':    m_r['max_dd'],
                'b_wr':              m_base['win_rate'], 'trending_wr':   m_t['win_rate'], 'ranging_wr':    m_r['win_rate'],
                'trending_pct':      trending_pct,
                'ranging_pass_pct':  100 - trending_pct,
                'trending_pass_pct': trending_pct,
            }

    return results


def _combined_metric_for_period(results: dict, period_key: str, optimize_metric: str = "profit") -> tuple[float, float]:
    t_key, r_key, b_key = _METRIC_MAP.get(optimize_metric, _METRIC_MAP["profit"])
    comb, base = 0.0, 0.0
    for sid, data in results.items():
        if sid == 'is_long' or period_key not in data or not isinstance(data[period_key], dict):
            continue
        d   = data[period_key]
        cls = data.get('classification', 'neutral')
        if optimize_metric == "calmar":
            base += _calmar(d[b_key], d['b_dd'])
            if cls == 'ranging':
                comb += _calmar(d[r_key], d['ranging_dd'])
            elif cls == 'trending':
                comb += _calmar(d[t_key], d['trending_dd'])
            else:
                comb += _calmar(d[b_key], d['b_dd'])
        elif optimize_metric == "mix":
            base += _mix_score(d[b_key], d['b_dd'])
            if cls == 'ranging':
                comb += _mix_score(d[r_key], d['ranging_dd'])
            elif cls == 'trending':
                comb += _mix_score(d[t_key], d['trending_dd'])
            else:
                comb += _mix_score(d[b_key], d['b_dd'])
        else:
            base += d[b_key]
            if cls == 'ranging':
                comb += d[r_key]
            elif cls == 'trending':
                comb += d[t_key]
            else:
                comb += d[b_key]
    return comb, base


# =============================================================================
# PROCESS SINGLE COMBO  (parallelizable unit)
# =============================================================================

def _process_combo(
    combo_idx:        int,
    combo:            tuple,
    active_keys:      list[str],
    total_combos:     int,
    baselines:        dict,
    strategies:       list[dict],
    indicator_cache:  dict,
    baseline_profit:  float,
    baseline_dd:      float,
) -> dict:
    windows, thresholds, mode = _unpack_combo(active_keys, combo)
    label = combo_label(active_keys, windows, thresholds, mode)

    results = _run_filtered_combo(baselines, strategies, indicator_cache, thresholds, mode)
    for sid in results:
        if sid != 'is_long':
            results[sid]['classification'] = classify_strategy(results, sid, optimize_metric=OPTIMIZE_METRIC, classification_mode=CLASSIFICATION_MODE)

    period_summaries: dict[str, dict] = {}
    for pk in EVAL_KEYS:
        period_summaries[pk] = print_combo_period_table(results, strategies, pk, label)

    cls_list       = [results[sid].get('classification', 'neutral') for sid in results if sid != 'is_long']
    comb_p, comb_d = combined_metrics(results)
    avg_trend      = sum(ps['avg_trend_pct'] for ps in period_summaries.values()) / max(len(period_summaries), 1)

    weighted_delta = sum(
        pct_improvement(*_combined_metric_for_period(results, pk, OPTIMIZE_METRIC)) * PERIOD_WEIGHTS.get(pk, 0)
        for pk in period_summaries
    ) / sum(PERIOD_WEIGHTS.get(pk, 0) for pk in period_summaries)

    return {
        'windows':          windows,
        'thresholds':       thresholds,
        'mode':             mode,
        'combo_idx':        combo_idx,
        'combined_profit':  comb_p,
        'combined_dd':      comb_d,
        'weighted_delta':   weighted_delta,
        'baseline_profit':  baseline_profit,
        'baseline_dd':      baseline_dd,
        'avg_trend_pct':    avg_trend,
        'n_ranging':        cls_list.count('ranging'),
        'n_trending':       cls_list.count('trending'),
        'n_neutral':        cls_list.count('neutral'),
        'period_summaries': period_summaries,
        'label':            label,
    }


# =============================================================================
# MAIN RUN
# =============================================================================

def run() -> None:
    _t0 = time.time()

    active_keys, grid = _build_grid()
    total_combos      = len(grid)

    logger.info(f"\n{'='*120}")
    logger.info(f"  REGIME CALIBRATION — {total_combos} combinations")
    logger.info(f"  Active indicators: {', '.join(active_keys)}")
    for k in active_keys:
        cfg = INDICATORS[k]
        logger.info(f"    {k.upper()}: windows={cfg['windows']}  thresholds={cfg['thresholds']}")
    logger.info(f"  ANALYSIS_MODE={ANALYSIS_MODE} | REGIME_TIMEFRAME_MODE={REGIME_TIMEFRAME_MODE} | OPTIMIZE_METRIC={OPTIMIZE_METRIC} | CLASSIFICATION_MODE={CLASSIFICATION_MODE}")
    if ANALYSIS_MODE == "BTC" and REGIME_TIMEFRAME_MODE == "STRATEGY":
        logger.info(f"  → BTCUSDT loaded per strategy timeframe")
    logger.info(f"  Lookahead fix: normalize()-1day")
    logger.info(f"  Periods: {' + '.join(EVAL_KEYS)}")
    logger.info(f"{'='*120}")

    if not active_keys:
        logger.info("  No indicators enabled — aborting.")
        return

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

    # Precalculate all indicator caches before parallel loop
    indicator_cache_map: dict[tuple, dict] = {}
    for combo in grid:
        windows, _, _ = _unpack_combo(active_keys, combo)
        win_key = tuple(windows[k] for k in active_keys)
        if win_key not in indicator_cache_map:
            indicator_cache_map[win_key] = build_indicator_cache(
                baselines, strategies_filtered, windows,
                analysis_mode=ANALYSIS_MODE,
                regime_timeframe_mode=REGIME_TIMEFRAME_MODE,
            )

    ranking: list[dict] = Parallel(n_jobs=N_JOBS)(
        delayed(_process_combo)(
            combo_idx        = combo_idx,
            combo            = combo,
            active_keys      = active_keys,
            total_combos     = total_combos,
            baselines        = baselines,
            strategies       = strategies_filtered,
            indicator_cache  = indicator_cache_map[tuple(_unpack_combo(active_keys, combo)[0][k] for k in active_keys)],
            baseline_profit  = baseline_profit,
            baseline_dd      = baseline_dd,
        )
        for combo_idx, combo in enumerate(grid, 1)
    )

    for row in sorted(ranking, key=lambda x: x['combo_idx']):
        logger.info(f"\n  COMBO {row['combo_idx']}/{total_combos}")
        print_combo_summary(
            row['period_summaries'],
            row['n_ranging'], row['n_trending'],
            row['n_neutral'],
            row['combined_profit'], row['combined_dd'],
            row['baseline_profit'], row['baseline_dd'],
            row['label'],
        )

    ranking.sort(key=lambda x: x['weighted_delta'], reverse=True)
    print_ranking(ranking, active_keys)

    # =========================================================================
    # TOP1 CLASSIFICATION & BINS
    # =========================================================================
    top1 = ranking[0]
    logger.info(f"\n  TOP1 COMBO — {top1['label']}")

    top1_results = _run_filtered_combo(
        baselines, strategies_filtered,
        indicator_cache_map[tuple(top1['windows'][k] for k in active_keys)],
        top1['thresholds'], top1['mode'],
    )
    for sid in top1_results:
        top1_results[sid]['classification'] = classify_strategy(top1_results, sid, optimize_metric=OPTIMIZE_METRIC, classification_mode=CLASSIFICATION_MODE)

    print_consistency_table(top1_results)
    print_classification_summary(top1_results)

    if AUTO_SAVE_BINS:
        save_bins(top1_results, top1['windows'], top1['thresholds'], top1['mode'], BINS_OUTPUT_PATH, STRATEGIES_SET_NAME)
    else:
        logger.info("\n  ⚠️  AUTO_SAVE_BINS=False — bins not saved. Set to True to persist.")

    elapsed = int(time.time() - _t0)
    print(f"\n  Completed in {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s\n")

if __name__ == "__main__":
    run()