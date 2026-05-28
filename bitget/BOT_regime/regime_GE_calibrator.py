#develop/market_regime/regime_GE_calibration.py

import os
import sys
import time
import itertools
import logging

for _key in list(sys.modules.keys()):
    if any(_key.startswith(_mod) for _mod in ("shared_batchs", "shared_batch_regime", "shared_trading_batch_regime", "shared", "bitget")):
        del sys.modules[_key]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batch_regime.regime_GE_core import EVAL_KEYS,pct_improvement
from shared_batch_regime.regime_GE_core import is_trending
from shared_batch_regime.regime_GE_core import combo_label, load_strategies_config
from shared_batch_regime.regime_GE_core import precompute_baselines
from shared_batch_regime.regime_GE_core import print_combo_period_table, print_combo_summary, print_ranking
from shared_batch_regime.regime_GE_core import build_indicator_cache, get_cache_key, classify_strategy, combined_metrics, lookup_indicators_batch
from shared_batch_regime.regime_GE_core import run_backtest


import numpy as np


# =============================================================================
# REGIME CONFIGURATION
# =============================================================================

# =============================================================================
# ANALYSIS_MODE         = "BTC"   # "BTC" | "SYMBOL"
# BTC_TIMEFRAME         = "1Dutc"
# COMBINE_MODES         = ["AND","OR"]
# REGIME_TIMEFRAME_MODE = "STRATEGY" # "DAILY" | "STRATEGY"
# =============================================================================

ANALYSIS_MODE         = "SYMBOL"   # "BTC" | "SYMBOL"
BTC_TIMEFRAME         = "1Dutc"
REGIME_TIMEFRAME_MODE = "DAILY"    # "DAILY" | "STRATEGY"
COMBINE_MODES         = ["OR"]    


PERIOD_WEIGHTS = {
    "OOS1": 0.50,
    "OOS2": 0.25,
    "OOS3": 0.25,
}

INDICATORS: dict[str, dict] = {
    "atr_norm": {
        "windows":    [5,10,15],
        "thresholds": [0.02,0.04,0.06],
        "enabled":    True,
    },
    "er": {
        "windows":    [10,20,30],
        "thresholds": [0.5,0.6,0.7],
        "enabled":    True,
    },
    "hurst": {
        "windows":    [30,50],
        "thresholds": [0.5,0.55,0.60,0.65],
        "enabled":    False,
    },
}



INDICATORS: dict[str, dict] = {
    "atr_norm": {
        "windows":    [10],
        "thresholds": [0.04],
        "enabled":    True,
    },
    "er": {
        "windows":    [20],
        "thresholds": [0.4],
        "enabled":    True,
    },
    "hurst": {
        "windows":    [30],
        "thresholds": [0.4],
        "enabled":    True,
    },
}

ORDER_AMOUNT     = 80
LONG_KEYWORD     = "long"
DEBUG_TF_FILTER: list[str] = []

logging.basicConfig(format="%(message)s", level=logging.INFO)
#logging.basicConfig(format="%(message)s", level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)


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
# FAST — experimental, remove if results differ from original
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


def _combined_profit_for_period(results: dict, period_key: str) -> tuple[float, float]:
    """Combined profit and baseline for a single period, respecting each strategy's classification."""
    comb, base = 0.0, 0.0
    for sid, data in results.items():
        if sid == 'is_long' or period_key not in data or not isinstance(data[period_key], dict):
            continue
        d   = data[period_key]
        cls = data.get('classification', 'neutral')
        base += d['b_prof']
        if cls == 'ranging':
            comb += d['ranging_prof']
        elif cls == 'trending':
            comb += d['trending_prof']
        elif cls == 'both':
            comb += max(d['ranging_prof'], d['trending_prof'])
        else:
            comb += d['b_prof']
    return comb, base
# =============================================================================
# MAIN RUN
# =============================================================================

def run() -> None:
    _t0 = time.time()

    active_keys, grid = _build_grid()
    total_combos      = len(grid)

    print(f"\n{'='*120}")
    print(f"  REGIME CALIBRATION — {total_combos} combinations  [MODE={ANALYSIS_MODE}]")
    print(f"  Active indicators: {', '.join(active_keys)}")
    for k in active_keys:
        cfg = INDICATORS[k]
        print(f"    {k.upper()}: windows={cfg['windows']}  thresholds={cfg['thresholds']}")
    print(f"  BTC_TF={BTC_TIMEFRAME} | Lookahead fix: normalize()-1day | TF_MODE={REGIME_TIMEFRAME_MODE}")
    print(f"  Periods: {' + '.join(EVAL_KEYS)}")
    print(f"{'='*120}")

    if not active_keys:
        print("  No indicators enabled — aborting.")
        return

    strategies_all = load_strategies_config()
    if not strategies_all:
        print("  No strategies found — aborting.")
        return

    baselines, strategies_filtered = precompute_baselines(strategies_all)
    if not strategies_filtered:
        print("  No strategies passed the baseline filter — aborting.")
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

    ranking: list[dict] = []
    indicator_cache_map: dict[tuple, dict] = {}

    for combo_idx, combo in enumerate(grid, 1):
        windows, thresholds, mode = _unpack_combo(active_keys, combo)
        label   = combo_label(active_keys, windows, thresholds, mode)
        win_key = tuple(windows[k] for k in active_keys)

        print(f"\n{'='*120}")
        print(f"  COMBO {combo_idx}/{total_combos} — {label}")
        print(f"{'='*120}")

        if win_key not in indicator_cache_map:
            indicator_cache_map[win_key] = build_indicator_cache(
                baselines, strategies_filtered, windows,
                analysis_mode=ANALYSIS_MODE,
                regime_timeframe_mode=REGIME_TIMEFRAME_MODE,
            )
        indicator_cache = indicator_cache_map[win_key]

        # FAST — experimental, remove if results differ from original
        results = _run_filtered_combo(baselines, strategies_filtered, indicator_cache, thresholds, mode)
        for sid in results:
            if sid != 'is_long':
                results[sid]['classification'] = classify_strategy(results, sid)

        period_summaries: dict[str, dict] = {}
        for pk in EVAL_KEYS:
            period_summaries[pk] = print_combo_period_table(results, strategies_filtered, pk, label)

        cls_list       = [results[sid].get('classification', 'neutral') for sid in results if sid != 'is_long']
        comb_p, comb_d = combined_metrics(results)
        avg_trend      = sum(ps['avg_trend_pct'] for ps in period_summaries.values()) / max(len(period_summaries), 1)

        print_combo_summary(
            period_summaries,
            cls_list.count('ranging'), cls_list.count('trending'),
            cls_list.count('both'),    cls_list.count('neutral'),
            comb_p, comb_d, baseline_profit, baseline_dd, label,
        )
# =============================================================================
#         for pk in period_summaries:
#             cp, bp = _combined_profit_for_period(results, pk)
#             print(f"  {pk}: comb={cp:.1f}  base={bp:.1f}  delta={pct_improvement(cp, bp):.2f}%  weight={PERIOD_WEIGHTS.get(pk,0)}")
# =============================================================================
        weighted_delta = sum(
            pct_improvement(*_combined_profit_for_period(results, pk)) * PERIOD_WEIGHTS.get(pk, 0)
            for pk in period_summaries
        ) / sum(PERIOD_WEIGHTS.get(pk, 0) for pk in period_summaries)

        ranking.append({
            'windows':    windows,
            'thresholds': thresholds,
            'mode':       mode,
            'combo_idx':  combo_idx,
            'combined_profit':  comb_p,          'combined_dd':   comb_d,
            'weighted_delta':   weighted_delta,
            'baseline_profit':  baseline_profit,  'baseline_dd':  baseline_dd,
            'avg_trend_pct':    avg_trend,
            'n_ranging':  cls_list.count('ranging'),
            'n_trending': cls_list.count('trending'),
            'n_both':     cls_list.count('both'),
            'n_neutral':  cls_list.count('neutral'),
        })

    ranking.sort(key=lambda x: x['weighted_delta'], reverse=True)
    print_ranking(ranking, active_keys)

    elapsed = int(time.time() - _t0)
    print(f"\n  Completed in {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s\n")


if __name__ == "__main__":
    run()