#develop/market_regime/regime_GE.py

import os
import sys
import time
import logging
import numpy as np
import pandas as pd

for _key in list(sys.modules.keys()):
    if any(_key.startswith(_mod) for _mod in ("shared_batchs", "shared_batch_regime", "shared_trading_batch_regime", "shared", "bitget")):
        del sys.modules[_key]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batch_regime.regime_GE_core import EVAL_KEYS
from shared_batch_regime.regime_GE_core import pct_improvement, is_trending, combo_label
from shared_batch_regime.regime_GE_core import load_strategies_config

from shared_batch_regime.regime_GE_core import load_ohlcv_for_period, run_backtest, precompute_baselines
from shared_batch_regime.regime_GE_core import build_indicator_cache, get_cache_key, classify_strategy, combined_metrics, print_combo_period_table
from shared_batch_regime.regime_GE_core import lookup_indicators
from shared_batch_regime.regime_GE_core import print_consistency_table, print_classification_summary
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays

# =============================================================================
# CONFIGURATION
# =============================================================================

STRATEGIES_SET_NAME  = "E1"
BINS_OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), f"regime_BINS_{STRATEGIES_SET_NAME}.py")

# =============================================================================
# REGIME CONFIGURATION
# =============================================================================
OPTIMIZE_METRIC       = "calmar"   # "profit" | "max_dd" | "win_rate"
ANALYSIS_MODE         = "SYMBOL"   # "BTC" | "SYMBOL"
REGIME_TIMEFRAME_MODE = "DAILY"    # "DAILY" | "STRATEGY"
COMBINE_MODES         = ["OR"]    

INDICATORS: dict[str, dict] = {
    "atr_norm": {
        "windows":    [30],
        "thresholds": [0.02],
        "enabled":    True,
    },
    "er": {
        "windows":    [40],
        "thresholds": [0.6],
        "enabled":    True,
    },
    "hurst": {
        "windows":    [30],
        "thresholds": [0.5],
        "enabled":    False,
    },
}

# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

ORDER_AMOUNT  = 80
LONG_KEYWORD  = "long"
SHORT_KEYWORD = "short"

DEBUG_TF_FILTER: list[str] = []
DEBUG_LOOKAHEAD_N          = 10
DEBUG_LOOKAHEAD_DONE       = False

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# HELPERS
# =============================================================================

def _active_config() -> tuple[dict[str, int], dict[str, float], str]:
    """Extract windows, thresholds and combine mode for enabled indicators."""
    windows    = {k: v["windows"][0]    for k, v in INDICATORS.items() if v.get("enabled")}
    thresholds = {k: v["thresholds"][0] for k, v in INDICATORS.items() if v.get("enabled")}
    mode       = COMBINE_MODES[0]
    return windows, thresholds, mode


# =============================================================================
# LOOKAHEAD DEBUG
# =============================================================================

def _print_lookahead_debug(
    strategy_id:        str,
    period_key:         str,
    sym:                str,
    arr:                dict,
    ts_arr:             np.ndarray,
    values_arr:         dict[str, np.ndarray],
    thresholds:         dict[str, float],
    mode:               str,
    signals:            np.ndarray,
    strategy_timeframe: str | None = None,
) -> None:
    signal_idxs = np.nonzero(signals)[0][:DEBUG_LOOKAHEAD_N]
    if len(signal_idxs) == 0:
        return

    active_keys = list(values_arr.keys())
    col_headers = "  ".join(f"{k.upper():>8}" for k in active_keys)

    print(f"\n{'='*120}")
    print(f"  LOOKAHEAD DEBUG — strategy={strategy_id} | period={period_key} | symbol={sym}")
    ind_info = ", ".join(f"{k}(w={INDICATORS[k]['windows'][0]}, th={thresholds[k]})" for k in active_keys)
    print(f"  Indicators: {ind_info}")
    print(f"  MODE={mode}")
    print(f"{'='*120}")
    print(f"  {'#':>3}  {'SIGNAL_TS':<26}  {'IND_CANDLE_TS':<26}  {col_headers}  {'TRENDING':>8}")
    print(f"  {'─'*110}")

    for n, sig_idx in enumerate(signal_idxs, 1):
        signal_ts = pd.Timestamp(arr['ts'][sig_idx])
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("UTC")

        indicator_values, lookup_idx = lookup_indicators(
            ts_arr, values_arr, signal_ts,
            timeframe=strategy_timeframe if REGIME_TIMEFRAME_MODE == "STRATEGY" else None,
            return_idx=True,
        )
        ind_ts = pd.Timestamp(ts_arr[lookup_idx]).tz_localize("UTC") if lookup_idx >= 0 else None

        trending   = is_trending(indicator_values, thresholds, mode)
        val_cols   = "  ".join(
            f"{indicator_values[k]:>8.4f}" if indicator_values.get(k) is not None else f"{'—':>8}"
            for k in active_keys
        )
        ind_ts_str = str(ind_ts) if ind_ts else "NO DATA"
        print(f"  {n:>3}  {str(signal_ts):<26}  {ind_ts_str:<26}  {val_cols}  {str(trending):>8}")

    print(f"  {'─'*110}\n")


# =============================================================================
# FILTERED ARRAYS BUILDER
# =============================================================================

DEBUG_SIGNALS_STRATEGY = "02_reversal_short_15m"
DEBUG_SIGNALS_SYMBOL   = "BTCUSDT"
DEBUG_SIGNALS_N        = 10


def _build_filtered_arrays(
    ohlcv_arrays:    dict,
    strategy:        dict,
    indicator_cache: dict,
    thresholds:      dict[str, float],
    mode:            str,
    period_key:      str,
) -> tuple[dict, dict, dict, float]:
    """
    Build baseline, trending and ranging signal arrays.
    - trending_arrays: signals where market IS trending     (blocks ranging signals)
    - ranging_arrays:  signals where market IS NOT trending (blocks trending signals)
    Returns (baseline_arrays, trending_arrays, ranging_arrays, trending_pct).
    """
    global DEBUG_LOOKAHEAD_DONE

    baseline_arrays = {}
    trending_arrays = {}
    ranging_arrays  = {}
    n_trending      = 0
    n_ranging       = 0

    _debug_signals = (
        logger.isEnabledFor(logging.DEBUG)
        and strategy['id'] == DEBUG_SIGNALS_STRATEGY
    )

    for sym, arr in ohlcv_arrays.items():
        signals     = strategy['signal_fn'](arr, **strategy['signal_params'], live_trading=False)
        signal_idxs = np.nonzero(signals)[0]

        filt_t = signals.copy()
        filt_r = signals.copy()

        sym_cache = indicator_cache.get(get_cache_key(sym, strategy, ANALYSIS_MODE, REGIME_TIMEFRAME_MODE))
        if sym_cache is None:
            filt_t[:] = 0
            n_trending += int(signals.sum())
            baseline_arrays[sym] = {**arr, 'signal': signals}
            trending_arrays[sym] = {**arr, 'signal': filt_t}
            ranging_arrays[sym]  = {**arr, 'signal': filt_r}
            continue
        ts_arr, values_arr = sym_cache

        if not DEBUG_LOOKAHEAD_DONE and logger.isEnabledFor(logging.DEBUG) and strategy['id'] == DEBUG_SIGNALS_STRATEGY:
            _print_lookahead_debug(
                strategy['id'], period_key, sym, arr,
                ts_arr, values_arr, thresholds, mode, signals,
                strategy_timeframe=strategy['timeframe'],
            )
            DEBUG_LOOKAHEAD_DONE = True

        _debug_this_sym = _debug_signals and sym == DEBUG_SIGNALS_SYMBOL
        _debug_count    = 0
        _ind_src        = "BTCUSDT" if ANALYSIS_MODE == "BTC" else sym

        if _debug_this_sym:
            active_keys = list(values_arr.keys())
            col_ind     = "  ".join(f"{k.upper():>9}" for k in active_keys)
            logger.debug(f"\n{'='*140}")
            logger.debug(f"  SIGNAL DEBUG — strategy={strategy['id']} | period={period_key} | symbol={sym} | ANALYSIS_MODE={ANALYSIS_MODE}")
            logger.debug(f"  thresholds={thresholds} | mode={mode}")
            logger.debug(f"{'='*140}")
            logger.debug(f"  {'TIMESTAMP':<28}  {'BASELINE':>8}  {'FILT_T':>6}  {'FILT_R':>6}  {col_ind}  {'TRENDING':>8}  {'IND_SRC':<10}")
            logger.debug(f"  {'─'*140}")

        for idx in signal_idxs:
            indicator_values = lookup_indicators(
                ts_arr, values_arr, pd.Timestamp(arr['ts'][idx]),
                timeframe=strategy['timeframe'] if REGIME_TIMEFRAME_MODE == "STRATEGY" else None,
            )
            trending         = is_trending(indicator_values, thresholds, mode)

            if trending:
                # Market is trending → keep in trending_arrays, block from ranging_arrays
                filt_r[idx] = 0
                n_trending  += 1
            else:
                # Market is ranging → keep in ranging_arrays, block from trending_arrays
                filt_t[idx] = 0
                n_ranging   += 1

            if _debug_this_sym:
                if _debug_count >= DEBUG_SIGNALS_N:
                    continue
                ts_str   = str(pd.Timestamp(arr['ts'][idx]))
                val_cols = "  ".join(
                    f"{indicator_values[k]:>9.4f}" if indicator_values.get(k) is not None else f"{'—':>9}"
                    for k in active_keys
                )
                logger.debug(
                    f"  {ts_str:<28}  "
                    f"{int(signals[idx]):>8}  "
                    f"{int(filt_t[idx]):>6}  "
                    f"{int(filt_r[idx]):>6}  "
                    f"{val_cols}  "
                    f"{str(trending):>8}  "
                    f"{_ind_src:<10}"
                )
                _debug_count += 1

        if _debug_this_sym:
            n_sig     = len(signal_idxs)
            n_trend_s = int(np.sum(filt_t[signal_idxs] == 0))
            n_range_s = int(np.sum(filt_r[signal_idxs] == 0))
            logger.debug(f"  {'─'*130}")
            logger.debug(f"  TOTAL signals={n_sig}  blocked_by_ranging={n_trend_s}  blocked_by_trending={n_range_s}\n")

        baseline_arrays[sym] = {**arr, 'signal': signals}
        trending_arrays[sym] = {**arr, 'signal': filt_t}
        ranging_arrays[sym]  = {**arr, 'signal': filt_r}

    total        = n_trending + n_ranging
    trending_pct = n_trending / max(total, 1) * 100
    return baseline_arrays, trending_arrays, ranging_arrays, trending_pct


# =============================================================================
# PERIOD EVALUATION
# =============================================================================

def _debug_print_trending_arrays(
    baseline_arrays: dict,
    trending_arrays: dict,
    sym:             str,
    n:               int,
    period_key:      str,
    strategy_id:     str,
) -> None:
    """
    Print raw arrays entering run_backtest(trending_arrays).
    No logic — reads arrays as-is after _build_filtered_arrays.
    Columns: TIMESTAMP | BASELINE | TRENDING_REGIME | FILT_T (enters backtester)
    Only rows where baseline != 0, first N.
    """
    if sym not in baseline_arrays or sym not in trending_arrays:
        logger.debug(f"  [TRENDING ARRAY DEBUG] symbol {sym} not found")
        return

    ts    = baseline_arrays[sym]['ts']
    sig_b = baseline_arrays[sym]['signal']
    sig_t = trending_arrays[sym]['signal']

    logger.debug(f"\n  {'='*80}")
    logger.debug(f"  TRENDING ARRAY DEBUG — strategy={strategy_id} | period={period_key} | symbol={sym}")
    logger.debug(f"  Columns: BASELINE=signal_fn output | TRENDING_REGIME=is_trending result | FILT_T=enters backtester")
    logger.debug(f"  {'='*80}")
    logger.debug(f"  {'TIMESTAMP':<28}  {'BASELINE':>8}  {'TRENDING_REGIME':>15}  {'FILT_T':>6}")
    logger.debug(f"  {'─'*65}")

    count = 0
    for i in range(len(ts)):
        if sig_b[i] == 0:
            continue
        trending_regime = sig_t[i] != 0
        logger.debug(
            f"  {str(pd.Timestamp(ts[i])):<28}  "
            f"{int(sig_b[i]):>8}  "
            f"{str(trending_regime):>15}  "
            f"{int(sig_t[i]):>6}"
        )
        count += 1
        if count >= n:
            break

    signal_rows = int(np.sum(sig_b != 0))
    logger.debug(f"  {'─'*65}")
    logger.debug(f"  showing {count} of {signal_rows} signal rows\n")


def _evaluate_period(
    strategy:            dict,
    period_key:          str,
    indicator_cache:     dict,
    thresholds:          dict[str, float],
    mode:                str,
    strategies_set_name: str,
) -> dict | None:
    ohlcv_data = load_ohlcv_for_period(strategy, period_key, strategies_set_name)

    if not ohlcv_data:
        return None

    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_data)

    baseline_arrays, trending_arrays, ranging_arrays, trending_pct = _build_filtered_arrays(
        ohlcv_arrays, strategy, indicator_cache, thresholds, mode, period_key,
    )

    _debug_arrays = (
        logger.isEnabledFor(logging.DEBUG)
        and strategy['id'] == DEBUG_SIGNALS_STRATEGY
    )

    if _debug_arrays:
        _debug_print_trending_arrays(
            baseline_arrays, trending_arrays,
            DEBUG_SIGNALS_SYMBOL, DEBUG_SIGNALS_N,
            period_key, strategy['id'],
        )

    m_b = run_backtest(baseline_arrays, strategy['best_params'])
    m_t = run_backtest(trending_arrays, strategy['best_params'])
    m_r = run_backtest(ranging_arrays,  strategy['best_params'])

    return {
        'baseline':     m_b,
        'trending':     m_t,
        'ranging':      m_r,
        'trending_pct': trending_pct,
    }


# =============================================================================
# MAIN RUN
# =============================================================================

def run(eval_keys: list[str]) -> None:
    global DEBUG_LOOKAHEAD_DONE
    DEBUG_LOOKAHEAD_DONE = False

    _t0 = time.time()

    windows, thresholds, mode = _active_config()
    active_keys               = list(windows.keys())

    if not active_keys:
        print("  No indicators enabled — aborting.")
        return

    print(f"\n{'='*120}")
    print(f"  REGIME DUAL FILTER  [MODE={ANALYSIS_MODE}]")
    print(f"  Active indicators: {', '.join(active_keys)}")
    for k in active_keys:
        print(f"    {k.upper()}: window={windows[k]}  threshold={thresholds[k]}  combine={mode}")
    print(f"  Eval: {' + '.join(eval_keys)}")
    print(f"{'='*120}\n")

    strategies_all = load_strategies_config(STRATEGIES_SET_NAME)
    if not strategies_all:
        print("  No strategies found — aborting.")
        return

    baselines, strategies_filtered = precompute_baselines(strategies_all, STRATEGIES_SET_NAME)
    if not strategies_filtered:
        print("  No strategies passed the baseline filter — aborting.")
        return

    indicator_cache = build_indicator_cache(
        baselines, strategies_filtered, windows,
        analysis_mode=ANALYSIS_MODE,
        regime_timeframe_mode=REGIME_TIMEFRAME_MODE,
    )

    label             = combo_label(active_keys, windows, thresholds, mode)
    strategy_results: dict[str, dict] = {}

    for period_key in eval_keys:
        results: dict = {}

        for strategy in strategies_filtered:
            if DEBUG_TF_FILTER and strategy['timeframe'] not in DEBUG_TF_FILTER:
                continue

            sid    = strategy['id']
            result = _evaluate_period(strategy, period_key, indicator_cache, thresholds, mode, STRATEGIES_SET_NAME)
            if not result:
                continue

            m_b, m_t, m_r = result['baseline'], result['trending'], result['ranging']
            trending_pct  = result.get('trending_pct', 0.0)

            if sid not in results:
                results[sid] = {'is_long': strategy['is_long']}
            results[sid][period_key] = {
                'b_prof':            m_b['profit'],   'trending_prof': m_t['profit'],   'ranging_prof': m_r['profit'],
                'b_dd':              m_b['max_dd'],   'trending_dd':   m_t['max_dd'],   'ranging_dd':   m_r['max_dd'],
                'b_wr':              m_b['win_rate'], 'trending_wr':   m_t['win_rate'], 'ranging_wr':   m_r['win_rate'],
                'trending_pct':      trending_pct,
                'trending_pass_pct': trending_pct,
                'ranging_pass_pct':  100 - trending_pct,
            }
            logger.debug(f"  [n_trades] {sid} {period_key}: baseline={m_b['n_trades']}")

            if sid not in strategy_results:
                strategy_results[sid] = {'is_long': strategy['is_long']}
            strategy_results[sid][period_key] = results[sid][period_key]

        print_combo_period_table(results, strategies_filtered, period_key, label)

    for sid in strategy_results:
        strategy_results[sid]['classification'] = classify_strategy(strategy_results, sid, optimize_metric=OPTIMIZE_METRIC)

    print_consistency_table(strategy_results)
    comb_p, comb_d  = combined_metrics(strategy_results)
    baseline_profit = sum(strategy_results[sid][pk]['b_prof'] for sid in strategy_results for pk in EVAL_KEYS if pk in strategy_results[sid] and isinstance(strategy_results[sid][pk], dict))
    comb_pct        = pct_improvement(comb_p, baseline_profit)
    print(f"\n  BASELINE={baseline_profit:.1f}  COMB_PROF={comb_p:.1f}  COMB_Δ%={comb_pct:+.1f}%  COMB_DD%={comb_d:.1f}%")
    print_classification_summary(strategy_results)

    elapsed = int(time.time() - _t0)
    print(f"\n  ⏱  Completed in {elapsed//60}m {elapsed%60}s\n")


if __name__ == "__main__":
    run(EVAL_KEYS)