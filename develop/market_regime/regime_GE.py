#develop/market_regime/regime_GE.py
"""
Regime dual filter with auto-classification.

Imports all shared functions from regime_GE_core.
Supports any combination of atr_norm / er / hurst via the INDICATORS dict.

Two filters are evaluated per strategy:
  - filter_trending : block signals when regime is trending
  - filter_ranging  : block signals when regime is ranging

Classification per strategy (consistent across ALL eval periods):
  - "trending"  : filter_ranging  improves in ALL periods
  - "ranging"   : filter_trending improves in ALL periods
  - "both"      : both filters improve in ALL periods
  - "neutral"   : neither filter improves consistently

Only strategies with B_PROF > 0 in ALL eval periods are included.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd

for _key in list(sys.modules.keys()):
    if any(_key.startswith(_mod) for _mod in ("shared_batchs", "shared_batch_develop", "shared_trading_batch_develop", "shared", "bitget")):
        del sys.modules[_key]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batch_develop.regime_GE_core import (
    PERIODS,
    EVAL_KEYS,
    DATA_FOLDER_IS,
    pct_improvement,
    _is_trending,
    _combo_label,
    load_strategies_config,
    load_symbols,
    _load_ohlcv,
    _load_ohlcv_for_period,
    _run_backtest,
    precompute_baselines,
    build_indicator_cache,
    _classify_strategy,
    _print_combo_period_table,
)
from shared_trading_batch_develop.regime_metrics import lookup_indicators
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays

# =============================================================================
# CONFIGURATION
# =============================================================================

STRATEGIES_SET_NAME  = "E1"
BINS_OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), f"regime_bins_06_{STRATEGIES_SET_NAME}_ge.py")

# =============================================================================
# REGIME CONFIGURATION
# =============================================================================

ANALYSIS_MODE = "SYMBOL"   # "BTC" | "SYMBOL"
BTC_TIMEFRAME = "1Dutc"
COMBINE_MODES = ["OR"]     # single value used at runtime: COMBINE_MODES[0]

INDICATORS: dict[str, dict] = {
    "atr_norm": {
        "windows":    [10],
        "thresholds": [0.04],
        "enabled":    True,
    },
    "er": {
        "windows":    [10],
        "thresholds": [0.6],
        "enabled":    True,
    },
    "hurst": {
        "windows":    [30],
        "thresholds": [0.8],
        "enabled":    True,
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

logging.basicConfig(format="%(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)


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
    strategy_id: str,
    period_key:  str,
    sym:         str,
    arr:         dict,
    ts_arr:      np.ndarray,
    values_arr:  dict[str, np.ndarray],
    thresholds:  dict[str, float],
    mode:        str,
    signals:     np.ndarray,
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
    print(f"  {'#':>3}  {'SIGNAL_TS':<26}  {'BTC_CANDLE_TS':<26}  {col_headers}  {'TRENDING':>8}")
    print(f"  {'─'*110}")

    for n, sig_idx in enumerate(signal_idxs, 1):
        signal_ts = pd.Timestamp(arr['ts'][sig_idx])
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("UTC")

        indicator_values = lookup_indicators(ts_arr, values_arr, signal_ts)

        ts_norm  = signal_ts.normalize() - pd.Timedelta(days=1)
        ts_norm  = ts_norm.tz_localize("UTC") if ts_norm.tzinfo is None else ts_norm
        idx      = np.searchsorted(ts_arr, np.datetime64(ts_norm.value, "ns"), side="right") - 1
        btc_ts   = pd.Timestamp(ts_arr[idx]).tz_localize("UTC") if idx >= 0 else None

        trending   = _is_trending(indicator_values, thresholds, mode)
        val_cols   = "  ".join(
            f"{indicator_values[k]:>8.4f}" if indicator_values.get(k) is not None else f"{'—':>8}"
            for k in active_keys
        )
        btc_ts_str = str(btc_ts) if btc_ts else "NO BTC DATA"
        print(f"  {n:>3}  {str(signal_ts):<26}  {btc_ts_str:<26}  {val_cols}  {str(trending):>8}")

    print(f"  {'─'*110}\n")


# =============================================================================
# FILTERED ARRAYS BUILDER
# =============================================================================

def _build_filtered_arrays(
    ohlcv_arrays:    dict,
    strategy:        dict,
    indicator_cache: dict,
    thresholds:      dict[str, float],
    mode:            str,
    period_key:      str,
) -> tuple[dict, dict, dict, float]:
    """
    Build baseline, filter_trending and filter_ranging signal arrays.
    Returns (baseline_arrays, trending_arrays, ranging_arrays, trending_pct).
    """
    global DEBUG_LOOKAHEAD_DONE

    baseline_arrays = {}
    trending_arrays = {}
    ranging_arrays  = {}
    n_trending      = 0
    n_ranging       = 0

    btc_cache = indicator_cache.get("BTCUSDT") if ANALYSIS_MODE == "BTC" else None

    for sym, arr in ohlcv_arrays.items():
        signals     = strategy['signal_fn'](arr, **strategy['signal_params'], live_trading=False)
        signal_idxs = np.nonzero(signals)[0]

        filt_t = signals.copy()
        filt_r = signals.copy()

        if ANALYSIS_MODE == "SYMBOL":
            sym_cache = indicator_cache.get(sym)
            if sym_cache is None:
                filt_t[:] = 0
                n_trending += int(signals.sum())
                baseline_arrays[sym] = {**arr, 'signal': signals}
                trending_arrays[sym] = {**arr, 'signal': filt_t}
                ranging_arrays[sym]  = {**arr, 'signal': filt_r}
                continue
            ts_arr, values_arr = sym_cache
        else:
            ts_arr, values_arr = btc_cache

        if not DEBUG_LOOKAHEAD_DONE and logger.isEnabledFor(logging.DEBUG):
            _print_lookahead_debug(strategy['id'], period_key, sym, arr, ts_arr, values_arr, thresholds, mode, signals)
            DEBUG_LOOKAHEAD_DONE = True

        for idx in signal_idxs:
            indicator_values = lookup_indicators(ts_arr, values_arr, pd.Timestamp(arr['ts'][idx]))
            trending         = _is_trending(indicator_values, thresholds, mode)
            if trending:
                filt_t[idx] = 0
                n_trending  += 1
            else:
                filt_r[idx] = 0
                n_ranging   += 1

        baseline_arrays[sym] = {**arr, 'signal': signals}
        trending_arrays[sym] = {**arr, 'signal': filt_t}
        ranging_arrays[sym]  = {**arr, 'signal': filt_r}

    total        = n_trending + n_ranging
    trending_pct = n_trending / max(total, 1) * 100
    return baseline_arrays, trending_arrays, ranging_arrays, trending_pct


# =============================================================================
# PERIOD EVALUATION
# =============================================================================

def _evaluate_period(
    strategy:        dict,
    period_key:      str,
    indicator_cache: dict,
    thresholds:      dict[str, float],
    mode:            str,
) -> dict | None:
    ohlcv_data = _load_ohlcv_for_period(strategy, period_key)
    if not ohlcv_data:
        return None
    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_data)
    baseline_arrays, trending_arrays, ranging_arrays, trending_pct = _build_filtered_arrays(
        ohlcv_arrays, strategy, indicator_cache, thresholds, mode, period_key,
    )
    m_b = _run_backtest(baseline_arrays, strategy['best_params'])
    m_t = _run_backtest(trending_arrays, strategy['best_params'])
    m_r = _run_backtest(ranging_arrays,  strategy['best_params'])

    return {
        'baseline':     m_b,
        'trending':     m_t,
        'ranging':      m_r,
        'trending_pct': trending_pct,
    }


# =============================================================================
# PRINT HELPERS
# =============================================================================

def _print_consistency_table(strategy_results: dict) -> None:
    for filter_key, filter_label in [("ranging_prof", "RANGING PASS"), ("trending_prof", "TRENDING PASS")]:
        print(f"\n{'='*120}")
        print(f"  STRATEGIES IMPROVING PROFIT IN ALL {len(EVAL_KEYS)} OOS PERIODS — {filter_label}")
        print(f"{'='*120}")
        header = f"  {'STRATEGY':<35} {'DIR':<6} {'CLASS':<10}"
        for pk in EVAL_KEYS:
            header += f"  {pk:>10}"
        header += f"  {'ALL Δ%':>8}"
        print(header)
        print(f"  {'─'*95}")
        consistent = [
            (sid, data) for sid, data in sorted(strategy_results.items())
            if all(pk in data and isinstance(data[pk], dict) for pk in EVAL_KEYS)
            and all(data[pk][filter_key] > data[pk]['b_prof'] for pk in EVAL_KEYS)
        ]
        if not consistent:
            print(f"  No strategy improved profit in all {len(EVAL_KEYS)} periods.\n")
            continue
        for sid, data in consistent:
            direction = "LONG" if data['is_long'] else "SHORT"
            cls       = data.get('classification', 'neutral').upper()
            row       = f"  {sid:<35} {direction:<6} {cls:<10}"
            total_b = total_f = 0.0
            for pk in EVAL_KEYS:
                dpct     = pct_improvement(data[pk][filter_key], data[pk]['b_prof'])
                row     += f"  \033[92m{dpct:>+9.1f}%\033[0m"
                total_b += data[pk]['b_prof']
                total_f += data[pk][filter_key]
            row += f"  \033[92m{pct_improvement(total_f, total_b):>+7.1f}%\033[0m"
            print(row)
        print(f"  {'─'*95}\n")


def _print_classification_summary(strategy_results: dict) -> None:
    print(f"\n{'='*120}")
    print(f"  STRATEGY CLASSIFICATION SUMMARY")
    print(f"{'='*120}")
    print(f"  {'STRATEGY':<35} {'DIR':<6} {'CLASS':<10}")
    print(f"  {'─'*55}")
    for sid, data in sorted(strategy_results.items()):
        direction = "LONG" if data['is_long'] else "SHORT"
        cls       = data.get('classification', 'neutral').upper()
        color     = {'RANGING': "\033[92m", 'TRENDING': "\033[94m", 'BOTH': "\033[93m", 'NEUTRAL': "\033[90m"}.get(cls, "")
        print(f"  {sid:<35} {direction:<6} {color}{cls:<10}\033[0m")
    print(f"  {'─'*55}\n")


# =============================================================================
# PERSIST
# =============================================================================

def _save_bins(strategy_results: dict, windows: dict, thresholds: dict, mode: str) -> None:
    active_keys  = list(windows.keys())
    header_lines = ["# Auto-generated by regime_GE.py"]
    header_lines.append("# " + " | ".join(f"{k.upper()}({windows[k]})>={thresholds[k]}" for k in active_keys) + f" | MODE={mode}")
    header_lines.append("# Classification: ranging=block_trending | trending=block_ranging | both | neutral")
    header_lines.append("")
    for k in active_keys:
        header_lines.append(f"{k.upper()}_WINDOW    = {windows[k]}")
        header_lines.append(f"{k.upper()}_THRESHOLD = {thresholds[k]}")
    header_lines += [f"COMBINE_MODE = '{mode}'", "", "REGIME_BINS = {"]

    bin_lines = [f'    "{sid}": "{data.get("classification", "neutral")}",' for sid, data in sorted(strategy_results.items())]

    with open(BINS_OUTPUT_PATH, "w") as f:
        f.write("\n".join(header_lines + bin_lines + ["}"]) + "\n")
    print(f"\n  ✅ Bins saved to: {BINS_OUTPUT_PATH}")


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
    print(f"  BTC_TF={BTC_TIMEFRAME} | Lookahead fix: normalize()-1day")
    print(f"  Eval: {' + '.join(eval_keys)}")
    print(f"{'='*120}\n")

    strategies_all = load_strategies_config()
    if not strategies_all:
        print("  No strategies found — aborting.")
        return

    baselines, strategies_filtered = precompute_baselines(strategies_all)
    if not strategies_filtered:
        print("  No strategies passed the baseline filter — aborting.")
        return

    indicator_cache = build_indicator_cache(baselines, strategies_filtered, windows, analysis_mode=ANALYSIS_MODE)

    label             = _combo_label(active_keys, windows, thresholds, mode)
    strategy_results: dict[str, dict] = {}

    for period_key in eval_keys:
        results: dict = {}

        for strategy in strategies_filtered:
            if DEBUG_TF_FILTER and strategy['timeframe'] not in DEBUG_TF_FILTER:
                continue

            sid    = strategy['id']
            result = _evaluate_period(strategy, period_key, indicator_cache, thresholds, mode)
            if not result:
                continue

            m_b, m_t, m_r = result['baseline'], result['trending'], result['ranging']
            trending_pct  = result.get('trending_pct', 0.0)

            if sid not in results:
                results[sid] = {'is_long': strategy['is_long']}
            results[sid][period_key] = {
            # NUEVO
            'b_prof': m_b['profit'], 'ranging_prof': m_t['profit'], 'trending_prof': m_r['profit'],
            'b_dd':   m_b['max_dd'], 'ranging_dd':   m_t['max_dd'], 'trending_dd':   m_r['max_dd'],
            'b_wr':   m_b['win_rate'], 'ranging_wr':  m_t['win_rate'], 'trending_wr':  m_r['win_rate'],
                'trending_pct': trending_pct,
                'ranging_pass_pct':  100 - trending_pct,
                'trending_pass_pct': trending_pct,
            }
            logger.debug(f"  [n_trades] {sid} {period_key}: baseline={m_b['n_trades']}")

            if sid not in strategy_results:
                strategy_results[sid] = {'is_long': strategy['is_long']}
            strategy_results[sid][period_key] = results[sid][period_key]

        _print_combo_period_table(results, strategies_filtered, period_key, label)

    for sid in strategy_results:
        strategy_results[sid]['classification'] = _classify_strategy(strategy_results, sid)

    _print_consistency_table(strategy_results)
    _print_classification_summary(strategy_results)
    _save_bins(strategy_results, windows, thresholds, mode)

    elapsed = int(time.time() - _t0)
    print(f"\n  ⏱  Completed in {elapsed//60}m {elapsed%60}s\n")


if __name__ == "__main__":
    run(EVAL_KEYS)