#shared/shared_batch_regime/regime_GE_core_uptrend.py

import os
import logging
import numpy as np
import pandas as pd
from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from importlib.util import spec_from_file_location, module_from_spec
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

from shared_batch_regime.config_paths import BITGET_ROOT, DATA_FOLDER_IS, DATA_FOLDER_OOS1, DATA_FOLDER_OOS2, DATA_FOLDER_OOS3, CRYPTO_FULL_DIR

PERIODS = {
    "IS":   DATA_FOLDER_IS,
    "OOS1": DATA_FOLDER_OOS1,
    "OOS2": DATA_FOLDER_OOS2,
    "OOS3": DATA_FOLDER_OOS3,
}
EVAL_KEYS = ["OOS2", "OOS3", "OOS1"]

# =============================================================================
# CONSTANTS
# =============================================================================

REGIME_DEFAULT_TIMEFRAME       = "1Dutc"
LONG_KEYWORD                   = "long"
ORDER_AMOUNT                   = 80
DEBUG_TF_FILTER: list[str]     = []
FILTER_NEGATIVE_BASELINE: bool = True

BINS: list[str] = ["uptrend", "downtrend"]

# =============================================================================
# HELPERS
# =============================================================================

def pct_improvement(val: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (val - base) / abs(base) * 100


def compute_ma(close: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average over close prices. Returns NaN for first window-1 values."""
    result = np.full(len(close), np.nan)
    for i in range(window - 1, len(close)):
        result[i] = close[i - window + 1: i + 1].mean()
    return result


def classify_market_regime(close: float, ma: float) -> str:
    """
    Classify a single bar into uptrend or downtrend based on close vs MA.
    Returns "uptrend" if close > MA, else "downtrend".
    Falls back to "downtrend" if any value is NaN/None.
    """
    if close is None or ma is None or np.isnan(close) or np.isnan(ma):
        return "downtrend"
    return "uptrend" if close > ma else "downtrend"


# =============================================================================
# COMBO LABEL
# =============================================================================

def combo_label(ma_window: int) -> str:
    return f"MA_W={ma_window}"


# =============================================================================
# CONFIG LOADERS
# =============================================================================

def load_strategies_config(strategies_set_name: str) -> list[dict]:
    loop_name = f"strategies_loop_{strategies_set_name}_01"
    loop_path = os.path.join(BITGET_ROOT, f"BOT_batch_{strategies_set_name}", "strategies_files", f"{loop_name}.py")
    spec      = spec_from_file_location(loop_name, loop_path)
    module    = module_from_spec(spec)
    spec.loader.exec_module(module)

    strategies = []
    for entry in module.STRATEGIES_LOOP:
        strategy_id = entry["id"]
        signal_key  = "_".join(strategy_id.split("_")[1:-1])
        if signal_key not in SIGNAL_REGISTRY:
            signal_key = "_".join(strategy_id.split("_")[:-1])
        if signal_key not in SIGNAL_REGISTRY:
            continue

        registry      = SIGNAL_REGISTRY[signal_key]
        param_grid    = entry["param_grid"]
        best_params   = {k.upper(): v[0] for k, v in param_grid.items()}
        signal_params = {k: best_params[k.upper()] for k in registry["params"] if k.upper() in best_params}

        strategies.append({
            "id":            strategy_id,
            "timeframe":     strategy_id.split("_")[-1],
            "signal_fn":     registry["fn"],
            "signal_params": signal_params,
            "best_params":   best_params,
            "is_long":       LONG_KEYWORD in strategy_id,
            "n_symbols":     entry.get("n_symbols", 10),
        })
    return strategies


def load_symbols(strategy_id: str, timeframe: str, strategies_set_name: str) -> list[str]:
    symbols_folder = os.path.join(BITGET_ROOT, "BOT_trading", "symbols_live", strategies_set_name)
    filepath       = os.path.join(symbols_folder, f"symbols_live_{strategy_id}_{timeframe}.csv")
    if not os.path.exists(filepath):
        return []
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


# =============================================================================
# OHLCV LOADERS
# =============================================================================

def load_ohlcv_for_period(strategy: dict, period_key: str, strategies_set_name: str) -> dict:
    if period_key == "OOS1":
        symbols = load_symbols(strategy['id'], strategy['timeframe'], strategies_set_name)
        if not symbols:
            return {}
        ohlcv_data, _ = filter_symbols(
            symbols, min_vol_usdt=0, timeframe=strategy['timeframe'],
            data_folder=PERIODS[period_key], min_price=None, vol_window=50,
            my_symbols=True, custom_symbols=symbols,
        )
        return ohlcv_data

    _, symbols_oos_final, _, ohlcv_oos = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        data_folder_oos   = PERIODS[period_key],
        timeframe         = strategy['timeframe'],
        n_symbols         = strategy['n_symbols'],
        min_price         = None,
        filter_symbols_fn = filter_symbols,
        my_symbols        = False,
    )
    ohlcv_oos = {sym: ohlcv_oos[sym] for sym in symbols_oos_final if sym in ohlcv_oos}
    logger.debug(f"[symbols] {strategy['id']} {period_key}: {sorted(ohlcv_oos.keys())}")
    return ohlcv_oos


def load_ohlcv_raw(symbol: str, timeframe: str) -> pd.DataFrame:
    path = os.path.join(CRYPTO_FULL_DIR, f"{symbol}_{timeframe}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if df.index.name and df.index.name.lower() in ("timestamp", "ts", "date", "time"):
        df.index.name = "ts"
        df = df.reset_index()
    if "volume_base" in df.columns and "volume_quote" in df.columns:
        df.drop(columns=["volume_base"], inplace=True)
    rename_map = {
        "timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts",
        "volume_quote": "volume", "vol": "volume",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["ts"] = df["ts"].dt.tz_localize("UTC") if df["ts"].dt.tz is None else df["ts"].dt.tz_convert("UTC")
    df.dropna(subset=["ts"], inplace=True)
    df.sort_values("ts", inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df


# =============================================================================
# INDICATOR CACHE  (MA over daily close)
# =============================================================================

def get_cache_key(sym: str, strategy: dict, analysis_mode: str) -> str:
    """Cache key is always daily — MA is always computed on 1Dutc."""
    return "BTCUSDT" if analysis_mode == "BTC" else sym


def build_indicator_cache(
    baselines:     dict,
    strategies:    list[dict],
    ma_window:     int,
    analysis_mode: str = "SYMBOL",
) -> dict:
    """
    Build MA indicator cache keyed by symbol (always daily timeframe).
    Returns {sym: (ts_arr, ma_arr)} where ma_arr is the MA(ma_window) series.
    """
    cache: dict      = {}
    keys_needed: set = set()

    for strategy in strategies:
        for period_key in EVAL_KEYS:
            if period_key in baselines.get(strategy['id'], {}):
                for sym in baselines[strategy['id']][period_key]['ohlcv_arrays']:
                    keys_needed.add(get_cache_key(sym, strategy, analysis_mode))

    for key in sorted(keys_needed):
        if key in cache:
            continue
        df = load_ohlcv_raw(key, REGIME_DEFAULT_TIMEFRAME)
        if not df.empty:
            cache[key] = precompute_indicators(df, ma_window)

    return cache


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(ohlcv_arrays: dict, best_params: dict) -> dict:
    result = run_grid_backtest(
        ohlcv_arrays,
        sell_after=best_params['SELL_AFTER'], tp_pct=best_params['TP_PCT'],
        sl_pct=best_params['SL_PCT'], order_amount=ORDER_AMOUNT,
    )
    trades = result['__PORTFOLIO__']['trade_log']
    if len(trades) == 0:
        return {'profit': 0.0, 'win_rate': 0.0, 'n_trades': 0, 'max_dd': 0.0, 'r2': 0.0}
    profits = trades['profit']
    equity  = INITIAL_BALANCE + profits.cumsum()
    eq_arr  = equity.values.reshape(-1, 1)
    x_arr   = np.arange(len(eq_arr)).reshape(-1, 1)
    r2      = float(round(LinearRegression().fit(x_arr, eq_arr).score(x_arr, eq_arr), 3))
    return {
        'profit':   float(profits.sum()),
        'win_rate': float((profits > 0).mean() * 100),
        'n_trades': len(profits),
        'max_dd':   float(((equity - equity.cummax()) / equity.cummax()).min() * 100),
        'r2':       r2,
    }


# =============================================================================
# BASELINE PRECOMPUTATION
# =============================================================================

def precompute_baselines(strategies_all: list[dict], strategies_set_name: str) -> tuple[dict, list[dict]]:
    label = "excluding strategies with B_PROF <= 0 in any period" if FILTER_NEGATIVE_BASELINE else "including all strategies"
    print(f"\n{'='*120}")
    print(f"  PRECOMPUTING BASELINES — {label}")
    print(f"{'='*120}")
    baselines: dict[str, dict] = {}
    for strategy in strategies_all:
        if DEBUG_TF_FILTER and strategy['timeframe'] not in DEBUG_TF_FILTER:
            continue
        sid            = strategy['id']
        baselines[sid] = {}
        for period_key in EVAL_KEYS:
            ohlcv_data = load_ohlcv_for_period(strategy, period_key, strategies_set_name)
            if not ohlcv_data:
                continue
            ohlcv_arrays    = prepare_ohlcv_arrays(ohlcv_data)
            signal_cache    = {}
            baseline_arrays = {}
            for sym, arr in ohlcv_arrays.items():
                signals              = strategy['signal_fn'](arr, **strategy['signal_params'], live_trading=False)
                signal_cache[sym]    = signals
                baseline_arrays[sym] = {**arr, 'signal': signals}
            baselines[sid][period_key] = {
                'metrics':      run_backtest(baseline_arrays, strategy['best_params']),
                'signal_cache': signal_cache,
                'ohlcv_arrays': ohlcv_arrays,
            }
        all_positive = all(
            baselines[sid].get(pk, {}).get('metrics', {}).get('profit', 0.0) > 0
            for pk in EVAL_KEYS
        )
        if not FILTER_NEGATIVE_BASELINE or all_positive:
            print(f"  ✓ {sid}")
        else:
            del baselines[sid]
            print(f"  ✗ {sid}  (excluded)")
    strategies_filtered = [s for s in strategies_all if s['id'] in baselines]
    print(f"\n  {len(strategies_filtered)} kept | {len(strategies_all) - len(strategies_filtered)} excluded\n")
    return baselines, strategies_filtered


# =============================================================================
# CLASSIFICATION
# =============================================================================

_METRIC_MAP: dict[str, dict[str, str]] = {
    bin_name: {
        "profit":   f"{bin_name}_prof",
        "win_rate": f"{bin_name}_wr",
        "calmar":   f"{bin_name}_prof",
        "r2":       f"{bin_name}_r2",
    }
    for bin_name in BINS
}

_DD_KEY_MAP: dict[str, str] = {bin_name: f"{bin_name}_dd" for bin_name in BINS}


def _calmar(prof: float, dd: float) -> float:
    return prof / abs(dd) if dd != 0 else 0.0


def _metric_value(d: dict, val_key: str, dd_key: str, optimize_metric: str) -> float:
    if optimize_metric == "calmar":
        return _calmar(d[val_key], d[dd_key])
    return d[val_key]


def classify_strategy(
    results:         dict,
    sid:             str,
    optimize_metric: str = "profit",
) -> str:
    """
    Classify a strategy into exactly one bin (strict mode):
      - uptrend   : beats baseline in ALL periods in uptrend bucket
      - downtrend : beats baseline in ALL periods in downtrend bucket
      - both pass : neutral (no discriminative power)
      - none pass : neutral
    """
    data              = results.get(sid, {})
    periods_with_data = [pk for pk in EVAL_KEYS if pk in data and isinstance(data[pk], dict)]
    if not periods_with_data:
        return "neutral"

    def _beats_baseline(pk: str, bin_name: str) -> bool:
        d       = data[pk]
        val_key = _METRIC_MAP[bin_name][optimize_metric]
        dd_key  = _DD_KEY_MAP[bin_name]
        return _metric_value(d, val_key, dd_key, optimize_metric) > _metric_value(d, "b_prof", "b_dd", optimize_metric)

    up_passes   = all(_beats_baseline(pk, "uptrend")   for pk in periods_with_data)
    down_passes = all(_beats_baseline(pk, "downtrend") for pk in periods_with_data)

    if up_passes and down_passes:
        return "neutral"
    if up_passes:
        return "uptrend"
    if down_passes:
        return "downtrend"
    return "neutral"


# =============================================================================
# COMBINED METRICS
# =============================================================================

def combined_metrics(results: dict) -> tuple[float, float]:
    profits, dds = [], []
    for sid, data in results.items():
        if sid == 'is_long':
            continue
        cls = data.get('classification', 'neutral')
        for pk in EVAL_KEYS:
            if pk not in data or not isinstance(data[pk], dict):
                continue
            d = data[pk]
            if cls == 'uptrend':
                profits.append(d['uptrend_prof'])
                dds.append(d['uptrend_dd'])
            elif cls == 'downtrend':
                profits.append(d['downtrend_prof'])
                dds.append(d['downtrend_dd'])
            else:
                profits.append(d['b_prof'])
                dds.append(d['b_dd'])
    return sum(profits), (sum(dds) / len(dds) if dds else 0.0)


# =============================================================================
# FILTERED BACKTEST FOR A SINGLE COMBO
# =============================================================================

def run_filtered_combo(
    baselines:       dict,
    strategies:      list[dict],
    indicator_cache: dict,
    ma_window:       int,
    analysis_mode:   str,
    debug_n:         int = 0,
) -> dict:
    """
    For each strategy and period, route each signal into uptrend or downtrend
    based on close vs MA(ma_window) on the daily timeframe.
    debug_n: if > 0, prints raw candle debug for first debug_n signals of first symbol.
    """
    results: dict = {}

    for strategy in strategies:
        sid = strategy['id']
        if sid not in baselines:
            continue

        results[sid] = {'is_long': strategy['is_long']}

        for period_key in EVAL_KEYS:
            if period_key not in baselines[sid]:
                continue

            cached = baselines[sid][period_key]
            m_base = cached['metrics']

            bin_counts: dict[str, int]  = {b: 0 for b in BINS}
            bin_arrays: dict[str, dict] = {b: {} for b in BINS}

            for sym, arr in cached['ohlcv_arrays'].items():
                signals     = cached['signal_cache'][sym]
                signal_idxs = np.nonzero(signals)[0]

                bin_signals: dict[str, np.ndarray] = {b: np.zeros_like(signals) for b in BINS}

                cache_key = get_cache_key(sym, strategy, analysis_mode)
                sym_cache = indicator_cache.get(cache_key)

                if sym_cache is None or signal_idxs.size == 0:
                    bin_signals["downtrend"] = signals.copy()
                    bin_counts["downtrend"] += int(signals.sum())
                else:
                    ts_arr, ma_arr  = sym_cache
                    signal_ts       = arr['ts'][signal_idxs]
                    _close_arr      = arr['close'] if 'close' in arr else None
                    _debug_this_sym = debug_n if (debug_n > 0 and not any(bin_counts.values())) else 0
                    lookups         = lookup_ma_batch(ts_arr, ma_arr, signal_ts, close_arr=_close_arr, debug_n=_debug_this_sym)

                    for i, idx in enumerate(signal_idxs):
                        close_val = float(arr['close'][idx]) if 'close' in arr else None
                        ma_val    = float(lookups[i]) if not np.isnan(lookups[i]) else None
                        regime    = classify_market_regime(close_val, ma_val)
                        bin_signals[regime][idx] = signals[idx]
                        bin_counts[regime] += 1

                for b in BINS:
                    bin_arrays[b][sym] = {**arr, 'signal': bin_signals[b]}

            bin_metrics: dict[str, dict] = {b: run_backtest(bin_arrays[b], strategy['best_params']) for b in BINS}
            total        = sum(bin_counts.values())
            uptrend_pct  = bin_counts["uptrend"] / max(total, 1) * 100

            results[sid][period_key] = {
                'b_prof': m_base['profit'],
                'b_dd':   m_base['max_dd'],
                'b_wr':   m_base['win_rate'],
                'b_r2':   m_base['r2'],
                'uptrend_pct': uptrend_pct,
                **{f"{b}_prof": bin_metrics[b]['profit']   for b in BINS},
                **{f"{b}_dd":   bin_metrics[b]['max_dd']   for b in BINS},
                **{f"{b}_wr":   bin_metrics[b]['win_rate'] for b in BINS},
                **{f"{b}_r2":   bin_metrics[b]['r2']       for b in BINS},
                **{f"{b}_pct":  bin_counts[b] / max(total, 1) * 100 for b in BINS},
            }

    return results


# =============================================================================
# PRINT TABLES
# =============================================================================

def print_combo_period_table(results: dict, strategies: list[dict], period_key: str, label: str) -> dict:
    logger.debug(f"\n  {'─'*120}")
    logger.debug(f"  {label}  |  PERIOD: {period_key}")
    logger.debug(f"  {'─'*120}")
    logger.debug(
        f"  {'STRATEGY':<35} {'B_PROF':>8}"
        + "  ".join(f"  {b.upper()[:10]:>12} {'Δ%':>6}" for b in BINS)
        + f"  {'UP%':>7}"
    )
    logger.debug(f"  {'─'*120}")

    sys_b   = 0.0
    sys_bin = {b: 0.0 for b in BINS}
    dd_b    = []
    dd_bin  = {b: [] for b in BINS}
    up_pcts = []

    for s in strategies:
        sid = s['id']
        if sid not in results or period_key not in results[sid]:
            continue
        if not isinstance(results[sid][period_key], dict):
            continue
        d = results[sid][period_key]

        bin_cols = ""
        for b in BINS:
            delta = pct_improvement(d[f"{b}_prof"], d['b_prof'])
            color = "\033[92m" if delta > 0 else "\033[91m"
            bin_cols += f"  {d[f'{b}_prof']:>12.1f} {color}{delta:>+5.1f}%\033[0m"

        logger.debug(f"  {sid:<35} {d['b_prof']:>8.1f}{bin_cols}  {d['uptrend_pct']:>6.1f}%")

        sys_b += d['b_prof']
        dd_b.append(d['b_dd'])
        up_pcts.append(d['uptrend_pct'])
        for b in BINS:
            sys_bin[b] += d[f"{b}_prof"]
            dd_bin[b].append(d[f"{b}_dd"])

    logger.debug(f"  {'─'*120}")
    sys_cols = ""
    avg_up   = sum(up_pcts) / len(up_pcts) if up_pcts else 0.0
    for b in BINS:
        delta = pct_improvement(sys_bin[b], sys_b)
        color = "\033[92m" if delta > 0 else "\033[91m"
        sys_cols += f"  {sys_bin[b]:>12.1f} {color}{delta:>+5.1f}%\033[0m"
    logger.debug(f"  {'SYSTEM TOTAL':<35} {sys_b:>8.1f}{sys_cols}  {avg_up:>6.1f}%")

    return {
        'sys_b':       sys_b,
        'avg_dd_b':    sum(dd_b) / len(dd_b) if dd_b else 0.0,
        'avg_up_pct':  avg_up,
        **{f"sys_{b}":    sys_bin[b]                             for b in BINS},
        **{f"pct_{b}":    pct_improvement(sys_bin[b], sys_b)     for b in BINS},
        **{f"avg_dd_{b}": sum(dd_bin[b]) / len(dd_bin[b]) if dd_bin[b] else 0.0 for b in BINS},
    }


def print_combo_summary(
    period_summaries: dict,
    bin_counts:       dict[str, int],
    n_neutral:        int,
    comb_p:           float,
    comb_dd:          float,
    base_p:           float,
    base_dd:          float,
    label:            str,
) -> None:
    logger.info(f"\n  COMBO SUMMARY — {label}")
    header = f"  {'PERIOD':<8} {'B_PROF':>10}" + "".join(f"  {b.upper():>12} {'Δ%':>7}" for b in BINS) + f"  {'UP%':>7}"
    logger.info(header)
    logger.info(f"  {'─'*90}")
    for pk, s in period_summaries.items():
        row = f"  {pk:<8} {s['sys_b']:>10.1f}"
        for b in BINS:
            color = "\033[92m" if s[f'pct_{b}'] > 0 else "\033[91m"
            row  += f"  {s[f'sys_{b}']:>12.1f} {color}{s[f'pct_{b}']:>+6.1f}%\033[0m"
        row += f"  {s['avg_up_pct']:>6.1f}%"
        logger.info(row)
    logger.info(f"  {'─'*90}")
    comb_pct = pct_improvement(comb_p, base_p)
    color    = "\033[92m" if comb_pct > 0 else "\033[91m"
    cls_str  = "  ".join(f"{b.upper()}:{bin_counts.get(b, 0)}" for b in BINS)
    logger.info(f"  Classifications — {cls_str}  NEUTRAL:{n_neutral}")
    logger.info(f"  Baseline  profit={base_p:>10.1f}  avg_dd={base_dd:>6.1f}%")
    logger.info(f"  Combined  profit={comb_p:>10.1f}  avg_dd={comb_dd:>6.1f}%  {color}Delta={comb_pct:>+6.1f}%\033[0m")


def print_ranking(ranking: list[dict]) -> None:
    bin_headers = "  ".join(f"{b.upper()[:8]:>8}" for b in BINS)
    header_line = (
        f"  {'#':>3}  {'COMBO':>5}  {'MA_W':>5}  "
        f"{bin_headers}  {'NEUT':>5}  "
        f"{'BASELINE':>10} {'COMB_PROF':>10} {'COMB_Δ%':>8} {'W_DELTA%':>9}  "
        f"{'BASE_DD%':>8} {'COMB_DD%':>8}"
    )
    total_w = len(header_line) - 2
    logger.info(f"\n\n{'='*total_w}")
    logger.info(f"  FINAL RANKING — ALL COMBOS BY WEIGHTED DELTA VS BASELINE  [MA UPTREND MODE]")
    logger.info(f"{'='*total_w}")
    logger.info(header_line)
    logger.info(f"  {'─'*total_w}")
    for i, row in enumerate(ranking[:5], 1):
        pct     = pct_improvement(row['combined_profit'], row['baseline_profit'])
        w_delta = row.get('weighted_delta', 0.0)
        cc      = "\033[92m" if pct > 0 else "\033[91m"
        wc      = "\033[92m" if w_delta > 0 else "\033[91m"
        ddc     = "\033[92m" if row['combined_dd'] > row['baseline_dd'] else "\033[91m"
        rs      = "\033[0m"
        bin_cols = "  ".join(f"{row['bin_counts'].get(b, 0):>8}" for b in BINS)
        logger.info(
            f"  {i:>3}  {row['combo_idx']:>5}  {row['ma_window']:>5}  "
            f"{bin_cols}  {row['n_neutral']:>5}  "
            f"{row['baseline_profit']:>10.1f} {cc}{row['combined_profit']:>10.1f}{rs} "
            f"{cc}{pct:>+7.1f}%{rs} {wc}{w_delta:>+8.1f}%{rs}  "
            f"{row['baseline_dd']:>7.1f}% {ddc}{row['combined_dd']:>7.1f}%{rs}"
        )
    logger.info(f"  {'─'*total_w}\n")


def print_classification_summary(strategy_results: dict) -> None:
    print(f"\n{'='*120}")
    print(f"  STRATEGY CLASSIFICATION SUMMARY  [MA UPTREND MODE]")
    print(f"{'='*120}")
    print(f"  {'STRATEGY':<35} {'DIR':<6} {'BIN'}")
    print(f"  {'─'*70}")
    bin_colors = {
        "uptrend":   "\033[92m",
        "downtrend": "\033[91m",
        "neutral":   "\033[90m",
    }
    for sid, data in sorted(strategy_results.items()):
        direction = "LONG" if data.get('is_long') else "SHORT"
        cls       = data.get('classification', 'neutral')
        color     = bin_colors.get(cls, "")
        print(f"  {sid:<35} {direction:<6} {color}{cls.upper()}\033[0m")
    print(f"  {'─'*70}\n")


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_bins(
    strategy_results:    dict,
    ma_window:           int,
    output_path:         str,
    strategies_set_name: str = "E1",
    all_strategies:      list[dict] | None = None,
    optimize_metric:     str = "",
    analysis_mode:       str = "SYMBOL",
) -> None:
    from datetime import datetime
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    header_lines = [
        '"""',
        f"regime_bins_{strategies_set_name}.py — MA uptrend regime classification. Do not edit manually.",
        f"Generated by regime_GE_calibration_uptrend.py — MA({ma_window}) on {REGIME_DEFAULT_TIMEFRAME}",
        f"Auto-generated on {generated_at} UTC.",
        '"""',
        "",
        f'MA_WINDOW             = {ma_window}',
        f'MA_TIMEFRAME          = "{REGIME_DEFAULT_TIMEFRAME}"',
        f'ANALYSIS_MODE         = "{analysis_mode}"',
        "",
    ]
    if optimize_metric:
        header_lines.append(f'OPTIMIZE_METRIC = "{optimize_metric}"')
    header_lines += ["", "REGIME_BINS = {"]

    all_ids = {s['id'] for s in all_strategies} if all_strategies else set()
    missing = all_ids - set(strategy_results.keys())

    all_entries: dict[str, str] = {
        sid: data.get('classification', 'neutral')
        for sid, data in strategy_results.items()
    }
    for sid in missing:
        all_entries[sid] = "neutral"

    bin_lines = [
        f'    "{sid}": "{cls}",{"  # excluded from calibration" if sid in missing else ""}'
        for sid, cls in sorted(all_entries.items())
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines + bin_lines + ["}"]) + "\n")
    print(f"\n  ✅ Bins saved to: {output_path}")


# =============================================================================
# LOAD REGIME BINS
# =============================================================================

def load_regime_bins_ge(bins_path: str, strategy_id: str) -> str:
    if not os.path.exists(bins_path):
        logger.warning(f"regime_bins file not found: {bins_path} — defaulting to neutral.")
        return "neutral"
    spec   = spec_from_file_location("regime_bins_ge", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    bins = getattr(module, "REGIME_BINS", {})
    return bins.get(strategy_id, "neutral")


# =============================================================================
# TIME-SERIES PRECOMPUTATION
# =============================================================================

def precompute_indicators(df: pd.DataFrame, ma_window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MA(ma_window) over daily close prices.
    Returns (ts_arr, ma_arr) — only rows where MA is valid (non-NaN).
    """
    close = df["close"].values
    ts    = df["ts"].values
    ma    = compute_ma(close, ma_window)

    valid  = ~np.isnan(ma)
    ts_arr = np.array(ts[valid], dtype="datetime64[ns]")
    ma_arr = ma[valid]
    return ts_arr, ma_arr


# =============================================================================
# LOOKUP
# =============================================================================

def lookup_ma_batch(
    ts_arr:        np.ndarray,
    ma_arr:        np.ndarray,
    signal_ts_arr: np.ndarray,
    close_arr:     np.ndarray | None = None,
    debug_n:       int = 0,
) -> np.ndarray:
    """
    Vectorized lookup of MA value for multiple signal timestamps.
    Always applies normalize()-1day lookahead fix (daily timeframe).
    Returns np.ndarray of MA values — NaN where no valid index found.

    debug_n   : if > 0, prints the first debug_n signals with raw candle info.
    close_arr : optional close array aligned with ts_arr, used for debug only.
    """
    ts_lookup = signal_ts_arr.astype("datetime64[ns]")
    ts_fixed  = (ts_lookup.astype("datetime64[D]") - np.timedelta64(1, "D")).astype("datetime64[ns]")
    idxs      = np.searchsorted(ts_arr, ts_fixed, side="right") - 1

    valid = idxs >= 0
    out   = np.full(len(idxs), np.nan)
    out[valid] = ma_arr[idxs[valid]]

    if debug_n > 0:
        print(f"\n  [LOOKUP DEBUG] showing first {min(debug_n, len(idxs))} signals")
        print(f"  {'SIGNAL_TS':<28} {'LOOKUP_TS (D-1)':<28} {'CANDLE_TS':<28} {'CLOSE':>10} {'MA':>10}")
        print(f"  {'─'*105}")
        for i in range(min(debug_n, len(idxs))):
            sig_ts    = str(signal_ts_arr[i])
            fixed_ts  = str(ts_fixed[i])
            idx       = idxs[i]
            candle_ts = str(ts_arr[idx]) if idx >= 0 else "N/A"
            ma_val    = f"{ma_arr[idx]:.6f}" if idx >= 0 else "N/A"
            close_val = f"{close_arr[idx]:.6f}" if (close_arr is not None and idx >= 0) else "N/A"
            print(f"  {sig_ts:<28} {fixed_ts:<28} {candle_ts:<28} {close_val:>10} {ma_val:>10}")

    return out