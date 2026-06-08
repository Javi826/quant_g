#shared/shared_batch_regime/regime_GE_core_4bins.py

import os
import logging
import numpy as np
import pandas as pd
from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_trading_batch_regime.regime_metrics import _CALC_FN
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

# 4 mutually exclusive market regime bins
BINS: list[str] = [
    "trending_highvol",
    "trending_lowvol",
    "ranging_highvol",
    "ranging_lowvol",
]

# =============================================================================
# HELPERS
# =============================================================================

def pct_improvement(val: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (val - base) / abs(base) * 100


def classify_market_regime(
    er_value:       float | None,
    atr_value:      float | None,
    er_threshold:   float,
    atr_threshold:  float,
) -> str:
    """
    Classify a single bar into one of 4 mutually exclusive regime bins.

    Direction (ER):   er  >= er_threshold  → trending, else ranging
    Volatility (ATR): atr >= atr_threshold → highvol,  else lowvol

    Returns one of: "trending_highvol" | "trending_lowvol" |
                    "ranging_highvol"  | "ranging_lowvol"
    Falls back to "ranging_lowvol" when any value is None/NaN.
    """
    if er_value is None or atr_value is None or np.isnan(er_value) or np.isnan(atr_value):
        return "ranging_lowvol"

    is_trending = er_value  >= er_threshold
    is_highvol  = atr_value >= atr_threshold

    if is_trending and is_highvol:
        return "trending_highvol"
    if is_trending and not is_highvol:
        return "trending_lowvol"
    if not is_trending and is_highvol:
        return "ranging_highvol"
    return "ranging_lowvol"


# =============================================================================
# COMBO LABEL
# =============================================================================

def combo_label(er_window: int, er_threshold: float, atr_window: int, atr_threshold: float) -> str:
    return (
        f"ER_W={er_window} | ER_TH={er_threshold:.3f} | "
        f"ATR_NORM_W={atr_window} | ATR_NORM_TH={atr_threshold:.3f}"
    )


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
# INDICATOR CACHE
# =============================================================================

def get_cache_key(sym: str, strategy: dict, analysis_mode: str, regime_timeframe_mode: str) -> str | tuple:
    ref_sym = "BTCUSDT" if analysis_mode == "BTC" else sym
    if regime_timeframe_mode == "STRATEGY":
        return (ref_sym, strategy['timeframe'])
    return ref_sym


def build_indicator_cache(
    baselines:             dict,
    strategies:            list[dict],
    er_window:             int,
    atr_window:            int,
    analysis_mode:         str = "SYMBOL",
    regime_timeframe_mode: str = "DAILY",
) -> dict:
    """
    Build indicator cache for ER and ATR_NORM with their respective windows.
    Returns {cache_key: (ts_arr, values_arr)} where values_arr has keys "er" and "atr_norm".
    """
    windows     = {"er": er_window, "atr_norm": atr_window}
    cache: dict = {}
    keys_needed: set = set()

    for strategy in strategies:
        for period_key in EVAL_KEYS:
            if period_key in baselines.get(strategy['id'], {}):
                for sym in baselines[strategy['id']][period_key]['ohlcv_arrays']:
                    keys_needed.add(get_cache_key(sym, strategy, analysis_mode, regime_timeframe_mode))

    for key in sorted(keys_needed, key=lambda x: x if isinstance(x, str) else f"{x[0]}_{x[1]}"):
        if key in cache:
            continue
        sym, tf = key if isinstance(key, tuple) else (key, REGIME_DEFAULT_TIMEFRAME)
        df = load_ohlcv_raw(sym, tf)
        if not df.empty:
            cache[key] = precompute_indicators(df, windows)

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

_METRIC_MAP: dict[str, dict[str, tuple[str, str]]] = {
    bin_name: {
        "profit":   (f"{bin_name}_prof", "b_prof"),
        "win_rate": (f"{bin_name}_wr",   "b_wr"),
        "calmar":   (f"{bin_name}_prof", "b_prof"),
        "mix":      (f"{bin_name}_prof", "b_prof"),
        "r2":       (f"{bin_name}_r2",   "b_r2"),
    }
    for bin_name in BINS
}

_DD_KEY_MAP: dict[str, str] = {bin_name: f"{bin_name}_dd" for bin_name in BINS}


def _calmar(prof: float, dd: float) -> float:
    return prof / abs(dd) if dd != 0 else 0.0


def _mix_score(prof: float, dd: float, w_profit: float = 0.1, w_dd: float = 0.9) -> float:
    calmar    = prof / abs(dd) if dd != 0 else 0.0
    calmar_sq = prof / (abs(dd) ** 2) if dd != 0 else 0.0
    return w_profit * calmar + w_dd * calmar_sq


def _metric_value(d: dict, val_key: str, dd_key: str, optimize_metric: str) -> float:
    if optimize_metric == "calmar":
        return _calmar(d[val_key], d[dd_key])
    if optimize_metric == "mix":
        return _mix_score(d[val_key], d[dd_key])
    return d[val_key]


def classify_strategy(
    results:             dict,
    sid:                 str,
    optimize_metric:     str = "profit",
    classification_mode: str = "strict",
    secondary_metric:    str | None = None,
) -> list[str]:
    """
    Classify a strategy into one or more bins where it beats the baseline.
    Returns a list of winning bin names, or ["neutral"] if none.
    """
    data              = results.get(sid, {})
    periods_with_data = [pk for pk in EVAL_KEYS if pk in data and isinstance(data[pk], dict)]
    if not periods_with_data:
        return ["neutral"]

    def _beats_baseline(pk: str, bin_name: str) -> bool:
        d       = data[pk]
        val_key = _METRIC_MAP[bin_name][optimize_metric][0]
        dd_key  = _DD_KEY_MAP[bin_name]
        b_key   = _METRIC_MAP[bin_name][optimize_metric][1]

        beats_primary = _metric_value(d, val_key, dd_key, optimize_metric) > _metric_value(d, b_key, "b_dd", optimize_metric)
        if beats_primary or secondary_metric is None:
            return beats_primary

        s_val_key = _METRIC_MAP[bin_name][secondary_metric][0]
        s_b_key   = _METRIC_MAP[bin_name][secondary_metric][1]
        return _metric_value(d, s_val_key, dd_key, secondary_metric) > _metric_value(d, s_b_key, "b_dd", secondary_metric)

    def _passes_bin(bin_name: str) -> bool:
        if classification_mode == "oos1_weighted":
            if "OOS1" not in periods_with_data or not _beats_baseline("OOS1", bin_name):
                return False
            oos23 = [pk for pk in periods_with_data if pk != "OOS1"]
            if not oos23:
                return True
            val_key = _METRIC_MAP[bin_name][optimize_metric][0]
            dd_key  = _DD_KEY_MAP[bin_name]
            b_key   = _METRIC_MAP[bin_name][optimize_metric][1]
            avg_bin = sum(_metric_value(data[pk], val_key, dd_key, optimize_metric) for pk in oos23) / len(oos23)
            avg_b   = sum(_metric_value(data[pk], b_key, "b_dd", optimize_metric) for pk in oos23) / len(oos23)
            return avg_bin > avg_b
        return all(_beats_baseline(pk, bin_name) for pk in periods_with_data)

    winning_bins = [b for b in BINS if _passes_bin(b)]
    return winning_bins if winning_bins else ["neutral"]


# =============================================================================
# COMBINED METRICS
# =============================================================================

def combined_metrics(results: dict) -> tuple[float, float]:
    """
    Aggregate profit and DD across strategies using their winning bins.
    Strategies with multiple bins use their best-profit bin per period.
    Neutral strategies fall back to baseline.
    """
    profits, dds = [], []
    for sid, data in results.items():
        if sid == 'is_long':
            continue
        bins = data.get('classification', ['neutral'])
        for pk in EVAL_KEYS:
            if pk not in data or not isinstance(data[pk], dict):
                continue
            d = data[pk]
            if bins == ["neutral"]:
                profits.append(d['b_prof'])
                dds.append(d['b_dd'])
            else:
                best_bin = max(bins, key=lambda b: d.get(f"{b}_prof", 0.0))
                profits.append(d[f"{best_bin}_prof"])
                dds.append(d[f"{best_bin}_dd"])
    return sum(profits), (sum(dds) / len(dds) if dds else 0.0)


# =============================================================================
# FILTERED BACKTEST FOR A SINGLE COMBO
# =============================================================================

def run_filtered_combo(
    baselines:       dict,
    strategies:      list[dict],
    indicator_cache: dict,
    er_threshold:    float,
    atr_threshold:   float,
    analysis_mode:   str,
    regime_timeframe_mode: str,
) -> dict:
    """
    For each strategy and period, route each signal into one of the 4 regime bins
    and run a backtest per bin. Returns results dict keyed by strategy id.
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

            bin_counts: dict[str, int]   = {b: 0 for b in BINS}
            bin_arrays: dict[str, dict]  = {b: {} for b in BINS}

            for sym, arr in cached['ohlcv_arrays'].items():
                signals     = cached['signal_cache'][sym]
                signal_idxs = np.nonzero(signals)[0]

                bin_signals: dict[str, np.ndarray] = {b: signals.copy() for b in BINS}
                for b in BINS:
                    bin_signals[b][:] = 0

                sym_cache = indicator_cache.get(get_cache_key(sym, strategy, analysis_mode, regime_timeframe_mode))

                if sym_cache is None or signal_idxs.size == 0:
                    bin_signals["ranging_lowvol"] = signals.copy()
                    bin_counts["ranging_lowvol"] += int(signals.sum())
                else:
                    ts_arr, values_arr = sym_cache
                    tf    = strategy['timeframe'] if regime_timeframe_mode == "STRATEGY" else None
                    batch = lookup_indicators_batch(ts_arr, values_arr, arr['ts'][signal_idxs], timeframe=tf)

                    for i, idx in enumerate(signal_idxs):
                        er_val  = float(batch["er"][i])  if not np.isnan(batch["er"][i])  else None
                        atr_val = float(batch["atr_norm"][i]) if not np.isnan(batch["atr_norm"][i]) else None
                        regime  = classify_market_regime(er_val, atr_val, er_threshold, atr_threshold)
                        bin_signals[regime][idx] = signals[idx]
                        bin_counts[regime] += 1

                for b in BINS:
                    bin_arrays[b][sym] = {**arr, 'signal': bin_signals[b]}

            bin_metrics: dict[str, dict] = {b: run_backtest(bin_arrays[b], strategy['best_params']) for b in BINS}
            total        = sum(bin_counts.values())
            trending_pct = (bin_counts["trending_highvol"] + bin_counts["trending_lowvol"]) / max(total, 1) * 100

            results[sid][period_key] = {
                'b_prof':  m_base['profit'],
                'b_dd':    m_base['max_dd'],
                'b_wr':    m_base['win_rate'],
                'b_r2':    m_base['r2'],
                'trending_pct': trending_pct,
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
    logger.debug(f"\n  {'─'*140}")
    logger.debug(f"  {label}  |  PERIOD: {period_key}")
    logger.debug(f"  {'─'*140}")
    logger.debug(
        f"  {'STRATEGY':<35} {'B_PROF':>8} "
        + "  ".join(f"{b.upper()[:12]:>13} {'Δ%':>6}" for b in BINS)
        + f"  {'TREND%':>7}"
    )
    logger.debug(f"  {'─'*140}")

    sys_b   = 0.0
    sys_bin = {b: 0.0 for b in BINS}
    dd_b    = []
    dd_bin  = {b: [] for b in BINS}
    trend_pcts = []

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
            bin_cols += f"  {d[f'{b}_prof']:>13.1f} {color}{delta:>+5.1f}%\033[0m"

        logger.debug(f"  {sid:<35} {d['b_prof']:>8.1f}{bin_cols}  {d['trending_pct']:>6.1f}%")

        sys_b += d['b_prof']
        dd_b.append(d['b_dd'])
        trend_pcts.append(d['trending_pct'])
        for b in BINS:
            sys_bin[b] += d[f"{b}_prof"]
            dd_bin[b].append(d[f"{b}_dd"])

    logger.debug(f"  {'─'*140}")
    sys_cols = ""
    for b in BINS:
        delta = pct_improvement(sys_bin[b], sys_b)
        color = "\033[92m" if delta > 0 else "\033[91m"
        sys_cols += f"  {sys_bin[b]:>13.1f} {color}{delta:>+5.1f}%\033[0m"
    avg_trend = sum(trend_pcts) / len(trend_pcts) if trend_pcts else 0.0
    logger.debug(f"  {'SYSTEM TOTAL':<35} {sys_b:>8.1f}{sys_cols}  {avg_trend:>6.1f}%")

    return {
        'sys_b':       sys_b,
        'avg_dd_b':    sum(dd_b) / len(dd_b) if dd_b else 0.0,
        'avg_trend_pct': avg_trend,
        **{f"sys_{b}":    sys_bin[b]                              for b in BINS},
        **{f"pct_{b}":    pct_improvement(sys_bin[b], sys_b)      for b in BINS},
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
    header = f"  {'PERIOD':<8} {'B_PROF':>10}" + "".join(f"  {b.upper()[:14]:>14} {'Δ%':>7}" for b in BINS) + f"  {'TREND%':>7}"
    logger.info(header)
    logger.info(f"  {'─'*120}")
    for pk, s in period_summaries.items():
        row = f"  {pk:<8} {s['sys_b']:>10.1f}"
        for b in BINS:
            color = "\033[92m" if s[f'pct_{b}'] > 0 else "\033[91m"
            row  += f"  {s[f'sys_{b}']:>14.1f} {color}{s[f'pct_{b}']:>+6.1f}%\033[0m"
        row += f"  {s['avg_trend_pct']:>6.1f}%"
        logger.info(row)
    logger.info(f"  {'─'*120}")
    comb_pct = pct_improvement(comb_p, base_p)
    color    = "\033[92m" if comb_pct > 0 else "\033[91m"
    cls_str  = "  ".join(f"{b.upper()}:{bin_counts[b]}" for b in BINS)
    logger.info(f"  Classifications — {cls_str}  NEUTRAL:{n_neutral}")
    logger.info(f"  Baseline  profit={base_p:>10.1f}  avg_dd={base_dd:>6.1f}%")
    logger.info(f"  Combined  profit={comb_p:>10.1f}  avg_dd={comb_dd:>6.1f}%  {color}Delta={comb_pct:>+6.1f}%\033[0m")


def print_ranking(ranking: list[dict]) -> None:
    bin_headers = "  ".join(f"{b.upper()[:8]:>8}" for b in BINS)
    header_line = (
        f"  {'#':>3}  {'COMBO':>5}  {'ER_W':>5}  {'ER_TH':>7}  {'ATR_W':>5}  {'ATR_TH':>7}  "
        f"{bin_headers}  {'NEUT':>5}  "
        f"{'BASELINE':>10} {'COMB_PROF':>10} {'COMB_Δ%':>8} {'W_DELTA%':>9}  "
        f"{'BASE_DD%':>8} {'COMB_DD%':>8}"
    )
    total_w = len(header_line) - 2
    logger.info(f"\n\n{'='*total_w}")
    logger.info(f"  FINAL RANKING — ALL COMBOS BY WEIGHTED DELTA VS BASELINE  [4-BIN MODE]")
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
            f"  {i:>3}  {row['combo_idx']:>5}  {row['er_window']:>5}  {row['er_threshold']:>7.3f}  "
            f"{row['atr_window']:>5}  {row['atr_threshold']:>7.3f}  "
            f"{bin_cols}  {row['n_neutral']:>5}  "
            f"{row['baseline_profit']:>10.1f} {cc}{row['combined_profit']:>10.1f}{rs} "
            f"{cc}{pct:>+7.1f}%{rs} {wc}{w_delta:>+8.1f}%{rs}  "
            f"{row['baseline_dd']:>7.1f}% {ddc}{row['combined_dd']:>7.1f}%{rs}"
        )
    logger.info(f"  {'─'*total_w}\n")


def print_classification_summary(strategy_results: dict) -> None:
    print(f"\n{'='*120}")
    print(f"  STRATEGY CLASSIFICATION SUMMARY  [4-BIN MODE]")
    print(f"{'='*120}")
    print(f"  {'STRATEGY':<35} {'DIR':<6} {'BINS'}")
    print(f"  {'─'*80}")
    bin_colors = {
        "trending_highvol": "\033[94m",
        "trending_lowvol":  "\033[96m",
        "ranging_highvol":  "\033[93m",
        "ranging_lowvol":   "\033[92m",
        "neutral":          "\033[90m",
    }
    for sid, data in sorted(strategy_results.items()):
        direction = "LONG" if data.get('is_long') else "SHORT"
        bins      = data.get('classification', ['neutral'])
        color     = bin_colors.get(bins[0], "")
        bins_str  = ", ".join(b.upper() for b in bins)
        print(f"  {sid:<35} {direction:<6} {color}{bins_str}\033[0m")
    print(f"  {'─'*80}\n")


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_bins(
    strategy_results:      dict,
    er_window:             int,
    er_threshold:          float,
    atr_window:            int,
    atr_threshold:         float,
    output_path:           str,
    strategies_set_name:   str = "E1",
    all_strategies:        list[dict] | None = None,
    optimize_metric:       str = "",
    classification_mode:   str = "",
    secondary_metric:      str | None = None,
    analysis_mode:         str = "SYMBOL",
    regime_timeframe_mode: str = "DAILY",
) -> None:
    from datetime import datetime
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    header_lines = [
        '"""',
        f"regime_bins_{strategies_set_name}.py — 4-bin regime classification. Do not edit manually.",
        f"Generated by regime_GE_calibration_4bins.py",
        f"ER({er_window})>={er_threshold} | ATR_NORM({atr_window})>={atr_threshold}",
        f"Auto-generated on {generated_at} UTC.",
        '"""',
        "",
        f'ER_WINDOW             = {er_window}',
        f'ER_THRESHOLD          = {er_threshold}',
        f'ATR_NORM_WINDOW       = {atr_window}',
        f'ATR_NORM_THRESHOLD    = {atr_threshold}',
        f'ANALYSIS_MODE         = "{analysis_mode}"',
        f'REGIME_TIMEFRAME_MODE = "{regime_timeframe_mode}"',
        "",
    ]
    if optimize_metric:
        header_lines.append(f'OPTIMIZE_METRIC     = "{optimize_metric}"')
    if classification_mode:
        header_lines.append(f'CLASSIFICATION_MODE = "{classification_mode}"')
    if secondary_metric:
        header_lines.append(f'CLASSIFY_SECONDARY_METRIC = "{secondary_metric}"')
    header_lines += ["", "REGIME_BINS = {"]

    all_ids = {s['id'] for s in all_strategies} if all_strategies else set()
    missing = all_ids - set(strategy_results.keys())

    all_entries: dict[str, list[str]] = {
        sid: data.get('classification', ['neutral'])
        for sid, data in strategy_results.items()
    }
    for sid in missing:
        all_entries[sid] = ["neutral"]

    bin_lines = [
        f'    "{sid}": {bins},{"  # excluded from calibration" if sid in missing else ""}'
        for sid, bins in sorted(all_entries.items())
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines + bin_lines + ["}"]) + "\n")
    print(f"\n  ✅ Bins saved to: {output_path}")


# =============================================================================
# LOAD REGIME BINS
# =============================================================================

def load_regime_bins_ge(bins_path: str, strategy_id: str) -> list[str]:
    """
    Load the 4-bin regime classification for a strategy.
    Returns list of bins e.g. ["trending_highvol"] or ["neutral"].
    """
    if not os.path.exists(bins_path):
        logger.warning(f"regime_bins file not found: {bins_path} — defaulting to neutral.")
        return ["neutral"]
    spec   = spec_from_file_location("regime_bins_ge", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    bins = getattr(module, "REGIME_BINS", {})
    return bins.get(strategy_id, ["neutral"])


# =============================================================================
# TIME-SERIES PRECOMPUTATION
# =============================================================================

def precompute_indicators(
    df:      pd.DataFrame,
    windows: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute rolling ER and ATR_NORM arrays for a full OHLCV DataFrame.
    Only rows where both indicators are valid (non-NaN) are kept.
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    ts    = df["ts"].values

    min_win = max(w + 1 for w in windows.values()) if windows else 1

    ts_list:     list            = []
    value_lists: dict[str, list] = {k: [] for k in windows}

    for i in range(min_win, len(close)):
        row_values: dict[str, float] = {}
        valid = True
        for key, w in windows.items():
            val = _CALC_FN[key](high[:i + 1], low[:i + 1], close[:i + 1], w)
            if np.isnan(val):
                valid = False
                break
            row_values[key] = val
        if not valid:
            continue
        ts_list.append(ts[i])
        for key in windows:
            value_lists[key].append(row_values[key])

    ts_arr     = np.array(ts_list, dtype="datetime64[ns]")
    values_arr = {k: np.array(v, dtype=float) for k, v in value_lists.items()}
    return ts_arr, values_arr


# =============================================================================
# LOOKUP
# =============================================================================

def lookup_indicators_batch(
    ts_arr:        np.ndarray,
    values_arr:    dict[str, np.ndarray],
    signal_ts_arr: np.ndarray,
    timeframe:     str | None = None,
) -> dict[str, np.ndarray]:
    """
    Vectorized lookup of ER and ATR_NORM for multiple signal timestamps.
    Returns {indicator_key: np.ndarray} — NaN where no valid index found.
    """
    ts = signal_ts_arr.astype("datetime64[ns]")

    if timeframe is None or timeframe in ("1Dutc", "1D"):
        ts   = (ts.astype("datetime64[D]") - np.timedelta64(1, "D")).astype("datetime64[ns]")
        idxs = np.searchsorted(ts_arr, ts, side="right") - 1
    else:
        idxs = np.searchsorted(ts_arr, ts, side="left") - 1

    valid  = idxs >= 0
    result = {}
    for k, arr in values_arr.items():
        out        = np.full(len(idxs), np.nan)
        out[valid] = arr[idxs[valid]]
        result[k]  = out

    return result