#shared/shared_batch_regime/regime_GE_core.py

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
# STRATEGY CONFIG PATHS  (set per batch)
# =============================================================================

BTC_TIMEFRAME    = "1Dutc"
LONG_KEYWORD     = "long"
ORDER_AMOUNT     = 80
DEBUG_TF_FILTER: list[str] = []

# =============================================================================
# HELPERS
# =============================================================================

def pct_improvement(val: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (val - base) / abs(base) * 100


def is_trending(values: dict[str, float | None], thresholds: dict[str, float], mode: str) -> bool:
    """
    Evaluate trending condition for enabled indicators only.
    values:     {indicator_key: computed_value | None}
    thresholds: {indicator_key: threshold}
    Returns False if no valid indicator has a value.
    """
    signals = []
    for key, val in values.items():
        if val is None or np.isnan(val):
            continue
        signals.append(val >= thresholds[key])
    if not signals:
        return False
    return all(signals) if mode == "AND" else any(signals)


# =============================================================================
# COMBO LABEL
# =============================================================================

def combo_label(active_keys: list[str], windows: dict, thresholds: dict, mode: str) -> str:
    parts = [f"{k.upper()}_W={windows[k]} | {k.upper()}_TH={thresholds[k]:.3f}" for k in active_keys]
    return " | ".join(parts) + f" | MODE={mode}"


# =============================================================================
# CONFIG LOADERS
# =============================================================================

def load_strategies_config(strategies_set_name: str) -> list[dict]:
    loop_name = f"strategies_loop_{strategies_set_name}_01"
    loop_path = os.path.join(BITGET_ROOT, f"BOT_batch_{strategies_set_name}", "strategies_files", f"{loop_name}.py")
    spec      = spec_from_file_location(loop_name, loop_path)
    module = module_from_spec(spec)
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

def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load full-history OHLCV from crypto_full_IS using BTC_TIMEFRAME."""
    return load_ohlcv_raw(symbol, BTC_TIMEFRAME)


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
    """Load OHLCV from CRYPTO_FULL_DIR for a given symbol and timeframe."""
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
    rename_map = {"timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts",
                  "volume_quote": "volume", "vol": "volume"}
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
    """Return the cache key for a symbol/strategy combination based on active modes."""
    ref_sym = "BTCUSDT" if analysis_mode == "BTC" else sym
    if regime_timeframe_mode == "STRATEGY":
        return (ref_sym, strategy['timeframe'])
    return ref_sym

def build_indicator_cache(
    baselines:             dict,
    strategies:            list[dict],
    windows:               dict[str, int],
    analysis_mode:         str = "SYMBOL",
    regime_timeframe_mode: str = "DAILY",
) -> dict:
    """
    Build indicator cache for all 4 mode combinations:
      - SYMBOL + DAILY   : {sym: (ts_arr, values_arr)}
      - SYMBOL + STRATEGY: {(sym, tf): (ts_arr, values_arr)}
      - BTC + DAILY      : {"BTCUSDT": (ts_arr, values_arr)}
      - BTC + STRATEGY   : {("BTCUSDT", tf): (ts_arr, values_arr)}
    """
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
        sym, tf = key if isinstance(key, tuple) else (key, BTC_TIMEFRAME)
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
        return {'profit': 0.0, 'win_rate': 0.0, 'n_trades': 0, 'max_dd': 0.0}
    profits = trades['profit']
    equity  = INITIAL_BALANCE + profits.cumsum()
    return {
        'profit':   float(profits.sum()),
        'win_rate': float((profits > 0).mean() * 100),
        'n_trades': len(profits),
        'max_dd':   float(((equity - equity.cummax()) / equity.cummax()).min() * 100),
    }


# =============================================================================
# BASELINE PRECOMPUTATION
# =============================================================================

def precompute_baselines(strategies_all: list[dict], strategies_set_name: str) -> tuple[dict, list[dict]]:
    print(f"\n{'='*120}")
    print(f"  PRECOMPUTING BASELINES — excluding strategies with B_PROF <= 0 in any period")
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
        if all_positive:
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

_METRIC_MAP = {
    "profit":   ("trending_prof", "ranging_prof", "b_prof"),
    "max_dd":   ("trending_dd",   "ranging_dd",   "b_dd"),
    "win_rate": ("trending_wr",   "ranging_wr",   "b_wr"),
    "calmar":   ("trending_prof", "ranging_prof", "b_prof"),
    "mix":      ("trending_prof", "ranging_prof", "b_prof"),
}

def _mix_score(prof: float, dd: float, w_profit: float = 0.6, w_dd: float = 0.4) -> float:
    """Weighted mix of profit and inverse drawdown. Higher is better."""
    inv_dd = 1 / abs(dd) if dd != 0 else 0.0
    return w_profit * prof + w_dd * inv_dd


def _calmar(prof: float, dd: float) -> float:
    """Calmar ratio: profit / abs(max_dd). Returns 0 if dd is 0."""
    return prof / abs(dd) if dd != 0 else 0.0

def _metric_value(d: dict, key: str, dd_key: str, optimize_metric: str) -> float:
    if optimize_metric == "calmar":
        return _calmar(d[key], d[dd_key])
    if optimize_metric == "mix":
        return _mix_score(d[key], d[dd_key])
    return d[key]


def classify_strategy(results: dict, sid: str, optimize_metric: str = "profit", classification_mode: str = "strict") -> str:
    data              = results.get(sid, {})
    periods_with_data = [pk for pk in EVAL_KEYS if pk in data and isinstance(data[pk], dict)]
    if not periods_with_data:
        return "neutral"

    t_key, r_key, b_key = _METRIC_MAP.get(optimize_metric, _METRIC_MAP["profit"])

    def _beats_baseline(pk: str, filt_key: str, dd_key: str) -> bool:
        return _metric_value(data[pk], filt_key, dd_key, optimize_metric) > _metric_value(data[pk], b_key, 'b_dd', optimize_metric)

    if classification_mode == "oos1_weighted":
        oos1_ok = "OOS1" in periods_with_data
        oos23   = [pk for pk in periods_with_data if pk != "OOS1"]

        def _passes(filt_key: str, dd_key: str) -> bool:
            if not oos1_ok or not _beats_baseline("OOS1", filt_key, dd_key):
                return False
            if not oos23:
                return True
            avg = sum(_metric_value(data[pk], filt_key, dd_key, optimize_metric) for pk in oos23) / len(oos23)
            avg_b = sum(_metric_value(data[pk], b_key, 'b_dd', optimize_metric) for pk in oos23) / len(oos23)
            return avg > avg_b

        t_all = _passes(t_key, 'trending_dd')
        r_all = _passes(r_key, 'ranging_dd')
    else:
        t_all = all(_beats_baseline(pk, t_key, 'trending_dd') for pk in periods_with_data)
        r_all = all(_beats_baseline(pk, r_key, 'ranging_dd') for pk in periods_with_data)

    if t_all and r_all:
        return "neutral"
    if t_all:
        return "trending"
    if r_all:
        return "ranging"
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
            if cls == 'ranging':
                profits.append(d['ranging_prof']); dds.append(d['ranging_dd'])
            elif cls == 'trending':
                profits.append(d['trending_prof']); dds.append(d['trending_dd'])
            else:
                profits.append(d['b_prof']); dds.append(d['b_dd'])
    return sum(profits), (sum(dds) / len(dds) if dds else 0.0)


# =============================================================================
# PRINT TABLES
# =============================================================================

def print_combo_period_table(results, strategies, period_key, combo_label) -> dict:
    logger.debug(f"\n  {'─'*120}")
    logger.debug(f"  {combo_label}  |  PERIOD: {period_key}")
    logger.debug(f"  {'─'*120}")
    logger.debug(f"  {'STRATEGY':<35} {'B_PROF':>8} {'TRENDING':>9} {'TRD_Δ%':>7} {'RANGING':>8} {'RNG_Δ%':>7} "
                 f"{'B_DD%':>7} {'TRD_DD%':>8} {'RNG_DD%':>8} {'TREND%':>7}")
    logger.debug(f"  {'─'*120}")
    sys_b = sys_t = sys_r = 0.0
    dd_b, dd_t, dd_r, trend_pcts = [], [], [], []
    for s in strategies:
        sid = s['id']
        if sid not in results or period_key not in results[sid]:
            continue
        if not isinstance(results[sid][period_key], dict):
            continue
        d     = results[sid][period_key]
        t_pct = pct_improvement(d['trending_prof'], d['b_prof'])
        r_pct = pct_improvement(d['ranging_prof'],  d['b_prof'])
        tc    = "\033[92m" if t_pct > 0 else "\033[91m"
        rc    = "\033[92m" if r_pct > 0 else "\033[91m"
        rs    = "\033[0m"
        logger.debug(f"  {sid:<35} {d['b_prof']:>8.1f} {d['trending_prof']:>9.1f} "
                     f"{tc}{t_pct:>+6.1f}%{rs} {d['ranging_prof']:>8.1f} "
                     f"{rc}{r_pct:>+6.1f}%{rs} {d['b_dd']:>6.1f}% "
                     f"{d['trending_dd']:>7.1f}% {d['ranging_dd']:>7.1f}% "
                     f"{d['trending_pct']:>6.1f}%")
        sys_b += d['b_prof'];  sys_t += d['trending_prof'];  sys_r += d['ranging_prof']
        dd_b.append(d['b_dd']); dd_t.append(d['trending_dd']); dd_r.append(d['ranging_dd'])
        trend_pcts.append(d['trending_pct'])
    t_pct_s   = pct_improvement(sys_t, sys_b)
    r_pct_s   = pct_improvement(sys_r, sys_b)
    avg_trend = sum(trend_pcts) / len(trend_pcts) if trend_pcts else 0.0
    tc = "\033[92m" if t_pct_s > 0 else "\033[91m"
    rc = "\033[92m" if r_pct_s > 0 else "\033[91m"
    rs = "\033[0m"
    logger.debug(f"  {'─'*110}")
    logger.debug(f"  {'SYSTEM TOTAL':<35} {sys_b:>8.1f} {sys_t:>9.1f} "
                 f"{tc}{t_pct_s:>+6.1f}%{rs} {sys_r:>8.1f} "
                 f"{rc}{r_pct_s:>+6.1f}%{rs} "
                 f"{sum(dd_b)/len(dd_b) if dd_b else 0:>6.1f}% "
                 f"{sum(dd_t)/len(dd_t) if dd_t else 0:>7.1f}% "
                 f"{sum(dd_r)/len(dd_r) if dd_r else 0:>7.1f}% {avg_trend:>6.1f}%")
    return {
        'sys_b':           sys_b,
        'sys_trending':    sys_t,
        'sys_ranging':     sys_r,
        'trending_pct':    t_pct_s,
        'ranging_pct':     r_pct_s,
        'avg_dd_b':        sum(dd_b)/len(dd_b) if dd_b else 0.0,
        'avg_dd_trending': sum(dd_t)/len(dd_t) if dd_t else 0.0,
        'avg_dd_ranging':  sum(dd_r)/len(dd_r) if dd_r else 0.0,
        'avg_trend_pct':   avg_trend,
    }

def print_combo_summary(period_summaries, n_r, n_t, n_n, comb_p, comb_dd, base_p, base_dd, label):
    logger.info(f"\n  COMBO SUMMARY — {label}")
    logger.info(f"  {'PERIOD':<8} {'B_PROF':>10} {'TRENDING':>10} {'TRD_Δ%':>7} {'RANGING':>10} {'RNG_Δ%':>7} "
                f"{'B_DD%':>7} {'TRD_DD%':>8} {'RNG_DD%':>8} {'TREND%':>7}")
    logger.info(f"  {'─'*95}")
    for pk, s in period_summaries.items():
        tc = "\033[92m" if s['trending_pct'] > 0 else "\033[91m"
        rc = "\033[92m" if s['ranging_pct']  > 0 else "\033[91m"
        rs = "\033[0m"
        logger.info(f"  {pk:<8} {s['sys_b']:>10.1f} {s['sys_trending']:>10.1f} "
                    f"{tc}{s['trending_pct']:>+6.1f}%{rs} {s['sys_ranging']:>10.1f} "
                    f"{rc}{s['ranging_pct']:>+6.1f}%{rs} {s['avg_dd_b']:>6.1f}% "
                    f"{s['avg_dd_trending']:>7.1f}% {s['avg_dd_ranging']:>7.1f}% {s['avg_trend_pct']:>6.1f}%")
    logger.info(f"  {'─'*95}")
    comb_pct = pct_improvement(comb_p, base_p)
    cc = "\033[92m" if comb_pct > 0 else "\033[91m"
    rs = "\033[0m"
    logger.info(f"  Classifications — RANGING:{n_r}  TRENDING:{n_t}  NEUTRAL:{n_n}")
    logger.info(f"  Baseline  profit={base_p:>10.1f}  avg_dd={base_dd:>6.1f}%")
    logger.info(f"  Combined  profit={comb_p:>10.1f}  avg_dd={comb_dd:>6.1f}%  {cc}Delta={comb_pct:>+6.1f}%{rs}")

def print_ranking(ranking: list[dict], active_keys: list[str]) -> None:
    col_w  = {k: max(len(f"{k.upper()}_W"), 5) for k in active_keys}
    col_t  = {k: max(len(f"{k.upper()}_TH"), 7) for k in active_keys}
    ind_header = "  ".join(f"{k.upper()+'_W':>{col_w[k]}}  {k.upper()+'_TH':>{col_t[k]}}" for k in active_keys)
    ind_width  = sum(col_w[k] + 2 + col_t[k] + 2 for k in active_keys)
    header_line = (f"  {'#':>3}  {'COMBO':>5}  {ind_header}  {'MODE':<5}  "
               f"{'TREND%':>7}  {'RANGING':>7} {'TREND':>6} {'NEUT':>5}  "
               f"{'BASELINE':>10} {'COMB_PROF':>10} {'COMB_Δ%':>8} {'W_DELTA%':>9}  "
               f"{'BASE_DD%':>8} {'COMB_DD%':>8}")
    total_w = len(header_line) - 2
    logger.info(f"\n\n{'='*total_w}")
    logger.info(f"  FINAL RANKING — ALL COMBOS BY WEIGHTED DELTA VS BASELINE")
    logger.info(f"  Active indicators: {', '.join(active_keys)}")
    logger.info(f"{'='*total_w}")
    logger.info(f"  {'#':>3}  {'COMBO':>5}  {ind_header}  {'MODE':<5}  "
                f"{'TREND%':>7}  {'RANGING':>7} {'TREND':>6} {'NEUT':>5}  "
                f"{'BASELINE':>10} {'COMB_PROF':>10} {'COMB_Δ%':>8} {'W_DELTA%':>9}  "
                f"{'BASE_DD%':>8} {'COMB_DD%':>8}")
    logger.info(f"  {'─'*total_w}")
    for i, row in enumerate(ranking[:5], 1):
        pct     = pct_improvement(row['combined_profit'], row['baseline_profit'])
        w_delta = row.get('weighted_delta', 0.0)
        cc      = "\033[92m" if pct > 0 else "\033[91m"
        wc      = "\033[92m" if w_delta > 0 else "\033[91m"
        ddc     = "\033[92m" if row['combined_dd'] > row['baseline_dd'] else "\033[91m"
        rs      = "\033[0m"
        ind_cols = "  ".join(
            f"{row['windows'][k]:>{col_w[k]}} {row['thresholds'][k]:>{col_t[k]}.3f}"
            for k in active_keys
        )
        logger.info(f"  {i:>3}  {row['combo_idx']:>5}  {ind_cols}  {row['mode']:<5}  "
                    f"{row['avg_trend_pct']:>6.1f}%  "
                    f"{row['n_ranging']:>7} {row['n_trending']:>6} {row['n_neutral']:>5}  "
                    f"{row['baseline_profit']:>10.1f} {cc}{row['combined_profit']:>10.1f}{rs} "
                    f"{cc}{pct:>+7.1f}%{rs} {wc}{w_delta:>+8.1f}%{rs}  "
                    f"{row['baseline_dd']:>7.1f}% {ddc}{row['combined_dd']:>7.1f}%{rs}")
    logger.info(f"  {'─'*total_w}\n")
# =============================================================================
# REPORTING & PERSISTENCE
# =============================================================================

def print_consistency_table(strategy_results: dict) -> None:
    def _has_all_periods(data: dict) -> bool:
        return all(pk in data and isinstance(data[pk], dict) for pk in EVAL_KEYS)

    def _print_table(title: str, consistent: list, col_fn, improvement_fn) -> None:
        if not consistent:
            return
        print(f"\n{'='*120}")
        print(f"  {title}")
        print(f"{'='*120}")
        header = f"  {'STRATEGY':<35} {'DIR':<6} {'CLASS':<10}"
        for pk in EVAL_KEYS:
            header += f"  {pk:>10}"
        header += f"  {'ALL Δ%':>8}"
        print(header)
        print(f"  {'─'*110}")
        for sid, data in consistent:
            direction = "LONG" if data['is_long'] else "SHORT"
            cls       = data.get('classification', 'neutral').upper()
            row       = f"  {sid:<35} {direction:<6} {cls:<10}"
            total_f = total_b = 0.0
            for pk in EVAL_KEYS:
                val_f, val_b = col_fn(data[pk])
                dpct         = improvement_fn(val_f, val_b)
                color        = "\033[92m" if dpct > 0 else "\033[91m"
                row         += f"  {color}{dpct:>+9.1f}%\033[0m"
                total_f     += val_f
                total_b     += val_b
            all_pct = improvement_fn(total_f, total_b)
            color   = "\033[92m" if all_pct > 0 else "\033[91m"
            row    += f"  {color}{all_pct:>+7.1f}%\033[0m"
            print(row)
        print(f"  {'─'*110}\n")

    for filter_prof, filter_dd, label in [
        ("trending_prof", "trending_dd", "TRENDING"),
        ("ranging_prof",  "ranging_dd",  "RANGING"),
    ]:
        # PROFIT table
        consistent_prof = [
            (sid, data) for sid, data in sorted(strategy_results.items())
            if _has_all_periods(data)
            and all(data[pk][filter_prof] > data[pk]['b_prof'] for pk in EVAL_KEYS)
        ]
        _print_table(
            title         = f"STRATEGIES IMPROVING PROFIT IN ALL {len(EVAL_KEYS)} OOS PERIODS — {label} PASS",
            consistent    = consistent_prof,
            col_fn        = lambda d, fp=filter_prof: (d[fp], d['b_prof']),
            improvement_fn= pct_improvement,
        )

        # DD table — improvement = filtered DD less negative than baseline DD
        consistent_dd = [
            (sid, data) for sid, data in sorted(strategy_results.items())
            if _has_all_periods(data)
            and all(data[pk][filter_dd] > data[pk]['b_dd'] for pk in EVAL_KEYS)
        ]
        _print_table(
            title         = f"STRATEGIES IMPROVING DRAWDOWN IN ALL {len(EVAL_KEYS)} OOS PERIODS — {label} PASS",
            consistent    = consistent_dd,
            col_fn        = lambda d, fd=filter_dd: (d[fd], d['b_dd']),
            improvement_fn= pct_improvement,
        )


def print_classification_summary(strategy_results: dict) -> None:
    print(f"\n{'='*120}")
    print(f"  STRATEGY CLASSIFICATION SUMMARY")
    print(f"{'='*120}")
    print(f"  {'STRATEGY':<35} {'DIR':<6} {'CLASS':<10}")
    print(f"  {'─'*55}")
    for sid, data in sorted(strategy_results.items()):
        direction = "LONG" if data['is_long'] else "SHORT"
        cls       = data.get('classification', 'neutral').upper()
        color = {'RANGING': "\033[92m", 'TRENDING': "\033[94m", 'NEUTRAL': "\033[90m"}.get(cls, "")
        print(f"  {sid:<35} {direction:<6} {color}{cls:<10}\033[0m")
    print(f"  {'─'*55}\n")


def save_bins(strategy_results: dict, windows: dict, thresholds: dict, mode: str, output_path: str, strategies_set_name: str = "E1") -> None:
    active_keys  = list(windows.keys())
    from datetime import datetime
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    indicators_str = " | ".join(f"{k.upper()}({windows[k]})>={thresholds[k]}" for k in active_keys)
    header_lines = [
        '"""',
        f"regime_BINS_{STRATEGIES_SET_NAME}.py — Regime classification bins. Do not edit manually.",
        f"Generated by regime_GE_calibration.py — {indicators_str} | MODE={mode}",
        f"Auto-generated on {generated_at} UTC.",
        '"""',
        "",
    ]
    for k in active_keys:
        header_lines.append(f"{k.upper()}_WINDOW    = {windows[k]}")
        header_lines.append(f"{k.upper()}_THRESHOLD = {thresholds[k]}")
    header_lines += [f"COMBINE_MODE = '{mode}'", "", "REGIME_BINS = {"]

    bin_lines = [f'    "{sid}": "{data.get("classification", "neutral")}",' for sid, data in sorted(strategy_results.items())]

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines + bin_lines + ["}"]) + "\n")
    print(f"\n  ✅ Bins saved to: {output_path}")


# =============================================================================
# LOAD REGIME BINS
# =============================================================================
# =============================================================================
# LOAD REGIME BINS
# =============================================================================

def load_regime_bins_ge(bins_path: str, strategy_id: str) -> str:
    """
    Load the GE regime classification for a strategy.
    Returns: "trending" | "ranging" | "both" | "neutral"
    """
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

def precompute_indicators(
    df:      pd.DataFrame,
    windows: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute rolling indicator arrays for a full OHLCV DataFrame.

    Parameters
    ----------
    df      : DataFrame with columns ts, high, low, close
    windows : {indicator_key: window}  — only enabled indicators

    Returns
    -------
    ts_arr     : np.ndarray[datetime64[ns]]
    values_arr : {indicator_key: np.ndarray[float]}

    Only rows where ALL indicators produced a valid (non-NaN) value are kept.
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    ts    = df["ts"].values

    min_win = (
        max((w + 1 if k != "hurst" else w) for k, w in windows.items())
        if windows else 1
    )

    ts_list:     list            = []
    value_lists: dict[str, list] = {k: [] for k in windows}

    for i in range(min_win, len(close)):
        row_values: dict[str, float] = {}
        valid = True

        for key, w in windows.items():
            val = _CALC_FN[key](high[: i + 1], low[: i + 1], close[: i + 1], w)
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
# FAST — experimental, remove if results differ from original
def lookup_indicators_batch(
    ts_arr:        np.ndarray,
    values_arr:    dict[str, np.ndarray],
    signal_ts_arr: np.ndarray,
    timeframe:     str | None = None,
) -> dict[str, np.ndarray]:
    """
    Vectorized version of lookup_indicators for multiple timestamps at once.
    Returns {indicator_key: np.ndarray of values} — one value per signal.
    NaN where no valid index found.
    """
    ts = signal_ts_arr.astype("datetime64[ns]")

    if timeframe is None or timeframe in ("1Dutc", "1D"):
        ts   = (ts.astype("datetime64[D]") - np.timedelta64(1, "D")).astype("datetime64[ns]")
        idxs = np.searchsorted(ts_arr, ts, side="right") - 1
    else:
        idxs = np.searchsorted(ts_arr, ts, side="left") - 1

    valid = idxs >= 0
    result = {}
    for k, arr in values_arr.items():
        out        = np.full(len(idxs), np.nan)
        out[valid] = arr[idxs[valid]]
        result[k]  = out

    return result
def lookup_indicators(
    ts_arr:     np.ndarray,
    values_arr: dict[str, np.ndarray],
    signal_ts,
    timeframe:       str | None = None,
    return_idx:      bool = False,
) -> dict[str, float | None] | tuple[dict[str, float | None], int]:
    """
    Return {indicator_key: value} for the last available row before signal_ts.
    - timeframe=None or daily: applies normalize() - 1 day lookahead fix.
    - intraday timeframe: uses searchsorted left - 1 (previous candle).
    """
    ts = pd.Timestamp(signal_ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    if timeframe is None or timeframe in ("1Dutc", "1D"):
        ts  = ts.normalize() - pd.Timedelta(days=1)
        idx = np.searchsorted(ts_arr, np.datetime64(ts.value, "ns"), side="right") - 1
    else:
        idx = np.searchsorted(ts_arr, np.datetime64(ts.value, "ns"), side="left") - 1

    if idx < 0:
        result = {k: None for k in values_arr}
        return (result, -1) if return_idx else result
    result = {k: float(arr[idx]) for k, arr in values_arr.items()}
    return (result, idx) if return_idx else result