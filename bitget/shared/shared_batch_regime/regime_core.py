#shared/shared_batch_regime/regime_core.py
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
# REGIME CONSTANTS — edit here to change global behaviour
# =============================================================================

REGIME_TIMEFRAME: str = "1Dutc"  # daily timeframe for MA computation — only "1Dutc" is supported

if REGIME_TIMEFRAME != "1Dutc":
    raise ValueError(f"❌ REGIME_TIMEFRAME='{REGIME_TIMEFRAME}' is not supported. Only '1Dutc' is allowed.")

logger.debug(f"  [regime_core] REGIME_TIMEFRAME={REGIME_TIMEFRAME}")

# =============================================================================
# CONSTANTS
# =============================================================================

LONG_KEYWORD                   = "long"
ORDER_AMOUNT                   = 80
DEBUG_TF_FILTER: list[str]     = []

BINS: list[str] = ["uptrend", "dwtrend"]

BIN_CONDITIONS: list[tuple[str, callable]] = [
    ("uptrend", lambda ctx: ctx["close"] > ctx["ma"]),
    ("dwtrend", lambda ctx: ctx["close"] <= ctx["ma"]),
]

INDICATOR_COMPUTERS: dict[str, callable] = {
    "ma": lambda df, cfg: compute_ma(df["close"].values, cfg["ma_window"]),
}

assert [b for b, _ in BIN_CONDITIONS] == BINS, \
    f"BIN_CONDITIONS keys must match BINS exactly. Got {[b for b,_ in BIN_CONDITIONS]} vs {BINS}"

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

def classify_market_regime(context: dict) -> str:
    if context.get("close") is None or context.get("ma") is None \
            or np.isnan(context["close"]) or np.isnan(context["ma"]):
        return BINS[-1]
    for bin_name, condition in BIN_CONDITIONS:
        if condition(context):
            return bin_name
    return BINS[-1]
# =============================================================================
# COMBO LABEL
# =============================================================================

def combo_label(indicator_cfg: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in indicator_cfg.items())

# =============================================================================
# CONFIG LOADERS
# =============================================================================

def load_strategies_config(strategies_set_name: str) -> list[dict]:
    loop_name = f"strategies_loop_{strategies_set_name}_01"
    loop_path = os.path.join(BITGET_ROOT, f"BOT_batch_{strategies_set_name}", "strategies_files", f"{loop_name}.py")
    logger.debug(f"  [load_strategies_config] Loading: {loop_path}")
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
            custom_symbols=symbols,
        )
        return ohlcv_data

    _, symbols_oos_final, _, ohlcv_oos = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        data_folder_oos   = PERIODS[period_key],
        timeframe         = strategy['timeframe'],
        n_symbols         = strategy['n_symbols'],
        min_price         = None,
        filter_symbols_fn = filter_symbols,
    )
    ohlcv_oos = {sym: ohlcv_oos[sym] for sym in symbols_oos_final if sym in ohlcv_oos}
    logger.debug(f"[symbols] {strategy['id']} {period_key}: {sorted(ohlcv_oos.keys())}")
    return ohlcv_oos


def load_ohlcv_raw(symbol: str) -> pd.DataFrame:
    """Load raw OHLCV data for a symbol on REGIME_TIMEFRAME."""
    path = os.path.join(CRYPTO_FULL_DIR, f"{symbol}_{REGIME_TIMEFRAME}.parquet")
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
# INDICATOR CACHE  (MA over daily close, keyed by symbol)
# =============================================================================

def build_indicator_cache(
    baselines:     dict,
    strategies:    list[dict],
    indicator_cfg: dict,
) -> tuple[dict, dict]:
    cache:       dict = {}
    keys_needed: set  = set()

    cache:         dict = {}
    keys_needed:   set  = set()


    for strategy in strategies:
        for period_key in EVAL_KEYS:
            if period_key in baselines.get(strategy['id'], {}):
                for sym in baselines[strategy['id']][period_key]['ohlcv_arrays']:
                    keys_needed.add(sym)

    for sym in sorted(keys_needed):
        if sym in cache:
            continue
        df = load_ohlcv_raw(sym)
        if not df.empty:
            cache[sym] = precompute_indicators(df, indicator_cfg)

    return cache, indicator_cfg


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

def precompute_baselines(strategies_all: list[dict], strategies_set_name: str, filter_negative_baseline: bool = True) -> tuple[dict, list[dict]]:
    label = "excluding strategies with B_PROF <= 0 in any period" if filter_negative_baseline else "including all strategies"
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
        if not filter_negative_baseline or all_positive:
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
    data              = results.get(sid, {})
    periods_with_data = [pk for pk in EVAL_KEYS if pk in data and isinstance(data[pk], dict)]
    if not periods_with_data:
        return "neutral"

    def _beats_baseline(pk: str, bin_name: str) -> bool:
        d = data[pk]
        return _metric_value(d, _METRIC_MAP[bin_name][optimize_metric], _DD_KEY_MAP[bin_name], optimize_metric) \
             > _metric_value(d, "b_prof", "b_dd", optimize_metric)

    winning_bins = [
        b for b in BINS
        if all(_beats_baseline(pk, b) for pk in periods_with_data)
    ]

    return winning_bins[0] if len(winning_bins) == 1 else "neutral"


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
            if cls in BINS:
                profits.append(d[f'{cls}_prof'])
                dds.append(d[f'{cls}_dd'])
            else:
                profits.append(d['b_prof'])
                dds.append(d['b_dd'])
    return sum(profits), (sum(dds) / len(dds) if dds else 0.0)


# =============================================================================
# FILTERED BACKTEST FOR A SINGLE COMBO
# =============================================================================
def _assign_regime_signals(
    sym:             str,
    arr:             dict,
    cached:          dict,
    indicator_cache: dict,
    debug_n:         int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    signals     = cached['signal_cache'][sym]
    signal_idxs = np.nonzero(signals)[0]

    bin_signals: dict[str, np.ndarray] = {b: np.zeros_like(signals) for b in BINS}
    bin_counts:  dict[str, int]        = {b: 0 for b in BINS}

    sym_cache = indicator_cache.get(sym)
    if sym_cache is None:
        raise KeyError(f"Missing indicator cache for symbol '{sym}' — check that the daily parquet file exists.")

    if signal_idxs.size == 0:
        return bin_signals, bin_counts

    signal_ts = arr['ts'][signal_idxs]
    _debug_n  = debug_n if (debug_n > 0 and not any(bin_counts.values())) else 0
    lookups   = {
        key: lookup_indicator_batch(sym_cache["ts"], sym_cache[key], signal_ts, debug_n=_debug_n if key == "ma" else 0)
        for key in sym_cache if key != "ts"
    }

    for i, idx in enumerate(signal_idxs):
        context = {"close": float(arr['close'][idx]) if 'close' in arr else None}
        for key, values in lookups.items():
            context[key] = float(values[i]) if not np.isnan(values[i]) else None
        regime                   = classify_market_regime(context)
        bin_signals[regime][idx] = signals[idx]
        bin_counts[regime]      += 1

    return bin_signals, bin_counts


def _build_period_metrics(
    sid:        str,
    strategy:   dict,
    cached:     dict,
    indicator_cache: dict,
    debug_n:    int = 0,
) -> dict:
    m_base      = cached['metrics']
    bin_counts: dict[str, int]  = {b: 0 for b in BINS}
    bin_arrays: dict[str, dict] = {b: {} for b in BINS}

    for sym, arr in cached['ohlcv_arrays'].items():
        bin_signals, sym_counts = _assign_regime_signals(sym, arr, cached, indicator_cache, debug_n)
        for b in BINS:
            bin_counts[b]          += sym_counts[b]
            bin_arrays[b][sym]      = {**arr, 'signal': bin_signals[b]}

    bin_metrics: dict[str, dict] = {b: run_backtest(bin_arrays[b], strategy['best_params']) for b in BINS}
    total = sum(bin_counts.values())

    return {
        'b_prof': m_base['profit'],
        'b_dd':   m_base['max_dd'],
        'b_wr':   m_base['win_rate'],
        'b_r2':   m_base['r2'],
        **{f"{b}_prof": bin_metrics[b]['profit']                    for b in BINS},
        **{f"{b}_dd":   bin_metrics[b]['max_dd']                    for b in BINS},
        **{f"{b}_wr":   bin_metrics[b]['win_rate']                  for b in BINS},
        **{f"{b}_r2":   bin_metrics[b]['r2']                        for b in BINS},
        **{f"{b}_pct":  bin_counts[b] / max(total, 1) * 100  for b in BINS},
    }

def run_filtered_combo(
    baselines:       dict,
    strategies:      list[dict],
    indicator_cache: dict,
    debug_n:         int = 0,
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
            results[sid][period_key] = _build_period_metrics(
                sid, strategy, baselines[sid][period_key], indicator_cache, debug_n
            )

    return results
# =============================================================================
# PERSISTENCE
# =============================================================================

def save_bins(
    strategy_results:    dict,
    indicator_cfg:       dict,
    output_path:         str,
    strategies_set_name: str = "E1",
    all_strategies:      list[dict] | None = None,
    optimize_metric:     str = "",
) -> None:
    from datetime import datetime
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    header_lines = [
        '"""',
        f"regime_bins_{strategies_set_name}.py — auto-generated regime classification. Do not edit manually.",
        f"Generated by regime_calibration.py on {REGIME_TIMEFRAME}",
        f"Auto-generated on {generated_at} UTC.",
        '"""',
        "",
        f'INDICATOR_CFG = {indicator_cfg}',
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
# TIME-SERIES PRECOMPUTATION
# =============================================================================
def precompute_indicators(df: pd.DataFrame, cfg: dict) -> dict:
    ts         = df["ts"].values
    indicators = {key: fn(df, cfg) for key, fn in INDICATOR_COMPUTERS.items()}
    valid      = np.ones(len(ts), dtype=bool)
    for arr in indicators.values():
        valid &= ~np.isnan(arr)
    return {
        "ts": np.array(ts[valid], dtype="datetime64[ns]"),
        **{key: arr[valid] for key, arr in indicators.items()},
    }

# =============================================================================
# LOOKUP
# =============================================================================
def lookup_indicator_batch(
    ts_arr:        np.ndarray,
    indicator_arr: np.ndarray,
    signal_ts_arr: np.ndarray,
    debug_n:       int = 0,
) -> np.ndarray:

    ts_lookup = signal_ts_arr.astype("datetime64[ns]")
    ts_fixed  = (ts_lookup.astype("datetime64[D]") - np.timedelta64(1, "D")).astype("datetime64[ns]")
    idxs      = np.searchsorted(ts_arr, ts_fixed, side="right") - 1

    valid = idxs >= 0
    out   = np.full(len(idxs), np.nan)
    out[valid] = indicator_arr[idxs[valid]]

    if debug_n > 0:
        print(f"\n  [LOOKUP DEBUG] first {min(debug_n, len(idxs))} signals — raw timestamps only")
        print(f"  {'SIGNAL_TS':<30} {'DAILY_CANDLE_TS (used for MA)'}")
        print(f"  {'─'*65}")
        for i in range(min(debug_n, len(idxs))):
            sig_ts    = str(signal_ts_arr[i])
            daily_idx = idxs[i]
            candle_ts = str(ts_arr[daily_idx]) if daily_idx >= 0 else "N/A"
            print(f"  {sig_ts:<30} {candle_ts}")

    return out