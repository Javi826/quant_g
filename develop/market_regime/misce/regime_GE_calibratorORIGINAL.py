#develop/market_regime/regime_XX_calibration.py
"""
Grid search calibration for ATR Norm + ER + Hurst regime filter.

ANALYSIS_MODE:
  "BTC"    — uses BTCUSDT_1Dutc indicators (global regime)
  "SYMBOL" — uses {symbol}_1Dutc indicators per symbol (local regime)

Each indicator can be enabled/disabled individually via INDICATORS dict.
Only enabled indicators participate in the grid and in _is_trending logic.

Lookahead fix: normalize() - 1 day in _lookup.
Data loaded from crypto_full_IS.
"""
import os
import sys
import time
import itertools
import logging
import numpy as np
import pandas as pd

for _key in list(sys.modules.keys()):
    if any(_key.startswith(_mod) for _mod in ("shared_batchs", "shared", "bitget")):
        del sys.modules[_key]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from importlib.util import spec_from_file_location, module_from_spec

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split_OLD", "expanding")

PERIODS = {
    "IS":   os.path.join(_BASE, "IS",  "crypto_2024-01_2025-05_IS"),
    "OOS1": os.path.join(_BASE, "OOS", "crypto_2025-05_2026-05_OOS"),
    "OOS2": os.path.join(_BASE, "OOS", "crypto_2022-01_2023-01_OOS"),
    "OOS3": os.path.join(_BASE, "OOS", "crypto_2023-01_2024-01_OOS"),
}

DATA_FOLDER_IS = PERIODS["IS"]

CRYPTO_FULL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split_OLD", "expanding", "IS", "crypto_full_IS")

STRATEGIES_SET_NAME  = "E1"
SYMBOLS_LIVE_FOLDER  = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1", "strategies_E1", "symbols_live")
STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_01"
STRATEGIES_LOOP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1", "strategies_files", f"{STRATEGIES_LOOP_NAME}.py")

EVAL_KEYS = ["OOS2", "OOS3", "OOS1"]

# =============================================================================
# REGIME CONFIGURATION
# =============================================================================

ANALYSIS_MODE = "SYMBOL"   # "BTC" | "SYMBOL"
BTC_TIMEFRAME = "1Dutc"
COMBINE_MODES = ["OR"]

# Each indicator: windows (grid), thresholds (grid), enabled flag.
# Disable an indicator by setting enabled=False — it is excluded from the grid.
INDICATORS: dict[str, dict] = {
    "atr_norm": {
        "windows":    [10,20,30],
        "thresholds": [0.02,0.04,0.06],
        "enabled":    True,
    },
    "er": {
        "windows":    [10,20,30],
        "thresholds": [0.2,0.4,0.6],
        "enabled":    True,
    },
    "hurst": {
        "windows":    [30,80],
        "thresholds": [0.4,0.6,0.8],
        "enabled":    True,
    },
}

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


ORDER_AMOUNT     = 80
LONG_KEYWORD     = "long"
DEBUG_TF_FILTER: list[str] = []

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger("BOT_batch.pipeline.universe").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# GRID BUILDER
# =============================================================================

def _build_grid() -> tuple[list[str], list[tuple]]:
    """
    Build the parameter grid from enabled INDICATORS.
    Returns (active_keys, grid) where active_keys is the ordered list of
    enabled indicator names and grid is the list of all parameter combos.

    Each combo is a tuple of (window, threshold) pairs for each active indicator,
    followed by the combine mode. E.g. for atr_norm + er + mode:
      (atr_w, atr_th, er_w, er_th, mode)
    """
    active_keys = [k for k, v in INDICATORS.items() if v.get("enabled", True)]

    # Build one axis per indicator: list of (window, threshold) pairs
    indicator_axes = [
        [(w, th) for w in INDICATORS[k]["windows"] for th in INDICATORS[k]["thresholds"]]
        for k in active_keys
    ]
    combos = list(itertools.product(*indicator_axes, COMBINE_MODES))
    return active_keys, combos


def _unpack_combo(active_keys: list[str], combo: tuple) -> tuple[dict[str, int], dict[str, float], str]:
    """
    Unpack a raw combo tuple into (windows, thresholds, mode) dicts.
    combo = ((w0, th0), (w1, th1), ..., mode)
    """
    *indicator_pairs, mode = combo
    windows    = {k: indicator_pairs[i][0] for i, k in enumerate(active_keys)}
    thresholds = {k: indicator_pairs[i][1] for i, k in enumerate(active_keys)}
    return windows, thresholds, mode


def _combo_label(active_keys: list[str], windows: dict, thresholds: dict, mode: str) -> str:
    parts = [f"{k.upper()}_W={windows[k]} | {k.upper()}_TH={thresholds[k]:.3f}" for k in active_keys]
    return " | ".join(parts) + f" | MODE={mode}"


# =============================================================================
# HELPERS
# =============================================================================

def _pct_improvement(val: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (val - base) / abs(base) * 100


def _is_trending(values: dict[str, float | None], thresholds: dict[str, float], mode: str) -> bool:
    """
    Evaluate trending condition for enabled indicators only.
    values:     {indicator_key: computed_value | None}
    thresholds: {indicator_key: threshold}
    Indicators absent from values are ignored.
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
# CONFIG LOADERS
# =============================================================================

def load_strategies_config() -> list[dict]:
    spec   = spec_from_file_location(STRATEGIES_LOOP_NAME, STRATEGIES_LOOP_PATH)
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


def load_symbols(strategy_id: str, timeframe: str) -> list[str]:
    filepath = os.path.join(SYMBOLS_LIVE_FOLDER, f"symbols_live_{strategy_id}_{timeframe}.csv")
    if not os.path.exists(filepath):
        return []
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


def _load_ohlcv_for_period(strategy: dict, period_key: str) -> dict:
    """
    Load OHLCV data for a strategy/period matching main_batch logic:
      - OOS1: load symbols_live directly (my_symbols=True)
      - OOS2/OOS3: select_universe with my_symbols=False (top N by volume)
    """
    if period_key == "OOS1":
        symbols = load_symbols(strategy['id'], strategy['timeframe'])
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
    logger.debug(f"[DEBUG] {strategy['id']} {period_key} symbols: {sorted(ohlcv_oos.keys())}")
    return ohlcv_oos


# =============================================================================
# OHLCV LOADER
# =============================================================================

def _load_ohlcv(symbol: str) -> pd.DataFrame:
    path = os.path.join(CRYPTO_FULL_DIR, f"{symbol}_{BTC_TIMEFRAME}.parquet")
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
# INDICATOR COMPUTATION
# =============================================================================

def _calc_atr_norm(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> float:
    needed = window + 1
    if len(close) < needed:
        return np.nan
    h   = high[-needed:]
    l   = low[-needed:]
    c   = close[-needed:]
    trs = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = trs.mean()
    return float(atr / c[-1]) if c[-1] > 0 else np.nan


def _calc_er(close: np.ndarray, window: int) -> float:
    if len(close) < window + 1:
        return np.nan
    series       = close[-(window + 1):]
    total_change = np.sum(np.abs(np.diff(series)))
    if total_change == 0:
        return 0.0
    return float(np.clip(abs(series[-1] - series[0]) / total_change, 0.0, 1.0))


def _calc_hurst(close: np.ndarray, window: int) -> float:
    if len(close) < window:
        return np.nan
    log_returns = np.diff(np.log(close[-window:] + 1e-10))
    if len(log_returns) < 4:
        return np.nan
    log_lags, log_vars = [], []
    for lag in range(2, max(3, len(log_returns) // 2)):
        agg = np.array([log_returns[i:i + lag].sum() for i in range(0, len(log_returns) - lag, lag)])
        if len(agg) < 2:
            continue
        var = np.var(agg)
        if var <= 0:
            continue
        log_lags.append(np.log(lag))
        log_vars.append(np.log(var))
    if len(log_lags) < 2:
        return np.nan
    return float(np.clip(np.polyfit(log_lags, log_vars, 1)[0] / 2.0, 0.0, 1.0))


# Map indicator key -> compute function signature: fn(high, low, close, window)
# atr_norm needs high/low/close; er and hurst only need close.
_CALC_FN = {
    "atr_norm": lambda high, low, close, w: _calc_atr_norm(high, low, close, w),
    "er":       lambda high, low, close, w: _calc_er(close, w),
    "hurst":    lambda high, low, close, w: _calc_hurst(close, w),
}


def precompute_indicators(
    df: pd.DataFrame,
    windows: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute arrays for each enabled indicator given their windows.
    windows: {indicator_key: window_size}  — only enabled indicators.
    Returns (ts_arr, {indicator_key: values_arr}).
    All arrays are aligned: only rows where ALL indicators produced valid values are kept.
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    ts    = df["ts"].values

    min_win = max((w + 1 if k != "hurst" else w) for k, w in windows.items()) if windows else 1

    ts_list: list     = []
    value_lists: dict = {k: [] for k in windows}

    for i in range(min_win, len(close)):
        row_values = {}
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

def _lookup(
    ts_arr: np.ndarray,
    values_arr: dict[str, np.ndarray],
    signal_ts,
) -> dict[str, float | None]:
    """
    Return {indicator_key: value} for the last available row before signal_ts.
    Returns None for each key if no valid row is found.
    """
    ts = pd.Timestamp(signal_ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    ts = ts.normalize() - pd.Timedelta(days=1)

    idx = np.searchsorted(ts_arr, np.datetime64(ts.value, "ns"), side="right") - 1
    if idx < 0:
        return {k: None for k in values_arr}
    return {k: float(arr[idx]) for k, arr in values_arr.items()}


# =============================================================================
# BACKTEST
# =============================================================================

def _run_backtest(ohlcv_arrays: dict, best_params: dict) -> dict:
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

def precompute_baselines(strategies_all: list[dict]) -> tuple[dict, list[dict]]:
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
            ohlcv_data = _load_ohlcv_for_period(strategy, period_key)
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
                'metrics':      _run_backtest(baseline_arrays, strategy['best_params']),
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
# INDICATOR CACHE
# =============================================================================

def build_indicator_cache(
    baselines: dict,
    strategies: list[dict],
    windows: dict[str, int],
) -> dict[str, tuple]:
    """
    Build {symbol: (ts_arr, values_arr)} for all needed symbols.
    In BTC mode: single entry keyed "BTCUSDT".
    In SYMBOL mode: one entry per unique symbol across all strategies/periods.
    windows: {indicator_key: window} for enabled indicators only.
    """
    cache: dict[str, tuple] = {}

    if ANALYSIS_MODE == "BTC":
        df = _load_ohlcv("BTCUSDT")
        if not df.empty:
            cache["BTCUSDT"] = precompute_indicators(df, windows)
        return cache

    symbols_needed: set[str] = set()
    for strategy in strategies:
        for period_key in EVAL_KEYS:
            if period_key in baselines.get(strategy['id'], {}):
                for sym in baselines[strategy['id']][period_key]['ohlcv_arrays']:
                    symbols_needed.add(sym)

    for sym in sorted(symbols_needed):
        df = _load_ohlcv(sym)
        if not df.empty:
            cache[sym] = precompute_indicators(df, windows)

    return cache


# =============================================================================
# FILTERED BACKTEST FOR A SINGLE COMBO
# =============================================================================

def _run_filtered_combo(
    baselines: dict,
    strategies: list[dict],
    indicator_cache: dict,
    thresholds: dict[str, float],
    mode: str,
) -> dict:
    results: dict = {}

    btc_cache = indicator_cache.get("BTCUSDT") if ANALYSIS_MODE == "BTC" else None

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
                signals = cached['signal_cache'][sym]
                filt_t  = signals.copy()
                filt_r  = signals.copy()

                if ANALYSIS_MODE == "SYMBOL":
                    sym_cache = indicator_cache.get(sym)
                    if sym_cache is None:
                        filt_r[:] = 0
                        n_ranging += int(signals.sum())
                        trending_arr[sym] = {**arr, 'signal': filt_t}
                        ranging_arr[sym]  = {**arr, 'signal': filt_r}
                        continue
                    ts_arr, values_arr = sym_cache
                else:
                    ts_arr, values_arr = btc_cache

                for idx in np.nonzero(signals)[0]:
                    indicator_values = _lookup(ts_arr, values_arr, pd.Timestamp(arr['ts'][idx]))
                    trending         = _is_trending(indicator_values, thresholds, mode)
                    if trending:
                        filt_t[idx] = 0
                        n_trending += 1
                    else:
                        filt_r[idx] = 0
                        n_ranging  += 1

                trending_arr[sym] = {**arr, 'signal': filt_t}
                ranging_arr[sym]  = {**arr, 'signal': filt_r}

            m_t          = _run_backtest(trending_arr, strategy['best_params'])
            m_r          = _run_backtest(ranging_arr,  strategy['best_params'])
            total        = n_trending + n_ranging
            trending_pct = n_trending / max(total, 1) * 100

            results[sid][period_key] = {
                'b_prof':       m_base['profit'],  't_prof':  m_t['profit'],   'r_prof':  m_r['profit'],
                'b_dd':         m_base['max_dd'],  't_dd':    m_t['max_dd'],   'r_dd':    m_r['max_dd'],
                'b_wr':         m_base['win_rate'],'t_wr':    m_t['win_rate'], 'r_wr':    m_r['win_rate'],
                'trending_pct': trending_pct,
                't_pass_pct':   100 - trending_pct,
                'r_pass_pct':   trending_pct,
            }

    return results


# =============================================================================
# CLASSIFICATION
# =============================================================================

def _classify_strategy(results: dict, sid: str) -> str:
    data              = results.get(sid, {})
    periods_with_data = [pk for pk in EVAL_KEYS if pk in data and isinstance(data[pk], dict)]
    if not periods_with_data:
        return "neutral"
    t_all = all(data[pk]['t_prof'] > data[pk]['b_prof'] for pk in periods_with_data)
    r_all = all(data[pk]['r_prof'] > data[pk]['b_prof'] for pk in periods_with_data)
    if t_all and r_all:
        return "both"
    if t_all:
        return "ranging"
    if r_all:
        return "trending"
    return "neutral"


# =============================================================================
# COMBINED METRICS
# =============================================================================

def _combined_metrics(results: dict) -> tuple[float, float]:
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
                profits.append(d['t_prof']); dds.append(d['t_dd'])
            elif cls == 'trending':
                profits.append(d['r_prof']); dds.append(d['r_dd'])
            elif cls == 'both':
                if d['t_prof'] >= d['r_prof']:
                    profits.append(d['t_prof']); dds.append(d['t_dd'])
                else:
                    profits.append(d['r_prof']); dds.append(d['r_dd'])
            else:
                profits.append(d['b_prof']); dds.append(d['b_dd'])

    return sum(profits), (sum(dds) / len(dds) if dds else 0.0)


# =============================================================================
# PRINT TABLES
# =============================================================================

def _print_combo_period_table(results, strategies, period_key, combo_label) -> dict:
    print(f"\n  {'─'*130}")
    print(f"  {combo_label}  |  PERIOD: {period_key}")
    print(f"  {'─'*130}")
    print(f"  {'STRATEGY':<35} {'B_PROF':>8} {'T_PROF':>8} {'T_Δ%':>7} {'R_PROF':>8} {'R_Δ%':>7} "
          f"{'B_DD%':>7} {'T_DD%':>7} {'R_DD%':>7} {'TREND%':>7} {'T_PASS%':>8} {'R_PASS%':>8}")
    print(f"  {'─'*130}")

    sys_b = sys_t = sys_r = 0.0
    dd_b, dd_t, dd_r, trend_pcts = [], [], [], []

    for s in strategies:
        sid = s['id']
        if sid not in results or period_key not in results[sid]:
            continue
        if not isinstance(results[sid][period_key], dict):
            continue
        d     = results[sid][period_key]
        t_pct = _pct_improvement(d['t_prof'], d['b_prof'])
        r_pct = _pct_improvement(d['r_prof'], d['b_prof'])
        tc    = "\033[92m" if t_pct > 0 else "\033[91m"
        rc    = "\033[92m" if r_pct > 0 else "\033[91m"
        rs    = "\033[0m"

        print(f"  {sid:<35} {d['b_prof']:>8.1f} {d['t_prof']:>8.1f} "
              f"{tc}{t_pct:>+6.1f}%{rs} {d['r_prof']:>8.1f} "
              f"{rc}{r_pct:>+6.1f}%{rs} {d['b_dd']:>6.1f}% "
              f"{d['t_dd']:>6.1f}% {d['r_dd']:>6.1f}% "
              f"{d['trending_pct']:>6.1f}% {d['t_pass_pct']:>7.1f}% {d['r_pass_pct']:>7.1f}%")

        sys_b += d['b_prof'];  sys_t += d['t_prof'];  sys_r += d['r_prof']
        dd_b.append(d['b_dd']); dd_t.append(d['t_dd']); dd_r.append(d['r_dd'])
        trend_pcts.append(d['trending_pct'])

    t_pct_s   = _pct_improvement(sys_t, sys_b)
    r_pct_s   = _pct_improvement(sys_r, sys_b)
    avg_trend = sum(trend_pcts) / len(trend_pcts) if trend_pcts else 0.0
    tc = "\033[92m" if t_pct_s > 0 else "\033[91m"
    rc = "\033[92m" if r_pct_s > 0 else "\033[91m"
    rs = "\033[0m"
    print(f"  {'─'*130}")
    print(f"  {'SYSTEM TOTAL':<35} {sys_b:>8.1f} {sys_t:>8.1f} "
          f"{tc}{t_pct_s:>+6.1f}%{rs} {sys_r:>8.1f} "
          f"{rc}{r_pct_s:>+6.1f}%{rs} "
          f"{sum(dd_b)/len(dd_b) if dd_b else 0:>6.1f}% "
          f"{sum(dd_t)/len(dd_t) if dd_t else 0:>6.1f}% "
          f"{sum(dd_r)/len(dd_r) if dd_r else 0:>6.1f}% {avg_trend:>6.1f}%")

    return {
        'sys_b': sys_b, 'sys_t': sys_t, 'sys_r': sys_r,
        't_pct': t_pct_s, 'r_pct': r_pct_s,
        'avg_dd_b': sum(dd_b)/len(dd_b) if dd_b else 0.0,
        'avg_dd_t': sum(dd_t)/len(dd_t) if dd_t else 0.0,
        'avg_dd_r': sum(dd_r)/len(dd_r) if dd_r else 0.0,
        'avg_trend_pct': avg_trend,
    }


def _print_combo_summary(period_summaries, n_r, n_t, n_b, n_n, comb_p, comb_dd, base_p, base_dd, label):
    print(f"\n  COMBO SUMMARY — {label}")
    print(f"  {'PERIOD':<8} {'B_PROF':>10} {'T_PROF':>10} {'T_Δ%':>7} {'R_PROF':>10} {'R_Δ%':>7} "
          f"{'B_DD%':>7} {'T_DD%':>7} {'R_DD%':>7} {'TREND%':>7}")
    print(f"  {'─'*95}")
    for pk, s in period_summaries.items():
        tc = "\033[92m" if s['t_pct'] > 0 else "\033[91m"
        rc = "\033[92m" if s['r_pct'] > 0 else "\033[91m"
        rs = "\033[0m"
        print(f"  {pk:<8} {s['sys_b']:>10.1f} {s['sys_t']:>10.1f} "
              f"{tc}{s['t_pct']:>+6.1f}%{rs} {s['sys_r']:>10.1f} "
              f"{rc}{s['r_pct']:>+6.1f}%{rs} {s['avg_dd_b']:>6.1f}% "
              f"{s['avg_dd_t']:>6.1f}% {s['avg_dd_r']:>6.1f}% {s['avg_trend_pct']:>6.1f}%")
    print(f"  {'─'*95}")
    comb_pct = _pct_improvement(comb_p, base_p)
    cc = "\033[92m" if comb_pct > 0 else "\033[91m"
    rs = "\033[0m"
    print(f"  Classifications — RANGING:{n_r}  TRENDING:{n_t}  BOTH:{n_b}  NEUTRAL:{n_n}")
    print(f"  Baseline  profit={base_p:>10.1f}  avg_dd={base_dd:>6.1f}%")
    print(f"  Combined  profit={comb_p:>10.1f}  avg_dd={comb_dd:>6.1f}%  {cc}Delta={comb_pct:>+6.1f}%{rs}")


def _print_ranking(ranking: list[dict], active_keys: list[str]) -> None:
    # Compute per-indicator column widths based on name length
    # Each indicator occupies: W_col (5) + sep (2) + TH_col (7) = 14 min, but header label may be wider
    col_w = {k: max(len(f"{k.upper()}_W"), 5) for k in active_keys}
    col_t = {k: max(len(f"{k.upper()}_TH"), 7) for k in active_keys}

    ind_header = "  ".join(f"{k.upper()+'_W':>{col_w[k]}}  {k.upper()+'_TH':>{col_t[k]}}" for k in active_keys)
    ind_width  = sum(col_w[k] + 2 + col_t[k] + 2 for k in active_keys)
    total_w    = 6 + ind_width + 7 + 8 + 8 + 7 + 6 + 6 + 12 + 11 + 9 + 9 + 9

    print(f"\n\n{'='*total_w}")
    print(f"  FINAL RANKING — ALL COMBOS BY COMBINED PROFIT VS BASELINE  [MODE={ANALYSIS_MODE}]")
    print(f"  Active indicators: {', '.join(active_keys)}")
    print(f"{'='*total_w}")
    print(f"  {'#':>3}  {ind_header}  {'MODE':<5}  "
          f"{'TREND%':>7}  {'RANGING':>7} {'TREND':>6} {'BOTH':>5} {'NEUT':>5}  "
          f"{'BASELINE':>10} {'COMB_PROF':>10} {'COMB_Δ%':>8}  "
          f"{'BASE_DD%':>8} {'COMB_DD%':>8}")
    print(f"  {'─'*total_w}")

    for i, row in enumerate(ranking, 1):
        pct = _pct_improvement(row['combined_profit'], row['baseline_profit'])
        cc  = "\033[92m" if pct > 0 else "\033[91m"
        ddc = "\033[92m" if row['combined_dd'] > row['baseline_dd'] else "\033[91m"
        rs  = "\033[0m"

        ind_cols = "  ".join(
            f"{row['windows'][k]:>{col_w[k]}} {row['thresholds'][k]:>{col_t[k]}.3f}"
            for k in active_keys
        )
        print(f"  {i:>3}  {ind_cols}  {row['mode']:<5}  "
              f"{row['avg_trend_pct']:>6.1f}%  "
              f"{row['n_ranging']:>7} {row['n_trending']:>6} {row['n_both']:>5} {row['n_neutral']:>5}  "
              f"{row['baseline_profit']:>10.1f} {cc}{row['combined_profit']:>10.1f}{rs} "
              f"{cc}{pct:>+7.1f}%{rs}  "
              f"{row['baseline_dd']:>7.1f}% {ddc}{row['combined_dd']:>7.1f}%{rs}")

    print(f"  {'─'*total_w}\n")


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
    print(f"  BTC_TF={BTC_TIMEFRAME} | Lookahead fix: normalize()-1day")
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

    # Cache indicator arrays per unique windows tuple to avoid recomputation
    indicator_cache_map: dict[tuple, dict] = {}

    for combo_idx, combo in enumerate(grid, 1):
        windows, thresholds, mode = _unpack_combo(active_keys, combo)
        label    = _combo_label(active_keys, windows, thresholds, mode)
        win_key  = tuple(windows[k] for k in active_keys)

        print(f"\n{'='*120}")
        print(f"  COMBO {combo_idx}/{total_combos} — {label}")
        print(f"{'='*120}")

        if win_key not in indicator_cache_map:
            indicator_cache_map[win_key] = build_indicator_cache(baselines, strategies_filtered, windows)
        indicator_cache = indicator_cache_map[win_key]

        results = _run_filtered_combo(baselines, strategies_filtered, indicator_cache, thresholds, mode)

        for sid in results:
            if sid != 'is_long':
                results[sid]['classification'] = _classify_strategy(results, sid)

        period_summaries: dict[str, dict] = {}
        for pk in EVAL_KEYS:
            period_summaries[pk] = _print_combo_period_table(results, strategies_filtered, pk, label)

        cls_list       = [results[sid].get('classification', 'neutral') for sid in results if sid != 'is_long']
        comb_p, comb_d = _combined_metrics(results)
        avg_trend      = sum(ps['avg_trend_pct'] for ps in period_summaries.values()) / max(len(period_summaries), 1)

        _print_combo_summary(
            period_summaries,
            cls_list.count('ranging'), cls_list.count('trending'),
            cls_list.count('both'),    cls_list.count('neutral'),
            comb_p, comb_d, baseline_profit, baseline_dd, label,
        )

        ranking.append({
            'windows':    windows,
            'thresholds': thresholds,
            'mode':       mode,
            'combined_profit':  comb_p,         'combined_dd':  comb_d,
            'baseline_profit':  baseline_profit, 'baseline_dd': baseline_dd,
            'avg_trend_pct':    avg_trend,
            'n_ranging':  cls_list.count('ranging'),
            'n_trending': cls_list.count('trending'),
            'n_both':     cls_list.count('both'),
            'n_neutral':  cls_list.count('neutral'),
        })

    ranking.sort(key=lambda x: x['combined_profit'], reverse=True)
    _print_ranking(ranking, active_keys)

    elapsed = int(time.time() - _t0)
    print(f"\n  Completed in {elapsed//60}m {elapsed%60}s\n")


if __name__ == "__main__":
    run()