#shared/shared_batchs/regime/regime_GE_core.py
"""
Core shared functions for the GE regime system.
Imported by:
  - regime_GE_calibration.py  (grid search calibration)
  - regime_GE.py              (dual filter + classification + bins persistence)
  - regime_GE_module.py       (main_batch integration)

Indicator calculations delegated to:
  - shared_trading_batch_develop.regime_metrics
"""
import os
import sys
import logging
import numpy as np
import pandas as pd

# Ensure shared_batchs and shared_trading_batch_develop are resolvable
# regardless of which script imports this module.
_SHARED = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

_BITGET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _BITGET not in sys.path:
    sys.path.insert(0, _BITGET)

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_trading_batch_develop.regime_metrics import (
    precompute_indicators,
    lookup_indicators,
)
from importlib.util import spec_from_file_location, module_from_spec

logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

_BITGET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASE   = os.path.join(_BITGET, "data_pipeline", "data", "04_split_OLD", "expanding")

PERIODS = {
    "IS":   os.path.join(_BASE, "IS",  "crypto_2024-01_2025-05_IS"),
    "OOS1": os.path.join(_BASE, "OOS", "crypto_2025-05_2026-05_OOS"),
    "OOS2": os.path.join(_BASE, "OOS", "crypto_2022-01_2023-01_OOS"),
    "OOS3": os.path.join(_BASE, "OOS", "crypto_2023-01_2024-01_OOS"),
}

DATA_FOLDER_IS  = PERIODS["IS"]
CRYPTO_FULL_DIR = os.path.join(_BASE, "IS", "crypto_full_IS")

EVAL_KEYS = ["OOS2", "OOS3", "OOS1"]

# =============================================================================
# STRATEGY CONFIG PATHS  (set per batch)
# =============================================================================

STRATEGIES_SET_NAME  = "E1"
SYMBOLS_LIVE_FOLDER  = os.path.join(_BITGET, "BOT_trading", "symbols_live", "E1")
STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_01"
STRATEGIES_LOOP_PATH = os.path.join(_BITGET, "BOT_batch_E1", "strategies_files", f"{STRATEGIES_LOOP_NAME}.py")

BTC_TIMEFRAME    = "1Dutc"
LONG_KEYWORD     = "long"
ORDER_AMOUNT     = 80
DEBUG_TF_FILTER: list[str] = []

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

def _combo_label(active_keys: list[str], windows: dict, thresholds: dict, mode: str) -> str:
    parts = [f"{k.upper()}_W={windows[k]} | {k.upper()}_TH={thresholds[k]:.3f}" for k in active_keys]
    return " | ".join(parts) + f" | MODE={mode}"


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


# =============================================================================
# OHLCV LOADERS
# =============================================================================

def _load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load full-history OHLCV from crypto_full_IS."""
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
    logger.debug(f"[symbols] {strategy['id']} {period_key}: {sorted(ohlcv_oos.keys())}")
    return ohlcv_oos


# =============================================================================
# INDICATOR CACHE
# =============================================================================

def build_indicator_cache(
    baselines:     dict,
    strategies:    list[dict],
    windows:       dict[str, int],
    analysis_mode: str = "SYMBOL",
) -> dict[str, tuple]:
    """
    Build {symbol: (ts_arr, values_arr)} for all needed symbols.
    analysis_mode: "BTC"    — single entry keyed "BTCUSDT"
                   "SYMBOL" — one entry per unique symbol across all strategies/periods
    """
    cache: dict[str, tuple] = {}

    if analysis_mode == "BTC":
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
# CLASSIFICATION
# =============================================================================

def _classify_strategy(results: dict, sid: str) -> str:
    data              = results.get(sid, {})
    periods_with_data = [pk for pk in EVAL_KEYS if pk in data and isinstance(data[pk], dict)]
    if not periods_with_data:
        return "neutral"
    t_all = all(data[pk]['ranging_prof']  > data[pk]['b_prof'] for pk in periods_with_data)
    r_all = all(data[pk]['trending_prof'] > data[pk]['b_prof'] for pk in periods_with_data)
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
                profits.append(d['ranging_prof']); dds.append(d['ranging_dd'])
            elif cls == 'trending':
                profits.append(d['trending_prof']); dds.append(d['trending_dd'])
            elif cls == 'both':
                if d['ranging_prof'] >= d['trending_prof']:
                    profits.append(d['ranging_prof']); dds.append(d['ranging_dd'])
                else:
                    profits.append(d['trending_prof']); dds.append(d['trending_dd'])
            else:
                profits.append(d['b_prof']); dds.append(d['b_dd'])

    return sum(profits), (sum(dds) / len(dds) if dds else 0.0)


# =============================================================================
# PRINT TABLES
# =============================================================================

def _print_combo_period_table(results, strategies, period_key, combo_label) -> dict:
    print(f"\n  {'─'*120}")
    print(f"  {combo_label}  |  PERIOD: {period_key}")
    print(f"  {'─'*120}")
    print(f"  {'STRATEGY':<35} {'B_PROF':>8} {'RANGING':>8} {'RNG_Δ%':>7} {'TRENDING':>9} {'TRD_Δ%':>7} "
          f"{'B_DD%':>7} {'RNG_DD%':>8} {'TRD_DD%':>8} {'TREND%':>7}")
    print(f"  {'─'*120}")
    sys_b = sys_t = sys_r = 0.0
    dd_b, dd_t, dd_r, trend_pcts = [], [], [], []
    for s in strategies:
        sid = s['id']
        if sid not in results or period_key not in results[sid]:
            continue
        if not isinstance(results[sid][period_key], dict):
            continue
        d     = results[sid][period_key]
        t_pct = _pct_improvement(d['ranging_prof'],  d['b_prof'])
        r_pct = _pct_improvement(d['trending_prof'], d['b_prof'])
        tc    = "\033[92m" if t_pct > 0 else "\033[91m"
        rc    = "\033[92m" if r_pct > 0 else "\033[91m"
        rs    = "\033[0m"
        print(f"  {sid:<35} {d['b_prof']:>8.1f} {d['ranging_prof']:>8.1f} "
              f"{tc}{t_pct:>+6.1f}%{rs} {d['trending_prof']:>9.1f} "
              f"{rc}{r_pct:>+6.1f}%{rs} {d['b_dd']:>6.1f}% "
              f"{d['ranging_dd']:>7.1f}% {d['trending_dd']:>7.1f}% "
              f"{d['trending_pct']:>6.1f}%")
        sys_b += d['b_prof'];  sys_t += d['ranging_prof'];  sys_r += d['trending_prof']
        dd_b.append(d['b_dd']); dd_t.append(d['ranging_dd']); dd_r.append(d['trending_dd'])
        trend_pcts.append(d['trending_pct'])
    t_pct_s   = _pct_improvement(sys_t, sys_b)
    r_pct_s   = _pct_improvement(sys_r, sys_b)
    avg_trend = sum(trend_pcts) / len(trend_pcts) if trend_pcts else 0.0
    tc = "\033[92m" if t_pct_s > 0 else "\033[91m"
    rc = "\033[92m" if r_pct_s > 0 else "\033[91m"
    rs = "\033[0m"
    print(f"  {'─'*110}")
    print(f"  {'SYSTEM TOTAL':<35} {sys_b:>8.1f} {sys_t:>8.1f} "
          f"{tc}{t_pct_s:>+6.1f}%{rs} {sys_r:>9.1f} "
          f"{rc}{r_pct_s:>+6.1f}%{rs} "
          f"{sum(dd_b)/len(dd_b) if dd_b else 0:>6.1f}% "
          f"{sum(dd_t)/len(dd_t) if dd_t else 0:>7.1f}% "
          f"{sum(dd_r)/len(dd_r) if dd_r else 0:>7.1f}% {avg_trend:>6.1f}%")
    return {
        'sys_b':          sys_b,
        'sys_ranging':    sys_t,
        'sys_trending':   sys_r,
        'ranging_pct':    t_pct_s,
        'trending_pct':   r_pct_s,
        'avg_dd_b':       sum(dd_b)/len(dd_b) if dd_b else 0.0,
        'avg_dd_ranging': sum(dd_t)/len(dd_t) if dd_t else 0.0,
        'avg_dd_trending':sum(dd_r)/len(dd_r) if dd_r else 0.0,
        'avg_trend_pct':  avg_trend,
    }


def _print_combo_summary(period_summaries, n_r, n_t, n_b, n_n, comb_p, comb_dd, base_p, base_dd, label):
    print(f"\n  COMBO SUMMARY — {label}")
    print(f"  {'PERIOD':<8} {'B_PROF':>10} {'RANGING':>10} {'RNG_Δ%':>7} {'TRENDING':>10} {'TRD_Δ%':>7} "
          f"{'B_DD%':>7} {'RNG_DD%':>8} {'TRD_DD%':>8} {'TREND%':>7}")
    print(f"  {'─'*95}")
    for pk, s in period_summaries.items():
        tc = "\033[92m" if s['ranging_pct']  > 0 else "\033[91m"
        rc = "\033[92m" if s['trending_pct'] > 0 else "\033[91m"
        rs = "\033[0m"
        print(f"  {pk:<8} {s['sys_b']:>10.1f} {s['sys_ranging']:>10.1f} "
              f"{tc}{s['ranging_pct']:>+6.1f}%{rs} {s['sys_trending']:>10.1f} "
              f"{rc}{s['trending_pct']:>+6.1f}%{rs} {s['avg_dd_b']:>6.1f}% "
              f"{s['avg_dd_ranging']:>7.1f}% {s['avg_dd_trending']:>7.1f}% {s['avg_trend_pct']:>6.1f}%")
    print(f"  {'─'*95}")
    comb_pct = _pct_improvement(comb_p, base_p)
    cc = "\033[92m" if comb_pct > 0 else "\033[91m"
    rs = "\033[0m"
    print(f"  Classifications — RANGING:{n_r}  TRENDING:{n_t}  BOTH:{n_b}  NEUTRAL:{n_n}")
    print(f"  Baseline  profit={base_p:>10.1f}  avg_dd={base_dd:>6.1f}%")
    print(f"  Combined  profit={comb_p:>10.1f}  avg_dd={comb_dd:>6.1f}%  {cc}Delta={comb_pct:>+6.1f}%{rs}")

def _print_ranking(ranking: list[dict], active_keys: list[str]) -> None:
    col_w  = {k: max(len(f"{k.upper()}_W"), 5) for k in active_keys}
    col_t  = {k: max(len(f"{k.upper()}_TH"), 7) for k in active_keys}
    ind_header = "  ".join(f"{k.upper()+'_W':>{col_w[k]}}  {k.upper()+'_TH':>{col_t[k]}}" for k in active_keys)
    ind_width  = sum(col_w[k] + 2 + col_t[k] + 2 for k in active_keys)
    total_w    = 6 + ind_width + 7 + 8 + 8 + 7 + 6 + 6 + 12 + 11 + 9 + 9 + 9

    print(f"\n\n{'='*total_w}")
    print(f"  FINAL RANKING — ALL COMBOS BY COMBINED PROFIT VS BASELINE")
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