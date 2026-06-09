#shared/shared_batchs/regime/regime_module.py
"""
Regime filter module — self-contained, no external regime dependencies.
Used by main_batch and regime03_bin_search.
"""
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from typing import Dict
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE, run_grid_backtest
from shared_batchs.utils.batch_metrics import compute_metrics

logger = logging.getLogger("shared_batch.regime.regime_module")


# =============================================================================
# CONFIGURATION
# =============================================================================
REGIME_ENABLED          = False
REGIME_SYMBOL_SOURCE    = 'symbol'   # 'symbol' | 'btc'
REGIME_REFERENCE        = 'BTCUSDT'
FORCE_DIRECTION_FILTER  = False
REGIME_MIN_TRADES       = 50
REGIME_LOOKBACK_BARS    = 50
REGIME_FAMILY_SOURCE    = 'strategy'    # 'strategy' | 'macro'
REGIME_DIRECTION_SOURCE = 'strategy'    # 'strategy' | 'daily'
REGIME0_MA_PERIOD       = 50

ER_WINDOW  = 14
ATR_WINDOW = 14

REGIME_FAMILIES = {
    'trending': {'efficiency_ratio': ('>', 0.6)},
    'volatile': {'atr_pct': ('>', 2.5)},
    'ranging':  {},
}

# =============================================================================
# METRICS
# =============================================================================

def calc_efficiency_ratio(close: np.ndarray, window: int = 14) -> float:
    if len(close) < window + 1:
        return np.nan
    series       = close[-(window + 1):]
    net_change   = abs(series[-1] - series[0])
    abs_changes  = np.abs(np.diff(series))
    total_change = np.sum(abs_changes)
    if total_change == 0:
        return 0.0
    return float(np.clip(net_change / total_change, 0.0, 1.0))


def calc_atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float:
    if len(close) < window + 1 or len(high) < window or len(low) < window:
        return np.nan
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))
    if len(tr) < window:
        return np.nan
    atr = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr = (atr * (window - 1) + tr[i]) / window
    current_price = close[-1]
    if current_price == 0 or np.isnan(current_price) or np.isnan(atr):
        return np.nan
    atr_pct = (atr / current_price) * 100
    return float(atr_pct) if 0 <= atr_pct <= 100 else np.nan


def calc_all_metrics(
    ohlc:       Dict[str, np.ndarray],
    er_window:  int = 14,
    atr_window: int = 14,
) -> Dict[str, float]:
    return {
        'efficiency_ratio': calc_efficiency_ratio(ohlc['close'], er_window),
        'atr_pct':          calc_atr_pct(ohlc['high'], ohlc['low'], ohlc['close'], atr_window),
    }

# =============================================================================
# DATA LOADING & CLASSIFICATION
# =============================================================================

def load_reference_symbol_for_timeframe(
    ohlc_folder: str,
    symbol:      str,
    timeframe:   str,
    cache:       dict,
) -> pd.DataFrame:
    cache_key = f"{ohlc_folder}_{symbol}_{timeframe}"
    if cache_key in cache:
        return cache[cache_key]
    filepath = Path(ohlc_folder) / f"{symbol}_{timeframe}.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"Reference symbol OHLC not found: {filepath}")
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    df = df.sort_values('ts').reset_index(drop=True)
    cache[cache_key] = df
    return df


def classify_trade_by_family(metrics: dict, families: dict) -> str:
    for family_name, rules in families.items():
        if not rules:
            continue
        match = True
        for metric, (op, val) in rules.items():
            if metrics.get(metric) is None or pd.isna(metrics[metric]):
                match = False
                break
            if op == '>' and not (metrics[metric] > val):
                match = False
                break
            elif op == '<' and not (metrics[metric] < val):
                match = False
                break
        if match:
            return family_name
    for family_name, rules in families.items():
        if not rules:
            return family_name
    return 'unknown'


def calc_all_metrics_at_time(
    ref_df:    pd.DataFrame,
    buy_time,
    lookback:  int,
    er_window: int,
    atr_window: int,
) -> dict | None:
    closed_candles = ref_df[ref_df['ts'] < buy_time]
    if len(closed_candles) < lookback:
        return None
    idx       = closed_candles.index[-1]
    start_idx = max(0, idx - lookback + 1)
    if idx - start_idx < 20:
        return None
    subset = ref_df.iloc[start_idx:idx + 1]
    ohlc   = {
        'open':  subset['open'].values.astype(np.float64),
        'high':  subset['high'].values.astype(np.float64),
        'low':   subset['low'].values.astype(np.float64),
        'close': subset['close'].values.astype(np.float64),
    }
    return calc_all_metrics(ohlc, er_window=er_window, atr_window=atr_window)

# =============================================================================
# DIRECTION & METRICS CACHE
# =============================================================================

def build_direction_cache(
    ref_1d_df:   pd.DataFrame,
    ma_period:   int,
    trade_times: pd.Series,
    is_daily:    bool = False,
) -> dict:
    closes  = ref_1d_df['close'].values.astype(np.float64)
    ts_int  = ref_1d_df['ts'].values.astype(np.int64)
    ts_vals = ref_1d_df['ts'].values
    n       = len(ref_1d_df)
    cache   = {}
    ma      = np.full(n, np.nan)
    for i in range(ma_period - 1, n):
        ma[i] = closes[i - ma_period + 1: i + 1].mean()
    for t in trade_times.drop_duplicates():
        if is_daily:
            t_int = np.int64(pd.Timestamp(t).normalize().value)
        else:
            t_int = np.int64(pd.Timestamp(t).value)
        idx = np.searchsorted(ts_int, t_int, side='left') - 1
        if idx < ma_period - 1:
            cache[pd.Timestamp(t)] = ('unknown', None)
            continue
        ma_val    = ma[idx]
        ref_close = closes[idx]
        ts_used   = pd.Timestamp(ts_vals[idx])
        if np.isnan(ma_val) or np.isnan(ref_close):
            cache[pd.Timestamp(t)] = ('unknown', ts_used)
        elif ref_close > ma_val:
            cache[pd.Timestamp(t)] = ('uptrend', ts_used)
        elif ref_close < ma_val:
            cache[pd.Timestamp(t)] = ('dwtrend', ts_used)
        else:
            cache[pd.Timestamp(t)] = ('neutral', ts_used)
    return cache


def build_metrics_cache(
    ref_df:     pd.DataFrame,
    lookback:   int,
    er_window:  int,
    atr_window: int,
) -> dict:
    cache = {}
    n     = len(ref_df)
    for i in range(lookback, n - 1):
        ts_next   = pd.Timestamp(ref_df.iloc[i + 1]['ts'])
        start_idx = max(0, i - lookback + 1)
        if i - start_idx < max(er_window, atr_window) + 1:
            continue
        subset = ref_df.iloc[start_idx:i + 1]
        ohlc   = {
            'open':  subset['open'].values.astype(np.float64),
            'high':  subset['high'].values.astype(np.float64),
            'low':   subset['low'].values.astype(np.float64),
            'close': subset['close'].values.astype(np.float64),
        }
        cache[ts_next] = calc_all_metrics(ohlc, er_window=er_window, atr_window=atr_window)
    return cache

# =============================================================================
# SIGNAL FILTERING
# =============================================================================

def filter_signals_by_regime(
    signals:        np.ndarray,
    ts:             np.ndarray,
    ref_1d_df:      pd.DataFrame,
    bins_to_filter: set,
    ma_period:      int  = 50,
    families:       dict = None,
    lookback_bars:  int  = 100,
    er_window:      int  = 14,
    atr_window:     int  = 14,
    metrics_cache:  dict = None,
    is_daily:       bool = False,
) -> np.ndarray:
    if not bins_to_filter:
        return signals
    filtered    = signals.copy()
    signal_idxs = np.nonzero(signals)[0]
    trade_times = pd.Series(pd.to_datetime(ts[signal_idxs]))

    direction_cache = build_direction_cache(ref_1d_df, ma_period, trade_times, is_daily=is_daily)

    for idx in signal_idxs:
        trade_time         = pd.Timestamp(ts[idx])
        direction, _       = direction_cache.get(trade_time, ('unknown', None))
        if metrics_cache is not None:
            metrics = metrics_cache.get(trade_time)
        else:
            metrics = calc_all_metrics_at_time(
                ref_df     = ref_1d_df,
                buy_time   = trade_time,
                lookback   = lookback_bars,
                er_window  = er_window,
                atr_window = atr_window,
            )
        family = classify_trade_by_family(metrics, families) if metrics else 'unknown'
        if family == 'unknown':
            continue
        if f"{family}_{direction}" in bins_to_filter:
            filtered[idx] = 0

    return filtered

# =============================================================================
# RUN OOS BACKTEST WITH REGIME
# =============================================================================

def run_oos_backtest_with_regime(
    strategy_id:    str,
    ohlcv_arrays:   dict,
    signal_fn,
    signal_params:  dict,
    best_params:    dict,
    order_amount:   int,
    data_folder:    str,
    timeframe:      str,
    bins_to_filter: set,
    initial_balance: float,
    debug_label:    str = "",
) -> tuple:
    ref_cache           = {}
    ohlcv_arrays_regime = {}

    if REGIME_ENABLED and bins_to_filter and REGIME_SYMBOL_SOURCE == 'btc':
        ref_tf_df         = load_reference_symbol_for_timeframe(data_folder, REGIME_REFERENCE, timeframe, ref_cache)
        btc_metrics_cache = build_metrics_cache(
            ref_df     = ref_tf_df,
            lookback   = REGIME_LOOKBACK_BARS,
            er_window  = ER_WINDOW,
            atr_window = ATR_WINDOW,
        )
    else:
        ref_tf_df         = None
        btc_metrics_cache = {}

    for sym, arr in ohlcv_arrays.items():
        signals = signal_fn(arr, **signal_params, live_trading=False)
        if REGIME_ENABLED and bins_to_filter:
            if REGIME_SYMBOL_SOURCE == 'symbol':
                sym_ref_df        = load_reference_symbol_for_timeframe(data_folder, sym, timeframe, ref_cache)
                sym_metrics_cache = build_metrics_cache(
                    ref_df     = sym_ref_df,
                    lookback   = REGIME_LOOKBACK_BARS,
                    er_window  = ER_WINDOW,
                    atr_window = ATR_WINDOW,
                )
                _dir_ref       = sym_ref_df
                _metrics_cache = sym_metrics_cache
            else:
                _dir_ref       = ref_tf_df
                _metrics_cache = btc_metrics_cache

            signals = filter_signals_by_regime(
                signals        = signals,
                ts             = arr['ts'],
                ref_1d_df      = _dir_ref,
                bins_to_filter = bins_to_filter,
                ma_period      = REGIME0_MA_PERIOD,
                families       = REGIME_FAMILIES,
                lookback_bars  = REGIME_LOOKBACK_BARS,
                er_window      = ER_WINDOW,
                atr_window     = ATR_WINDOW,
                metrics_cache  = _metrics_cache,
                is_daily       = (REGIME_DIRECTION_SOURCE == 'daily'),
            )

        ohlcv_arrays_regime[sym] = {**arr, "signal": signals}

    result_regime         = run_grid_backtest(
        ohlcv_arrays_regime,
        sell_after   = best_params["SELL_AFTER"],
        tp_pct       = best_params["TP_PCT"],
        sl_pct       = best_params["SL_PCT"],
        order_amount = order_amount,
    )
    trades_df             = result_regime["__PORTFOLIO__"]["trade_log"].copy()
    trades_df.columns     = trades_df.columns.str.lower().str.strip()
    trades_df["buy_time"] = pd.to_datetime(trades_df["buy_time"])
    metrics = compute_metrics(trades_df, capital=initial_balance, name=strategy_id) if len(trades_df) > 0 else None
    return trades_df, metrics

# =============================================================================
# LOAD REGIME BINS
# =============================================================================

def load_regime_bins(bins_path: str, strategy_id: str) -> set:
    """
    Load precomputed regime bins for a strategy from a generated regime_bins_{SET}.py file.
    Returns empty set if file not found or strategy not present.
    """
    if not os.path.exists(bins_path):
        logger.warning(f"⚠️  regime_bins file not found: {bins_path} — using empty bins.")
        return set()
    spec   = spec_from_file_location("regime_bins", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    bins   = getattr(module, "REGIME_BINS", {})
    return set(bins.get(strategy_id, set()))