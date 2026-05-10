# BOT_batch_rwa/utils/regime_common_rwa.py
# Adapted from shared/shared_market_regime/regime_common.py
# Changes: BTC reference replaced by a generic symbol reference ('self' or explicit symbol)

import numpy as np
import pandas as pd
from pathlib import Path

from shared_market_regime.regime_metrics import calc_all_metrics


# =============================================================================
# DATA LOADING
# =============================================================================

def load_reference_symbol_for_timeframe(
    ohlc_folder: str,
    symbol: str,
    timeframe: str,
    cache: dict,
) -> pd.DataFrame:
    """
    Load OHLC data for the reference symbol at the given timeframe, with caching.

    Args:
        ohlc_folder : Folder containing parquet files
        symbol      : Reference symbol (e.g. 'XAUUSD', 'WTICOUSD')
        timeframe   : Timeframe string (e.g. '1H', '1Dutc')
        cache       : Dict used for in-memory caching across calls

    Returns:
        DataFrame with columns: ts, open, high, low, close
    """
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


# =============================================================================
# METRICS AT POINT IN TIME
# =============================================================================

def calc_all_metrics_at_time(
    ref_df: pd.DataFrame,
    buy_time: pd.Timestamp,
    lookback: int,
    hurst_window: int,
    er_window: int,
    atr_window: int,
    pe_window: int,
    pe_order: int,
) -> dict | None:
    """
    Calculate regime metrics at a specific trade entry time (no lookahead).

    Args:
        ref_df    : Reference symbol OHLC DataFrame with 'ts' column
        buy_time  : Trade entry timestamp
        lookback  : Number of bars to use for metric calculation

    Returns:
        Dict of metrics or None if insufficient data
    """
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

    return calc_all_metrics(
        ohlc,
        hurst_window = hurst_window,
        er_window    = er_window,
        atr_window   = atr_window,
        pe_window    = pe_window,
        pe_order     = pe_order,
    )


# =============================================================================
# FAMILY CLASSIFICATION
# =============================================================================

def classify_trade_by_family(metrics: dict, families: dict) -> str:
    """
    Classify trade into a regime family based on metrics.

    Args:
        metrics  : Dict of computed regime metrics
        families : Dict of family rules from config

    Returns:
        Family name string or 'unknown'
    """
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


# =============================================================================
# MACRO DIRECTION
# =============================================================================

def get_macro_direction(
    ref_1d_df: pd.DataFrame,
    trade_time: pd.Timestamp,
    ma_period: int,
    long_th: float,
    short_th: float,
) -> str:
    """
    Compute macro direction of the reference symbol at trade entry time.
    Uses only closed daily candles (no lookahead).

    Args:
        ref_1d_df  : Daily OHLC DataFrame with 'ts' and 'close' columns
        trade_time : Trade entry timestamp
        ma_period  : MA period for trend detection
        long_th    : Multiplier threshold for uptrend  (e.g. 1.0 -> close > MA * 1.0)
        short_th   : Multiplier threshold for dwtrend (e.g. 1.0 -> close < MA * 1.0)

    Returns:
        'uptrend' | 'dwtrend' | 'neutral' | 'unknown'
    """
    closed = ref_1d_df[ref_1d_df['ts'] < trade_time]

    if len(closed) < ma_period:
        return 'unknown'

    last      = closed.iloc[-1]
    ma_series = closed['close'].iloc[-ma_period:]

    if len(ma_series) < ma_period:
        return 'unknown'

    ma_value  = ma_series.mean()
    ref_close = last['close']

    if pd.isna(ma_value) or pd.isna(ref_close):
        return 'unknown'

    if ref_close > ma_value * long_th:
        return 'uptrend'
    if ref_close < ma_value * short_th:
        return 'dwtrend'

    return 'neutral'


def build_direction_cache(
    ref_1d_df: pd.DataFrame,
    ma_period: int,
    long_th: float,
    short_th: float,
    trade_times: pd.Series,
) -> dict:
    """
    Vectorized precomputation of macro direction for a set of trade timestamps.

    Args:
        ref_1d_df   : Daily OHLC DataFrame with 'ts' and 'close' columns
        ma_period   : MA period for trend detection
        long_th     : Multiplier threshold for uptrend
        short_th    : Multiplier threshold for dwtrend
        trade_times : Series of trade entry timestamps

    Returns:
        Dict {pd.Timestamp: 'uptrend' | 'dwtrend' | 'neutral' | 'unknown'}
    """
    closes  = ref_1d_df['close'].values.astype(np.float64)
    ts_int  = ref_1d_df['ts'].values.astype(np.int64)
    n       = len(ref_1d_df)
    cache   = {}

    ma = np.full(n, np.nan)
    for i in range(ma_period - 1, n):
        ma[i] = closes[i - ma_period + 1: i + 1].mean()

    for t in trade_times.drop_duplicates():
        t_int = np.int64(pd.Timestamp(t).value)
        idx   = np.searchsorted(ts_int, t_int, side='left') - 1

        if idx < ma_period - 1:
            cache[pd.Timestamp(t)] = 'unknown'
            continue

        ma_val    = ma[idx]
        ref_close = closes[idx]

        if np.isnan(ma_val) or np.isnan(ref_close):
            cache[pd.Timestamp(t)] = 'unknown'
        elif ref_close > ma_val * long_th:
            cache[pd.Timestamp(t)] = 'uptrend'
        elif ref_close < ma_val * short_th:
            cache[pd.Timestamp(t)] = 'dwtrend'
        else:
            cache[pd.Timestamp(t)] = 'neutral'

    return cache


# =============================================================================
# SIGNAL FILTERING
# =============================================================================

def filter_signals_by_regime(
    signals: np.ndarray,
    ts: np.ndarray,
    ref_1d_df: pd.DataFrame,
    ref_tf_df: pd.DataFrame,
    bins_to_filter: set,
    ma_period: int = 5,
    long_th: float = 1.0,
    short_th: float = 1.0,
    families: dict = None,
    lookback_bars: int = 100,
    hurst_window: int = 100,
    er_window: int = 14,
    atr_window: int = 14,
    pe_window: int = 50,
    pe_order: int = 3,
    metrics_cache: dict = None,
) -> np.ndarray:
    """
    Zero out signals that fall into filtered regime bins.

    Args:
        signals        : Array of signal values
        ts             : Array of timestamps aligned with signals
        ref_1d_df      : Daily OHLC of reference symbol (for direction)
        ref_tf_df      : Strategy-timeframe OHLC of reference symbol (for family)
        bins_to_filter : Set of bins to suppress (e.g. {'trending_dwtrend'})
        metrics_cache  : Optional precomputed metrics cache for performance

    Returns:
        Filtered signals array
    """
    if not bins_to_filter:
        return signals

    filtered    = signals.copy()
    signal_idxs = np.nonzero(signals)[0]
    trade_times = pd.Series(pd.to_datetime(ts[signal_idxs]))

    direction_cache = build_direction_cache(ref_1d_df, ma_period, long_th, short_th, trade_times)

    for idx in signal_idxs:
        trade_time = pd.Timestamp(ts[idx])
        direction  = direction_cache.get(trade_time, 'unknown')

        if metrics_cache is not None:
            metrics = metrics_cache.get(trade_time)
        else:
            metrics = calc_all_metrics_at_time(
                ref_df       = ref_tf_df,
                buy_time     = trade_time,
                lookback     = lookback_bars,
                hurst_window = hurst_window,
                er_window    = er_window,
                atr_window   = atr_window,
                pe_window    = pe_window,
                pe_order     = pe_order,
            )

        family = classify_trade_by_family(metrics, families) if metrics else 'unknown'

        if family == 'unknown':
            continue

        if f"{family}_{direction}" in bins_to_filter:
            filtered[idx] = 0

    return filtered


# =============================================================================
# UTILITIES
# =============================================================================

def calculate_max_dd_pct(equity_curve: pd.Series) -> float:
    """Calculate Maximum Drawdown percentage from an equity curve."""
    if len(equity_curve) == 0:
        return 0.0
    cummax       = equity_curve.cummax()
    drawdown_pct = np.where(cummax > 0, ((cummax - equity_curve) / cummax) * 100, 0.0)
    return float(np.max(drawdown_pct))