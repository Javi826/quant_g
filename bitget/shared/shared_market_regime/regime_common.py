#!/usr/bin/env python3
"""
shared/shared_market_regime/regimm_common.py - Shared functions for regime analysis scripts

Contains all common functions used across:
- regime1_performance_IS.py
- regime1_performance_WFO.py
- regime1_compare_IS_vs_OOS.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from shared_config import REGIME_REFERENCE_SYMBOL
from shared_market_regime.regime_metrics import calc_all_metrics


def extract_timeframe(df):
    strategy_val = df['strategy'].iloc[0]  # e.g. "03_parity_long_4H"
    parts = strategy_val.split('_')
    # el timeframe es el último elemento
    return parts[-1]


def load_btc_for_timeframe(ohlc_folder, timeframe, cache):
    """Load BTC OHLC for specific timeframe with caching"""
    cache_key = f"{ohlc_folder}_{timeframe}"
    if cache_key in cache:
        return cache[cache_key]
    
    filepath = Path(ohlc_folder) / f"{REGIME_REFERENCE_SYMBOL}_{timeframe}.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"BTC OHLC not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    df = df.sort_values('ts').reset_index(drop=True)
    
    cache[cache_key] = df
    return df


def calc_all_metrics_at_time(btc_df, buy_time, lookback, hurst_window, er_window, atr_window, pe_window, pe_order):
    """Calculate metrics at specific time - no lookahead"""
    closed_candles = btc_df[btc_df['ts'] < buy_time]
    if len(closed_candles) < lookback:
        return None
    idx = closed_candles.index[-1]
    start_idx = max(0, idx - lookback + 1)
    if idx - start_idx < 20:
        return None
    subset = btc_df.iloc[start_idx:idx + 1]
    ohlc = {
        'open': subset['open'].values.astype(np.float64),
        'high': subset['high'].values.astype(np.float64),
        'low': subset['low'].values.astype(np.float64),
        'close': subset['close'].values.astype(np.float64)
    }
    metrics = calc_all_metrics(ohlc, hurst_window=hurst_window, er_window=er_window, 
                                atr_window=atr_window, pe_window=pe_window, pe_order=pe_order)

    return metrics


def classify_trade_by_family(metrics, families):
    """Classify trade into family"""
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


def load_trades(filepath):
    """Load trades from CSV file"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    else:
        raise ValueError("File missing buy_time column")
    return df


def calculate_max_dd_pct(equity_curve):
    """Calculate Maximum Drawdown percentage"""
    if len(equity_curve) == 0:
        return 0.0
    cummax = equity_curve.cummax()
    drawdown_pct = np.where(cummax > 0, ((cummax - equity_curve) / cummax) * 100, 0.0)
    return float(np.max(drawdown_pct))


def permutation_test(profits1, profits2, n_permutations=1000, random_seed=42):

    if len(profits1) < 10 or len(profits2) < 10:
        return 1.0
    observed_diff = np.mean(profits1) - np.mean(profits2)
    combined = profits1 + profits2
    n1 = len(profits1)
    count_extreme = 0
    
    np.random.seed(random_seed)
    
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    return count_extreme / n_permutations


def format_significance(p_value):
    """Format significance indicator"""
    if p_value < 0.1:
        return f"✅ (p={p_value:.3f})"
    else:
        return f"❌ (p={p_value:.2f})"


def analyze_by_dimension(df, dimension, initial_capital, min_trades_confidence=None):
    """Analyze performance by dimension (family/direction/regime)
    
    Args:
        df: DataFrame with trades
        dimension: Column name to group by ('family', 'trend', 'regime')
        initial_capital: Starting capital
        min_trades_confidence: If provided, adds confidence indicator
    
    Returns:
        Dictionary with stats per category
    """
    stats = {}
    for category in df[dimension].unique():
        cat_df = df[df[dimension] == category].copy()
        cat_df = cat_df.sort_values('buy_time').reset_index(drop=True)
        cat_df['equity'] = initial_capital + cat_df['profit'].cumsum()
        num_trades = len(cat_df)
        profit = cat_df['profit'].sum()
        profits_list = cat_df['profit'].tolist()
        
        stats[category] = {
            'num_trades': num_trades,
            'profit': profit,
            'dd_pct': calculate_max_dd_pct(cat_df['equity']),
            'win_rate': (cat_df['profit'] > 0).mean() * 100 if num_trades > 0 else 0.0,
            'profits_list': profits_list
        }
        
        # Add confidence indicator if min_trades_confidence is provided
        if min_trades_confidence is not None:
            stats[category]['confidence'] = "✓" if num_trades >= min_trades_confidence else "✗"
    
    return stats
#EXTRA
def get_btc_macro_direction(
    btc_1d_df: pd.DataFrame,
    trade_time: pd.Timestamp,
    ma_period: int,
    long_th: float,
    short_th: float
) -> str:
    """
    Returns BTC macro direction at trade time using closed candles only.

    Args:
        btc_1d_df : BTC daily OHLC DataFrame with 'ts' and 'close' columns
        trade_time: Entry time of the trade (no lookahead)
        ma_period : MA period (e.g. 5, 10, 20, 50)
        long_th   : Multiplier threshold for uptrend  (e.g. 1.02 -> BTC > MA * 1.02)
        short_th  : Multiplier threshold for dwtrend (e.g. 0.98 -> BTC < MA * 0.98)

    Returns:
        'uptrend' | 'dwtrend' | 'neutral' | 'unknown'
    """
    closed = btc_1d_df[btc_1d_df['ts'] < trade_time]

    if len(closed) < ma_period:
        return 'unknown'

    last      = closed.iloc[-1]
    ma_series = closed['close'].iloc[-ma_period:]

    if len(ma_series) < ma_period:
        return 'unknown'

    ma_value  = ma_series.mean()
    btc_close = last['close']

    if pd.isna(ma_value) or pd.isna(btc_close):
        return 'unknown'

    if btc_close > ma_value * long_th:
        return 'uptrend'
    if btc_close < ma_value * short_th:
        return 'dwtrend'

    return 'neutral'

def filter_signals_by_regime(
    signals: np.ndarray,
    ts: np.ndarray,
    btc_1d_df: pd.DataFrame,
    btc_tf_df: pd.DataFrame,
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
    Filters signal array by market regime — sets signal to 0 where bin is blocked.
    Uses closed candles only — no lookahead bias.

    Args:
        signals        : numpy array of signals (1=LONG, -1=SHORT, 0=no signal)
        ts             : numpy array of timestamps (datetime64) matching signals
        btc_1d_df      : BTC 1D OHLC DataFrame with 'ts' and 'close' columns
        btc_tf_df      : BTC OHLC at strategy timeframe for family metrics
        bins_to_filter : set of bin keys to block e.g. {'trending_dwtrend', 'ranging_uptrend'}
        ma_period      : MA period for macro direction
        long_th        : multiplier threshold for uptrend
        short_th       : multiplier threshold for dwtrend
        families       : family classification rules dict
        lookback_bars  : lookback for metric calculation
        hurst_window   : window for Hurst exponent
        er_window      : window for Efficiency Ratio
        atr_window     : window for ATR
        pe_window      : window for Permutation Entropy
        pe_order       : order for Permutation Entropy
        metrics_cache  : optional precomputed {timestamp: metrics} dict from build_metrics_cache

    Returns:
        Filtered signals array (same shape, 0 where bin is blocked)
    """
    if not bins_to_filter:
        return signals

    filtered    = signals.copy()
    signal_idxs = np.nonzero(signals)[0]

    for idx in signal_idxs:
        trade_time = pd.Timestamp(ts[idx])

        direction = get_btc_macro_direction(
            btc_1d_df  = btc_1d_df,
            trade_time = trade_time,
            ma_period  = ma_period,
            long_th    = long_th,
            short_th   = short_th,
        )

        if metrics_cache is not None:
            metrics = metrics_cache.get(trade_time)
        else:
            metrics = calc_all_metrics_at_time(
                btc_df       = btc_tf_df,
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

def build_metrics_cache(
    btc_df: pd.DataFrame,
    lookback: int,
    hurst_window: int,
    er_window: int,
    atr_window: int,
    pe_window: int,
    pe_order: int,
) -> dict:
    """
    Precalculate regime metrics for all BTC bars.
    Returns a dict {timestamp: metrics} for fast lookup during signal filtering.
    Key is the timestamp of the NEXT bar — metrics are valid for any trade
    occurring at or after that timestamp (no lookahead).

    Args:
        btc_df       : BTC OHLC DataFrame with 'ts' column
        lookback     : lookback bars for metric calculation
        hurst_window : window for Hurst exponent
        er_window    : window for Efficiency Ratio
        atr_window   : window for ATR
        pe_window    : window for Permutation Entropy
        pe_order     : order for Permutation Entropy

    Returns:
        dict {pd.Timestamp: metrics_dict}
    """
    cache = {}
    n = len(btc_df)

    for i in range(lookback, n - 1):
        ts_next   = pd.Timestamp(btc_df.iloc[i + 1]['ts'])
        start_idx = max(0, i - lookback + 1)

        if i - start_idx < 20:
            continue

        subset = btc_df.iloc[start_idx:i + 1]
        ohlc = {
            'open':  subset['open'].values.astype(np.float64),
            'high':  subset['high'].values.astype(np.float64),
            'low':   subset['low'].values.astype(np.float64),
            'close': subset['close'].values.astype(np.float64),
        }
        metrics = calc_all_metrics(
            ohlc,
            hurst_window = hurst_window,
            er_window    = er_window,
            atr_window   = atr_window,
            pe_window    = pe_window,
            pe_order     = pe_order,
        )
        cache[ts_next] = metrics

    return cache
