# BOT_batch/utils/regime_utils.py
# Unified for crypto and RWA:
#   - REGIME_ENABLED: master switch
#   - REGIME_REFERENCE: reference symbol (e.g. 'BTCUSDT', 'QQQUSDT')

import logging

import numpy as np
import pandas as pd

from backtesters.ZX_compute_BT import INITIAL_BALANCE, run_grid_backtest
from shared_config import REGIME_ATR_WINDOW as ATR_WINDOW, REGIME_PE_WINDOW as PE_WINDOW, REGIME_PE_ORDER as PE_ORDER
from shared_config import REGIME_FAMILIES as FAMILIES, REGIME_HURST_WINDOW as HURST_WINDOW, REGIME_ER_WINDOW as ER_WINDOW
from shared_config import REGIME0_MA_PERIOD as R0_MA_PERIOD, REGIME0_LONG_TH as R0_LONG_TH, REGIME0_SHORT_TH as R0_SHORT_TH
from shared_market_regime.regime_common import (
    load_reference_symbol_for_timeframe,
    calc_all_metrics_at_time,
    calc_all_metrics,
    calculate_max_dd_pct,
    classify_trade_by_family,
    get_macro_direction,
    filter_signals_by_regime,
)
from utils.metrics import compute_metrics

logger = logging.getLogger("BOT_batch.utils.regime_utils")

# -----------------------------------------------------------------------------
# Regime configuration
# -----------------------------------------------------------------------------
REGIME_ENABLED         = True          # Master switch — set False to bypass all regime filtering
REGIME_REFERENCE       = 'QQQUSDT'     # Reference symbol for regime calculation
FORCE_DIRECTION_FILTER = True
REGIME_MIN_TRADES      = 10
REGIME_LOOKBACK_BARS   = 50
REGIME_FAMILY_SOURCE   = 'strategy'    # 'strategy' | 'macro'


# =============================================================================
# BUILD METRICS CACHE
# =============================================================================

def build_metrics_cache(
    ref_df: pd.DataFrame,
    lookback: int,
    hurst_window: int,
    er_window: int,
    atr_window: int,
    pe_window: int,
    pe_order: int,
) -> dict:
    """
    Precalculate regime metrics for all reference symbol bars.
    Returns a dict {timestamp: metrics} for fast lookup during signal filtering.
    Key is the timestamp of the NEXT bar — metrics are valid for any trade
    occurring at or after that timestamp (no lookahead).
    """
    cache = {}
    n     = len(ref_df)

    for i in range(lookback, n - 1):
        ts_next   = pd.Timestamp(ref_df.iloc[i + 1]['ts'])
        start_idx = max(0, i - lookback + 1)

        min_bars_required = max(ER_WINDOW, ATR_WINDOW) + 1
        if i - start_idx < min_bars_required:
            continue

        subset = ref_df.iloc[start_idx:i + 1]
        ohlc   = {
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


def prepare_regime_metrics_cache_is(data_folder_is: str, timeframe: str) -> dict:
    """
    Load reference symbol data and build metrics cache for IS regime analysis.
    Returns empty dict when REGIME_ENABLED=False.
    """
    if not REGIME_ENABLED:
        return {}

    ref_cache = {}
    ref_tf    = load_reference_symbol_for_timeframe(data_folder_is, REGIME_REFERENCE, timeframe, ref_cache) \
                if REGIME_FAMILY_SOURCE == 'strategy' \
                else load_reference_symbol_for_timeframe(data_folder_is, REGIME_REFERENCE, '1Dutc', ref_cache)

    return build_metrics_cache(
        ref_df       = ref_tf,
        lookback     = REGIME_LOOKBACK_BARS,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    )


# =============================================================================
# ANALYZE REGIME IS
# =============================================================================

def analyze_regime_is(
    trades_df_is: pd.DataFrame,
    timeframe: str,
    data_folder_is: str,
    strategy_direction: str,
    metrics_cache: dict = None,
) -> tuple[set, float]:
    """
    Analyze IS trades to determine which regime bins to filter.
    When REGIME_ENABLED=False, returns empty filter set and 100% remaining.
    Uses REGIME_REFERENCE symbol as regime reference for all trades.

    Returns:
        tuple: (bins_to_filter, pct_remain)
    """
    if not REGIME_ENABLED:
        return set(), 100.0

    ref_cache = {}
    ref_1d_df = load_reference_symbol_for_timeframe(data_folder_is, REGIME_REFERENCE, '1Dutc', ref_cache)
    ref_tf_df = load_reference_symbol_for_timeframe(data_folder_is, REGIME_REFERENCE, timeframe, ref_cache) \
                if REGIME_FAMILY_SOURCE == 'strategy' else ref_1d_df

    directions = []
    families_  = []

    for _, trade in trades_df_is.iterrows():
        direction = get_macro_direction(
            ref_1d_df  = ref_1d_df,
            trade_time = trade['buy_time'],
            ma_period  = R0_MA_PERIOD,
            long_th    = R0_LONG_TH,
            short_th   = R0_SHORT_TH,
        )

        if metrics_cache is not None:
            metrics = metrics_cache.get(pd.Timestamp(trade['buy_time']))
        else:
            metrics = calc_all_metrics_at_time(
                ref_df       = ref_tf_df,
                buy_time     = trade['buy_time'],
                lookback     = REGIME_LOOKBACK_BARS,
                hurst_window = HURST_WINDOW,
                er_window    = ER_WINDOW,
                atr_window   = ATR_WINDOW,
                pe_window    = PE_WINDOW,
                pe_order     = PE_ORDER,
            )

        family = classify_trade_by_family(metrics, FAMILIES) if metrics else 'unknown'
        directions.append(direction)
        families_.append(family)

    df              = trades_df_is.copy()
    df['direction'] = directions
    df['family']    = families_

    df_valid = df[
        (df['family'] != 'unknown') &
        (df['direction'].isin(['uptrend', 'dwtrend']))
    ].copy()

    bins_to_filter = set()

    for family in ['trending', 'ranging', 'volatile']:
        for direction in ['uptrend', 'dwtrend']:
            subset = df_valid[(df_valid['family'] == family) & (df_valid['direction'] == direction)]
            n      = len(subset)
            profit = subset['profit'].sum() if n > 0 else 0.0
            if n >= REGIME_MIN_TRADES and profit < 0:
                bins_to_filter.add(f"{family}_{direction}")

    n_valid    = len(df_valid)
    n_filtered = df_valid[
        df_valid.apply(lambda r: f"{r['family']}_{r['direction']}" in bins_to_filter, axis=1)
    ].shape[0]
    pct_remain = round((n_valid - n_filtered) / n_valid * 100, 1) if n_valid > 0 else 0.0

    if logger.isEnabledFor(logging.DEBUG):
        lines = []
        lines.append(f"\n  {'BIN':<30} {'CONF':>5} {'TRADES':>8} {'PROFIT':>12} {'WIN%':>8} {'DD%':>8} {'FILTER':>8}")
        lines.append("  " + "-" * 88)
        for fam in ['trending', 'ranging', 'volatile']:
            for dir_ in ['uptrend', 'dwtrend']:
                bin_key = f"{fam}_{dir_}"
                subset  = df_valid[(df_valid['family'] == fam) & (df_valid['direction'] == dir_)]
                n       = len(subset)
                profit  = subset['profit'].sum() if n > 0 else 0.0
                wr      = (subset['profit'] > 0).mean() * 100 if n > 0 else 0.0
                eq      = INITIAL_BALANCE + subset.sort_values('buy_time')['profit'].cumsum()
                dd      = calculate_max_dd_pct(eq) if n > 0 else 0.0
                conf    = "✓" if n >= REGIME_MIN_TRADES else "✗"
                flag    = "🚫 FILTER" if bin_key in bins_to_filter else ""
                lines.append(f"  {bin_key:<30} {conf:>5} {n:>8} {profit:>12.2f} {wr:>7.1f}% {dd:>7.2f}% {flag}")
        lines.append("  " + "-" * 88)
        logger.debug("\n".join(lines))

    if FORCE_DIRECTION_FILTER:
        forced = 'dwtrend' if strategy_direction == 'long' else 'uptrend'
        for fam in ['trending', 'ranging', 'volatile']:
            bins_to_filter.add(f"{fam}_{forced}")
            
    logger.debug(f"  Regime IS — total={len(trades_df_is)} | valid={n_valid} | unknown/neutral={len(trades_df_is)-n_valid}")

    return bins_to_filter, pct_remain


# =============================================================================
# RUN OOS BACKTEST WITH REGIME
# =============================================================================

def run_oos_backtest_with_regime(
    strategy_id: str,
    ohlcv_arrays: dict,
    signal_fn: callable,
    signal_params: dict,
    best_params: dict,
    order_amount: int,
    data_folder: str,
    timeframe: str,
    bins_to_filter: set,
    initial_balance: float,
) -> tuple:
    """
    Run backtest with optional regime filter.
    When REGIME_ENABLED=False, runs without any regime filtering.
    Uses REGIME_REFERENCE symbol as regime reference for all symbols.

    Returns:
        tuple: (trades_df, metrics_dict)
    """
    ref_cache = {}
    ref_1d_df = load_reference_symbol_for_timeframe(data_folder, REGIME_REFERENCE, '1Dutc', ref_cache) \
                if REGIME_ENABLED and bins_to_filter else None
    ref_tf_df = load_reference_symbol_for_timeframe(data_folder, REGIME_REFERENCE, timeframe, ref_cache) \
                if REGIME_ENABLED and bins_to_filter and REGIME_FAMILY_SOURCE == 'strategy' else ref_1d_df

    metrics_cache = build_metrics_cache(
        ref_df       = ref_tf_df,
        lookback     = REGIME_LOOKBACK_BARS,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    ) if REGIME_ENABLED and bins_to_filter else {}

    ohlcv_arrays_regime = {}
    for sym, arr in ohlcv_arrays.items():
        signals = signal_fn(arr, **signal_params, live_trading=False)

        if REGIME_ENABLED and bins_to_filter:
            signals = filter_signals_by_regime(
                signals        = signals,
                ts             = arr['ts'],
                ref_1d_df      = ref_1d_df,
                ref_tf_df      = ref_tf_df,
                bins_to_filter = bins_to_filter,
                ma_period      = R0_MA_PERIOD,
                long_th        = R0_LONG_TH,
                short_th       = R0_SHORT_TH,
                families       = FAMILIES,
                lookback_bars  = REGIME_LOOKBACK_BARS,
                hurst_window   = HURST_WINDOW,
                er_window      = ER_WINDOW,
                atr_window     = ATR_WINDOW,
                pe_window      = PE_WINDOW,
                pe_order       = PE_ORDER,
                metrics_cache  = metrics_cache,
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