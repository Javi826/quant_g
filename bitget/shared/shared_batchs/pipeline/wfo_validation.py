#shared_batchs/pipeline/wfo_validation.py
import logging
import numpy as np
import pandas as pd

from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE, run_grid_backtest
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.tools.wfo_ST import WARMUP_BARS

logger = logging.getLogger("BOT_batch.pipeline.wfo_validation")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _extract_wfo_window_data(
    window_idx: int,
    df_results: pd.DataFrame,
    ohlcv_is: dict,
) -> tuple[dict, pd.Timestamp, pd.Timestamp] | None:
    """
    Slice ohlcv_is to the exact test window dates for the given window index.
    Returns (ohlcv_window, test_start, test_end) or None if window is out of range.
    """
    n_windows = len(df_results) - 1  # last row is summary
    if window_idx >= n_windows:
        logger.error(f"window_idx={window_idx} out of range (0–{n_windows - 1})")
        return None

    row        = df_results.iloc[window_idx]
    test_start = pd.Timestamp(row["_test_start_ts"])
    test_end   = pd.Timestamp(row["_test_end_ts"])
    syms       = row["ts_syms"]

    ohlcv_window = {}
    for sym in syms:
        if sym not in ohlcv_is:
            logger.warning(f"Symbol {sym} not found in ohlcv_is — skipping.")
            continue
        df        = ohlcv_is[sym]
        test_iloc = df.index.searchsorted(test_start)
        warm_iloc = max(0, test_iloc - WARMUP_BARS)
        end_iloc  = df.index.searchsorted(test_end, side="right")
        ohlcv_window[sym] = df.iloc[warm_iloc:end_iloc]

    return ohlcv_window, test_start, test_end


def _run_baseline_backtest(
    ohlcv_window: dict,
    signal_fn: callable,
    signal_params: dict,
    best_params: dict,
    order_amount: int,
) -> pd.DataFrame:
    """Run baseline backtest on a window slice and return the trade log."""
    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_window)

    ohlcv_with_signals = {}
    for sym, arr in ohlcv_arrays.items():
        signals = signal_fn(arr, **signal_params, live_trading=False)
        ohlcv_with_signals[sym] = {**arr, "signal": np.asarray(signals)}

    results = run_grid_backtest(
        ohlcv_with_signals,
        sell_after   = best_params["SELL_AFTER"],
        tp_pct       = best_params["TP_PCT"],
        sl_pct       = best_params["SL_PCT"],
        order_amount = order_amount,
    )

    trades             = results["__PORTFOLIO__"]["trade_log"].copy()
    trades.columns     = trades.columns.str.lower().str.strip()
    trades["buy_time"] = pd.to_datetime(trades["buy_time"])
    return trades


def _filter_to_test_period(
    trades: pd.DataFrame,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    """Remove warmup trades that fall before the actual test window start."""
    if trades.empty:
        return trades
    return trades[trades["buy_time"] >= test_start].copy()


def _compare_metrics(
    wfo_trades: pd.DataFrame,
    oos_trades: pd.DataFrame,
) -> dict:
    """Compute and diff metrics between WFO test trades and aligned OOS trades."""
    def _safe_metrics(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"Net_Gain_pct": 0.0, "Max_DD_pct": 0.0, "Win_Rate": 0.0, "n_trades": 0}
        m = compute_metrics(df, capital=INITIAL_BALANCE, name="")
        return {
            "Net_Gain_pct": round(m["Net_Gain_pct"], 2),
            "Max_DD_pct":   round(m["Max_DD_pct"],   2),
            "Win_Rate":     round(m["Win_Rate"],      2),
            "n_trades":     len(df),
        }

    wfo_m = _safe_metrics(wfo_trades)
    oos_m = _safe_metrics(oos_trades)

    match_trades  = wfo_m["n_trades"]     == oos_m["n_trades"]
    match_netgain = abs(wfo_m["Net_Gain_pct"] - oos_m["Net_Gain_pct"]) < 0.01
    match_dd      = abs(wfo_m["Max_DD_pct"]   - oos_m["Max_DD_pct"])   < 0.01
    passed        = match_trades and match_netgain and match_dd

    return {
        "passed":       passed,
        "wfo":          wfo_m,
        "oos_aligned":  oos_m,
        "diff": {
            "n_trades":     oos_m["n_trades"]     - wfo_m["n_trades"],
            "Net_Gain_pct": round(oos_m["Net_Gain_pct"] - wfo_m["Net_Gain_pct"], 2),
            "Max_DD_pct":   round(oos_m["Max_DD_pct"]   - wfo_m["Max_DD_pct"],   2),
        },
    }


def _log_comparison(
    window_idx: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    comparison: dict,
) -> None:
    verdict = "✅ PASS" if comparison["passed"] else "❌ FAIL"
    wfo     = comparison["wfo"]
    oos     = comparison["oos_aligned"]
    diff    = comparison["diff"]

    logger.info(f"\n{'─'*115}")
    logger.info(f"  WFO VALIDATION — Window {window_idx} | {test_start} → {test_end} | {verdict}")
    logger.info(f"{'─'*115}")
    logger.info(f"  {'Metric':<18} {'WFO Test':>12} {'OOS Aligned':>14} {'Diff':>10}")
    logger.info(f"  {'─'*54}")
    logger.info(f"  {'n_trades':<18} {wfo['n_trades']:>12} {oos['n_trades']:>14} {diff['n_trades']:>+10}")
    logger.info(f"  {'Net_Gain_pct':<18} {wfo['Net_Gain_pct']:>12.2f} {oos['Net_Gain_pct']:>14.2f} {diff['Net_Gain_pct']:>+10.2f}")
    logger.info(f"  {'Max_DD_pct':<18} {wfo['Max_DD_pct']:>12.2f} {oos['Max_DD_pct']:>14.2f} {diff['Max_DD_pct']:>+10.2f}")
    logger.info(f"{'─'*115}\n")


# =============================================================================
# PUBLIC API
# =============================================================================

def validate_wfo_window(
    window_idx: int,
    df_results: pd.DataFrame,
    wfo_test_trades: pd.DataFrame,
    ohlcv_is: dict,
    signal_fn: callable,
    signal_params_keys: list,
    best_params: dict,
    param_names: list,
    order_amount: int,
    timeframe: str,
) -> dict | None:

    result = _extract_wfo_window_data(window_idx, df_results, ohlcv_is)
    if result is None:
        return None

    ohlcv_window, test_start, test_end = result

    signal_params = {k: best_params[k.upper()] for k in signal_params_keys if k.upper() in best_params}

    oos_trades_raw = _run_baseline_backtest(
        ohlcv_window = ohlcv_window,
        signal_fn    = signal_fn,
        signal_params = signal_params,
        best_params  = best_params,
        order_amount = order_amount,
    )
    oos_trades = _filter_to_test_period(oos_trades_raw, test_start)

    wfo_window_trades = pd.DataFrame()
    if wfo_test_trades is not None and not wfo_test_trades.empty:
        wfo_window_trades = wfo_test_trades[wfo_test_trades["wfo_window"] == window_idx + 1].copy()

    comparison = _compare_metrics(wfo_window_trades, oos_trades)
    _log_comparison(window_idx, test_start, test_end, comparison)

    return comparison