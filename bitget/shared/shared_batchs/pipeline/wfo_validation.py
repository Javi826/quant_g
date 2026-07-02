import logging
import numpy as np
import pandas as pd

from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE, run_grid_backtest
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.engines.wfo_WF import WARMUP_BARS

logger = logging.getLogger("BOT_batch.pipeline.wfo_validation")


# =============================================================================
# PRIVATE HELPERS — BASELINE COMPARISON
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
# PRIVATE HELPERS — OVERLAP / GAP / DUPLICATE CHECKS
# =============================================================================

def _check_window_bounds(
    wfo_test_trades: pd.DataFrame,
    df_results: pd.DataFrame,
) -> list[dict]:
    """Flag trades whose buy_time falls outside its own window's [test_start, test_end]."""
    violations = []
    n_windows  = len(df_results) - 1  # last row is summary

    for window_idx in range(n_windows):
        row        = df_results.iloc[window_idx]
        test_start = pd.Timestamp(row["_test_start_ts"])
        test_end   = pd.Timestamp(row["_test_end_ts"])

        window_trades = wfo_test_trades[wfo_test_trades["wfo_window"] == window_idx + 1]
        if window_trades.empty:
            continue

        out_of_bounds = window_trades[
            (window_trades["buy_time"] < test_start) | (window_trades["buy_time"] > test_end)
        ]
        if not out_of_bounds.empty:
            violations.append({
                "wfo_window": window_idx + 1,
                "test_start": test_start,
                "test_end":   test_end,
                "n_trades":   len(out_of_bounds),
                "min_buy":    out_of_bounds["buy_time"].min(),
                "max_buy":    out_of_bounds["buy_time"].max(),
            })
    return violations


def _check_duplicates(wfo_test_trades: pd.DataFrame) -> pd.DataFrame:
    """Flag duplicate trades (same symbol + buy_time) across different windows."""
    key_cols = ["symbol", "buy_time"]
    dup_mask = wfo_test_trades.duplicated(subset=key_cols, keep=False)
    duplicates = wfo_test_trades[dup_mask].sort_values(key_cols)
    return duplicates[["wfo_window", "symbol", "buy_time"]] if not duplicates.empty else duplicates


def _check_gaps(
    wfo_test_trades: pd.DataFrame,
    df_results: pd.DataFrame,
) -> list[dict]:
    """Flag test windows with zero trades across all symbols (informational only)."""
    gaps      = []
    n_windows = len(df_results) - 1

    for window_idx in range(n_windows):
        row           = df_results.iloc[window_idx]
        window_trades = wfo_test_trades[wfo_test_trades["wfo_window"] == window_idx + 1]
        if window_trades.empty:
            gaps.append({
                "wfo_window": window_idx + 1,
                "test_start": pd.Timestamp(row["_test_start_ts"]),
                "test_end":   pd.Timestamp(row["_test_end_ts"]),
            })
    return gaps


def _check_cross_window_gap(
    wfo_test_trades: pd.DataFrame,
    n_windows: int,
    gap_threshold_days: int,
) -> list[dict]:
    """Compare last trade of window N vs first trade of window N+1 (real activity gap)."""
    issues = []

    for window_idx in range(n_windows - 1):
        curr_trades = wfo_test_trades[wfo_test_trades["wfo_window"] == window_idx + 1]
        next_trades = wfo_test_trades[wfo_test_trades["wfo_window"] == window_idx + 2]
        if curr_trades.empty or next_trades.empty:
            continue

        last_buy_curr  = curr_trades["buy_time"].max()
        first_buy_next = next_trades["buy_time"].min()
        gap_days       = (first_buy_next - last_buy_curr).days

        if gap_days > gap_threshold_days:
            issues.append({
                "window_a":       window_idx + 1,
                "window_b":       window_idx + 2,
                "last_buy_curr":  last_buy_curr,
                "first_buy_next": first_buy_next,
                "days":           gap_days,
            })
    return issues


def _check_intra_window_gaps(
    wfo_test_trades: pd.DataFrame,
    n_windows: int,
    gap_threshold_days: int,
) -> list[dict]:
    """Find the largest gap in days between consecutive trades within each window."""
    issues = []

    for window_idx in range(n_windows):
        window_trades = wfo_test_trades[wfo_test_trades["wfo_window"] == window_idx + 1]
        if len(window_trades) < 2:
            continue

        buy_times   = window_trades["buy_time"].sort_values().reset_index(drop=True)
        gaps_days   = buy_times.diff().dt.days.dropna()
        if gaps_days.empty:
            continue

        max_gap_days = int(gaps_days.max())
        if max_gap_days > gap_threshold_days:
            max_gap_idx = gaps_days.idxmax()
            issues.append({
                "wfo_window": window_idx + 1,
                "gap_days":   max_gap_days,
                "before":     buy_times.iloc[max_gap_idx - 1],
                "after":      buy_times.iloc[max_gap_idx],
            })
    return issues


def _log_overlap_report(
    bound_violations: list[dict],
    duplicates: pd.DataFrame,
    gaps: list[dict],
    cross_window_gaps: list[dict],
    intra_window_gaps: list[dict],
) -> None:
    logger.info(f"\n{'─'*115}")
    logger.info(f"  WFO TEST TRADES — Overlap / Duplicate / Gap check")
    logger.info(f"{'─'*115}")

    if bound_violations:
        logger.warning(f"  ❌ Out-of-bounds trades found in {len(bound_violations)} window(s):")
        for v in bound_violations:
            logger.warning(
                f"     Window {v['wfo_window']}: {v['n_trades']} trade(s) outside "
                f"[{v['test_start']} → {v['test_end']}] (range {v['min_buy']} → {v['max_buy']})"
            )
    else:
        logger.info(f"  ✅ No out-of-bounds trades — all buy_time within their window's [test_start, test_end].")

    if not duplicates.empty:
        logger.warning(f"  ❌ Duplicate trades found ({len(duplicates)} rows, same symbol + buy_time across windows):")
        logger.warning(f"\n{duplicates.to_string(index=False)}")
    else:
        logger.info(f"  ✅ No duplicate trades across windows.")

    if gaps:
        logger.info(f"  ⚪ {len(gaps)} window(s) with zero trades (informational, not necessarily an error):")
        for g in gaps:
            logger.info(f"     Window {g['wfo_window']}: {g['test_start']} → {g['test_end']}")
    else:
        logger.info(f"  ✅ No empty windows.")

    if cross_window_gaps:
        logger.warning(f"  ❌ Trade activity gaps between consecutive windows ({len(cross_window_gaps)}):")
        for c in cross_window_gaps:
            logger.warning(
                f"     Window {c['window_a']} → {c['window_b']}: "
                f"{c['days']} day(s) with no trades ({c['last_buy_curr']} → {c['first_buy_next']})"
            )
    else:
        logger.info(f"  ✅ No trade activity gaps between consecutive windows.")

    if intra_window_gaps:
        logger.warning(f"  ❌ Trade activity gaps within windows ({len(intra_window_gaps)}):")
        for g in intra_window_gaps:
            logger.warning(
                f"     Window {g['wfo_window']}: largest gap {g['gap_days']} day(s) "
                f"({g['before']} → {g['after']})"
            )
    else:
        logger.info(f"  ✅ No trade activity gaps within windows.")

    logger.info(f"{'─'*115}\n")


# =============================================================================
# PUBLIC API
# =============================================================================

def validate_no_overlap_or_gaps(
    wfo_test_trades: pd.DataFrame,
    df_results: pd.DataFrame,
    gap_threshold_days: int = 3,
) -> dict:
    """
    Validate WFO test trades for out-of-bounds buy_time, cross-window duplicates,
    empty (gap) windows, and real trade-activity gaps (cross-window and intra-window).
    Logs a full report and returns a summary dict.
    """
    if wfo_test_trades is None or wfo_test_trades.empty:
        logger.warning("WFO test trades are empty — skipping overlap/gap validation.")
        return {"passed": False, "reason": "empty_trades"}

    n_windows = len(df_results) - 1  # last row is summary

    bound_violations  = _check_window_bounds(wfo_test_trades, df_results)
    duplicates        = _check_duplicates(wfo_test_trades)
    gaps              = _check_gaps(wfo_test_trades, df_results)
    cross_window_gaps = _check_cross_window_gap(wfo_test_trades, n_windows, gap_threshold_days)
    intra_window_gaps = _check_intra_window_gaps(wfo_test_trades, n_windows, gap_threshold_days)

    _log_overlap_report(bound_violations, duplicates, gaps, cross_window_gaps, intra_window_gaps)

    return {
        "passed":            not bound_violations and duplicates.empty
                              and not cross_window_gaps and not intra_window_gaps,
        "bound_violations":  bound_violations,
        "duplicates":        duplicates,
        "gaps":              gaps,
        "cross_window_gaps": cross_window_gaps,
        "intra_window_gaps": intra_window_gaps,
    }


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

    # ---- Overlap / duplicate / gap check across ALL windows -----------------
    validate_no_overlap_or_gaps(wfo_test_trades, df_results)

    # ---- Baseline comparison for the requested window -----------------------
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