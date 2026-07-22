#shared_batchs/pipeline/multiverse.py
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shared_config import VOLUME_COL
from shared_batchs.pipeline.wfo import run_wfo_is
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays

logger = logging.getLogger("BOT_batch.pipeline.multiverse")

# =============================================================================
# MCPT EXECUTION CONFIG
# =============================================================================

N_PERMUTATIONS  = 1000
BLOCK_SIZE      = 20   # 1 = simple bootstrap with replacement (no block structure).
N_JOBS          = -1
DEBUG_MAX_PATHS = 1

# === DEBUG BLOCK: DRIFT ANALYSIS (remove entire block to disable) ===
DEBUG_DRIFT_ANALYSIS = False

import matplotlib.pyplot as plt


def _plot_synthetic_vs_historical(ohlcv_data: dict, paths: dict) -> None:
    for sym, df_hist in ohlcv_data.items():
        arr_paths = paths.get(sym)
        if arr_paths is None or arr_paths.shape[0] == 0:
            continue

        hist_close  = df_hist["close"].to_numpy(dtype=np.float64)
        synth_close = arr_paths[:, :, 3].astype(np.float64)

        fig, ax = plt.subplots(figsize=(12, 5))
        for path_idx in range(synth_close.shape[0]):
            ax.plot(synth_close[path_idx], color="gray", alpha=0.15, linewidth=0.7)
        ax.plot(hist_close, color="red", linewidth=1.8, label="Historical close")

        ax.set_title(f"{sym} — historical vs MCPT permuted paths (n_paths={synth_close.shape[0]})")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Close price")
        ax.legend()
        fig.tight_layout()
        plt.show()


def _log_drift_analysis(ohlcv_data: dict, paths: dict) -> None:
    rows = []
    for sym, df_hist in ohlcv_data.items():
        arr_paths = paths.get(sym)
        if arr_paths is None or arr_paths.shape[0] == 0:
            continue

        hist_close = df_hist["close"].to_numpy(dtype=np.float64)
        hist_n_bars        = len(hist_close)
        hist_total_ret_pct = float((hist_close[-1] / hist_close[0] - 1.0) * 100.0)

        synth_close = arr_paths[:, :, 3].astype(np.float64)
        synth_n_bars           = arr_paths.shape[1]
        synth_total_ret_pct    = (synth_close[:, -1] / synth_close[:, 0] - 1.0) * 100.0
        synth_pct_paths_positive = float(np.mean(synth_total_ret_pct > 0) * 100.0)

        rows.append({
            "symbol":                   sym,
            "hist_n_bars":              hist_n_bars,
            "synth_n_bars":             synth_n_bars,
            "hist_total_ret_pct":       hist_total_ret_pct,
            "synth_pct_paths_positive": synth_pct_paths_positive,
        })

    if not rows:
        logger.warning("DRIFT ANALYSIS ── no valid symbols to analyze")
        return

    df_drift = pd.DataFrame(rows)
    summary  = df_drift.drop(columns=["symbol"]).mean()
    df_drift = pd.concat(
        [df_drift, pd.DataFrame([{"symbol": "MEAN", **summary.to_dict()}])],
        ignore_index=True,
    )

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    logger.info(f"\n{'─' * 115}")
    logger.info("  DRIFT ANALYSIS ── historical vs MCPT permuted paths")
    logger.info(f"{'─' * 115}")
    logger.info(f"\n{df_drift.to_string(index=False)}")
    logger.info(f"{'─' * 115}\n")
# === END DEBUG BLOCK ===


# =============================================================================
# MCPT PATH GENERATION — moving block bootstrap (overlapping blocks + replacement)
# =============================================================================
def _compute_log_features(df: pd.DataFrame, raw_columns: list) -> tuple:
    df = df.copy()
    prev_close = df["close"].shift(1)
    prev_close.iloc[0] = df["open"].iloc[0]

    df["log_ret_close"]  = np.log(df["close"] / prev_close)
    df["log_open_low"]   = np.log(df["low"]   / df["open"])
    df["log_open_high"]  = np.log(df["high"]  / df["open"])
    df["log_open_close"] = np.log(df["close"] / df["open"])

    if len(df.index) >= 2:
        time_deltas = (df.index[1:] - df.index[:-1]).total_seconds()
        mode = pd.Series(time_deltas).mode()[0]
        time_deltas = np.insert(time_deltas, 0, mode)
    else:
        time_deltas = np.zeros(len(df.index))
    df["time_variation"] = time_deltas

    index_sec = df.index.view(np.int64) // 10**9
    low_sec   = pd.to_datetime(df["low_time"]).view(np.int64) // 10**9
    high_sec  = pd.to_datetime(df["high_time"]).view(np.int64) // 10**9
    df["var_low_time"]  = (low_sec  - index_sec).astype(float)
    df["var_high_time"] = (high_sec - index_sec).astype(float)

    df_raw = df[raw_columns].copy() if raw_columns else pd.DataFrame(index=df.index)
    return df, df_raw


def _make_overlapping_row_blocks(data_array: np.ndarray, block_size: int) -> np.ndarray:
    """Returns array of shape (n_blocks, block_size, n_features) — sliding windows over rows."""
    windows = np.lib.stride_tricks.sliding_window_view(data_array, block_size, axis=0)
    return np.moveaxis(windows, -1, 1)


def _block_bootstrap_sample(
    data_array: np.ndarray,
    n_rows: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Moving block bootstrap: overlapping blocks sampled with replacement, concatenated and truncated to n_rows."""
    if block_size <= 1:
        chosen_idx = rng.integers(0, n_rows, size=n_rows)
        return data_array[chosen_idx]

    blocks   = _make_overlapping_row_blocks(data_array, block_size)
    n_blocks = blocks.shape[0]
    n_blocks_needed = int(np.ceil(n_rows / block_size))
    chosen = rng.integers(0, n_blocks, size=n_blocks_needed)
    return np.concatenate(blocks[chosen], axis=0)[:n_rows]


def _generate_mcpt_paths(
    df_hist: pd.DataFrame,
    n_paths: int,
    raw_columns: list,
    base_seed: int,
    dtype,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    df_features, df_raw = _compute_log_features(df_hist, raw_columns)
    n_rows = len(df_features)
    if n_rows == 0:
        return np.empty((0, 0, 0))

    cols = [
        df_features["log_ret_close"].to_numpy(np.float64),
        df_features["log_open_low"].to_numpy(np.float64),
        df_features["log_open_high"].to_numpy(np.float64),
        df_features["log_open_close"].to_numpy(np.float64),
        df_features["time_variation"].to_numpy(np.float64),
        df_features["var_low_time"].to_numpy(np.float64),
        df_features["var_high_time"].to_numpy(np.float64),
    ]
    for rc in raw_columns:
        cols.append(df_raw[rc].to_numpy(np.float64))
    data_array = np.column_stack(cols)
    n_raw          = data_array.shape[1] - 7
    n_features_out = 7 + n_raw

    start_price     = float(df_features["open"].iloc[0])
    start_timestamp = df_features.index[0].value // 10**9

    effective_block_size = min(block_size, n_rows)

    paths_array = np.empty((n_paths, n_rows, n_features_out), dtype=np.float64)

    for i in range(n_paths):
        rng     = np.random.default_rng(base_seed + i)
        sampled = _block_bootstrap_sample(data_array, n_rows, effective_block_size, rng)

        log_ret_close  = sampled[:, 0]
        log_open_low   = sampled[:, 1]
        log_open_high  = sampled[:, 2]
        log_open_close = sampled[:, 3]

        close_prices = start_price * np.exp(np.cumsum(log_ret_close))
        open_prices  = close_prices * np.exp(-log_open_close)
        low_prices   = open_prices  * np.exp(log_open_low)
        high_prices  = open_prices  * np.exp(log_open_high)

        cumul_seconds = np.cumsum(sampled[:, 4])
        times      = start_timestamp + cumul_seconds
        low_times  = times + sampled[:, 5]
        high_times = times + sampled[:, 6]

        base_cols = [open_prices, low_prices, high_prices, close_prices, low_times, high_times, times]
        if n_raw > 0:
            for idx_col in range(n_raw):
                base_cols.append(sampled[:, 7 + idx_col])
        paths_array[i, :, :] = np.column_stack(base_cols)

    return paths_array.astype(dtype, copy=False)


def _generate_mcpt_paths_all_symbols(
    ohlcv_data: dict,
    n_paths: int,
    raw_columns: list,
    dtype,
    base_seed: int = 42,
    block_size: int = BLOCK_SIZE,
) -> dict:
    paths_per_symbol = {}
    for symbol, df_hist in ohlcv_data.items():
        arr_paths = _generate_mcpt_paths(
            df_hist, n_paths=n_paths, raw_columns=raw_columns, base_seed=base_seed, dtype=dtype,
            block_size=block_size,
        )
        if arr_paths is not None and arr_paths.shape[0] > 0:
            paths_per_symbol[symbol] = arr_paths
    return paths_per_symbol


# =============================================================================
# PRIVATE HELPERS
# =============================================================================
def _synthetic_ohlcv_data(paths_per_symbol: dict, path_idx: int, ts_index: np.ndarray, dtype) -> dict:

    ohlcv_data = {}
    for sym, arr_paths in paths_per_symbol.items():
        if path_idx >= arr_paths.shape[0]:
            continue
        arr = arr_paths[path_idx]  # (n_obs, n_features)
        ohlcv_data[sym] = pd.DataFrame(
            {
                "open":       arr[:, 0].astype(dtype),
                "low":        arr[:, 1].astype(dtype),
                "high":       arr[:, 2].astype(dtype),
                "close":      arr[:, 3].astype(dtype),
                "low_time":   np.array(arr[:, 4], dtype="datetime64[ns]"),
                "high_time":  np.array(arr[:, 5], dtype="datetime64[ns]"),
                VOLUME_COL:   arr[:, 7].astype(dtype),
            },
            index=pd.DatetimeIndex(ts_index),
        )
    return ohlcv_data


def _evaluate_universe(
    path_idx: int,
    paths: dict,
    ts_index: np.ndarray,
    n_symbols_expected: int,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    timeframe: str,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    dtype,
    n_symbols: int,
) -> tuple:

    synthetic_ohlcv = _synthetic_ohlcv_data(paths, path_idx, ts_index, dtype)
    if len(synthetic_ohlcv) < n_symbols_expected:
        return None, None

    synthetic_arr = prepare_ohlcv_arrays(synthetic_ohlcv)

    (
        _best_params, _approved_wfo, _net_gain, _max_dd, wfo_test_trades, _df_results, _wfr, _metrics,
    ) = run_wfo_is(
        ohlcv_arr            = synthetic_arr,
        param_names          = param_names,
        lists_for_grid       = lists_for_grid,
        signal_fn            = signal_fn,
        signal_params_keys   = signal_params_keys,
        order_amount         = order_amount,
        timeframe            = timeframe,
        net_gain_th          = net_gain_th,
        dd_th                = dd_th,
        r2_th                = r2_th,
        wfr_th               = wfr_th,
        dtype                = dtype,
        n_jobs               = 1,
        show_progress        = False,
        n_symbols            = n_symbols,
    )

    if wfo_test_trades is None or wfo_test_trades.empty:
        if path_idx < DEBUG_MAX_PATHS:
            logger.debug(f"MCPT path={path_idx} ── no test trades ── profit_sum=0.0")
        return True, 0.0, False

    profit_sum = float(wfo_test_trades["profit"].sum())

    if path_idx < DEBUG_MAX_PATHS:
        per_window = wfo_test_trades.groupby("wfo_window")["profit"].sum()
        window_breakdown = " | ".join(f"w{w}={p:.2f}" for w, p in per_window.items())
        logger.debug(
            f"MCPT path={path_idx} ── {len(per_window)} windows with trades ── "
            f"{window_breakdown} ── TOTAL={profit_sum:.2f}"
        )

    return True, profit_sum, True


# =============================================================================
# APPROVAL CRITERION — Monte Carlo Permutation Test p-value
# =============================================================================
def _compute_p_value(real_profit: float, permuted_profits: list) -> float:
    n_matching_or_beating = sum(1 for p in permuted_profits if p >= real_profit)
    return n_matching_or_beating / len(permuted_profits)


# =============================================================================
# CORE MULTIVERSE EVALUATION (single rule)
# =============================================================================
def _evaluate_multiverse(
    ohlcv_data: dict,
    timeframe: str,
    param_grid: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    dtype,
    n_symbols: int,
    real_profit: float,
    p_value_th: float,
    n_paths: int = N_PERMUTATIONS,
    block_size: int = BLOCK_SIZE,
    n_jobs: int = N_JOBS,
) -> tuple:
    """Runs MCPT for a single rule. Returns (approved, p_value)."""

    if not ohlcv_data:
        return False, 1.0

    ref_sym  = max(ohlcv_data.keys(), key=lambda sym: len(ohlcv_data[sym]))
    n_obs    = len(ohlcv_data[ref_sym])
    ts_index = ohlcv_data[ref_sym].index[:n_obs].to_numpy()

    paths = _generate_mcpt_paths_all_symbols(
        ohlcv_data, n_paths=n_paths, raw_columns=[VOLUME_COL], dtype=dtype, block_size=block_size,
    )

    # === DEBUG BLOCK: DRIFT ANALYSIS (remove entire block to disable) ===
    if DEBUG_DRIFT_ANALYSIS:
        _log_drift_analysis(ohlcv_data, paths)
        _plot_synthetic_vs_historical(ohlcv_data, paths)
    # === END DEBUG BLOCK ===

    param_names    = list(param_grid.keys())
    lists_for_grid = [param_grid[k] for k in param_names]
    n_symbols_expected = len(ohlcv_data)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_universe)(
            path_idx, paths, ts_index, n_symbols_expected, param_names, lists_for_grid,
            signal_fn, signal_params_keys, order_amount, timeframe,
            net_gain_th, dd_th, r2_th, wfr_th, dtype, n_symbols,
        )
        for path_idx in range(n_paths)
    )

    permuted_profits = [r[1] for r in results if r[0] is not None]
    n_valid           = len(permuted_profits)
    if n_valid == 0:
        return False, 1.0

    # === DEBUG: no-trades vs with-trades path breakdown ===
    n_no_trades   = sum(1 for r in results if r[0] is not None and not r[2])
    n_with_trades = sum(1 for r in results if r[0] is not None and r[2])
    pct_no_trades = n_no_trades / n_valid * 100.0
    logger.debug(
        f"MCPT DEBUG ── no_trades_paths={n_no_trades}/{n_valid} ({pct_no_trades:.1f}%) "
        f"with_trades_paths={n_with_trades}/{n_valid}"
    )
    if n_with_trades > 0:
        with_trades_profits = [r[1] for r in results if r[0] is not None and r[2]]
        logger.debug(
            f"MCPT DEBUG ── with_trades profit_sum stats: "
            f"min={min(with_trades_profits):.2f} max={max(with_trades_profits):.2f} "
            f"mean={float(np.mean(with_trades_profits)):.2f}"
        )
    # === END DEBUG ===

    p_value  = _compute_p_value(real_profit, permuted_profits)
    approved = p_value <= p_value_th

    logger.debug(
        f"MCPT ── n_paths={n_paths} block_size={block_size} valid_universes={n_valid} "
        f"real_profit={real_profit:.2f} p_value={p_value:.4f} -> {'PASS' if approved else 'FAIL'}"
    )
    return approved, p_value


# =============================================================================
# PIPE MULTIVERSE — evaluates every rule's WFO test trades independently
# =============================================================================
def _empty_multiverse_fields() -> dict:
    """Placeholder Multiverse fields for rules that were never evaluated (pipe disabled)."""
    return {
        "passed_multiverse":  True,
        "multiverse_p_value": 0.0,
    }


def pipe_multiverse(
    rules: list,
    ohlcv_data_by_timeframe: dict,
    param_grid: dict,
    order_amount: int,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    dtype,
    n_symbols: int,
    p_value_th: float,
    enabled: bool = True,
    n_paths: int = N_PERMUTATIONS,
    block_size: int = BLOCK_SIZE,
    n_jobs: int = N_JOBS,
) -> list:


    if not enabled:
        logger.info(f"MULTIVERSE ── disabled — passing all {len(rules)} rules through untouched")
        return [{**r, **_empty_multiverse_fields()} for r in rules]

    results = []
    for r in rules:
        approved, p_value = _evaluate_multiverse(
            ohlcv_data          = ohlcv_data_by_timeframe[r["timeframe"]],
            timeframe           = r["timeframe"],
            param_grid          = param_grid,
            signal_fn           = r["signal_fn"],
            signal_params_keys  = [],
            order_amount        = order_amount,
            net_gain_th         = net_gain_th,
            dd_th               = dd_th,
            r2_th               = r2_th,
            wfr_th              = wfr_th,
            dtype               = dtype,
            n_symbols           = n_symbols,
            real_profit         = float(r["wfo_test_trades"]["profit"].sum()),
            p_value_th          = p_value_th,
            n_paths             = n_paths,
            block_size          = block_size,
            n_jobs              = n_jobs,
        )
        results.append({
            **r,
            "passed_multiverse":  approved,
            "multiverse_p_value": p_value,
        })

    return results