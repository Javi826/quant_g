#shared_batchs/pipeline/multiverse.py
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from shared_config import VOLUME_COL
import matplotlib.pyplot as plt
from shared_batchs.backtesters.ZX_compute_BT import prepare_backtest_data, run_backtest_from_prepared_light
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays

logger = logging.getLogger("BOT_batch.pipeline.multiverse")

# =============================================================================
# MCPT EXECUTION CONFIG
# =============================================================================

N_PERMUTATIONS  = 1000
BLOCK_SIZE      = 20   # 1 = simple bootstrap with replacement (no block structure).
N_JOBS          = -1

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
def _synthetic_ohlcv_arr(paths_per_symbol: dict, path_idx: int, ts_index: np.ndarray, dtype) -> dict:
    ts64 = ts_index.astype("datetime64[ns]")

    ohlcv_arr = {}
    for sym, arr_paths in paths_per_symbol.items():
        if path_idx >= arr_paths.shape[0]:
            continue
        arr = arr_paths[path_idx]  # (n_obs, n_features)
        ohlcv_arr[sym] = {
            "ts":        ts64,
            "open":      arr[:, 0].astype(np.float64),
            "high":      arr[:, 2].astype(np.float64),
            "low":       arr[:, 1].astype(np.float64),
            "close":     arr[:, 3].astype(np.float64),
            VOLUME_COL:  arr[:, 7].astype(np.float64),
            "low_time":  np.array(arr[:, 4], dtype="datetime64[ns]"),
            "high_time": np.array(arr[:, 5], dtype="datetime64[ns]"),
        }
    return ohlcv_arr


def _evaluate_universe(
    path_idx: int,
    paths: dict,
    ts_index: np.ndarray,
    n_symbols_expected: int,
    signal_fn: callable,
    best_params: dict,
    order_amount: int,
    dtype,
) -> tuple:

    synthetic_arr = _synthetic_ohlcv_arr(paths, path_idx, ts_index, dtype)
    if len(synthetic_arr) < n_symbols_expected:
        return None, None

    ohlcv_arrays = {}
    for sym, arr in synthetic_arr.items():
        signals = signal_fn(arr, live_trading=False)
        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=dtype)}

    prepared_data = prepare_backtest_data(ohlcv_arrays)
    results = run_backtest_from_prepared_light(
        prepared_data,
        sell_after   = best_params["SELL_AFTER"],
        tp_pct       = best_params["TP_PCT"],
        sl_pct       = best_params["SL_PCT"],
        order_amount = order_amount,
    )

    trade_log = results["__PORTFOLIO__"]["trade_log"]
    if trade_log is None or trade_log.empty:
        return True, 0.0, False

    profit_sum = float(trade_log["profit"].sum())
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
    signal_fn: callable,
    best_params: dict,
    order_amount: int,
    dtype,
    real_profit: float,
    p_value_th: float,
    n_paths: int = N_PERMUTATIONS,
    block_size: int = BLOCK_SIZE,
    n_jobs: int = N_JOBS,
) -> tuple:
    """Runs MCPT for a single rule, using its fixed best_params (no re-optimization
    per synthetic universe). Returns (approved, p_value)."""

    if not ohlcv_data:
        return False, 1.0

    ref_sym  = max(ohlcv_data.keys(), key=lambda sym: len(ohlcv_data[sym]))
    n_obs    = len(ohlcv_data[ref_sym])
    ts_index = ohlcv_data[ref_sym].index[:n_obs].to_numpy()

    paths = _generate_mcpt_paths_all_symbols(
        ohlcv_data, n_paths=n_paths, raw_columns=[VOLUME_COL], dtype=dtype, block_size=block_size,
    )

    n_symbols_expected = len(ohlcv_data)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_universe)(
            path_idx, paths, ts_index, n_symbols_expected,
            signal_fn, best_params, order_amount, dtype,
        )
        for path_idx in range(n_paths)
    )

    permuted_profits = [r[1] for r in results if r[0] is not None]
    n_valid           = len(permuted_profits)
    if n_valid == 0:
        return False, 1.0

    p_value  = _compute_p_value(real_profit, permuted_profits)
    approved = p_value <= p_value_th

    if logger.isEnabledFor(logging.DEBUG):
        _log_multiverse_debug(ohlcv_data, paths, results, n_valid, n_paths, block_size, real_profit, p_value, approved)

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
    for r in tqdm(rules, desc="MULTIVERSE", dynamic_ncols=True):
        approved, p_value = _evaluate_multiverse(
            ohlcv_data   = ohlcv_data_by_timeframe[r["timeframe"]],
            signal_fn    = r["signal_fn"],
            best_params  = r["best_params"],
            order_amount = order_amount,
            dtype        = dtype,
            real_profit  = float(r["wfo_test_trades"]["profit"].sum()),
            p_value_th   = p_value_th,
            n_paths      = n_paths,
            block_size   = block_size,
            n_jobs       = n_jobs,
        )
        results.append({
            **r,
            "passed_multiverse":  approved,
            "multiverse_p_value": p_value,
        })

    return results

def _log_multiverse_debug(
    ohlcv_data: dict,
    paths: dict,
    results: list,
    n_valid: int,
    n_paths: int,
    block_size: int,
    real_profit: float,
    p_value: float,
    approved: bool,
) -> None:
    """Single entry point for all multiverse debug logging, gated by logger level."""
    _log_drift_analysis(ohlcv_data, paths)
    _plot_synthetic_vs_historical(ohlcv_data, paths)

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

    logger.debug(
        f"MCPT ── n_paths={n_paths} block_size={block_size} valid_universes={n_valid} "
        f"real_profit={real_profit:.2f} p_value={p_value:.4f} -> {'PASS' if approved else 'FAIL'}"
    )