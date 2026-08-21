#shared_batchs/pipeline/multiverse.py
import os
import time
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from shared_config import VOLUME_COL
from shared_batchs.backtesters.ZX_compute_BT import prepare_static_arrays,prepare_signal_arrays,run_backtest_from_prepared_light
from shared_batchs.utils.reporting import report_multiverse_debug
logger = logging.getLogger("BOT_batch.pipeline.multiverse")
DTYPE  = np.float32
# =============================================================================
# MCPT EXECUTION CONFIG
# =============================================================================
MULTIVERSE_PVALUE_TH    = 0.10
BLOCK_SIZE_BY_TIMEFRAME = {
    "1H":     150, #1H-400 -> script
    "4H":     120,
    "6Hutc":  70,
    "12Hutc": 30,
}

N_PERMUTATIONS       = 1000
MCPT_N_JOBS          = -1

def _resolve_block_size(timeframe: str) -> int:
    block_size = BLOCK_SIZE_BY_TIMEFRAME.get(timeframe)
    if block_size is None:
        raise ValueError(
            f"MULTIVERSE ── timeframe={timeframe!r} has no entry in "
            f"BLOCK_SIZE_BY_TIMEFRAME — add it explicitly, no fallback is defined."
        )
    return block_size

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

    n_blocks = n_rows - block_size + 1
    n_blocks_needed = int(np.ceil(n_rows / block_size))
    chosen = rng.integers(0, n_blocks, size=n_blocks_needed)

    n_features = data_array.shape[1]
    out = np.empty((n_blocks_needed * block_size, n_features), dtype=data_array.dtype)
    for k, start in enumerate(chosen):
        out[k * block_size:(k + 1) * block_size] = data_array[start:start + block_size]
    return out[:n_rows]

def _evaluate_universe_batch_chunk(
    path_indices: list,
    paths: dict,
    ts_index: np.ndarray,
    n_symbols_expected: int,
    rules: list,
    order_amount: int,
) -> list:
    return [
        _evaluate_universe_batch(
            path_idx, paths, ts_index, n_symbols_expected,
            rules, order_amount,
        )
        for path_idx in path_indices
    ]


def _generate_mcpt_paths(
    df_hist: pd.DataFrame,
    n_paths: int,
    raw_columns: list,
    base_seed: int,
    block_size: int,
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

    return paths_array.astype(DTYPE, copy=False)

def _generate_mcpt_paths_all_symbols(
    ohlcv_data: dict,
    n_paths: int,
    raw_columns: list,
    block_size: int,
    base_seed: int = 42,
) -> dict:
    
    
    paths_per_symbol = {}
    for symbol, df_hist in ohlcv_data.items():
        arr_paths = _generate_mcpt_paths(
            df_hist, n_paths=n_paths, raw_columns=raw_columns, base_seed=base_seed,
            block_size=block_size,
        )
        if arr_paths is not None and arr_paths.shape[0] > 0:
            paths_per_symbol[symbol] = arr_paths
    return paths_per_symbol

# =============================================================================
# PRIVATE HELPERS
# =============================================================================
def _synthetic_ohlcv_arr(paths_per_symbol: dict, path_idx: int, ts_index: np.ndarray) -> dict:
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

# DESPUÉS
def _evaluate_universe_batch(
    path_idx: int,
    paths: dict,
    ts_index: np.ndarray,
    n_symbols_expected: int,
    rules: list,
    order_amount: int,
) -> dict:

    synthetic_arr = _synthetic_ohlcv_arr(paths, path_idx, ts_index)

    if len(synthetic_arr) < n_symbols_expected:
        return {r["rule_id"]: (None, None, None) for r in rules}

    static_bundle = prepare_static_arrays(synthetic_arr)

    results = {}
    for r in rules:
        best_params = r["best_params"]

        ohlcv_arrays = {}
        for sym, arr in synthetic_arr.items():
            signals = r["signal_fn"](arr, live_trading=False)
            ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=DTYPE)}

        prepared_data = prepare_signal_arrays(static_bundle, ohlcv_arrays)
        bt_results = run_backtest_from_prepared_light(
            prepared_data,
            sell_after   = best_params["SELL_AFTER"],
            tp_pct       = best_params["TP_PCT"],
            sl_pct       = best_params["SL_PCT"],
            order_amount = order_amount,
        )

        trade_log = bt_results["__PORTFOLIO__"]["trade_log"]
        if trade_log is None or trade_log.empty:
            results[r["rule_id"]] = (True, 0.0, False)
        else:
            profit_sum = float(trade_log["profit"].sum())
            results[r["rule_id"]] = (True, profit_sum, True)

    return results

# =============================================================================
# APPROVAL CRITERION — Monte Carlo Permutation Test p-value
# =============================================================================
def _compute_p_value(real_profit: float, permuted_profits: list) -> float:
    n_matching_or_beating = sum(1 for p in permuted_profits if p >= real_profit)
    return n_matching_or_beating / len(permuted_profits)

# =============================================================================
# CORE MULTIVERSE EVALUATION (one timeframe, every rule of that timeframe)
# =============================================================================
def _evaluate_multiverse_batch(
    ohlcv_data: dict,
    rules: list,
    order_amount: int,
    p_value_th: float,
    block_size: int,
    n_paths: int = N_PERMUTATIONS,
    n_jobs: int = MCPT_N_JOBS,
    timeframe: str = "",
) -> tuple:

    if not ohlcv_data or not rules:
        return (
            {r["rule_id"]: 1.0   for r in rules},
            {r["rule_id"]: False for r in rules},
        )

    ref_sym  = max(ohlcv_data.keys(), key=lambda sym: len(ohlcv_data[sym]))
    n_obs    = len(ohlcv_data[ref_sym])
    ts_index = ohlcv_data[ref_sym].index[:n_obs].to_numpy()

    paths = _generate_mcpt_paths_all_symbols(
        ohlcv_data, n_paths=n_paths, raw_columns=[VOLUME_COL], block_size=block_size,
    )

    n_symbols_expected = len(ohlcv_data)

    desc = f"MULTIVERSE {timeframe}".strip().ljust(18)
    n_workers  = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    n_chunks   = min(n_paths, n_workers)
    path_chunks = [chunk for chunk in np.array_split(np.arange(n_paths), n_chunks) if len(chunk) > 0]

    chunked_results = list(tqdm(
        Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_evaluate_universe_batch_chunk)(
                chunk.tolist(), paths, ts_index, n_symbols_expected,
                rules, order_amount,
            )
            for chunk in path_chunks
        ),
        desc=desc,
        total=len(path_chunks),
        dynamic_ncols=True,
    ))
    per_path_results = [res for chunk_res in chunked_results for res in chunk_res]

    p_value_by_id  = {}
    approved_by_id = {}
    for r in rules:
        rid = r["rule_id"]
        permuted_profits = [res[rid][1] for res in per_path_results if res[rid][0] is not None]
        n_valid = len(permuted_profits)

        if n_valid == 0:
            p_value_by_id[rid]  = 1.0
            approved_by_id[rid] = False
            continue

        real_profit = float(r["wfo_test_trades"]["profit"].sum())
        p_value     = _compute_p_value(real_profit, permuted_profits)
        approved    = p_value <= p_value_th

        p_value_by_id[rid]  = p_value
        approved_by_id[rid] = approved

    if logger.isEnabledFor(logging.DEBUG):
        report_multiverse_debug(
            ohlcv_data, paths, per_path_results, rules,
            p_value_by_id, approved_by_id, n_paths, block_size,
        )

    return p_value_by_id, approved_by_id
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
    p_value_th: float = None,
    enabled: bool = True,
    n_paths: int = N_PERMUTATIONS,
    block_size: int | None = None,
    n_jobs: int = MCPT_N_JOBS,
) -> list:

    p_value_th = p_value_th if p_value_th is not None else MULTIVERSE_PVALUE_TH

    start = time.time()

    if not enabled:
        logger.info(f"MULTIVERSE ── disabled — passing all {len(rules)} rules through untouched")
        return [{**r, **_empty_multiverse_fields()} for r in rules]

    rules_by_timeframe: dict = {}
    for r in rules:
        rules_by_timeframe.setdefault(r["timeframe"], []).append(r)

    p_value_by_id:  dict = {}
    approved_by_id: dict = {}

    for timeframe, tf_rules in rules_by_timeframe.items():
        resolved_block_size = block_size if block_size is not None else _resolve_block_size(timeframe)

        tf_p_values, tf_approved = _evaluate_multiverse_batch(
            ohlcv_data   = ohlcv_data_by_timeframe[timeframe],
            rules        = tf_rules,
            order_amount = order_amount,
            p_value_th   = p_value_th,
            n_paths      = n_paths,
            block_size   = resolved_block_size,
            n_jobs       = n_jobs,
            timeframe    = timeframe,
        )
        p_value_by_id.update(tf_p_values)
        approved_by_id.update(tf_approved)

    results = [
        {
            **r,
            "passed_multiverse":  approved_by_id[r["rule_id"]],
            "multiverse_p_value": p_value_by_id[r["rule_id"]],
        }
        for r in rules
    ]

    elapsed = int(time.time() - start)
    logger.info(f"\nMULTIVERSE ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return results