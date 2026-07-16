#shared/shared_batchs/engines/wfo_WF.py
import logging
import contextlib
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from multiprocessing.shared_memory import SharedMemory
from shared_config import VOLUME_COL
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE

logger = logging.getLogger("BOT_batch.engines.wfo_WF")

EMA_ALPHA   = 1.0
WARMUP_BARS = 100

WFO_MODES = ("wfo", "wfo_ema")

# =============================================================================
# PARAM ROUNDING HELPERS
# =============================================================================

def _decimals_for_values(values) -> int:
    """Max number of decimal places present in a list of numeric param values."""
    max_decimals = 0
    for v in values:
        s = f"{float(v):.10f}".rstrip("0")
        if "." in s:
            max_decimals = max(max_decimals, len(s.split(".")[1]))
    return max_decimals


def _round_param(value, decimals: int):
    """Round a value to match the precision of its original param grid."""
    return int(round(value)) if decimals == 0 else round(float(value), decimals)


def _round_params_dict(params: dict, param_ranges: dict) -> dict:
    return {
        k: _round_param(v, _decimals_for_values(param_ranges[k]))
        for k, v in params.items()
    }


# =============================================================================
# EMA STATE — running exponential moving average of per-window optimal params
# =============================================================================

def _update_ema_state(ema_raw: dict | None, new_best: dict, alpha: float = EMA_ALPHA) -> dict:

    if ema_raw is None:
        return dict(new_best)
    return {k: alpha * new_best[k] + (1.0 - alpha) * ema_raw[k] for k in new_best}


# =============================================================================
# WINDOW SYMBOL SELECTION
# =============================================================================
def _find_window_indices(
    sym_ts: np.ndarray,
    train_start_ts,
    test_start_ts,
    test_end_ts,
) -> tuple | None:

    if sym_ts[0] > train_start_ts or sym_ts[-1] < test_end_ts:
        return None

    t0    = int(np.searchsorted(sym_ts, train_start_ts, side="left"))
    t1    = int(np.searchsorted(sym_ts, test_start_ts,  side="left"))
    test0 = t1
    test1 = int(np.searchsorted(sym_ts, test_end_ts,    side="right"))

    if t1 <= t0 or test1 <= test0:
        return None
    return t0, t1, test0, test1

def _select_window_symbols(
    candidate_indices: dict,
    ohlcv_arr: dict,
    n_symbols: int | None,
) -> dict:

    if n_symbols is None:
        return candidate_indices

    def _avg_train_vol(sym: str) -> float:
        t0, t1, _, _ = candidate_indices[sym]
        arr = ohlcv_arr[sym]
        vol = arr.get(VOLUME_COL, arr["close"] * 0)
        return float(np.mean(vol[t0:t1])) if t1 > t0 else 0.0

    selected_syms = sorted(candidate_indices, key=_avg_train_vol, reverse=True)[:n_symbols]
    return {sym: candidate_indices[sym] for sym in selected_syms}

# =============================================================================
# SHARED MEMORY HELPERS
# =============================================================================
def _arrays_to_shared_memory(base_arrays: dict) -> tuple:
    """Copy base_arrays numpy arrays to shared memory. Returns (shm_list, metadata)."""
    shm_list = []
    metadata = {}
    for sym, arr_dict in base_arrays.items():
        metadata[sym] = {}
        for key, arr in arr_dict.items():
            if isinstance(arr, np.ndarray):
                shm      = SharedMemory(create=True, size=max(arr.nbytes, 1))
                buf      = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                buf[:]   = arr
                shm_list.append(shm)
                metadata[sym][key] = {"name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}
            else:
                metadata[sym][key] = {"value": arr}
    return shm_list, metadata

def _arrays_from_shared_memory(metadata: dict) -> tuple:
    """Reconstruct base_arrays from shared memory metadata. Returns (base_arrays, shm_handles)."""
    base_arrays = {}
    shm_handles = []
    for sym, fields in metadata.items():
        base_arrays[sym] = {}
        for key, info in fields.items():
            if "name" in info:
                shm = SharedMemory(name=info["name"], create=False)
                shm_handles.append(shm)
                base_arrays[sym][key] = np.ndarray(info["shape"], dtype=np.dtype(info["dtype"]), buffer=shm.buf)
            else:
                base_arrays[sym][key] = info["value"]
    return base_arrays, shm_handles

def _evaluate_with_shm(params: dict, shm_metadata: dict, evaluate_fn) -> tuple:
    """Worker: reconstruct base_arrays from shared memory and evaluate."""
    base_arrays, shm_handles = _arrays_from_shared_memory(shm_metadata)
    try:
        return evaluate_fn(params, base_arrays)
    finally:
        for shm in shm_handles:
            shm.close()
# =============================================================================
# WALK FORWARD OPTIMIZATION
# =============================================================================

def walk_forward_optimization(
    ohlcv_arr,
    param_ranges,
    length_train_set,
    pct_train_set,
    anchored,
    evaluate_fn,
    wfo_mode="wfo",
    n_jobs=-1,
    show_progress=False,
    n_symbols=None,
    collect_train_trades_fn=None,
    collect_test_trades_fn=None,
):
    if evaluate_fn is None:
        raise ValueError("You must pass an evaluate_fn(params, base_arrays) function")

    if wfo_mode not in WFO_MODES:
        raise ValueError(f"Unknown wfo_mode: {wfo_mode} (expected one of {WFO_MODES})")

    keys               = list(param_ranges.keys())
    all_combinations   = list(itertools.product(*[param_ranges[k] for k in keys]))
    dict_combinations  = [dict(zip(keys, comb)) for comb in all_combinations]

    length_test        = int(length_train_set / pct_train_set - length_train_set)
    best_params_list   = []
    best_criteria_list = []
    window_idx         = 1
    last_test_end_ref  = length_train_set + length_test

    train_start_dates  = []
    train_end_dates    = []
    test_start_dates   = []
    test_end_dates     = []
    train_symbols_list = []
    test_symbols_list  = []

    # Trade accumulators per window
    train_trades_list      = []
    test_trades_list       = []
    test_n_trades_list     = []
    window_test_arrays_list = []  # per-window test OHLCV, needed for synthetic-path (Multiverse) validation
    window_test_start_ts_list = []  # per-window real test_start_ts, needed to drop warmup trades downstream

    ema_raw = None  # running unrounded EMA state, only used when wfo_mode == "wfo_ema"

    start = 0
    end   = length_train_set

    ref_sym    = max(ohlcv_arr.keys(), key=lambda k: len(ohlcv_arr[k]['ts']))
    ref_ts     = ohlcv_arr[ref_sym]['ts']
    max_length = len(ref_ts)

    while start < max_length:
        remaining_data = max_length - (end if anchored else start)
        is_last_window = remaining_data < (length_train_set + length_test)

        # -----------------------------------------------------------------
        # Define window date boundaries from ref_sym
        # -----------------------------------------------------------------
        if is_last_window:
            remaining_from_last = max_length - last_test_end_ref
            if remaining_from_last < int(length_test * 0.5):
                break
            t0_ref    = 0 if anchored else start
            t1_ref    = last_test_end_ref
            test0_ref = t1_ref
            test1_ref = max_length
        else:
            t0_ref    = 0 if anchored else start
            t1_ref    = end if anchored else start + length_train_set
            test0_ref = t1_ref
            test1_ref = min(t1_ref + length_test, max_length)
            last_test_end_ref = test1_ref

        train_start_ts = ref_ts[t0_ref]
        test_start_ts  = ref_ts[t1_ref] if t1_ref < max_length else ref_ts[-1]
        test_end_ts    = ref_ts[test1_ref - 1]

        # -----------------------------------------------------------------
        # Collect all valid candidates using date-aligned indices
        # -----------------------------------------------------------------
        candidate_indices = {}
        for sym, arr_dict in ohlcv_arr.items():
            indices = _find_window_indices(
                arr_dict["ts"], train_start_ts, test_start_ts, test_end_ts
            )
            if indices is not None:
                candidate_indices[sym] = indices

        # -----------------------------------------------------------------
        # Select exactly n_symbols (OOS1 priority + fill by volume)
        # -----------------------------------------------------------------
        selected_indices = _select_window_symbols(
            candidate_indices, ohlcv_arr, n_symbols
        )

        train_indices = {sym: (t0, t1)       for sym, (t0, t1, _,     _    ) in selected_indices.items()}
        test_indices  = {sym: (test0, test1) for sym, (_,  _,  test0, test1) in selected_indices.items()}

        train_symbols_list.append(sorted(train_indices.keys()))
        test_symbols_list.append(sorted(test_indices.keys()))

        if not train_indices:
            break

        if ref_sym in selected_indices:
            t0, t1, test0, test1 = selected_indices[ref_sym]
        else:
            t0, t1, test0, test1 = t0_ref, t1_ref, test0_ref, test1_ref

        # -----------------------------------------------------------
        # Prepare base arrays (train + test) with warmup prefix
        # -----------------------------------------------------------
        base_arrays = {}
        for sym, (t0_sym, t1_sym) in train_indices.items():
            arr_dict   = ohlcv_arr[sym]
            warm_start = max(0, t0_sym - WARMUP_BARS)
            base_arrays[sym] = {
                'ts':        arr_dict['ts'][warm_start:t1_sym],
                'open':      arr_dict['open'][warm_start:t1_sym],
                'high':      arr_dict['high'][warm_start:t1_sym],
                'low':       arr_dict['low'][warm_start:t1_sym],
                'close':     arr_dict['close'][warm_start:t1_sym],
                VOLUME_COL:  arr_dict.get(VOLUME_COL, arr_dict['close'] * 0)[warm_start:t1_sym],
                'low_time':  arr_dict['low_time'][warm_start:t1_sym],
                'high_time': arr_dict['high_time'][warm_start:t1_sym],
            }

        base_arrays_test = {}
        for sym, (t0_sym, t1_sym) in test_indices.items():
            arr_dict   = ohlcv_arr[sym]
            warm_start = max(0, t0_sym - WARMUP_BARS)
            base_arrays_test[sym] = {
                'ts':        arr_dict['ts'][warm_start:t1_sym],
                'open':      arr_dict['open'][warm_start:t1_sym],
                'high':      arr_dict['high'][warm_start:t1_sym],
                'low':       arr_dict['low'][warm_start:t1_sym],
                'close':     arr_dict['close'][warm_start:t1_sym],
                VOLUME_COL:  arr_dict.get(VOLUME_COL, arr_dict['close'] * 0)[warm_start:t1_sym],
                'low_time':  arr_dict['low_time'][warm_start:t1_sym],
                'high_time': arr_dict['high_time'][warm_start:t1_sym],
            }

        window_test_arrays_list.append(base_arrays_test)
        window_test_start_ts_list.append(test_start_ts)

        # -----------------------------------------------------------
        # Parallel evaluation via shared memory
        # -----------------------------------------------------------
        if n_jobs == 1:
            # Sequential path: no inter-process parallelism needed, so avoid creating
            # multiprocessing.SharedMemory blocks entirely. This matters because this
            # function may itself run inside an already-forked worker process (e.g.
            # called from Multiverse's outer Parallel with n_jobs=1) — repeatedly
            # creating/destroying SharedMemory blocks inside nested worker processes,
            # thousands of times over a run, is a known source of memory corruption
            # and SIGSEGV crashes.
            results = [evaluate_fn(params, base_arrays) for params in dict_combinations]
        else:
            shm_list, shm_metadata = _arrays_to_shared_memory(base_arrays)
            try:
                with (tqdm_joblib(
                    tqdm(desc=f"🔁 WFO Window {window_idx}", total=len(dict_combinations), dynamic_ncols=True)
                ) if show_progress else contextlib.nullcontext()):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(_evaluate_with_shm)(params, shm_metadata, evaluate_fn)
                        for params in dict_combinations
                    )
            finally:
                for shm in shm_list:
                    shm.close()
                    shm.unlink()

        # -----------------------------------------------------------
        # Select best result (on train)
        # -----------------------------------------------------------
        _, raw_best_params = max(results, key=lambda x: x[0])

        # -----------------------------------------------------------
        # WFO_MODE:
        #   "wfo"      -> use this window's raw train optimum as-is (no memory)
        #   "wfo_ema"  -> use the running EMA of raw optima across windows 1..i
        #                 (window 1 starts the EMA at its own raw optimum)
        # In both cases, effective_params is what is actually applied to the test window.
        # -----------------------------------------------------------
        if wfo_mode == "wfo_ema":
            ema_raw          = _update_ema_state(ema_raw, raw_best_params, alpha=EMA_ALPHA)
            effective_params = _round_params_dict(ema_raw, param_ranges)
        else:
            effective_params = raw_best_params

        best_params_list.append(effective_params)

        window_test_n_trades = 0
        test_criterion       = np.nan

        df_train = None
        if collect_train_trades_fn is not None and base_arrays:
            df_train = collect_train_trades_fn(effective_params, base_arrays)
            if df_train is not None and not df_train.empty:
                df_train = df_train[df_train["buy_time"] >= pd.Timestamp(train_start_ts)].copy()

        df_test = None
        if collect_test_trades_fn is not None and base_arrays_test:
            df_test = collect_test_trades_fn(effective_params, base_arrays_test)
            if df_test is not None and not df_test.empty:
                df_test = df_test[df_test["buy_time"] >= pd.Timestamp(test_start_ts)].copy()

        train_has_trades = df_train is not None and not df_train.empty
        test_has_trades  = df_test is not None and not df_test.empty

        if train_has_trades and test_has_trades:
            df_train["wfo_window"] = window_idx
            train_trades_list.append(df_train)

            df_test["wfo_window"] = window_idx
            test_trades_list.append(df_test)
            window_test_n_trades = len(df_test)
            test_criterion       = float(df_test["profit"].sum()) / INITIAL_BALANCE * 100
        else:
            logger.debug(
                f"WFO window {window_idx} ── dropped from train/test WFR pool "
                f"(train_has_trades={train_has_trades}, test_has_trades={test_has_trades})"
            )

        best_criteria_list.append(test_criterion)
        test_n_trades_list.append(window_test_n_trades)

        train_start_dates.append(ref_ts[t0]       if t0       < len(ref_ts) else None)
        train_end_dates.append(ref_ts[t1 - 1]     if t1 - 1   < len(ref_ts) else None)
        test_start_dates.append(ref_ts[test0]     if test0    < len(ref_ts) else None)
        test_end_dates.append(ref_ts[test1 - 1]   if test1 - 1 < len(ref_ts) else None)

        window_idx += 1
        if is_last_window:
            break

        if anchored:
            end += length_test
        else:
            start += length_test
            end = start + length_train_set

    # -----------------------------------------------------------
    # Final parameter summary (deploy params)
    # In both modes this is simply the effective_params actually applied in the LAST
    # window — for "wfo" that is the last window's raw optimum; for "wfo_ema" that is
    # the running EMA already used to generate that window's test trades.
    # -----------------------------------------------------------
    final_params = best_params_list[-1] if best_params_list else {}

    # -----------------------------------------------------------
    # Final DataFrame with train/test dates, params, and criterion
    # -----------------------------------------------------------
    # Raw timestamps for exact alignment (used by debug)
    train_start_ts_raw = list(train_start_dates)
    test_start_ts_raw  = list(test_start_dates)
    test_end_ts_raw    = list(test_end_dates)

    train_start_dates = [pd.to_datetime(d).date() if d is not None else None for d in train_start_dates]
    train_end_dates   = [pd.to_datetime(d).date() if d is not None else None for d in train_end_dates]
    test_start_dates  = [pd.to_datetime(d).date() if d is not None else None for d in test_start_dates]
    test_end_dates    = [pd.to_datetime(d).date() if d is not None else None for d in test_end_dates]

    df_results = pd.DataFrame(best_params_list)
    df_results.insert(0, 'train_start', train_start_dates)
    df_results.insert(1, 'train_end',   train_end_dates)
    df_results.insert(2, 'test_start',  test_start_dates)
    df_results.insert(3, 'test_end',    test_end_dates)
    df_results['best_crite']      = best_criteria_list
    df_results['tn_trades']       = test_n_trades_list
    df_results['tr_symbols']      = [len(s) for s in train_symbols_list]
    df_results['ts_symbols']      = [len(s) for s in test_symbols_list]
    df_results['tr_syms']         = train_symbols_list
    df_results['ts_syms']         = test_symbols_list
    df_results['_train_start_ts'] = train_start_ts_raw
    df_results['_test_start_ts']  = test_start_ts_raw
    df_results['_test_end_ts']    = test_end_ts_raw

    summary_row = dict(final_params)
    summary_row['train_start'] = wfo_mode
    summary_row['train_end']   = ''
    summary_row['test_start']  = ''
    summary_row['test_end']    = ''
    summary_row['best_crite']  = df_results['best_crite'].mean() if 'best_crite' in df_results else None

    df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)


    sep_row = {col: "·" * min(8, len(str(col))) for col in df_results.columns}
    sep_row["train_start"] = "·" * 10
    df_display   = pd.concat([df_results.iloc[:-1], pd.DataFrame([sep_row]), df_results.iloc[[-1]]], ignore_index=True)
    display_cols = [c for c in df_display.columns if not c.startswith("_") and c not in ("tr_syms", "ts_syms")]
    logger.debug(f"WFO Final summary — parameters, criterion, and train/test dates per window:\n{df_display[display_cols].to_string()}\n{'─'*115}")

    # -----------------------------------------------------------
    # Concatenate per-window trade logs
    # -----------------------------------------------------------
    wfo_train_trades = pd.concat(train_trades_list, ignore_index=True) if train_trades_list else pd.DataFrame()
    wfo_test_trades  = pd.concat(test_trades_list,  ignore_index=True) if test_trades_list  else pd.DataFrame()

    return final_params, df_results, wfo_train_trades, wfo_test_trades, window_idx, best_params_list, window_test_arrays_list, window_test_start_ts_list