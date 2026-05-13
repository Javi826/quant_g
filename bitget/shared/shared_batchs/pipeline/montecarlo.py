#shared_batch/pipeline/montecarlo.py
import contextlib
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.utils.analysis import report_montecarlo
from shared_batchs.tools.optimize_MC import generate_paths_for_all_symbols_functional
from shared_batchs.utils.torque import get_n_obs, extract_ohlcv_from_path, compile_MC_results

logger = logging.getLogger("BOT_batch.pipeline.montecarlo")


def extract_best_params(df_summary, param_names, lists_for_grid, selection_percentile=None):
    """
    Extract optimal params from MC summary.
    Sorts by Net_Gain_pct_m (mean) or Net_Gain_pct_pN (percentile N) depending on selection_percentile.
    Preserves int/float types based on the original grid lists.
    """
    int_params  = {k for k, lst in zip(param_names, lists_for_grid) if all(isinstance(x, int) for x in lst)}
    sort_col = "Net_Gain_pct_m" if selection_percentile is None else "Net_Gain_pct_pN"
    best_row = df_summary.loc[df_summary[sort_col].idxmax()]
    best_params = {
        k: int(round(best_row[k])) if k in int_params else round(float(best_row[k]), 4)
        for k in param_names
    }
    logger.debug(f"Extracting optimal params (best {sort_col})...")
    logger.debug("Best params: " + " | ".join(f"{k}: {v}" for k, v in best_params.items()))
    return best_params


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _process_path(
    path_idx: int,
    paths: dict,
    params_list: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
) -> list:
    """Process a single MC path across all param combinations."""
    all_results = []
    for param_dict in params_list:
        ohlcv_arrays = extract_ohlcv_from_path(paths, path_idx, dtype=dtype)
        for sym in ohlcv_arrays:
            sig_kwargs = {k: param_dict[k.upper()] for k in signal_params_keys if k.upper() in param_dict}
            signals    = signal_fn(ohlcv_arrays[sym], **sig_kwargs, live_trading=False)
            ohlcv_arrays[sym]["signal"] = np.asarray(signals, dtype=dtype)
        result = run_grid_backtest(
            ohlcv_arrays,
            sell_after   = param_dict["SELL_AFTER"],
            tp_pct       = param_dict["TP_PCT"],
            sl_pct       = param_dict["SL_PCT"],
            order_amount = order_amount,
        )
        all_results.append(compile_MC_results(result, param_dict, path_idx, INITIAL_BALANCE, dtype=dtype))
    return all_results


# =============================================================================
# RUN MONTE CARLO IS
# =============================================================================

def run_montecarlo_is(
    ohlcv_data: dict,
    param_dict_list: list,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    n_paths: int,
    timeframe: str,
    dtype,
    n_jobs: int = -1,
    show_progress: bool = False,
    selection_percentile: int = None,
) -> tuple:
    """
    Run Monte Carlo simulation on IS data.

    Returns:
        tuple: (best_params, df_summary)
    """


    n_obs = get_n_obs(timeframe)
    paths = generate_paths_for_all_symbols_functional(
        ohlcv_data, n_paths=n_paths, n_obs=n_obs, raw_columns=[],
    )


    with (tqdm_joblib(tqdm(total=n_paths, desc="🔄 Evaluating MC IS paths")) if show_progress else contextlib.nullcontext()):
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_process_path)(i, paths, param_dict_list, signal_fn, signal_params_keys, order_amount, dtype)
            for i in range(n_paths)
        )

    all_results  = [r for sublist in results_list for r in sublist]
    df_portfolio = pd.DataFrame(all_results)
    df_summary, _, _ = report_montecarlo(
        df_portfolio         = df_portfolio,
        param_names          = param_names,
        initial_balance      = INITIAL_BALANCE,
        selection_percentile = selection_percentile,
    )

    best_params = extract_best_params(df_summary, param_names, lists_for_grid, selection_percentile=selection_percentile)

    params_str = " | ".join(f"{k}={v}" for k, v in best_params.items() if k not in ("SELL_AFTER",))
    logger.info(f"STAGE 1 ── MC Best params         ── {params_str} — {n_paths} paths | {len(param_dict_list)} combos")

    return best_params, df_summary


# =============================================================================
# RUN MONTE CARLO OOS
# =============================================================================

def run_montecarlo_oos(
    ohlcv_data: dict,
    best_params: dict,
    param_names: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    n_paths: int,
    timeframe: str,
    dtype,
    n_jobs: int = -1,
    show_progress: bool = False,
) -> tuple:
    """
    Run Monte Carlo simulation on OOS data using best params from IS.

    Returns:
        tuple: (df_portfolio_oos, p5_winrate, p50_winrate)
    """
    n_obs = get_n_obs(timeframe)
    paths = generate_paths_for_all_symbols_functional(
        ohlcv_data, n_paths=n_paths, n_obs=n_obs, raw_columns=[],
    )


    with (tqdm_joblib(tqdm(total=n_paths, desc="🔄 Evaluating MC OOS paths")) if show_progress else contextlib.nullcontext()):
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_process_path)(i, paths, [best_params], signal_fn, signal_params_keys, order_amount, dtype)
            for i in range(n_paths)
        )

    all_results      = [r for sublist in results_list for r in sublist]
    df_portfolio_oos = pd.DataFrame(all_results)
    _, p5_winrate, p50_winrate = report_montecarlo(
        df_portfolio    = df_portfolio_oos,
        param_names     = param_names,
        initial_balance = INITIAL_BALANCE,
    )

    return df_portfolio_oos, p5_winrate, p50_winrate