#shared_batch/pipeline/montecarlo.py
import ast
import contextlib
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.tools.optimize_MC import generate_paths_for_all_symbols_functional
from shared_batchs.utils.ohlcv_utils import get_n_obs, extract_ohlcv_from_path
from shared_batchs.utils.backtest_compiler import compile_MC_results

logger = logging.getLogger("BOT_batch.pipeline.montecarlo")


def extract_best_params(df_summary, param_names, lists_for_grid, selection_percentile=None):
    """
    Extract optimal params from MC summary.
    Sorts by Net_Gain_pct_m (mean) or Net_Gain_pct_pN (percentile N) depending on selection_percentile.
    Preserves int/float types based on the original grid lists.
    """
    int_params = {k for k, lst in zip(param_names, lists_for_grid) if all(isinstance(x, int) for x in lst)}
    sort_col   = "Net_Gain_pct_m" if selection_percentile is None else "Net_Gain_pct_pN"
    best_row   = df_summary.loc[df_summary[sort_col].idxmax()]

    best_params = {
        k: int(round(best_row[k])) if k in int_params else round(float(best_row[k]), 4)
        for k in param_names
    }
    logger.debug(f"Extracting optimal params (best {sort_col})...")
    logger.debug("Best params: " + " | ".join(f"{k}: {v}" for k, v in best_params.items()))
    return best_params


# =============================================================================
# REPORT MONTECARLO
# =============================================================================

def report_montecarlo(df_portfolio, param_names, initial_balance, selection_percentile=None):

    # -----------------------------
    # RESUMEN POR COMBINACIÓN
    # -----------------------------
    summary_results = []

    # ===== CONVERTIR LISTAS A STRINGS PARA drop_duplicates() =====
    df_temp      = df_portfolio.copy()
    list_columns = []

    for col in param_names:
        if df_temp[col].apply(lambda x: isinstance(x, list)).any():
            list_columns.append(col)
            df_temp[col] = df_temp[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    combos_present = df_temp[param_names].drop_duplicates().to_dict(orient='records')
    # ==============================================================

    for comb in combos_present:
        filt = np.ones(len(df_portfolio), dtype=bool)

        for k, v in comb.items():
            col_values = df_portfolio[k]

            if k in list_columns:
                if v is None or pd.isna(v):
                    v_normalized = None
                elif isinstance(v, list):
                    v_normalized = v
                elif isinstance(v, str):
                    if v in ["None", "nan", "NaN"]:
                        v_normalized = None
                    else:
                        try:
                            v_normalized = ast.literal_eval(v)
                        except Exception:
                            v_normalized = None
                else:
                    v_normalized = None

                if v_normalized is None:
                    filt &= col_values.isna() | (col_values == None)
                else:
                    filt &= col_values.apply(lambda x: x == v_normalized if isinstance(x, list) else False)
            else:
                filt &= (col_values == v)

        subset = df_portfolio[filt]

        if subset.empty:
            continue

        port_balances  = subset['Portfolio_Final_Balance'].dropna()
        port_dd        = subset['DD'].dropna() if 'DD' in subset.columns else pd.Series(dtype=float)
        port_win_ratio = subset['Win_Ratio'].dropna() if 'Win_Ratio' in subset.columns else pd.Series(dtype=float)
        port_sharpe    = subset['Sharpe'].dropna() if 'Sharpe' in subset.columns else pd.Series(dtype=float)

        if len(port_balances) > 0:
            port_gain_abs          = port_balances - initial_balance
            port_gain_pct          = (port_gain_abs / initial_balance) * 100
            port_net_gain_mean     = port_gain_abs.mean()
            port_net_gain_pct_mean = port_gain_pct.mean()
            port_gain_pct_series   = port_gain_pct
        else:
            port_net_gain_mean     = np.nan
            port_net_gain_pct_mean = np.nan
            port_gain_pct_series   = pd.Series(dtype=float)

        port_dd_mean        = port_dd.mean() if len(port_dd) > 0 else np.nan
        port_win_ratio_mean = port_win_ratio.mean() if len(port_win_ratio) > 0 else np.nan
        port_sharpe_mean    = port_sharpe.mean() if len(port_sharpe) > 0 else np.nan

        summary_results.append({
            **comb,
            "Net_Gain_m": port_net_gain_mean,
            "Net_Gain_pct_m": port_net_gain_pct_mean,
            "Net_Gain_pct_pN": float(np.percentile(port_gain_pct_series, selection_percentile)) if selection_percentile is not None and len(port_gain_pct_series) > 0 else np.nan,
            "Win_Ratio_m": port_win_ratio_mean,
            "DD_m": port_dd_mean,
            "Sharpe_m": port_sharpe_mean,
            "Paths_IDX": subset['path_index'].nunique() if 'path_index' in subset.columns else np.nan,
            "Rows": len(subset)
        })

    df_summary = pd.DataFrame(summary_results).sort_values(by='Net_Gain_pct_m', ascending=False).reset_index(drop=True)

    # -----------------------------
    # HISTOGRAMS
    # -----------------------------
    path_grouped = df_portfolio.groupby('path_index').agg({
        'Portfolio_Final_Balance': 'mean',
        'DD': 'mean',
        'Win_Ratio': 'mean'
    }).reset_index()

    path_grouped['Net_Gain_pct'] = (path_grouped['Portfolio_Final_Balance'] - initial_balance) / initial_balance * 100

    # -----------------------------
    # MEJORES COMBOS POR MÉTRICA
    # -----------------------------
    SHARPE_ADJUSTMENT_FACTOR = 1e6
    df_summary['Sharpe_m']   = df_summary['Sharpe_m'] / SHARPE_ADJUSTMENT_FACTOR

    best_netgain = df_summary.loc[df_summary['Net_Gain_pct_m'].idxmax()]
    best_sharpe  = df_summary.loc[df_summary['Sharpe_m'].dropna().idxmax()] if df_summary['Sharpe_m'].notna().any() else best_netgain
    best_dd      = df_summary.loc[df_summary['DD_m'].idxmin()]

    df_best = pd.DataFrame([
        {'Metric': 'Net_Gain_pct', **best_netgain},
        {'Metric': 'Sharpe      ', **best_sharpe},
        {'Metric': 'Lowest DD   ', **best_dd}
    ])

    df_best = df_best.drop(columns=['Net_Gain_m', 'Rows'], errors='ignore')
    cols    = ['Metric'] + [c for c in df_best.columns if c != 'Metric']
    df_best = df_best[cols]
    df_best = df_best.round(2)

    median_gain   = np.percentile(path_grouped['Net_Gain_pct'].dropna(), 50)
    std_gain      = path_grouped['Net_Gain_pct'].dropna().std()
    prob_negative = (path_grouped['Net_Gain_pct'] < 0).mean() * 100

    win_rates   = path_grouped['Win_Ratio'].dropna()
    p5_winrate  = np.percentile(win_rates, 5)
    p50_winrate = np.percentile(win_rates, 50)

    logger.debug(
        f"{df_best.to_string(index=False)}\n"
        f"\nP50 Net_Gain_pct per Path    : {median_gain:.2f}%"
        f"\nStd Dev Net_Gain_pct per Path: {std_gain:.2f}%"
        f"\nProbability of Negative Path : {prob_negative:.2f}%"
        f"\nP5  Win Rate per Path: {p5_winrate:.2f}%"
        f"\nP50 Win Rate per Path: {p50_winrate:.2f}%"
    )

    return df_summary, p5_winrate, p50_winrate


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