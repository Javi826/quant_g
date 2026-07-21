#shared_batchs/pipeline/dsr.py
import itertools
import logging
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from shared_batchs.backtesters.ZX_compute_BT import (
    INITIAL_BALANCE,
    prepare_backtest_data,
    run_backtest_from_prepared,
)
from shared_batchs.utils.batch_metrics import compute_metrics

logger = logging.getLogger("BOT_batch.pipeline.dsr")

# =============================================================================
# DSR EXECUTION CONFIG
# =============================================================================
EULER_GAMMA         = 0.5772156649015328606  # Euler-Mascheroni constant
SHARPE_PERIODS_YEAR = 365.0                  # must match the annualization factor in compute_metrics (sqrt(365))
DSR_N_JOBS          = -1                     # safe to parallelize fully: this Parallel runs as its own phase, sequential relative to WFO — no nesting
DSR_MIN_TRADES      = 100   
DSR_MAX_SHARPE_ANN  = 10.0                   # combos with unrealistically high annualized Sharpe are rejected (near-zero variance artifact)                 # combos with fewer trades are rejected (near-zero variance inflates Sharpe artificially)
# =============================================================================
# FULL-PERIOD GRID SEARCH — selection-bias metrics (single pass, no WFO windows)
# =============================================================================

def _combo_id(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))


def _daily_profit_from_trades(trade_log: pd.DataFrame) -> pd.Series:
    tl = trade_log.copy()
    tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
    return tl.groupby("_date")["profit"].sum()


def _build_full_period_ohlcv(ohlcv_arr: dict, signal_fn: callable, dtype) -> dict:
    ohlcv_arrays = {}
    for sym, arr in ohlcv_arr.items():
        signals = signal_fn(arr, live_trading=False)
        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=dtype)}
    return ohlcv_arrays


def prepare_full_period_data(ohlcv_arr: dict, signal_fn: callable, dtype):

    ohlcv_arrays = _build_full_period_ohlcv(ohlcv_arr, signal_fn, dtype)
    return prepare_backtest_data(ohlcv_arrays)


def _evaluate_combo_sharpe(params: dict, prepared_data, order_amount: int) -> tuple:
    results = run_backtest_from_prepared(
        prepared_data,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    trade_log = results["__PORTFOLIO__"]["trade_log"]
    if trade_log is None or trade_log.empty or len(trade_log) < DSR_MIN_TRADES:
        return -np.inf, params, None, None

    trade_log             = trade_log.copy()
    trade_log.columns     = trade_log.columns.str.lower().str.strip()
    trade_log["buy_time"] = pd.to_datetime(trade_log["buy_time"])

    m      = compute_metrics(trade_log, capital=INITIAL_BALANCE, name="")
    sharpe = m["Sharpe"] if np.isfinite(m["Sharpe"]) else -np.inf
    if sharpe > DSR_MAX_SHARPE_ANN:
        return -np.inf, params, None, None

    daily_profit = _daily_profit_from_trades(trade_log)
    return sharpe, params, m, daily_profit


def _run_full_period_for_rule(
    rule_id: str,
    ohlcv_arr: dict,
    signal_fn: callable,
    param_grid: dict,
    order_amount: int,
    dtype,
) -> tuple:

    prepared_data = prepare_full_period_data(ohlcv_arr, signal_fn, dtype)

    keys   = list(param_grid.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*[param_grid[k] for k in keys])]

    rows = [_evaluate_combo_sharpe(params, prepared_data, order_amount) for params in combos]

    combo_daily_profit = {
        _combo_id(params): daily_profit
        for _sharpe, params, _m, daily_profit in rows
        if daily_profit is not None and len(daily_profit) > 1
    }
    best_sharpe, best_params, best_metrics, _best_daily = max(rows, key=lambda x: x[0])
    best_combo_id = _combo_id(best_params)

    if best_metrics is None:
        winner_metrics = {
            "sharpe_train":   np.nan,
            "skew_train":     np.nan,
            "kurtosis_train": np.nan,
            "n_days_train":   0,
            "net_gain_train": np.nan,
            "max_dd_train":   np.nan,
        }
    else:
        winner_metrics = {
            "sharpe_train":   best_metrics["Sharpe"],
            "skew_train":     best_metrics["Skew"],
            "kurtosis_train": best_metrics["Kurtosis"],
            "n_days_train":   best_metrics["N_days"],
            "net_gain_train": best_metrics["Net_Gain_pct"],
            "max_dd_train":   best_metrics["Max_DD_pct"],
        }

    return rule_id, {**winner_metrics, "combo_daily_profit": combo_daily_profit, "best_combo_id": best_combo_id}

def run_full_period_search(rules: list, param_grid: dict, order_amount: int, dtype, progress_label: str = "") -> dict:

    desc = f"DSR FULL-PERIOD SEARCH {progress_label}".strip()
    with tqdm_joblib(tqdm(desc=desc, total=len(rules), dynamic_ncols=True)):
        results = Parallel(n_jobs=DSR_N_JOBS, max_nbytes=None)(
            delayed(_run_full_period_for_rule)(
                r["rule_id"], r["ohlcv_arr"], r["signal_fn"], param_grid, order_amount, dtype,
            )
            for r in rules
        )

    return dict(results)


# =============================================================================
# PRIVATE HELPERS — N_eff estimation (flat rule x combo matrix, eigenvalue method)
# =============================================================================
def _build_flat_daily_matrix(all_raw_results: list) -> pd.DataFrame | None:

    series_by_col = {}
    for r in all_raw_results:
        combo_profit = r.get("combo_daily_profit") or {}
        for combo_id, s in combo_profit.items():
            series_by_col[f"{r['rule_id']}__{combo_id}"] = s

    if len(series_by_col) < 2:
        return None

    all_dates = sorted(set().union(*[s.index for s in series_by_col.values()]))
    matrix    = pd.DataFrame(index=all_dates)
    for col_id, s in series_by_col.items():
        matrix[col_id] = s.reindex(all_dates, fill_value=0.0)

    return matrix


def _standardize_and_split(matrix: pd.DataFrame) -> tuple:
    arr        = matrix.to_numpy(dtype=np.float64)
    means      = arr.mean(axis=0)
    stds       = arr.std(axis=0, ddof=1)
    valid_mask = stds > 0
    n_const    = int(np.sum(~valid_mask))

    if not np.any(valid_mask):
        return None, n_const

    x_std = (arr[:, valid_mask] - means[valid_mask]) / stds[valid_mask]
    return x_std, n_const


def _eigenvalues_desc(square_array: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(square_array)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return np.sort(eigenvalues)[::-1]


def _participation_ratio(eigenvalues: np.ndarray, n_const: int) -> float:
    sum_eig    = eigenvalues.sum() + n_const
    sum_eig_sq = np.sum(eigenvalues ** 2) + n_const
    if sum_eig_sq <= 0:
        return 1.0
    return float((sum_eig ** 2) / sum_eig_sq)


def _estimate_n_eff_eigen(matrix: pd.DataFrame) -> float:
    x_std, n_const = _standardize_and_split(matrix)

    if x_std is None:
        return float(n_const) if n_const > 0 else 1.0

    t_days = x_std.shape[0]
    gram   = (x_std @ x_std.T) / (t_days - 1)

    eigenvalues = _eigenvalues_desc(gram)
    return _participation_ratio(eigenvalues, n_const)


def estimate_n_eff_flat(all_raw_results: list) -> float | None:

    matrix = _build_flat_daily_matrix(all_raw_results)
    if matrix is None:
        return None
    return _estimate_n_eff_eigen(matrix)


# =============================================================================
# PRIVATE HELPERS — DSR formula (paper Eq. 1-2)
# =============================================================================
def _unannualize_sharpe(sharpe_annualized: float, periods_per_year: float = SHARPE_PERIODS_YEAR) -> float:
    if sharpe_annualized is None or not np.isfinite(sharpe_annualized):
        return np.nan
    return float(sharpe_annualized / np.sqrt(periods_per_year))


def _expected_max_sharpe(var_sr: float, n_trials: float) -> float:
    """Eq. 1 — expected maximum Sharpe ratio under N independent trials, assuming null skill."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z_n  = norm.ppf(1.0 - 1.0 / n_trials)
    z_ne = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    term = (1.0 - EULER_GAMMA) * z_n + EULER_GAMMA * z_ne
    return float(np.sqrt(var_sr) * term)


def _deflated_sharpe_ratio(sr: float, sr0: float, t_obs: int, skew_r: float, kurt_r: float) -> float:
    """Eq. 2. sr and sr0 must both be UNANNUALIZED. kurt_r is raw kurtosis (fisher=False)."""
    if t_obs <= 1 or not np.isfinite(sr):
        return 0.0
    moment_term = 1.0 - skew_r * sr + ((kurt_r - 1.0) / 4.0) * (sr ** 2)
    if moment_term <= 0:
        return 0.0
    numerator = (sr - sr0) * np.sqrt(t_obs - 1)
    return float(norm.cdf(numerator / np.sqrt(moment_term)))


# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_dsr_approval(dsr_value: float, dsr_th: float) -> bool:
    return dsr_value >= dsr_th


def _short_id(rule_id: str) -> str:
    return "_".join(rule_id.split("_")[:3])


def _print_train_metrics_table(raw_by_id: dict, dsr_by_id: dict, sr_by_id: dict, candidate_ids: set, passed_ids: set, sr0: float) -> None:

    rows = [raw_by_id[rid] for rid in candidate_ids if rid in raw_by_id]
    rows.sort(key=lambda r: dsr_by_id.get(r["rule_id"], 0.0), reverse=True)

    if not rows:
        return

    id_width    = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2
    label_width = max((len(r.get("label", "")) for r in rows), default=8) + 2

    logger.debug(f"\n{'─' * 170}")
    logger.debug(f"  DSR TRAIN METRICS (full-period grid search) ── SR0={sr0:.4f} ── {len(rows)} candidates")
    logger.debug(f"{'─' * 170}")
    logger.debug(
        f"{'ID':<{id_width}}{'SIDE':<6}{'NET_GAIN_TR':<13}{'MAX_DD_TR':<11}{'SR_ANN':<10}{'SR_UNANN':<11}"
        f"{'SKEW_TR':<10}{'KURT_TR':<10}{'N_DAYS_TR':<11}{'DSR':<9}{'RULE':<{label_width}}{'STATUS':<8}"
    )
    logger.debug(f"{'─' * 170}")

    for r in rows:
        rule_id = r["rule_id"]
        status  = "✅" if rule_id in passed_ids else "❌"
        logger.debug(
            f"{_short_id(rule_id):<{id_width}}{r.get('side', ''):<6}"
            f"{r.get('net_gain_train', float('nan')):<13.1f}{r.get('max_dd_train', float('nan')):<11.1f}"
            f"{r.get('sharpe_train', float('nan')):<10.4f}{sr_by_id.get(rule_id, float('nan')):<11.4f}"
            f"{r.get('skew_train', float('nan')):<10.4f}{r.get('kurtosis_train', float('nan')):<10.4f}"
            f"{r.get('n_days_train', 0):<11}{dsr_by_id.get(rule_id, 0.0):<9.4f}"
            f"{r.get('label', ''):<{label_width}}{status:<8}"
        )

    logger.debug(f"{'─' * 170}\n")


# =============================================================================
# CORE DSR CALCULATION (across a set of candidate trials — typically one timeframe)
# =============================================================================
def _compute_dsr(all_raw_results: list, dsr_th: float, n_combos: int) -> dict:

    total_candidates = len(all_raw_results)
    n_bruto           = total_candidates * max(n_combos, 1)

    n_eff = estimate_n_eff_flat(all_raw_results)

    logger.info(
        f"DSR ── N comparison ── N_bruto={n_bruto} (M={total_candidates} x n_combos={n_combos})  "
        f"N_eff(eigen, flat rule x combo pool)={n_eff if n_eff is not None else 'n/a (insufficient data)'}"
    )

    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    if n_eff is None:
        logger.warning("DSR ── N_eff unavailable — setting DSR=0.0 for all rules (no rules pass).")
        dsr_by_id = {rule_id: 0.0 for rule_id in raw_by_id}
        return {
            "passed_dsr_ids": [],
            "dsr_by_rule_id": dsr_by_id,
            "n_eff":          None,
            "n_bruto":        n_bruto,
            "sr0":            np.nan,
        }

    sr_by_id = {
        rule_id: _unannualize_sharpe(r.get("sharpe_train", np.nan))
        for rule_id, r in raw_by_id.items()
    }
    sr_array = np.array(list(sr_by_id.values()), dtype=np.float64)
    sr_array = sr_array[np.isfinite(sr_array)]
    var_sr   = float(np.var(sr_array, ddof=1)) if sr_array.size > 1 else 0.0

    sr0 = _expected_max_sharpe(var_sr, n_eff)

    logger.debug(
        f"DSR ── SR0 terms ── total_candidates={total_candidates} n_combos={n_combos} "
        f"n_eff={n_eff:.4f} n_sr={sr_array.size} var_sr={var_sr:.6f} -> SR0={sr0:.4f}"
    )

    dsr_by_id = {}
    for rule_id, r in raw_by_id.items():
        t_days = int(r.get("n_days_train", 0))
        skew_r = float(r.get("skew_train", np.nan))
        kurt_r = float(r.get("kurtosis_train", np.nan))

        if not (np.isfinite(skew_r) and np.isfinite(kurt_r)):
            dsr_by_id[rule_id] = 0.0
            continue

        dsr_by_id[rule_id] = _deflated_sharpe_ratio(sr_by_id[rule_id], sr0, t_days, skew_r, kurt_r)

    passed_dsr_ids = [rid for rid, dsr_val in dsr_by_id.items() if _evaluate_dsr_approval(dsr_val, dsr_th)]

    if logger.isEnabledFor(logging.DEBUG):
        _print_train_metrics_table(raw_by_id, dsr_by_id, sr_by_id, set(passed_dsr_ids), set(passed_dsr_ids), sr0)

    logger.debug(
        f"DSR ── M={total_candidates} n_combos={n_combos} N_bruto={n_bruto} N_eff={n_eff:.4f} SR0={sr0:.3f} "
        f"-> {len(passed_dsr_ids)}/{total_candidates} significant at th={dsr_th}"
    )

    return {
        "passed_dsr_ids": passed_dsr_ids,
        "dsr_by_rule_id": dsr_by_id,
        "n_eff":          n_eff,
        "n_bruto":        n_bruto,
        "sr0":            sr0,
    }


# =============================================================================
# PIPE DSR — one timeframe at a time
# =============================================================================
def _empty_dsr_fields() -> dict:
    """Placeholder DSR fields for rules that were never evaluated (pipe disabled)."""
    return {
        "passed_dsr":         True,
        "dsr":                0.0,
        "sharpe_train":       None,
        "skew_train":         None,
        "kurtosis_train":     None,
        "n_days_train":       None,
        "net_gain_train":     None,
        "max_dd_train":       None,
        "combo_daily_profit": None,
        "best_combo_id":      None,
    }


def pipe_dsr(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    dtype,
    dsr_th: float,
    enabled: bool = True,
    timeframe: str = "",
) -> list:


    if not enabled:
        logger.info(f"DSR ── {timeframe} ── disabled — passing all {len(rules)} rules through untouched")
        return [{**r, **_empty_dsr_fields()} for r in rules]

    rules_for_search = [
        {"rule_id": r["rule_id"], "ohlcv_arr": ohlcv_arr, "signal_fn": r["signal_fn"]}
        for r in rules
    ]
    full_period_by_rule = run_full_period_search(
        rules          = rules_for_search,
        param_grid     = param_grid,
        order_amount   = order_amount,
        dtype          = dtype,
        progress_label = timeframe,
    )

    n_combos = 1
    for _values in param_grid.values():
        n_combos *= len(_values)

    raw_for_dsr = [
        {**r, **full_period_by_rule[r["rule_id"]]}
        for r in rules
    ]
    dsr_result     = _compute_dsr(raw_for_dsr, dsr_th=dsr_th, n_combos=n_combos)
    passed_dsr_ids = set(dsr_result["passed_dsr_ids"])
    dsr_by_id      = dsr_result["dsr_by_rule_id"]

    logger.info(f"DSR ── {timeframe} ── {len(passed_dsr_ids)}/{len(rules)} rules pass")
    debug_plot_approved_dsr_daily_profit(raw_for_dsr, passed_dsr_ids)  # DEBUG — remove after use

    results = []
    for r in rules:
        rid = r["rule_id"]
        fp  = full_period_by_rule[rid]
        results.append({
            **r,
            "passed_dsr":         rid in passed_dsr_ids,
            "dsr":                dsr_by_id.get(rid, 0.0),
            "sharpe_train":       fp["sharpe_train"],
            "skew_train":         fp["skew_train"],
            "kurtosis_train":     fp["kurtosis_train"],
            "n_days_train":       fp["n_days_train"],
            "net_gain_train":     fp["net_gain_train"],
            "max_dd_train":       fp["max_dd_train"],
            "combo_daily_profit": fp["combo_daily_profit"],
            "best_combo_id":      fp["best_combo_id"],
        })

    return results

# =============================================================================
# DEBUG ONLY — remove after use
# =============================================================================
def debug_plot_approved_dsr_daily_profit(results: list, passed_dsr_ids: set) -> None:
    import matplotlib.pyplot as plt

    for r in results:
        rule_id = r["rule_id"]
        if rule_id not in passed_dsr_ids:
            continue

        combo_daily_profit = r.get("combo_daily_profit") or {}
        if not combo_daily_profit:
            continue

        best_combo_id = r.get("best_combo_id")
        if best_combo_id is None or best_combo_id not in combo_daily_profit:
            continue
        daily_profit = combo_daily_profit[best_combo_id]

        values = daily_profit.values[np.isfinite(daily_profit.values)]
        if values.size == 0 or np.ptp(values) < 1e-6:
            logger.warning(
                f"DSR DEBUG PLOT ── {rule_id} ── skipped, degenerate combo={best_combo_id} "
                f"n_days={values.size} n_nonzero={(values != 0).sum()} "
                f"min={values.min() if values.size else 'n/a'} max={values.max() if values.size else 'n/a'}"
            )
            continue

        nonzero_values = values[values != 0.0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{rule_id} — {best_combo_id}")

        axes[0].hist(values, bins=50)
        axes[0].set_title("Daily profit (with zeros)")

        if nonzero_values.size > 0 and np.ptp(nonzero_values) >= 1e-6:
            axes[1].hist(nonzero_values, bins=50)
        axes[1].set_title("Daily profit (without zeros)")

        plt.tight_layout()
        plt.show()