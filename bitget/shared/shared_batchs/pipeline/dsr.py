#shared_batchs/pipeline/dsr.py
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger("BOT_batch.pipeline.dsr")

# =============================================================================
# DSR EXECUTION CONFIG
# =============================================================================
EULER_GAMMA         = 0.5772156649015328606  # Euler-Mascheroni constant
SHARPE_PERIODS_YEAR = 365.0                  # must match the annualization factor in compute_metrics (sqrt(365))

# =============================================================================
# PRIVATE HELPERS — trial correlation / N estimation
# =============================================================================
def _daily_profit_series(wfo_test_trades: pd.DataFrame) -> pd.Series:
    if wfo_test_trades is None or wfo_test_trades.empty:
        return None
    tl = wfo_test_trades.copy()
    tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
    return tl.groupby("_date")["profit"].sum()


def _build_daily_profit_matrix(all_raw_results: list) -> tuple:

    daily_series = {}
    for r in all_raw_results:
        s = _daily_profit_series(r.get("wfo_test_trades"))
        if s is not None and len(s) > 1:
            daily_series[r["rule_id"]] = s

    if len(daily_series) < 2:
        return None, list(daily_series.keys())

    all_dates = sorted(set().union(*[s.index for s in daily_series.values()]))
    matrix    = pd.DataFrame(index=all_dates)
    for rule_id, s in daily_series.items():
        matrix[rule_id] = s.reindex(all_dates, fill_value=0.0)

    return matrix, list(daily_series.keys())


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
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # guard against numerical noise (tiny negative values)
    return np.sort(eigenvalues)[::-1]


def _estimate_n_eff_eigen(matrix: pd.DataFrame) -> float:

    x_std, n_const = _standardize_and_split(matrix)

    if x_std is None:
        # every column had zero variance -> each is independent by the same convention.
        return float(n_const) if n_const > 0 else 1.0

    t_days = x_std.shape[0]
    gram   = (x_std @ x_std.T) / (t_days - 1)

    eigenvalues = _eigenvalues_desc(gram)

    sum_eig    = eigenvalues.sum() + n_const
    sum_eig_sq = np.sum(eigenvalues ** 2) + n_const
    if sum_eig_sq <= 0:
        return 1.0
    return float((sum_eig ** 2) / sum_eig_sq)


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
    """Eq. 2. sr and sr0 must both be UNANNUALIZED. kurt_r is raw kurtosis (fisher=False).
    t_obs is the number of observations in the SAME series used for sr/skew_r/kurt_r
    (here: daily profit observations, matching the sqrt(365) annualization in batch_metrics)."""
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


# =============================================================================
# PIPE DSR (across the full set of candidate trials)
# =============================================================================
def pipe_dsr(all_raw_results: list, dsr_th: float, debug_ids: set = None) -> dict:

    total_candidates = len(all_raw_results)
    daily_matrix, trial_ids = _build_daily_profit_matrix(all_raw_results)
    m_trials = len(trial_ids)

    logger.debug(
        f"DSR ── candidate universe ── total_candidates={total_candidates} "
        f"(all timeframes, incl. zero-trade rules) vs matrix_dim(M)={m_trials} "
        f"-> excluded={total_candidates - m_trials}"
    )

    if daily_matrix is None or m_trials < 2:
        logger.debug(f"DSR ── skipped: not enough trials with trades (M={m_trials})")
        return {"significant_ids": [], "dsr_by_rule_id": {}, "n_eff": 0.0, "sr0": 0.0}

    n_eff = _estimate_n_eff_eigen(daily_matrix)
    
    logger.debug(f"DSR ── N_eff terms ── method=ratio(dual/Gram) M={m_trials} -> N_eff={n_eff:.4f}")

    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    sr_by_id = {
        rule_id: _unannualize_sharpe(raw_by_id[rule_id].get("sharpe", np.nan))
        for rule_id in trial_ids
    }
    sr_array = np.array(list(sr_by_id.values()), dtype=np.float64)
    sr_array = sr_array[np.isfinite(sr_array)]
    var_sr   = float(np.var(sr_array, ddof=1)) if sr_array.size > 1 else 0.0

    sr0 = _expected_max_sharpe(var_sr, n_eff)

    logger.debug(
        f"DSR ── SR0 terms ── n_sr={sr_array.size} var_sr={var_sr:.6f} n_eff={n_eff:.4f} -> SR0={sr0:.4f}"
    )

    dsr_by_id = {}
    _sr_vals, _skew_vals, _kurt_vals, _dsr_vals = [], [], [], []
    for rule_id in trial_ids:
        t_days = int(raw_by_id[rule_id].get("n_days", 0))
        skew_r = float(raw_by_id[rule_id].get("skew", np.nan))
        kurt_r = float(raw_by_id[rule_id].get("kurtosis", np.nan))

        if not (np.isfinite(skew_r) and np.isfinite(kurt_r)):
            dsr_by_id[rule_id] = 0.0
            if debug_ids is None or rule_id in debug_ids:
                logger.debug(f"DSR[{rule_id}] ── skipped: non-finite skew/kurtosis (skew={skew_r} kurt={kurt_r})")
            continue

        dsr_value = _deflated_sharpe_ratio(sr_by_id[rule_id], sr0, t_days, skew_r, kurt_r)
        dsr_by_id[rule_id] = dsr_value

        if np.isfinite(sr_by_id[rule_id]):
            _sr_vals.append(sr_by_id[rule_id])
        _skew_vals.append(skew_r)
        _kurt_vals.append(kurt_r)
        _dsr_vals.append(dsr_value)

        if debug_ids is None or rule_id in debug_ids:
            logger.debug(
                f"DSR[{rule_id}] ── SR={sr_by_id[rule_id]:.4f} SR0={sr0:.4f} T_days={t_days} "
                f"skew={skew_r:.4f} kurt={kurt_r:.4f} -> DSR={dsr_value:.4f}"
            )

    significant_ids = [rid for rid, dsr_val in dsr_by_id.items() if _evaluate_dsr_approval(dsr_val, dsr_th)]

    def _stats(name: str, values: list) -> str:
        if not values:
            return f"{name}: n/a"
        arr = np.array(values, dtype=np.float64)
        return f"{name}[min={arr.min():.4f} mean={arr.mean():.4f} max={arr.max():.4f}]"

    logger.debug(
        "DSR ── metric ranges ── " +
        " ".join([
            _stats("SR", _sr_vals),
            _stats("skew", _skew_vals),
            _stats("kurt", _kurt_vals),
            _stats("DSR", _dsr_vals),
        ])
    )

    logger.debug(
        f"DSR ── M={m_trials} N_eff={n_eff:.1f} SR0={sr0:.3f} "
        f"-> {len(significant_ids)}/{m_trials} significant at th={dsr_th}"
    )

    return {
        "significant_ids": significant_ids,
        "dsr_by_rule_id":  dsr_by_id,
        "n_eff":           n_eff,
        "sr0":             sr0,
    }