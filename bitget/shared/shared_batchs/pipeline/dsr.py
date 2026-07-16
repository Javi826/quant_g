#shared_batchs/pipeline/dsr.py
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

logger = logging.getLogger("BOT_batch.pipeline.dsr")

# =============================================================================
# DSR EXECUTION CONFIG
# =============================================================================
EULER_GAMMA         = 0.5772156649015328606  # Euler-Mascheroni constant
SHARPE_PERIODS_YEAR = 365.0                  # must match the annualization factor in compute_metrics (sqrt(365))
DSR_TH              = 0.95                   # min DSR probability required to accept a rule

# =============================================================================
# PRIVATE HELPERS — trial correlation / N estimation (paper Eq. 8-9)
# =============================================================================
def _daily_profit_series(wfo_test_trades: pd.DataFrame) -> pd.Series:
    if wfo_test_trades is None or wfo_test_trades.empty:
        return None
    tl = wfo_test_trades.copy()
    tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
    return tl.groupby("_date")["profit"].sum()


def _build_correlation_matrix(all_raw_results: list) -> tuple:
    """Aligns all candidates' daily profit series on a common date index and
    returns (corr_matrix, trial_rule_ids). Only candidates with >1 profit day qualify."""
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

    return matrix.corr(), list(daily_series.keys())


def _average_off_diagonal_correlation(corr_matrix: pd.DataFrame) -> float:
    """Equal-weighted average correlation across all off-diagonal pairs (Eq. 8)."""
    values      = corr_matrix.to_numpy(dtype=np.float64)
    n           = values.shape[0]
    off_diag    = values.sum() - np.trace(values)
    n_pairs     = n * (n - 1)
    return float(off_diag / n_pairs) if n_pairs > 0 else 0.0


def _estimate_n_independent_trials(avg_corr: float, m_trials: int) -> float:
    """Interpolates between N=1 (rho=1) and N=M (rho=0) — Eq. 9."""
    avg_corr = float(np.clip(avg_corr, 0.0, 1.0))
    return avg_corr + (1.0 - avg_corr) * m_trials


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


def _deflated_sharpe_ratio(sr: float, sr0: float, t_trades: int, skew_r: float, kurt_r: float) -> float:
    """Eq. 2. sr and sr0 must both be UNANNUALIZED. kurt_r is raw kurtosis (fisher=False)."""
    if t_trades <= 1 or not np.isfinite(sr):
        return 0.0
    moment_term = 1.0 - skew_r * sr + ((kurt_r - 1.0) / 4.0) * (sr ** 2)
    if moment_term <= 0:
        return 0.0
    numerator = (sr - sr0) * np.sqrt(t_trades - 1)
    return float(norm.cdf(numerator / np.sqrt(moment_term)))


# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_dsr_approval(dsr_value: float, dsr_th: float) -> bool:
    return dsr_value >= dsr_th


# =============================================================================
# RUN DSR SIGNIFICANCE (across the full set of candidate trials)
# =============================================================================
def run_dsr_significance(all_raw_results: list, dsr_th: float = DSR_TH) -> dict:
    """
    Computes the Deflated Sharpe Ratio for every candidate rule that produced trades,
    correcting Sharpe ratio inflation caused by (1) multiple testing / selection bias
    and (2) non-Normal returns (Bailey & Lopez de Prado, 2014).

    N (independent trials) is estimated via the average off-diagonal correlation between
    the daily profit series of ALL candidates (Eq. 8-9 in the paper) — NOT via the raw
    count of trials (M), and NOT via PCA.
    """
    corr_matrix, trial_ids = _build_correlation_matrix(all_raw_results)
    m_trials = len(trial_ids)

    if corr_matrix is None or m_trials < 2:
        logger.debug(f"DSR ── skipped: not enough trials with trades (M={m_trials})")
        return {"significant_ids": [], "dsr_by_rule_id": {}, "n_eff": 0.0, "avg_corr": 0.0, "sr0": 0.0}

    avg_corr = _average_off_diagonal_correlation(corr_matrix)
    n_eff    = _estimate_n_independent_trials(avg_corr, m_trials)

    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    sr_by_id = {
        rule_id: _unannualize_sharpe(raw_by_id[rule_id].get("sharpe", np.nan))
        for rule_id in trial_ids
    }
    sr_array = np.array(list(sr_by_id.values()), dtype=np.float64)
    sr_array = sr_array[np.isfinite(sr_array)]
    var_sr   = float(np.var(sr_array, ddof=1)) if sr_array.size > 1 else 0.0

    sr0 = _expected_max_sharpe(var_sr, n_eff)

    dsr_by_id = {}
    for rule_id in trial_ids:
        profits  = raw_by_id[rule_id]["wfo_test_trades"]["profit"].to_numpy(dtype=np.float64)
        t_trades = len(profits)
        skew_r   = float(skew(profits))
        kurt_r   = float(kurtosis(profits, fisher=False))
        dsr_by_id[rule_id] = _deflated_sharpe_ratio(sr_by_id[rule_id], sr0, t_trades, skew_r, kurt_r)

    significant_ids = [rid for rid, dsr_val in dsr_by_id.items() if _evaluate_dsr_approval(dsr_val, dsr_th)]

    logger.debug(
        f"DSR ── M={m_trials} N_eff={n_eff:.1f} avg_corr={avg_corr:.3f} SR0={sr0:.3f} "
        f"-> {len(significant_ids)}/{m_trials} significant at th={dsr_th}"
    )

    return {
        "significant_ids": significant_ids,
        "dsr_by_rule_id":  dsr_by_id,
        "n_eff":           n_eff,
        "avg_corr":        avg_corr,
        "sr0":             sr0,
    }