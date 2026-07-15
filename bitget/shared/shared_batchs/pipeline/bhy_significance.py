#shared_batchs/pipeline/bhy_significance.py
import logging
import numpy as np
from scipy.stats import norm, skew, kurtosis

logger = logging.getLogger("BOT_batch.pipeline.bhy_significance")

# =============================================================================
# BHY EXECUTION CONFIG
# =============================================================================
BHY_ALPHA = 0.05  # target False Discovery Rate (FDR)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================
def _compute_psr_std(sr_obs: float, gamma3: float, gamma4: float, n_trades: int) -> float:
    """
    Standard error of the Sharpe Ratio estimator, correcting for non-normality
    (skewness and kurtosis) — Bailey & Lopez de Prado, 2012/2014. Same formula as
    dsr.py's _compute_sr_std, kept self-contained here to avoid a cross-module
    dependency for a single small helper.
    """
    if n_trades <= 1:
        return 0.0
    numerator = 1 - gamma3 * sr_obs + ((gamma4 - 1) / 4) * (sr_obs ** 2)
    if numerator <= 0:
        return 0.0
    return float(np.sqrt(numerator / (n_trades - 1)))


# =============================================================================
# PER-RULE PSR P-VALUE
# =============================================================================
def compute_psr_p_value(sr_obs: float, profits: np.ndarray, theta: float = 0.0) -> float:
    """
    One-sided p-value for H0: true Sharpe Ratio <= theta, using the Probabilistic
    Sharpe Ratio (PSR) framework (Bailey & Lopez de Prado, 2012), which corrects for
    sample length, skewness, and kurtosis — but NOT for multiple testing (that
    correction is applied separately via Benjamini-Yekutieli, see below).

    Returns:
        p_value in [0, 1]. Small p_value = strong evidence the true SR exceeds theta.
    """
    n_trades = len(profits)
    if n_trades <= 1 or sr_obs is None or not np.isfinite(sr_obs):
        return 1.0

    gamma3 = float(skew(profits, bias=False))
    gamma4 = float(kurtosis(profits, fisher=False, bias=False))  # non-excess kurtosis

    psr_std = _compute_psr_std(sr_obs, gamma3, gamma4, n_trades)
    if psr_std <= 0:
        return 1.0

    z = (sr_obs - theta) / psr_std
    psr = float(norm.cdf(z))  # P(true SR > theta)
    return 1.0 - psr


# =============================================================================
# BENJAMINI-YEKUTIELI FDR CONTROL
# =============================================================================
def benjamini_yekutieli(p_values_by_rule_id: dict, alpha: float = BHY_ALPHA) -> set:
    """
    Benjamini-Yekutieli (2001) procedure for controlling the False Discovery Rate
    (FDR) under ARBITRARY dependence between tests — valid even when candidate rules
    are highly correlated, unlike methods (Bonferroni, standard BH) that assume
    independence or only positive dependence. This is the multiple-testing
    correction Harvey & Liu (2015) recommend ("BHY") as their preferred method.

    No correlation/N_eff estimation is required: the dependence-robustness comes
    from the harmonic-number correction factor c(M) below, not from modeling the
    correlation structure directly.

    Args:
        p_values_by_rule_id : dict rule_id -> p-value (e.g. from compute_psr_p_value).
        alpha                : target FDR level (e.g. 0.05).

    Returns:
        set of rule_ids whose null hypothesis is rejected (statistically significant
        after multiple-testing correction).
    """
    if not p_values_by_rule_id:
        return set()

    items = sorted(p_values_by_rule_id.items(), key=lambda kv: kv[1])
    rule_ids_sorted = [rid for rid, _ in items]
    p_sorted         = np.array([p for _, p in items], dtype=np.float64)

    m = len(p_sorted)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))  # harmonic number, the BY dependence correction

    thresholds = (np.arange(1, m + 1) / (m * c_m)) * alpha
    below_threshold = np.where(p_sorted <= thresholds)[0]

    if below_threshold.size == 0:
        logger.debug(f"BHY ── M={m} c(M)={c_m:.3f} alpha={alpha} -> 0 significant rule(s)")
        return set()

    k = int(below_threshold.max())  # largest index satisfying p_(k) <= threshold_(k)
    significant_ids = set(rule_ids_sorted[:k + 1])

    logger.debug(
        f"BHY ── M={m} c(M)={c_m:.3f} alpha={alpha} -> {len(significant_ids)} significant rule(s) "
        f"(threshold at k={k + 1}: p<={thresholds[k]:.6f})"
    )
    return significant_ids


# =============================================================================
# RUN BHY SIGNIFICANCE TEST
# =============================================================================
def run_bhy_significance(all_raw_results: list, alpha: float = BHY_ALPHA, theta: float = 0.0) -> dict:
    """
    Full pipeline: compute a PSR p-value per candidate rule (using ALL evaluated
    rules with trades, not just survivors — the multiple-testing universe must
    reflect everything that was searched), then apply Benjamini-Yekutieli to
    control the FDR across that universe.

    Args:
        all_raw_results : list of rule result dicts, each with 'rule_id',
                          'wfo_test_trades', and 'sharpe'.
        alpha            : target FDR level (e.g. 0.05).
        theta            : Sharpe threshold for the null hypothesis (default 0.0,
                           i.e. "is the true Sharpe better than not trading at all").

    Returns:
        dict: {"significant_ids": set, "p_values_by_rule_id": dict}
    """
    p_values_by_rule_id = {}
    for r in all_raw_results:
        trades = r.get("wfo_test_trades")
        if trades is None or trades.empty:
            continue
        profits = trades["profit"].to_numpy(dtype=np.float64)
        p_values_by_rule_id[r["rule_id"]] = compute_psr_p_value(r["sharpe"], profits, theta=theta)

    significant_ids = benjamini_yekutieli(p_values_by_rule_id, alpha=alpha)
    return {"significant_ids": significant_ids, "p_values_by_rule_id": p_values_by_rule_id}