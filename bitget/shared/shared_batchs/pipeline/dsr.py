#shared_batchs/pipeline/dsr.py
import logging
import numpy as np
from scipy.stats import norm, skew, kurtosis

logger = logging.getLogger("BOT_batch.pipeline.dsr")

# =============================================================================
# DSR EXECUTION CONFIG
# =============================================================================
EULER_MASCHERONI = 0.5772156649

# =============================================================================
# PRIVATE HELPERS
# =============================================================================
def _compute_sr_star(n_trials: float, sigma_sr: float) -> float:
    """
    Expected maximum Sharpe Ratio under pure noise after n_trials independent trials
    (Bailey & Lopez de Prado, 2014).
    """
    if n_trials <= 1 or sigma_sr <= 0:
        return 0.0
    term1 = (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n_trials)
    term2 = EULER_MASCHERONI * norm.ppf(1 - 1 / (n_trials * np.e))
    return float(sigma_sr * (term1 + term2))


def _compute_sr_std(sr_obs: float, gamma3: float, gamma4: float, n_trades: int) -> float:
    """
    Standard deviation of the Sharpe Ratio estimator, correcting for
    non-normality of returns (skewness and kurtosis) — Bailey & Lopez de Prado, 2014.

    Args:
        sr_obs   : observed Sharpe Ratio.
        gamma3   : skewness of returns.
        gamma4   : kurtosis of returns (non-excess, i.e. normal = 3.0).
        n_trades : number of observations (T).
    """
    if n_trades <= 1:
        return 0.0
    numerator = 1 - gamma3 * sr_obs + ((gamma4 - 1) / 4) * (sr_obs ** 2)
    if numerator <= 0:
        return 0.0
    return float(np.sqrt(numerator / (n_trades - 1)))


def _compute_dsr_probability(sr_obs: float, sr_star: float, sr_std: float) -> float:
    """
    DSR as a probability: Phi((SR_obs - SR_star) / sigma(SR)).
    Returns a value in [0, 1].
    """
    if sr_std <= 0:
        return 0.0
    z = (sr_obs - sr_star) / sr_std
    return float(norm.cdf(z))


# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_dsr_approval(dsr_value: float, dsr_th: float) -> bool:
    return dsr_value >= dsr_th


# =============================================================================
# RUN DSR
# =============================================================================
def compute_sigma_sr(all_sharpe_values: list) -> float:
    """
    Standard deviation of Sharpe Ratios across all candidate rules tested.

    NOTE: caller is responsible for filtering out rules with too few trades before
    passing their Sharpe values here — low-trade-count rules produce noisy, extreme
    Sharpe estimates that inflate this dispersion and distort SR* downstream.
    """
    values = np.array([s for s in all_sharpe_values if s is not None and np.isfinite(s)], dtype=np.float64)
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def run_dsr(
    sr_obs: float,
    profits: np.ndarray,
    n_trials: float,
    sigma_sr: float,
    dsr_th: float,
) -> tuple:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Deflates the observed Sharpe Ratio of a single candidate rule, correcting for:
      - selection bias from n_trials independent candidate rules tested (SR*)
      - non-normality of returns via skewness/kurtosis (sigma(SR))

    Args:
        n_trials : effective number of independent trials (N_eff), typically a float
                   since it's estimated from correlation structure, not a raw integer count.
        sigma_sr : dispersion of Sharpe ratios across the candidate universe, used to
                   compute SR* (the expected max Sharpe under pure noise). This is a
                   different quantity from sr_std computed below (which is the standard
                   error of THIS rule's own Sharpe estimator) — do not confuse the two
                   in logs or downstream code.

    Returns:
        tuple: (approved, dsr_value, sr_star) — dsr_value is a PROBABILITY in [0, 1].
        dsr_value >= dsr_th (e.g. 0.95) means the observed Sharpe is unlikely
        to be the product of chance/overfitting.
    """
    n_trades = len(profits)
    if n_trades <= 1 or sr_obs is None or not np.isfinite(sr_obs):
        return False, 0.0, 0.0

    gamma3 = float(skew(profits, bias=False))
    gamma4 = float(kurtosis(profits, fisher=False, bias=False))  # non-excess kurtosis

    sr_star   = _compute_sr_star(n_trials, sigma_sr)
    sr_std    = _compute_sr_std(sr_obs, gamma3, gamma4, n_trades)
    dsr_value = _compute_dsr_probability(sr_obs, sr_star, sr_std)
    approved  = _evaluate_dsr_approval(dsr_value, dsr_th)

    logger.debug(
        f"DSR ── n_trials={n_trials:.1f} sr_obs={sr_obs:.3f} sr_star={sr_star:.3f} "
        f"skew={gamma3:.3f} kurt={gamma4:.3f} sr_std={sr_std:.3f} sigma_sr_input={sigma_sr:.3f} "
        f"-> DSR={dsr_value:.3f} {'PASS' if approved else 'FAIL'}"
    )
    return approved, dsr_value, sr_star