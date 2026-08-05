#validate_stepm.py
"""
Ground-truth validation of the StepM (Romano & Wolf, 2005) implementation in
reality_check.py. This does NOT touch real trading data — it plants a known
ground truth and checks whether the real production code recovers it.

TEST A — Realized FWER on all-null cohorts.
    Simulate cohorts of pure noise (no real edge in ANY column). Run the real
    StepM implementation on each cohort and record whether it approves at
    least one column. Over many independent cohorts, that approval rate
    should sit close to STEPM_ALPHA if StepM is correctly controlling the
    family-wise error rate. This is the same logic as Cohort A / the FWER
    reading in Inglese (2026), Sec. 3.3, Eq. 6.

TEST B — Power on a cohort with one planted real edge.
    Same setup, but one column gets a real, known drift added on top of the
    noise. Measures how often StepM approves THAT specific column.

TEST C — First-round consistency check.
    Step 1 of the stepdown (before any column is removed from the active
    set) is mathematically defined to reduce to the standalone global
    p-value (White, 2000). This checks that the two functions agree on the
    same data, as a sanity check that the stepdown loop was implemented
    correctly on top of the global test.

All three tests call the REAL functions from reality_check.py directly, so
this validates the production code, not a reimplementation of it.

Adjust REALITY_CHECK_MODULE_PATH below to point at the folder containing
reality_check.py in your repo before running.
"""
import sys
import contextlib
import io
import numpy as np
import pandas as pd

REALITY_CHECK_MODULE_PATH = "/home/javi/projects/quant/quant_b/bitget/shared/shared_batchs/pipeline"
sys.path.append(REALITY_CHECK_MODULE_PATH)

from reality_check import (
    _compute_deviation_matrix,
    _compute_global_pvalue,
    _stepwise_reality_check_pvalues,
    SHARPE_PERIODS_YEAR,
    STEPM_ALPHA,
    WHITE_N_BOOTSTRAP,
    WHITE_BLOCK_SIZE,
)

# =============================================================================
# SIMULATION CONFIG
# =============================================================================
N_COHORTS       = 100    # independent cohorts per test — raise for a tighter CI
N_COLUMNS       = 50     # M, number of candidate columns per cohort
N_OBS           = 1000   # T, days per cohort
PLANTED_SHARPE  = 2.0    # annualized Sharpe of the single true edge in Test B
BASE_SEED       = 12345


def _make_null_cohort(n_obs: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    """All-null cohort: every column is pure unit-variance noise, no edge anywhere."""
    returns = rng.standard_normal(size=(n_obs, n_cols))
    dates   = pd.bdate_range("2015-01-01", periods=n_obs)
    columns = [f"null_col_{i}" for i in range(n_cols)]
    return pd.DataFrame(returns, index=dates, columns=columns)


def _make_edge_cohort(
    n_obs: int, n_cols: int, planted_sharpe: float, rng: np.random.Generator
) -> tuple:
    """One column carries a real, known drift; the rest are pure noise.

    The per-period drift is derived directly from the target ANNUALIZED
    Sharpe so this can be read in the same units as dsr.py / reality_check.py.
    """
    returns = rng.standard_normal(size=(n_obs, n_cols))
    daily_mu = planted_sharpe / np.sqrt(SHARPE_PERIODS_YEAR)
    edge_col_idx = rng.integers(0, n_cols)
    returns[:, edge_col_idx] += daily_mu

    dates   = pd.bdate_range("2015-01-01", periods=n_obs)
    columns = [f"null_col_{i}" for i in range(n_cols)]
    columns[edge_col_idx] = "TRUE_EDGE_col"
    matrix = pd.DataFrame(returns, index=dates, columns=columns)
    return matrix, "TRUE_EDGE_col"


def _run_stepm_on_matrix(matrix: pd.DataFrame, seed: int) -> dict:
    """One full pass through the real production code: bootstrap deviations,
    studentization, global p-value, and StepM p-values.

    Progress bars are silenced here: reality_check.py prints one tqdm bar per
    call, which is fine for a single production run but floods the output
    across hundreds of small validation cohorts.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bootstrap_result = _compute_deviation_matrix(
            matrix,
            n_bootstrap=WHITE_N_BOOTSTRAP,
            block_size=WHITE_BLOCK_SIZE,
            seed=seed,
            progress_label="[validation]",
        )
    kept_columns            = bootstrap_result["kept_columns"]
    studentized_deviations  = bootstrap_result["studentized_deviations"]
    z_stat                  = bootstrap_result["z_stat"]

    global_result = _compute_global_pvalue(studentized_deviations, z_stat)
    stepm_pvals   = _stepwise_reality_check_pvalues(studentized_deviations, z_stat, alpha=STEPM_ALPHA)

    return {
        "kept_columns":   kept_columns,
        "global_p":       global_result["global_p"],
        "best_col_idx":   global_result["best_col_idx"],
        "stepm_p_by_col": dict(zip(kept_columns, stepm_pvals)),
    }


# =============================================================================
# TEST A — realized FWER under the null
# =============================================================================
def run_test_a() -> None:
    print("=" * 70)
    print("TEST A — Realized FWER on all-null cohorts")
    print(f"  M={N_COLUMNS} columns, T={N_OBS} days, {N_COHORTS} cohorts, "
          f"nominal alpha={STEPM_ALPHA}")
    print("=" * 70)

    n_false_positive_cohorts = 0

    for cohort_idx in range(N_COHORTS):
        rng    = np.random.default_rng(BASE_SEED + cohort_idx)
        matrix = _make_null_cohort(N_OBS, N_COLUMNS, rng)
        result = _run_stepm_on_matrix(matrix, seed=BASE_SEED + cohort_idx)

        any_approved = any(p <= STEPM_ALPHA for p in result["stepm_p_by_col"].values())
        n_false_positive_cohorts += int(any_approved)

    fwer_hat = n_false_positive_cohorts / N_COHORTS
    se       = np.sqrt(fwer_hat * (1 - fwer_hat) / N_COHORTS)

    print(f"  Realized FWER      : {fwer_hat:.4f}  (SE ≈ {se:.4f})")
    print(f"  Nominal alpha       : {STEPM_ALPHA:.4f}")
    print(f"  Within 2 SE of nominal? {'YES — consistent with correct FWER control' if abs(fwer_hat - STEPM_ALPHA) <= 2 * se else 'NO — investigate'}")
    print()


# =============================================================================
# TEST B — power on a cohort with one planted real edge
# =============================================================================
def run_test_b() -> None:
    print("=" * 70)
    print("TEST B — Power on a cohort with one planted real edge")
    print(f"  M={N_COLUMNS} columns, T={N_OBS} days, {N_COHORTS} cohorts, "
          f"planted annualized Sharpe={PLANTED_SHARPE}")
    print("=" * 70)

    n_detected = 0

    for cohort_idx in range(N_COHORTS):
        rng = np.random.default_rng(BASE_SEED + 100_000 + cohort_idx)
        matrix, edge_col_name = _make_edge_cohort(N_OBS, N_COLUMNS, PLANTED_SHARPE, rng)
        result = _run_stepm_on_matrix(matrix, seed=BASE_SEED + 100_000 + cohort_idx)

        edge_p = result["stepm_p_by_col"].get(edge_col_name, float("nan"))
        detected = np.isfinite(edge_p) and edge_p <= STEPM_ALPHA
        n_detected += int(detected)

    power_hat = n_detected / N_COHORTS
    se        = np.sqrt(power_hat * (1 - power_hat) / N_COHORTS)

    print(f"  Realized power      : {power_hat:.4f}  (SE ≈ {se:.4f})")
    print(f"  (Power near 0 would mean StepM never detects even a real edge — "
          f"investigate. Power near 1 is expected for a strong planted edge.)")
    print()


# =============================================================================
# TEST C — first-round consistency: StepM round 1 must reduce to the global test
# =============================================================================
def run_test_c() -> None:
    print("=" * 70)
    print("TEST C — First-round consistency (StepM round 1 == global p-value)")
    print("=" * 70)

    n_checked  = 0
    n_consistent = 0

    for cohort_idx in range(N_COHORTS):
        rng = np.random.default_rng(BASE_SEED + 200_000 + cohort_idx)
        matrix, edge_col_name = _make_edge_cohort(N_OBS, N_COLUMNS, PLANTED_SHARPE, rng)
        result = _run_stepm_on_matrix(matrix, seed=BASE_SEED + 200_000 + cohort_idx)

        # Only meaningful when the best column is significant enough to be
        # rejected in round 1 (global_p <= alpha) — otherwise it may only get
        # rejected in a later round with a smaller active set, and the two
        # numbers are not expected to match. See docstring for the reasoning.
        if result["global_p"] > STEPM_ALPHA:
            continue

        n_checked += 1
        best_col_name  = str(result["kept_columns"][result["best_col_idx"]])
        stepm_p_best   = result["stepm_p_by_col"].get(best_col_name, float("nan"))
        matches        = np.isclose(stepm_p_best, result["global_p"], atol=1e-12)
        n_consistent  += int(matches)

    print(f"  Cohorts where the best column was significant enough to check : {n_checked}/{N_COHORTS}")
    if n_checked > 0:
        print(f"  Consistent with global p-value                               : {n_consistent}/{n_checked}")
        print(f"  {'PASS' if n_consistent == n_checked else 'FAIL — investigate the stepdown loop'}")
    else:
        print("  No cohort had a strong enough edge to check — raise PLANTED_SHARPE and rerun.")
    print()


if __name__ == "__main__":
    run_test_a()
    run_test_b()
    run_test_c()