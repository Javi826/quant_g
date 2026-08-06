#validate_stepm.py
"""
Ground-truth validation of the StepM (Romano & Wolf, 2005) implementation in
stepm.py. This does NOT touch real trading data — it plants a known
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

All three tests call the REAL functions from stepm.py directly, so
this validates the production code, not a reimplementation of it.

Adjust STEPM_MODULE_PATH below to point at the folder containing
stepm.py in your repo before running.
"""
import sys
import time
import argparse
import contextlib
import io
import numpy as np
import pandas as pd

STEPM_MODULE_PATH = "/mnt/user-data/outputs"  # <-- point this at shared_batchs/pipeline
sys.path.append(STEPM_MODULE_PATH)

from stepm import (
    compute_deviation_matrix,
    compute_global_pvalue,
    stepwise_reality_check_pvalues,
    SHARPE_PERIODS_YEAR,
    STEPM_ALPHA as DEFAULT_STEPM_ALPHA,
    WHITE_N_BOOTSTRAP as DEFAULT_N_BOOTSTRAP,
    WHITE_BLOCK_SIZE,
)


# =============================================================================
# CLI — every knob can be overridden without editing this file or stepm.py
# =============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cohorts", type=int, default=500,
                         help="Independent cohorts per test. Higher = tighter confidence interval, slower.")
    parser.add_argument("--n-columns", type=int, default=50, help="M, candidate columns per cohort.")
    parser.add_argument("--n-obs", type=int, default=1000, help="T, days per cohort.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP, help="Bootstrap replicas per cohort.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_STEPM_ALPHA,
                         help="StepM significance level to test — overrides stepm.py's STEPM_ALPHA "
                              "for this run only, no need to edit that file.")
    parser.add_argument("--planted-sharpe", type=float, default=2.0,
                         help="Annualized Sharpe of the single true edge planted in Test B/C.")
    parser.add_argument("--n-jobs", type=int, default=1,
                         help="joblib n_jobs for the bootstrap. 1 avoids per-cohort process-pool "
                              "startup overhead, which dominates runtime at this data size.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Master seed. Omit for a fresh random seed each run.")
    parser.add_argument("--progress-every", type=int, default=50,
                         help="Print a progress line every N cohorts.")
    return parser.parse_args()


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
    Sharpe so this can be read in the same units as dsr.py / stepm.py.
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


def _run_stepm_on_matrix(matrix: pd.DataFrame, seed: int, n_bootstrap: int, alpha: float, n_jobs: int) -> dict:
    """One full pass through the real production code: bootstrap deviations,
    studentization, global p-value, and StepM p-values.

    Progress bars are silenced here: stepm.py prints one tqdm bar per
    call, which is fine for a single production run but floods the output
    across hundreds of small validation cohorts.

    n_jobs=1 by default: with M in the tens/low hundreds, the whole bootstrap
    fits in a single batch, so joblib's default n_jobs=-1 pays the cost of
    spinning up a fresh process pool on every cohort for no parallelism
    benefit. That startup cost, repeated across hundreds of cohorts, is the
    dominant runtime cost of this script — not the actual computation.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bootstrap_result = compute_deviation_matrix(
            matrix,
            n_bootstrap=n_bootstrap,
            block_size=WHITE_BLOCK_SIZE,
            seed=seed,
            n_jobs=n_jobs,
            progress_label="[validation]",
        )
    kept_columns            = bootstrap_result["kept_columns"]
    studentized_deviations  = bootstrap_result["studentized_deviations"]
    z_stat                  = bootstrap_result["z_stat"]

    global_result = compute_global_pvalue(studentized_deviations, z_stat)
    stepm_pvals   = stepwise_reality_check_pvalues(studentized_deviations, z_stat, alpha=alpha)

    return {
        "kept_columns":   kept_columns,
        "global_p":       global_result["global_p"],
        "best_col_idx":   global_result["best_col_idx"],
        "stepm_p_by_col": dict(zip(kept_columns, stepm_pvals)),
    }


# =============================================================================
# TEST A — realized FWER under the null
# =============================================================================
def run_test_a(cfg: argparse.Namespace, seed_seq: np.random.SeedSequence) -> None:
    print("=" * 70)
    print("TEST A — Realized FWER on all-null cohorts")
    print(f"  M={cfg.n_columns} columns, T={cfg.n_obs} days, {cfg.n_cohorts} cohorts, "
          f"nominal alpha={cfg.alpha}, n_bootstrap={cfg.n_bootstrap}")
    print("=" * 70)

    cohort_seeds = seed_seq.spawn(cfg.n_cohorts)
    n_false_positive_cohorts = 0
    t0 = time.time()

    for cohort_idx, cohort_seed in enumerate(cohort_seeds):
        rng    = np.random.default_rng(cohort_seed)
        matrix = _make_null_cohort(cfg.n_obs, cfg.n_columns, rng)
        result = _run_stepm_on_matrix(
            matrix, seed=int(rng.integers(0, 2**31 - 1)),
            n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha, n_jobs=cfg.n_jobs,
        )

        any_approved = any(p <= cfg.alpha for p in result["stepm_p_by_col"].values())
        n_false_positive_cohorts += int(any_approved)

        if cfg.progress_every and (cohort_idx + 1) % cfg.progress_every == 0:
            elapsed = time.time() - t0
            print(f"  ... {cohort_idx + 1}/{cfg.n_cohorts} cohorts  ({elapsed:.1f}s elapsed)")

    fwer_hat = n_false_positive_cohorts / cfg.n_cohorts
    se       = np.sqrt(fwer_hat * (1 - fwer_hat) / cfg.n_cohorts)
    ci_lo, ci_hi = max(0.0, fwer_hat - 1.96 * se), min(1.0, fwer_hat + 1.96 * se)

    print(f"  Realized FWER      : {fwer_hat:.4f}  (SE ≈ {se:.4f}, 95% CI ≈ [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  Nominal alpha      : {cfg.alpha:.4f}")
    print(f"  Within 2 SE of nominal? "
          f"{'YES — consistent with correct FWER control' if abs(fwer_hat - cfg.alpha) <= 2 * se else 'NO — investigate'}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print()


# =============================================================================
# TEST B — power on a cohort with one planted real edge
# =============================================================================
def run_test_b(cfg: argparse.Namespace, seed_seq: np.random.SeedSequence) -> None:
    print("=" * 70)
    print("TEST B — Power on a cohort with one planted real edge")
    print(f"  M={cfg.n_columns} columns, T={cfg.n_obs} days, {cfg.n_cohorts} cohorts, "
          f"planted annualized Sharpe={cfg.planted_sharpe}")
    print("=" * 70)

    cohort_seeds = seed_seq.spawn(cfg.n_cohorts)
    n_detected = 0
    t0 = time.time()

    for cohort_idx, cohort_seed in enumerate(cohort_seeds):
        rng = np.random.default_rng(cohort_seed)
        matrix, edge_col_name = _make_edge_cohort(cfg.n_obs, cfg.n_columns, cfg.planted_sharpe, rng)
        result = _run_stepm_on_matrix(
            matrix, seed=int(rng.integers(0, 2**31 - 1)),
            n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha, n_jobs=cfg.n_jobs,
        )

        edge_p = result["stepm_p_by_col"].get(edge_col_name, float("nan"))
        detected = np.isfinite(edge_p) and edge_p <= cfg.alpha
        n_detected += int(detected)

        if cfg.progress_every and (cohort_idx + 1) % cfg.progress_every == 0:
            elapsed = time.time() - t0
            print(f"  ... {cohort_idx + 1}/{cfg.n_cohorts} cohorts  ({elapsed:.1f}s elapsed)")

    power_hat = n_detected / cfg.n_cohorts
    se        = np.sqrt(power_hat * (1 - power_hat) / cfg.n_cohorts)

    print(f"  Realized power     : {power_hat:.4f}  (SE ≈ {se:.4f})")
    print(f"  (Power near 0 would mean StepM never detects even a real edge — "
          f"investigate. Power near 1 is expected for a strong planted edge.)")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print()


# =============================================================================
# TEST C — first-round consistency: StepM round 1 must reduce to the global test
# =============================================================================
def run_test_c(cfg: argparse.Namespace, seed_seq: np.random.SeedSequence) -> None:
    print("=" * 70)
    print("TEST C — First-round consistency (StepM round 1 == global p-value)")
    print("=" * 70)

    cohort_seeds = seed_seq.spawn(cfg.n_cohorts)
    n_checked    = 0
    n_consistent = 0
    t0 = time.time()

    for cohort_idx, cohort_seed in enumerate(cohort_seeds):
        rng = np.random.default_rng(cohort_seed)
        matrix, edge_col_name = _make_edge_cohort(cfg.n_obs, cfg.n_columns, cfg.planted_sharpe, rng)
        result = _run_stepm_on_matrix(
            matrix, seed=int(rng.integers(0, 2**31 - 1)),
            n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha, n_jobs=cfg.n_jobs,
        )

        # Only meaningful when the best column is significant enough to be
        # rejected in round 1 (global_p <= alpha) — otherwise it may only get
        # rejected in a later round with a smaller active set, and the two
        # numbers are not expected to match. See docstring for the reasoning.
        if result["global_p"] > cfg.alpha:
            continue

        n_checked += 1
        best_col_name  = str(result["kept_columns"][result["best_col_idx"]])
        stepm_p_best   = result["stepm_p_by_col"].get(best_col_name, float("nan"))
        matches        = np.isclose(stepm_p_best, result["global_p"], atol=1e-12)
        n_consistent  += int(matches)

        if cfg.progress_every and (cohort_idx + 1) % cfg.progress_every == 0:
            elapsed = time.time() - t0
            print(f"  ... {cohort_idx + 1}/{cfg.n_cohorts} cohorts  ({elapsed:.1f}s elapsed)")

    print(f"  Cohorts where the best column was significant enough to check : {n_checked}/{cfg.n_cohorts}")
    if n_checked > 0:
        print(f"  Consistent with global p-value                               : {n_consistent}/{n_checked}")
        print(f"  {'PASS' if n_consistent == n_checked else 'FAIL — investigate the stepdown loop'}")
    else:
        print("  No cohort had a strong enough edge to check — raise --planted-sharpe and rerun.")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print()


if __name__ == "__main__":
    args = _parse_args()
    master_seed = args.seed if args.seed is not None else int.from_bytes(np.random.bytes(4), "little")
    print(f"Master seed for this run: {master_seed}  (pass --seed {master_seed} to reproduce)\n")

    seed_seq_a = np.random.SeedSequence(master_seed, spawn_key=(0,))
    seed_seq_b = np.random.SeedSequence(master_seed, spawn_key=(1,))
    seed_seq_c = np.random.SeedSequence(master_seed, spawn_key=(2,))

    run_test_a(args, seed_seq_a)
    run_test_b(args, seed_seq_b)
    run_test_c(args, seed_seq_c)