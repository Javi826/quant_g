#shared_batchs/pipeline/montecarlo.py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
logger = logging.getLogger("BOT_batch.pipeline.montecarlo")

# =============================================================================
# MONTECARLO EXECUTION CONFIG
# =============================================================================
N_SIMULATIONS      = 1000
BLOCK_SIZE         = 20   # 1 = simple bootstrap with replacement (no block structure).
RUIN_THRESHOLD_PCT = 25   # % capital drawdown considered "ruin" within a single simulation
SEED               = 42   # fixed seed for reproducible bootstrap runs
MAX_PLOT_CURVES    = 100
# =============================================================================
# PRIVATE HELPERS
# =============================================================================
def _make_overlapping_blocks(profits: np.ndarray, block_size: int) -> np.ndarray:
    n_trades = len(profits)
    n_blocks = n_trades - block_size + 1
    
    return np.lib.stride_tricks.sliding_window_view(profits, block_size)[:n_blocks]

def _bootstrap_max_drawdowns(
    profits: np.ndarray,
    initial_balance: float,
    n_simulations: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    n_trades        = len(profits)
    blocks          = _make_overlapping_blocks(profits, block_size)
    n_blocks        = len(blocks)
    max_dds         = np.empty(n_simulations, dtype=np.float64)
    rng             = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n_trades / block_size))
    for i in range(n_simulations):
        chosen      = rng.integers(0, n_blocks, size=n_blocks_needed)
        sampled     = np.concatenate(blocks[chosen])[:n_trades]
        equity      = initial_balance + np.cumsum(sampled)
        cummax      = np.maximum.accumulate(equity)
        safe_cummax = np.where(cummax <= 0, np.nan, cummax)
        dd          = (cummax - equity) / safe_cummax
        max_dds[i]  = float(np.nanmax(dd)) * 100.0 if np.any(np.isfinite(dd)) else 100.0
    return max_dds


def _probability_of_ruin(max_dds: np.ndarray, ruin_threshold_pct: float) -> float:
    """% of simulations whose max drawdown exceeds the ruin threshold."""
    return float(np.mean(max_dds >= ruin_threshold_pct)) * 100.0

def _plot_montecarlo_equity_curves(
    profits: np.ndarray,
    initial_balance: float,
    block_size: int,
    ruin_threshold_pct: float,
    n_curves: int = MAX_PLOT_CURVES,
    seed: int = SEED,
) -> None:
    """Debug plot: bootstrap equity curves vs the original, with a dynamic ruin band."""
    n_trades = len(profits)
    blocks   = _make_overlapping_blocks(profits, block_size)
    n_blocks = len(blocks)
    rng      = np.random.default_rng(seed + 1)
    n_blocks_needed = int(np.ceil(n_trades / block_size))

    fig, ax = plt.subplots(figsize=(12, 5))

    for _ in range(n_curves):
        chosen  = rng.integers(0, n_blocks, size=n_blocks_needed)
        sampled = np.concatenate(blocks[chosen])[:n_trades]
        equity  = initial_balance + np.cumsum(sampled)
        ax.plot(equity, color="gray", alpha=0.15, linewidth=0.7)

    original_equity      = initial_balance + np.cumsum(profits)
    original_running_max = np.maximum.accumulate(original_equity)
    ruin_band            = original_running_max * (1.0 - ruin_threshold_pct / 100.0)

    ax.plot(original_equity, color="red", linewidth=1.8, label="Original equity")
    ax.plot(ruin_band, color="red", linewidth=1.2, linestyle="--", label=f"Ruin threshold ({ruin_threshold_pct}%% DD)")

    ax.set_title(f"MONTECARLO — bootstrap equity curves (n_curves={n_curves})")
    ax.set_xlabel("Trade index")
    ax.set_ylabel("Equity")
    ax.legend()
    fig.tight_layout()
    plt.show()

# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_montecarlo_approval(prob_ruin: float, prob_ruin_th: float) -> bool:
    return prob_ruin <= prob_ruin_th
# =============================================================================
# CORE MONTECARLO EVALUATION (single rule)
# =============================================================================
def _evaluate_montecarlo(
    wfo_test_trades: pd.DataFrame,
    initial_balance: float,
    prob_ruin_th: float,
    n_simulations: int = N_SIMULATIONS,
    block_size: int = BLOCK_SIZE,
    ruin_threshold_pct: float = RUIN_THRESHOLD_PCT,
    seed: int = SEED,
) -> tuple:
    """Runs the bootstrap for a single rule's trades. Returns (approved, prob_ruin)."""

    if wfo_test_trades is None or wfo_test_trades.empty:
        return False, 100.0
    trades_sorted = wfo_test_trades.sort_values("buy_time")
    profits       = trades_sorted["profit"].to_numpy(dtype=np.float64)
    if len(profits) <= block_size:
        logger.debug(f"MONTECARLO ── skipped: n_trades={len(profits)} <= block_size={block_size}")
        return False, 100.0
    max_dds   = _bootstrap_max_drawdowns(profits, initial_balance, n_simulations, block_size, seed)
    prob_ruin = _probability_of_ruin(max_dds, ruin_threshold_pct)
    approved  = _evaluate_montecarlo_approval(prob_ruin, prob_ruin_th)
    logger.debug(
        f"MONTECARLO ── n_sims={n_simulations} block_size={block_size} "
        f"prob_ruin={prob_ruin:.1f}% -> {'PASS' if approved else 'FAIL'}"
    )

    if logger.isEnabledFor(logging.DEBUG):
        _plot_montecarlo_equity_curves(profits, initial_balance, block_size, ruin_threshold_pct, seed=seed)

    return approved, prob_ruin
# =============================================================================
# PIPE MONTECARLO — evaluates every rule's WFO test trades independently
# =============================================================================
def _empty_montecarlo_fields() -> dict:
    """Placeholder Montecarlo fields for rules that were never evaluated (pipe disabled)."""
    return {
        "passed_montecarlo":    True,
        "montecarlo_prob_ruin": 0.0,
    }

def pipe_montecarlo(
    rules: list,
    initial_balance: float,
    prob_ruin_th: float,
    enabled: bool = True,
    n_simulations: int = N_SIMULATIONS,
    block_size: int = BLOCK_SIZE,
    ruin_threshold_pct: float = RUIN_THRESHOLD_PCT,
    seed: int = SEED,
) -> list:


    if not enabled:
        logger.info(f"MONTECARLO ── disabled — passing all {len(rules)} rules through untouched")
        return [{**r, **_empty_montecarlo_fields()} for r in rules]

    results = []
    for r in rules:
        approved, prob_ruin = _evaluate_montecarlo(
            wfo_test_trades    = r["wfo_test_trades"],
            initial_balance    = initial_balance,
            prob_ruin_th       = prob_ruin_th,
            n_simulations      = n_simulations,
            block_size         = block_size,
            ruin_threshold_pct = ruin_threshold_pct,
            seed               = seed,
        )
        results.append({
            **r,
            "passed_montecarlo":    approved,
            "montecarlo_prob_ruin": prob_ruin,
        })

    return results