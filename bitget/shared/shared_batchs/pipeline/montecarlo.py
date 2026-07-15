#shared_batchs/pipeline/montecarlo.py
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("BOT_batch.pipeline.montecarlo")

# =============================================================================
# MONTECARLO EXECUTION CONFIG
# =============================================================================
N_SIMULATIONS      = 1000
BLOCK_SIZE         = 10   # 1 = simple bootstrap with replacement (no block structure).
RUIN_THRESHOLD_PCT = 20   # % capital drawdown considered "ruin" within a single simulation
SEED               = 42   # fixed seed for reproducible bootstrap runs

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
) -> tuple:
    n_trades = len(profits)
    blocks   = _make_overlapping_blocks(profits, block_size)
    n_blocks = len(blocks)


    max_dds = np.empty(n_simulations, dtype=np.float64)
    equity_curves = np.empty((n_simulations, n_trades), dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n_trades / block_size))
    for i in range(n_simulations):
        chosen  = rng.integers(0, n_blocks, size=n_blocks_needed)
        sampled = np.concatenate(blocks[chosen])[:n_trades]
        equity  = initial_balance + np.cumsum(sampled)
        equity_curves[i] = equity
        cummax = np.maximum.accumulate(equity)
        safe_cummax = np.where(cummax <= 0, np.nan, cummax)
        dd = (cummax - equity) / safe_cummax
        max_dds[i] = float(np.nanmax(dd)) * 100.0 if np.any(np.isfinite(dd)) else 100.0
    return max_dds, equity_curves


def _probability_of_ruin(max_dds: np.ndarray, ruin_threshold_pct: float) -> float:
    """% of simulations whose max drawdown exceeds the ruin threshold."""
    return float(np.mean(max_dds >= ruin_threshold_pct)) * 100.0


# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_montecarlo_approval(prob_ruin: float, prob_ruin_th: float) -> bool:
    return prob_ruin <= prob_ruin_th


# =============================================================================
# RUN MONTECARLO
# =============================================================================
def pipe_montecarlo(
    wfo_test_trades: pd.DataFrame,
    initial_balance: float,
    prob_ruin_th: float,
    n_simulations: int = N_SIMULATIONS,
    block_size: int = BLOCK_SIZE,
    ruin_threshold_pct: float = RUIN_THRESHOLD_PCT,
    seed: int = SEED,
) -> tuple:
    if wfo_test_trades is None or wfo_test_trades.empty:
        return False, 100.0
    trades_sorted = wfo_test_trades.sort_values("buy_time")
    profits       = trades_sorted["profit"].to_numpy(dtype=np.float64)
    if len(profits) <= block_size:
        logger.debug(f"MONTECARLO ── skipped: n_trades={len(profits)} <= block_size={block_size}")
        return False, 100.0

    max_dds, equity_curves = _bootstrap_max_drawdowns(profits, initial_balance, n_simulations, block_size, seed)
    prob_ruin = _probability_of_ruin(max_dds, ruin_threshold_pct)
    approved  = _evaluate_montecarlo_approval(prob_ruin, prob_ruin_th)

    logger.debug(
        f"MONTECARLO ── n_sims={n_simulations} block_size={block_size} "
        f"prob_ruin={prob_ruin:.1f}% -> {'PASS' if approved else 'FAIL'}"
    )
    return approved, prob_ruin