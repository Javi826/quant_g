#sets/set_REDUN.py
import os
import sys
import time
import logging
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import time as _time
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_redundancy")
logger.setLevel(LOG_LEVEL)

logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from shared_batchs.symbols.universe import filter_symbols, select_universe, select_top_n_by_volume
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.rule_mining.rule_generator import generate_all_rules, MAX_DEPTH
from shared_batchs.pipeline.backtest_runner import pipe_backtesting

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION — mirrors main_MINER.py
# =============================================================================
TIMEFRAMES = ["1H", "4H", "6Hutc", "12Hutc"]
TIMEFRAMES = ["12Hutc"]
N_SYMBOLS  = 10

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [2, 4, 6, 8, 10],
    "SL_PCT":     [2, 4, 6, 8, 10],
}

# =============================================================================
# REDUNDANCY ANALYSIS CONFIG
# =============================================================================
RANDOM_SEED              = 42
DECORRELATE_THRESHOLD    = 0.9
DECORRELATE_BATCH_SIZE   = 5000
GPU_SURVIVOR_CHUNK       = 100_000  # survivor columns compared per GPU matmul chunk, bounds VRAM peak
GPU_INITIAL_CAPACITY     = 20_000   # initial survivors buffer capacity in VRAM, doubled on overflow
SIGNAL_MASK_N_JOBS       = -1
METRIC_LABEL_WIDTH       = 36  # fixed label width so all metric-block prints align

# =============================================================================
# SIGNAL MASK CLEANING — pre-backtest, exact-duplicate detection based on
# which bars each rule's raw signal fires on (before TP/SL is applied).
# Mirrors shared_batchs/pipeline/signal_cleaning.py used in production.
# =============================================================================
def _compute_signal_mask_row(rule: dict, ohlcv_arr: dict, symbols: list) -> np.ndarray:
    signal_parts = [
        np.asarray(rule["signal_fn"](ohlcv_arr[sym], live_trading=False), dtype=bool)
        for sym in symbols
    ]
    return np.concatenate(signal_parts)


def build_signal_mask_matrix(
    all_rules: list,
    ohlcv_arr: dict,
    n_jobs: int = SIGNAL_MASK_N_JOBS,
) -> "sparse.csr_matrix":
    """Boolean signal mask per rule, concatenated across symbols in a fixed
    order, stored sparse since signals are mostly inactive bars. Computed
    in parallel across rules with process-based workers (loky) — signal_fn
    is typically pure-Python/pandas indicator logic that holds the GIL, so
    a threading backend does not parallelize this in practice."""

    symbols = list(ohlcv_arr.keys())

    with tqdm_joblib(tqdm(desc="SIGNAL MASK BUILD", total=len(all_rules), dynamic_ncols=True)):
        row_arrays = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_signal_mask_row)(rule, ohlcv_arr, symbols)
            for rule in all_rules
        )

    mask_dense = np.vstack(row_arrays)
    return sparse.csr_matrix(mask_dense)


def deduplicate_exact_signal_masks(mask_matrix: "sparse.csr_matrix", sides: list) -> tuple:
    """
    Exact positional dedup: two rules are the same hypothesis only if they
    fire on exactly the same bars, bit for bit, AND trade the same side —
    no similarity threshold. side is part of the key: long and short share
    the same entry mask by construction (signal_fn ignores side), but TP/SL
    acts asymmetrically on up vs down moves, so long and short are distinct
    hypotheses with different backtest results even on identical bars.
    Two rules with identical (side, mask) always produce identical backtest
    results (same param_grid, same OHLCV), so keeping only one representative
    per group changes nothing about what gets evaluated, only how many times.
    """
    mask_matrix = mask_matrix.tocsr()
    mask_matrix.sort_indices()

    n_rules = mask_matrix.shape[0]
    first_seen: dict = {}
    representative_idx = []
    group_of = np.empty(n_rules, dtype=np.int64)

    for i in range(n_rules):
        row_bytes = mask_matrix.indices[mask_matrix.indptr[i]:mask_matrix.indptr[i + 1]].tobytes()
        row_key   = (sides[i], row_bytes)
        if row_key not in first_seen:
            first_seen[row_key] = i
            representative_idx.append(i)
        group_of[i] = first_seen[row_key]

    representative_idx = np.array(representative_idx, dtype=np.int64)
    return representative_idx, group_of

# =============================================================================
# COLUMN DECORRELATION (GPU) — pre-StepM redundancy filter
# =============================================================================
def _normalize_columns_for_correlation(matrix_arr: np.ndarray) -> np.ndarray:
    """Center and L2-normalize each column so that a plain dot product
    between any two columns equals their Pearson correlation coefficient."""
    matrix_norm = matrix_arr.astype(np.float32, copy=True)
    matrix_norm -= matrix_norm.mean(axis=0, keepdims=True)
    col_norms = np.linalg.norm(matrix_norm, axis=0)
    col_norms[col_norms == 0] = 1.0  # guard degenerate constant columns
    matrix_norm /= col_norms[None, :]
    return matrix_norm


def _sequential_decorrelate_within_batch(candidate_norm: np.ndarray, threshold: float) -> np.ndarray:
    """
    Exact greedy decorrelation among candidates of a single batch, preserving
    their given (already shuffled) order. Two candidates in the same batch
    may be correlated with each other even though neither is correlated with
    any already-accepted survivor — this phase catches that case, keeping
    the batched algorithm exactly equivalent to a column-by-column pass.
    Runs on CPU: the working set is small (batch_size x batch_size) and the
    dependency chain is strictly sequential, so GPU offload would add kernel
    launch overhead without any parallelism to exploit.
    """
    n_candidates = candidate_norm.shape[1]
    keep_mask = np.zeros(n_candidates, dtype=bool)
    if n_candidates == 0:
        return keep_mask

    intra_corr = candidate_norm.T @ candidate_norm  # (n_candidates, n_candidates)

    accepted_idx = []
    for i in range(n_candidates):
        if accepted_idx and intra_corr[i, accepted_idx].max() > threshold:
            continue
        keep_mask[i] = True
        accepted_idx.append(i)

    return keep_mask


def _ensure_survivor_capacity(survivors_gpu: cp.ndarray, n_survivors: int, n_new: int, n_days: int) -> cp.ndarray:
    """Doubles the survivors VRAM buffer when it can no longer fit n_new
    additional columns, copying existing survivors into the grown buffer."""
    capacity = survivors_gpu.shape[1]
    if n_survivors + n_new <= capacity:
        return survivors_gpu

    new_capacity = capacity
    while n_survivors + n_new > new_capacity:
        new_capacity *= 2

    grown = cp.empty((n_days, new_capacity), dtype=cp.float32)
    grown[:, :n_survivors] = survivors_gpu[:, :n_survivors]
    return grown


def _max_corr_against_survivors_gpu(
    batch_gpu: cp.ndarray,
    survivors_gpu: cp.ndarray,
    n_survivors: int,
    chunk_size: int,
) -> cp.ndarray:
    """Max correlation of each batch column against all current survivors,
    chunked over the survivor axis to bound peak VRAM regardless of how
    large the survivors set grows."""
    n_batch_cols = batch_gpu.shape[1]
    max_corr = cp.zeros(n_batch_cols, dtype=cp.float32)

    for start in range(0, n_survivors, chunk_size):
        end = min(start + chunk_size, n_survivors)
        corr_block = batch_gpu.T @ survivors_gpu[:, start:end]
        cp.maximum(max_corr, corr_block.max(axis=1), out=max_corr)

    return max_corr


def batch_decorrelate_columns(
    matrix_arr: np.ndarray,
    col_names: np.ndarray,
    threshold: float = DECORRELATE_THRESHOLD,
    batch_size: int = DECORRELATE_BATCH_SIZE,
    survivor_chunk_size: int = GPU_SURVIVOR_CHUNK,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Greedy random-order decorrelation, exact and order-equivalent to a
    column-by-column pass. GPU-accelerated: the dominant cost — comparing
    each batch against the growing set of survivors — runs as a chunked
    matmul on the GPU. The small, strictly sequential intra-batch pass
    (phase b) stays on CPU.

      1. Columns are visited in a fixed random order.
      2. Each batch is tested against the survivors accepted so far via a
         GPU matmul, chunked over the survivor axis.
      3. Survivors of step 2 are decorrelated exactly against each other
         in-order on CPU, since two candidates in the same batch may be
         correlated with each other but not yet with any accepted survivor.

    No column is ever chosen based on backtest performance — only its
    position in the random order determines survival. This must hold to
    avoid the max-of-noise selection bias that stepM.py is designed to
    control for.

    The normalized matrix and the growing survivors buffer for phase 2 stay
    in VRAM; only the current batch and the intra-batch phase move through
    host RAM. Requires a CUDA GPU — there is no CPU fallback.

    Returns:
        survivor_positions: indices into the ORIGINAL matrix_arr/col_names
            of the columns that survived deduplication.
        dropped_positions: indices into the ORIGINAL matrix_arr/col_names
            of the columns discarded as redundant.
    """
    n_days, n_cols = matrix_arr.shape
    rng = np.random.default_rng(seed)
    shuffled_order = rng.permutation(n_cols)

    matrix_norm = _normalize_columns_for_correlation(matrix_arr)

    survivors_gpu = cp.empty((n_days, GPU_INITIAL_CAPACITY), dtype=cp.float32)
    n_survivors = 0
    survivor_chunks = []
    dropped_chunks = []

    # DIAGNOSTIC ONLY — timing breakdown, remove once the bottleneck is confirmed.
    transfer_time = 0.0
    gpu_compare_time = 0.0
    cpu_intra_time = 0.0

    n_batches = int(np.ceil(n_cols / batch_size))
    desc = f"DECORRELATE GPU ({batch_size} cols/batch)"

    for batch_start in tqdm(range(0, n_cols, batch_size), desc=desc, total=n_batches, dynamic_ncols=True):
        batch_end = min(batch_start + batch_size, n_cols)
        batch_positions = shuffled_order[batch_start:batch_end]
        batch_norm = matrix_norm[:, batch_positions]

        _t0 = _time.time()
        batch_gpu = cp.asarray(batch_norm)
        cp.cuda.Stream.null.synchronize()
        transfer_time += _time.time() - _t0

        if n_survivors > 0:
            _t0 = _time.time()
            max_corr = _max_corr_against_survivors_gpu(batch_gpu, survivors_gpu, n_survivors, survivor_chunk_size)
            passes_survivors = cp.asnumpy(max_corr) <= threshold
            cp.cuda.Stream.null.synchronize()
            gpu_compare_time += _time.time() - _t0
        else:
            passes_survivors = np.ones(batch_gpu.shape[1], dtype=bool)

        candidate_positions   = batch_positions[passes_survivors]
        candidate_norm        = batch_norm[:, passes_survivors]
        rejected_vs_survivors = batch_positions[~passes_survivors]

        _t0 = _time.time()
        keep_mask = _sequential_decorrelate_within_batch(candidate_norm, threshold)
        cpu_intra_time += _time.time() - _t0

        accepted_positions     = candidate_positions[keep_mask]
        rejected_within_batch  = candidate_positions[~keep_mask]

        n_accepted = accepted_positions.shape[0]
        survivors_gpu = _ensure_survivor_capacity(survivors_gpu, n_survivors, n_accepted, n_days)
        if n_accepted > 0:
            survivors_gpu[:, n_survivors:n_survivors + n_accepted] = cp.asarray(candidate_norm[:, keep_mask])
        n_survivors += n_accepted

        survivor_chunks.append(accepted_positions)
        dropped_chunks.append(rejected_vs_survivors)
        dropped_chunks.append(rejected_within_batch)

    logger.info(
        f"TIMING BREAKDOWN — transfer={transfer_time:.1f}s  "
        f"gpu_compare={gpu_compare_time:.1f}s  cpu_intra_batch={cpu_intra_time:.1f}s"
    )

    survivor_positions = np.concatenate(survivor_chunks)
    dropped_positions  = np.concatenate(dropped_chunks)

    del survivors_gpu
    cp.get_default_memory_pool().free_all_blocks()

    n_survivors_count = survivor_positions.shape[0]
    n_dropped_count   = dropped_positions.shape[0]

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  DECORRELATION FILTER (GPU, pre-StepM, threshold ρ>{threshold})")
    logger.info(f"{'─' * 70}")
    logger.info(f"  {'total columns (post-backtest)':<{METRIC_LABEL_WIDTH}} : {n_cols:,}")
    logger.info(f"  {'dropped (redundant)':<{METRIC_LABEL_WIDTH}} : {n_dropped_count:,} / {n_cols:,} │ {n_dropped_count / n_cols:.4%}")
    logger.info(f"  {'kept (survivors)':<{METRIC_LABEL_WIDTH}} : {n_survivors_count:,} / {n_cols:,} │ {n_survivors_count / n_cols:.4%}")
    logger.info(f"{'─' * 70}\n")

    return survivor_positions, dropped_positions

# =============================================================================
# PER-TIMEFRAME REDUNDANCY DIAGNOSTIC
# =============================================================================
def run_redundancy_diagnostic_for_timeframe(timeframe: str) -> None:

    start = time.time()
    timings = {}

    _t0 = time.time()
    ohlcv_is = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        timeframe         = timeframe,
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )
    ohlcv_is  = select_top_n_by_volume(ohlcv_is, N_SYMBOLS)
    ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)
    timings["universe_load"] = time.time() - _t0

    _t0 = time.time()
    sample_arr = next(iter(ohlcv_arr.values()))
    all_rules  = generate_all_rules(sample_arr, max_depth=MAX_DEPTH)
    for i, rule in enumerate(all_rules):
        rule["rule_id"] = f"{i:06d}_{timeframe}_{rule['side']}"
    timings["rule_generation"] = time.time() - _t0

    _t0 = time.time()
    signal_mask_matrix = build_signal_mask_matrix(all_rules, ohlcv_arr)
    timings["signal_mask_build"] = time.time() - _t0

    _t0 = time.time()
    representative_idx, group_of = deduplicate_exact_signal_masks(
        signal_mask_matrix, [r["side"] for r in all_rules],
    )
    timings["unique_masks"] = time.time() - _t0

    n_rules_total         = len(all_rules)
    n_unique_signal_masks = representative_idx.shape[0]
    n_dropped_exact        = n_rules_total - n_unique_signal_masks

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  SIGNAL MASK CLEANING (pre-backtest, exact match) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  {'total rules (pre TP/SL grid)':<{METRIC_LABEL_WIDTH}} : {n_rules_total:,}")
    logger.info(f"  {'dropped as exact duplicates':<{METRIC_LABEL_WIDTH}} : {n_dropped_exact:,} / {n_rules_total:,} │ {n_dropped_exact / n_rules_total:.4%}")
    logger.info(f"  {'kept (unique signal masks)':<{METRIC_LABEL_WIDTH}} : {n_unique_signal_masks:,} / {n_rules_total:,} │ {n_unique_signal_masks / n_rules_total:.4%}")
    logger.info(f"{'─' * 70}\n")

    # Feed only the deduplicated rules forward — mirrors pipe_signal_cleaning
    # in production, so the backtest and everything downstream (decorrelation,
    # M_eff) reflects the same reduced set as the real pipeline.
    all_rules = [all_rules[i] for i in representative_idx]

    _t0 = time.time()
    _, n_combos, matrix_arr, col_names = pipe_backtesting(
        rules        = all_rules,
        ohlcv_arr    = ohlcv_arr,
        param_grid   = PARAM_GRID,
        order_amount = ORDER_AMOUNT,
        timeframe    = timeframe,
    )
    timings["backtest_full"] = time.time() - _t0

    n_cols_total = matrix_arr.shape[1]
    if n_cols_total < 2:
        logger.warning(f"REDUNDANCY ── {timeframe} ── insufficient columns — skipping")
        return

    _t0 = time.time()
    survivor_idx, dropped_idx = batch_decorrelate_columns(
        matrix_arr, col_names,
        threshold=DECORRELATE_THRESHOLD,
        batch_size=DECORRELATE_BATCH_SIZE,
        survivor_chunk_size=GPU_SURVIVOR_CHUNK,
        seed=RANDOM_SEED,
    )
    matrix_arr = matrix_arr[:, survivor_idx]
    col_names = np.asarray(col_names)[survivor_idx]
    n_cols_total = matrix_arr.shape[1]
    timings["decorrelation"] = time.time() - _t0

    elapsed = int(time.time() - start)

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  TIMING SUMMARY ── {timeframe}")
    logger.info(f"{'─' * 70}")
    for step_name, step_seconds in timings.items():
        logger.info(f"  {step_name:<20} : {step_seconds:>8.1f}s")
    logger.info(f"  {'─' * 20}")
    logger.info(f"  {'total':<20} : {sum(timings.values()):>8.1f}s")
    logger.info(f"{'─' * 70}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  REDUNDANCY DIAGNOSTIC START")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES     : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS      : {N_SYMBOLS}")
    logger.info(f"{'─' * 115}\n")

    for timeframe in TIMEFRAMES:
        run_redundancy_diagnostic_for_timeframe(timeframe)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")