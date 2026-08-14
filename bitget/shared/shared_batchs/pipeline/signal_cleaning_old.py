import os
import time
import logging
import numpy as np
import cupy as cp
from scipy import sparse
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from signals.condition_bank import ConditionBank

logger = logging.getLogger("BOT_batch.pipeline.signal_cleaning")

# =============================================================================
# CONFIG
# =============================================================================
SIGNAL_MASK_N_JOBS       = -1
SIGNAL_MASK_CHUNK_SIZE   = None  # None -> auto-sized from n_jobs and rule count
DECORRELATE_THRESHOLD    = 0.9
DECORRELATE_BATCH_SIZE   = 1000
GPU_SURVIVOR_CHUNK       = 20_000  # matmul sub-chunk size (both levels below); fixed for reproducibility
GPU_SURVIVOR_VRAM_BUDGET_BYTES = 6 * 1024**3  # fixed 6 GiB budget for the VRAM-resident survivor buffer
                                    # (level 1, no transfer cost). Overflow beyond this spills to level 2
                                    # (host RAM, fetched per chunk) — conservative so it coexists safely
                                    # with batch_gpu and the rest of the pipeline sharing the GPU.
RANDOM_SEED              = 42
METRIC_LABEL_WIDTH       = 36  # fixed label width so all metric-block prints align

# =============================================================================
# SIGNAL MASK BUILD
# =============================================================================
def _auto_chunk_size(n_rules: int, n_jobs: int) -> int:
    """Sizes rule chunks so there are enough chunks to keep all workers busy
    (~8 chunks per worker) while still amortizing ConditionBank construction
    over many rules per chunk."""
    workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    return max(1, n_rules // (workers * 8))


def _chunk_list(items: list, chunk_size: int) -> list:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _compute_signal_mask_chunk(rule_chunk: list, ohlcv_arr: dict, symbols: list) -> "sparse.csr_matrix":

    banks = {sym: ConditionBank(ohlcv_arr[sym]) for sym in symbols}

    chunk_rows = []
    for rule in rule_chunk:
        signal_parts = [
            np.asarray(rule["signal_fn"](ohlcv_arr[sym], live_trading=False, bank=banks[sym]), dtype=bool)
            for sym in symbols
        ]
        chunk_rows.append(np.concatenate(signal_parts))

    chunk_dense = np.vstack(chunk_rows)
    return sparse.csr_matrix(chunk_dense)


def build_signal_mask_matrix(
    all_rules: list,
    ohlcv_arr: dict,
    n_jobs: int = SIGNAL_MASK_N_JOBS,
    chunk_size: int = SIGNAL_MASK_CHUNK_SIZE,
) -> "sparse.csr_matrix":


    symbols = list(ohlcv_arr.keys())
    effective_chunk_size = chunk_size or _auto_chunk_size(len(all_rules), n_jobs)
    rule_chunks = _chunk_list(all_rules, effective_chunk_size)

    with tqdm_joblib(tqdm(desc="SIGNAL MASK BUILD", total=len(rule_chunks), dynamic_ncols=True)):
        chunk_sparses = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_signal_mask_chunk)(chunk, ohlcv_arr, symbols)
            for chunk in rule_chunks
        )

    return sparse.vstack(chunk_sparses, format="csr")

def _pack_signal_mask_row(bool_row: np.ndarray) -> bytes:

    return np.packbits(bool_row).tobytes()


def _compute_signal_mask_keys_chunk(rule_chunk: list, ohlcv_arr: dict, symbols: list) -> list:

    banks = {sym: ConditionBank(ohlcv_arr[sym]) for sym in symbols}

    chunk_keys = []
    for rule in rule_chunk:
        signal_parts = [
            np.asarray(rule["signal_fn"](ohlcv_arr[sym], live_trading=False, bank=banks[sym]), dtype=bool)
            for sym in symbols
        ]
        row = np.concatenate(signal_parts)
        chunk_keys.append((rule["side"], _pack_signal_mask_row(row)))
    return chunk_keys


def build_signal_mask_keys(
    all_rules: list,
    ohlcv_arr: dict,
    n_jobs: int = SIGNAL_MASK_N_JOBS,
    chunk_size: int = SIGNAL_MASK_CHUNK_SIZE,
) -> list:

    symbols = list(ohlcv_arr.keys())
    effective_chunk_size = chunk_size or _auto_chunk_size(len(all_rules), n_jobs)
    rule_chunks = _chunk_list(all_rules, effective_chunk_size)

    with tqdm_joblib(tqdm(desc="SIGNAL MASK BUILD", total=len(rule_chunks), dynamic_ncols=True)):
        chunk_keys_lists = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_signal_mask_keys_chunk)(chunk, ohlcv_arr, symbols)
            for chunk in rule_chunks
        )

    all_keys = []
    for chunk_keys in chunk_keys_lists:
        all_keys.extend(chunk_keys)
    return all_keys


def deduplicate_exact_signal_keys(signal_keys: list) -> tuple:

    n_rules = len(signal_keys)
    first_seen: dict = {}
    representative_idx = []
    group_of = np.empty(n_rules, dtype=np.int64)

    for i, key in enumerate(signal_keys):
        if key not in first_seen:
            first_seen[key] = i
            representative_idx.append(i)
        group_of[i] = first_seen[key]

    representative_idx = np.array(representative_idx, dtype=np.int64)
    return representative_idx, group_of



def deduplicate_exact_signal_masks(mask_matrix: "sparse.csr_matrix", sides: list) -> tuple:

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
# PIPE SIGNAL CLEANING — pre-backtest step
# =============================================================================
def pipe_signal_cleaning(
    rules: list,
    ohlcv_arr: dict,
    timeframe: str = "",
    n_jobs: int = SIGNAL_MASK_N_JOBS,
) -> list:

    signal_keys = build_signal_mask_keys(rules, ohlcv_arr, n_jobs=n_jobs)
    representative_idx, group_of = deduplicate_exact_signal_keys(signal_keys)

    n_rules_total = len(rules)
    n_unique      = representative_idx.shape[0]
    n_dropped     = n_rules_total - n_unique

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  SIGNAL MASK CLEANING (pre-backtest, exact match) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  {'total rules (pre TP/SL grid)':<{METRIC_LABEL_WIDTH}} : {n_rules_total:,}")
    logger.info(f"  {'dropped as exact duplicates':<{METRIC_LABEL_WIDTH}} : {n_dropped:,} / {n_rules_total:,} │ {n_dropped / n_rules_total:.4%}")
    logger.info(f"  {'kept (unique signal masks)':<{METRIC_LABEL_WIDTH}} : {n_unique:,} / {n_rules_total:,} │ {n_unique / n_rules_total:.4%}")
    logger.info(f"{'─' * 70}\n")

    return [rules[i] for i in representative_idx]

# =============================================================================
# COLUMN DECORRELATION (GPU) — post-backtest, pre-StepM redundancy filter
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


def _survivor_vram_capacity(n_days: int, budget_bytes: int = GPU_SURVIVOR_VRAM_BUDGET_BYTES) -> int:

    return max(1, int(budget_bytes // (n_days * 4)))


def _max_corr_against_survivors(
    batch_gpu: cp.ndarray,
    survivors_gpu: cp.ndarray,
    n_survivors_vram: int,
    overflow_position_chunks: list,
    matrix_norm: np.ndarray,
    chunk_size: int,
) -> cp.ndarray:

    n_batch_cols = batch_gpu.shape[1]
    max_corr = cp.zeros(n_batch_cols, dtype=cp.float32)

    # Tier 1 — VRAM-resident, no transfer.
    for start in range(0, n_survivors_vram, chunk_size):
        end = min(start + chunk_size, n_survivors_vram)
        corr_block = batch_gpu.T @ survivors_gpu[:, start:end]
        cp.maximum(max_corr, corr_block.max(axis=1), out=max_corr)

    # Tier 2 — overflow, fetched from host RAM per chunk.
    for positions in overflow_position_chunks:
        for start in range(0, len(positions), chunk_size):
            sub_positions = positions[start:start + chunk_size]
            overflow_chunk_gpu = cp.asarray(matrix_norm[:, sub_positions])
            corr_block = batch_gpu.T @ overflow_chunk_gpu
            cp.maximum(max_corr, corr_block.max(axis=1), out=max_corr)
            del overflow_chunk_gpu

    return max_corr


def pipe_decorrelation(
    matrix_arr: np.ndarray,
    col_names: np.ndarray,
    timeframe: str = "",
    threshold: float = DECORRELATE_THRESHOLD,
    batch_size: int = DECORRELATE_BATCH_SIZE,
    survivor_chunk_size: int = GPU_SURVIVOR_CHUNK,  # None -> auto-sized from free VRAM per batch
    seed: int = RANDOM_SEED,
) -> tuple:

    n_days, n_cols = matrix_arr.shape
    rng = np.random.default_rng(seed)
    shuffled_order = rng.permutation(n_cols)

    matrix_norm = _normalize_columns_for_correlation(matrix_arr)

    vram_capacity    = _survivor_vram_capacity(n_days)
    survivors_gpu    = cp.empty((n_days, vram_capacity), dtype=cp.float32)
    n_survivors_vram = 0
    overflow_chunks  = []  # host-side position arrays, beyond the fixed VRAM budget
    survivor_chunks  = []  # full accepted-order bookkeeping, regardless of storage tier
    dropped_chunks   = []

    n_batches = int(np.ceil(n_cols / batch_size))
    desc = f"DECORRELATE GPU ({batch_size} cols/batch)"

    for batch_start in tqdm(range(0, n_cols, batch_size), desc=desc, total=n_batches, dynamic_ncols=True):
        batch_end = min(batch_start + batch_size, n_cols)
        batch_positions = shuffled_order[batch_start:batch_end]
        batch_norm = matrix_norm[:, batch_positions]
        batch_gpu = cp.asarray(batch_norm)

        if n_survivors_vram > 0 or overflow_chunks:
            max_corr = _max_corr_against_survivors(
                batch_gpu, survivors_gpu, n_survivors_vram, overflow_chunks, matrix_norm, survivor_chunk_size
            )
            passes_survivors = cp.asnumpy(max_corr) <= threshold
        else:
            passes_survivors = np.ones(batch_gpu.shape[1], dtype=bool)

        candidate_positions   = batch_positions[passes_survivors]
        candidate_norm        = batch_norm[:, passes_survivors]
        rejected_vs_survivors = batch_positions[~passes_survivors]

        keep_mask = _sequential_decorrelate_within_batch(candidate_norm, threshold)

        accepted_positions     = candidate_positions[keep_mask]
        accepted_norm          = candidate_norm[:, keep_mask]
        rejected_within_batch  = candidate_positions[~keep_mask]

        n_accepted = accepted_positions.shape[0]
        if n_accepted > 0:
            space_left = vram_capacity - n_survivors_vram
            n_to_vram  = min(space_left, n_accepted)
            if n_to_vram > 0:
                survivors_gpu[:, n_survivors_vram:n_survivors_vram + n_to_vram] = cp.asarray(accepted_norm[:, :n_to_vram])
                n_survivors_vram += n_to_vram
            if n_to_vram < n_accepted:
                overflow_chunks.append(accepted_positions[n_to_vram:])
            survivor_chunks.append(accepted_positions)

        dropped_chunks.append(rejected_vs_survivors)
        dropped_chunks.append(rejected_within_batch)

        del batch_gpu

    survivor_positions = np.concatenate(survivor_chunks) if survivor_chunks else np.array([], dtype=np.int64)
    dropped_positions  = np.concatenate(dropped_chunks)

    del survivors_gpu
    cp.get_default_memory_pool().free_all_blocks()

    n_survivors_count = survivor_positions.shape[0]
    n_dropped_count   = dropped_positions.shape[0]

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  DECORRELATION FILTER (GPU, pre-StepM, threshold ρ>{threshold}) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  {'total columns (post-backtest)':<{METRIC_LABEL_WIDTH}} : {n_cols:,}")
    logger.info(f"  {'dropped (redundant)':<{METRIC_LABEL_WIDTH}} : {n_dropped_count:,} / {n_cols:,} │ {n_dropped_count / n_cols:.4%}")
    logger.info(f"  {'kept (survivors)':<{METRIC_LABEL_WIDTH}} : {n_survivors_count:,} / {n_cols:,} │ {n_survivors_count / n_cols:.4%}")
    logger.info(f"{'─' * 70}\n")

    return matrix_arr[:, survivor_positions], np.asarray(col_names)[survivor_positions]