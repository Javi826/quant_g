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
DECORRELATE_BATCH_SIZE   = 8192
GPU_SURVIVOR_CHUNK       = 16_384   # survivor rows compared per GPU matmul chunk
GPU_SURVIVOR_VRAM_BUDGET_BYTES = 8 * 1024**3  # fixed budget for the fp16 survivor buffer.
                                    # Preallocated ONCE at full capacity: it never grows,
                                    # never reallocates, never fragments the CuPy pool.
                                    # Survivors beyond capacity spill to the host tier.
CORR_FP16_BAND           = 4e-3     # |rho_hat - threshold| below this is re-checked in fp32,
                                    # so the accept/reject decision stays exact in fp32 even
                                    # though the bulk matmul runs in fp16.
SURVIVOR_REORDER_EVERY_N = 10       # batches between kill-count reorderings of the survivor scan
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
    """Builds one ConditionBank per symbol ONCE for the whole chunk, then
    reuses it across every rule in the chunk. Indicator computation (RSI,
    ADX, ATR, ...) is cached inside each bank, so rules sharing the same
    underlying indicator no longer recompute it per rule — only per chunk.
    Numerically identical to building a fresh bank per rule, since the
    cached indicator values are deterministic and depend only on (arr, spec).

    Returns the chunk's rows already as a sparse matrix: converting here,
    inside the worker, keeps only chunk_size rows dense at once (never the
    full rule set) and shrinks what gets shipped back through loky's IPC."""
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
    """Boolean signal mask per rule, concatenated across symbols in a fixed
    order, stored sparse since signals are mostly inactive bars. Rules are
    grouped into chunks and each chunk is computed in parallel across
    process-based workers (loky) — signal_fn is typically pure-Python/pandas
    indicator logic that holds the GIL, so a threading backend does not
    parallelize this in practice. Within a chunk, one ConditionBank per
    symbol is built once and reused across all its rules, avoiding
    redundant indicator recomputation across rules that share indicators.

    Each chunk is converted to sparse inside its worker (see
    _compute_signal_mask_chunk) and chunks are combined with sparse.vstack,
    so the full dense mask matrix (all rules at once) is never materialized —
    only one chunk's worth of dense rows exists at any time, per worker."""

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
    """Fixed-size fingerprint of a boolean row (n_bits/8 bytes, always,
    regardless of how many True values it has). Two rows are byte-identical
    iff their boolean values are identical, so this is exact — not an
    approximate hash. Unlike sparse-index storage, its size never grows
    with signal density, which is what caused the sparse matrix itself to
    blow up in RAM for high-firing-rate rules."""
    return np.packbits(bool_row).tobytes()


def _compute_signal_mask_keys_chunk(rule_chunk: list, ohlcv_arr: dict, symbols: list) -> list:
    """Same per-symbol ConditionBank reuse as _compute_signal_mask_chunk,
    but returns only a compact (side, packed_bytes) fingerprint per rule —
    never a matrix. This is what pipe_signal_cleaning consumes: it only
    ever needs to know which rules are exact duplicates, never the actual
    mask values, so nothing bigger than the fingerprints has to survive
    past this function, in the worker or in the caller."""
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
    """Streaming counterpart to build_signal_mask_matrix: computes the same
    per-rule signal masks (same chunking, same ConditionBank reuse) but
    never assembles a matrix, dense or sparse. Each chunk returns a small
    list of fixed-size fingerprints instead of full boolean rows, so peak
    memory no longer scales with signal density or total rule count."""

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
    """Exact positional dedup over (side, packed_bytes) fingerprints — same
    semantics as deduplicate_exact_signal_masks (side is part of the key,
    first-seen order preserved), operating on compact fingerprints instead
    of a sparse matrix."""
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
# PIPE SIGNAL CLEANING — pre-backtest step
# =============================================================================
def pipe_signal_cleaning(
    rules: list,
    ohlcv_arr: dict,
    timeframe: str = "",
    n_jobs: int = SIGNAL_MASK_N_JOBS,
) -> list:
    """
    Drops rules whose signal mask is an exact positional duplicate of
    another rule's trading the SAME side, before the TP/SL grid multiplies
    the cost of running them through the backtest. Keeps one representative
    per duplicate group; the dropped rules would have produced byte-identical
    backtest results regardless of TP/SL, since it is the entry signal — not
    the exit — that is identical here. long and short are never merged with
    each other even when their entry mask matches exactly.
    """
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
def _survivor_capacity_fp16(n_days: int, budget_bytes: int = GPU_SURVIVOR_VRAM_BUDGET_BYTES) -> int:
    """Fixed VRAM budget -> number of fp16 survivor rows the level-1 buffer
    holds. Deterministic given n_days, so the run is reproducible regardless
    of what else is using the GPU, and the buffer is allocated ONCE at this
    full capacity: it never grows and never reallocates."""
    return max(1, int(budget_bytes // (n_days * 2)))


def _normalize_rows_gpu(block_gpu: cp.ndarray) -> cp.ndarray:
    """(n_days, m) column block -> (m, n_days) row-major fp32 block whose rows
    are centered and L2-normalized, so a plain dot product between any two
    rows equals their Pearson correlation. Row-major layout means a chunk of
    consecutive rows is contiguous, which is what makes gathers cheap."""
    block = block_gpu.astype(cp.float32, copy=False)
    block = block - block.mean(axis=0, keepdims=True)
    norms = cp.linalg.norm(block, axis=0)
    norms = cp.where(norms == 0, cp.float32(1.0), norms)  # guard degenerate constant columns
    block = block / norms[None, :]
    return cp.ascontiguousarray(block.T)


def _upload_normalized_rows(matrix_arr: np.ndarray, positions: np.ndarray) -> cp.ndarray:
    """Gathers the given columns from the host matrix and returns them
    normalized as (len(positions), n_days) fp32 on the GPU. Replaces the
    full host-side normalized copy of the matrix: each column is gathered
    exactly once for its own batch, so nothing of matrix-scale is duplicated
    in host RAM."""
    block_host = np.take(matrix_arr, positions, axis=1)
    return _normalize_rows_gpu(cp.asarray(block_host))


def _survivor_chunk_verdict(
    alive_fp16: cp.ndarray,
    alive_fp32: cp.ndarray,
    chunk_fp16: cp.ndarray,
    chunk_positions: np.ndarray,
    matrix_arr: np.ndarray,
    threshold: float,
    band: float,
) -> tuple:
    """
    Rejection verdict of one survivor chunk against the columns still alive
    in the current batch, plus the per-survivor kill count used to reorder
    the scan.

    The bulk matmul runs in fp16 (tensor cores, several times the fp32
    throughput). fp16 cannot resolve the threshold on its own, so the verdict
    is split into three zones: above threshold+band is a definite reject,
    below threshold-band is a definite pass, and only the narrow ambiguous
    band in between is recomputed in fp32 — against just the handful of
    survivors actually near the threshold for those columns, not the whole
    chunk. The accept/reject decision is therefore exact in fp32 while
    almost all of the arithmetic happens in fp16.

    Returns:
        rejected: (n_alive,) bool on GPU.
        kills:    (n_chunk,) int32 on GPU — how many alive columns each
                  survivor rejected, accumulated to reorder the scan.
    """
    corr_fp16 = alive_fp16 @ chunk_fp16.T
    max_corr  = corr_fp16.max(axis=1).astype(cp.float32)

    rejected = max_corr > np.float32(threshold + band)
    kills    = (corr_fp16 > cp.float16(threshold)).sum(axis=0).astype(cp.int32)

    ambiguous = (~rejected) & (max_corr > np.float32(threshold - band))
    if bool(ambiguous.any()):
        ambiguous_rows = cp.nonzero(ambiguous)[0]
        near_threshold = corr_fp16[ambiguous_rows] > cp.float16(threshold - band)
        involved_cols  = cp.nonzero(near_threshold.any(axis=0))[0]

        sub_positions = chunk_positions[cp.asnumpy(involved_cols)]
        sub_fp32      = _upload_normalized_rows(matrix_arr, sub_positions)
        exact_corr    = alive_fp32[ambiguous_rows] @ sub_fp32.T
        rejected[ambiguous_rows] = exact_corr.max(axis=1) > np.float32(threshold)

    return rejected, kills


def _decorrelate_within_batch_gpu(candidate_fp32: cp.ndarray, threshold: float) -> np.ndarray:
    """
    Exact greedy decorrelation among the candidates of a single batch,
    preserving their given (already shuffled) order. Two candidates in the
    same batch may be correlated with each other even though neither is
    correlated with any already-accepted survivor — this phase catches that
    case, keeping the batched algorithm equivalent to a column-by-column pass.

    The Gram matrix is a single fp32 GEMM on the GPU: it is quadratic in the
    batch size and was the dominant CPU cost before. The greedy loop itself
    stays on CPU — its dependency chain is strictly sequential, so on GPU it
    would cost one kernel launch and one device sync per iteration. It is
    driven by a running per-candidate maximum against the accepted set,
    updated with a contiguous row of the (symmetric) Gram matrix, instead of
    re-gathering a growing index list every iteration.
    """
    n_candidates = candidate_fp32.shape[0]
    keep_mask = np.zeros(n_candidates, dtype=bool)
    if n_candidates == 0:
        return keep_mask

    gram = cp.asnumpy(candidate_fp32 @ candidate_fp32.T)
    max_to_accepted = np.full(n_candidates, -np.inf, dtype=np.float32)

    for i in range(n_candidates):
        if max_to_accepted[i] > threshold:
            continue
        keep_mask[i] = True
        np.maximum(max_to_accepted, gram[i], out=max_to_accepted)

    return keep_mask


def pipe_decorrelation(
    matrix_arr: np.ndarray,
    col_names: np.ndarray,
    timeframe: str = "",
    threshold: float = DECORRELATE_THRESHOLD,
    batch_size: int = DECORRELATE_BATCH_SIZE,
    survivor_chunk_size: int = GPU_SURVIVOR_CHUNK,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Post-backtest, pre-StepM step: greedy random-order decorrelation of P&L
    columns, equivalent to a column-by-column pass.

      1. Columns are visited in a fixed random order, one batch at a time.
      2. Each batch is tested against the survivors accepted so far. The
         criterion is existential — one survivor above the threshold is
         enough — so survivors are scanned in chunks and the batch is
         compacted to the columns still alive after every chunk. A column
         rejected by the first chunk never pays for the rest, and once the
         batch is empty the remaining chunks are skipped entirely. Chunks
         are visited most-lethal-first (by accumulated kill count), which
         makes that early exit fire as soon as possible. Neither the chunk
         order nor the compaction changes the verdict: max is an
         order-independent reduction and rejection is absorbing.
      3. Survivors of step 2 are decorrelated exactly against each other
         in-order, since two candidates in the same batch may be correlated
         with each other but not yet with any accepted survivor.

    Memory: the survivor buffer is fp16 and preallocated ONCE at a fixed
    capacity derived from a fixed VRAM budget — it never grows, never
    reallocates, and never fragments the pool, which is what used to make
    this stage run out of VRAM. Survivors past that capacity are tracked as
    host column positions and fetched per chunk, so the stage degrades in
    speed rather than failing. No normalized copy of the matrix is kept in
    host RAM: each batch is gathered from matrix_arr and normalized on GPU.

    No column is ever chosen based on backtest performance — only its
    position in the random order determines survival. This must hold to
    avoid the max-of-noise selection bias that stepM.py is designed to
    control for.

    Requires a CUDA GPU — there is no CPU fallback.

    Returns:
        matrix_arr: filtered to the surviving columns.
        col_names:  filtered to the surviving columns, same order as matrix_arr.
    """
    start = time.time()
    n_days, n_cols = matrix_arr.shape
    rng = np.random.default_rng(seed)
    shuffled_order = rng.permutation(n_cols)

    vram_capacity = _survivor_capacity_fp16(n_days)
    survivors_fp16 = cp.empty((vram_capacity, n_days), dtype=cp.float16)
    chunk_scratch  = cp.empty((survivor_chunk_size, n_days), dtype=cp.float16)

    n_vram              = 0
    vram_positions      = np.empty(vram_capacity, dtype=np.int64)
    vram_kills          = np.zeros(vram_capacity, dtype=np.int64)
    vram_scan_order     = np.empty(0, dtype=np.int64)
    overflow_positions  = []   # host-only tier, beyond the fixed VRAM budget
    survivor_chunks     = []   # accepted order, regardless of storage tier

    n_batches = int(np.ceil(n_cols / batch_size))
    desc = f"DECORRELATE GPU ({batch_size} cols/batch)"

    for batch_index, batch_start in enumerate(
        tqdm(range(0, n_cols, batch_size), desc=desc, total=n_batches, dynamic_ncols=True)
    ):
        batch_end       = min(batch_start + batch_size, n_cols)
        batch_positions = shuffled_order[batch_start:batch_end]

        batch_fp32 = _upload_normalized_rows(matrix_arr, batch_positions)
        batch_fp16 = batch_fp32.astype(cp.float16)

        alive_rows = cp.arange(batch_fp32.shape[0], dtype=cp.int64)

        # ---- phase a: reject against already-accepted survivors, most lethal first
        for scan_start in range(0, n_vram, survivor_chunk_size):
            if alive_rows.size == 0:
                break
            scan_idx = vram_scan_order[scan_start:scan_start + survivor_chunk_size]
            n_chunk  = scan_idx.shape[0]
            cp.take(survivors_fp16, cp.asarray(scan_idx), axis=0, out=chunk_scratch[:n_chunk])

            rejected, kills = _survivor_chunk_verdict(
                alive_fp16      = batch_fp16[alive_rows],
                alive_fp32      = batch_fp32[alive_rows],
                chunk_fp16      = chunk_scratch[:n_chunk],
                chunk_positions = vram_positions[scan_idx],
                matrix_arr      = matrix_arr,
                threshold       = threshold,
                band            = CORR_FP16_BAND,
            )
            vram_kills[scan_idx] += cp.asnumpy(kills)
            alive_rows = alive_rows[~rejected]

        for positions in overflow_positions:
            for scan_start in range(0, positions.shape[0], survivor_chunk_size):
                if alive_rows.size == 0:
                    break
                sub_positions = positions[scan_start:scan_start + survivor_chunk_size]
                overflow_fp32 = _upload_normalized_rows(matrix_arr, sub_positions)
                rejected, _ = _survivor_chunk_verdict(
                    alive_fp16      = batch_fp16[alive_rows],
                    alive_fp32      = batch_fp32[alive_rows],
                    chunk_fp16      = overflow_fp32.astype(cp.float16),
                    chunk_positions = sub_positions,
                    matrix_arr      = matrix_arr,
                    threshold       = threshold,
                    band            = CORR_FP16_BAND,
                )
                alive_rows = alive_rows[~rejected]
                del overflow_fp32
            if alive_rows.size == 0:
                break

        # ---- phase b: decorrelate the surviving candidates against each other
        candidate_rows = cp.asnumpy(alive_rows)
        candidate_fp32 = batch_fp32[alive_rows]
        keep_mask      = _decorrelate_within_batch_gpu(candidate_fp32, threshold)

        accepted_rows      = candidate_rows[keep_mask]
        accepted_positions = batch_positions[accepted_rows]
        n_accepted         = accepted_positions.shape[0]

        if n_accepted > 0:
            n_to_vram = min(vram_capacity - n_vram, n_accepted)
            if n_to_vram > 0:
                accepted_gpu = batch_fp16[cp.asarray(accepted_rows[:n_to_vram])]
                survivors_fp16[n_vram:n_vram + n_to_vram] = accepted_gpu
                vram_positions[n_vram:n_vram + n_to_vram] = accepted_positions[:n_to_vram]
                vram_kills[n_vram:n_vram + n_to_vram] = 0
                vram_scan_order = np.concatenate(
                    [vram_scan_order, np.arange(n_vram, n_vram + n_to_vram, dtype=np.int64)]
                )
                n_vram += n_to_vram
                del accepted_gpu
            if n_to_vram < n_accepted:
                overflow_positions.append(accepted_positions[n_to_vram:])
            survivor_chunks.append(accepted_positions)

        if (batch_index + 1) % SURVIVOR_REORDER_EVERY_N == 0 and n_vram > 0:
            vram_scan_order = np.argsort(-vram_kills[:n_vram], kind="stable")

        del batch_fp32, batch_fp16, candidate_fp32

    survivor_positions = (
        np.concatenate(survivor_chunks) if survivor_chunks else np.array([], dtype=np.int64)
    )

    del survivors_fp16, chunk_scratch
    cp.get_default_memory_pool().free_all_blocks()

    n_survivors_count = survivor_positions.shape[0]
    n_dropped_count   = n_cols - n_survivors_count
    n_overflow        = int(sum(p.shape[0] for p in overflow_positions))
    elapsed           = time.time() - start

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  DECORRELATION FILTER (GPU, pre-StepM, threshold ρ>{threshold}) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  {'total columns (post-backtest)':<{METRIC_LABEL_WIDTH}} : {n_cols:,}")
    logger.info(f"  {'dropped (redundant)':<{METRIC_LABEL_WIDTH}} : {n_dropped_count:,} / {n_cols:,} │ {n_dropped_count / n_cols:.4%}")
    logger.info(f"  {'kept (survivors)':<{METRIC_LABEL_WIDTH}} : {n_survivors_count:,} / {n_cols:,} │ {n_survivors_count / n_cols:.4%}")
    logger.info(f"  {'survivor VRAM capacity (fp16)':<{METRIC_LABEL_WIDTH}} : {vram_capacity:,}")
    logger.info(f"  {'survivors spilled to host tier':<{METRIC_LABEL_WIDTH}} : {n_overflow:,}")
    logger.info(f"  {'elapsed':<{METRIC_LABEL_WIDTH}} : {elapsed:,.1f}s")
    logger.info(f"{'─' * 70}\n")

    return matrix_arr[:, survivor_positions], np.asarray(col_names)[survivor_positions]