#BOT_batch/main_REDUN.py
import os
import sys
import time
import logging
import numpy as np
from scipy.linalg import eigvalsh
from scipy import sparse
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
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
N_SAMPLE_COLS            = 10000
RANDOM_SEED              = 42
HIGH_CORR_THRESHOLD      = 0.8
SIGNAL_JACCARD_THRESHOLD = 0.99
SIGNAL_JACCARD_CHUNK_SIZE = 2000  # rules processed per chunk during pairwise Jaccard
SIGNAL_MASK_N_JOBS       = -1

# =============================================================================
# STRATIFIED COLUMN SAMPLING — spreads the sample across distinct rule_ids
# instead of clustering inside a few rules' 25-combo blocks.
# =============================================================================
def stratified_sample_column_indices(col_names, n_samples: int, seed: int) -> np.ndarray:

    col_names = np.asarray(col_names)
    rng = np.random.default_rng(seed)

    rule_ids = np.array([str(name).split("__", 1)[0] for name in col_names])
    unique_rule_ids = np.unique(rule_ids)
    rng.shuffle(unique_rule_ids)

    rule_id_to_col_indices = {
        rid: rng.permutation(np.flatnonzero(rule_ids == rid)).tolist()
        for rid in unique_rule_ids
    }

    n_total_available = col_names.shape[0]
    n_target = min(n_samples, n_total_available)

    sampled_indices = []
    active_rule_ids = list(unique_rule_ids)
    while len(sampled_indices) < n_target and active_rule_ids:
        next_active_rule_ids = []
        for rid in active_rule_ids:
            pool = rule_id_to_col_indices[rid]
            if pool:
                sampled_indices.append(pool.pop())
                if len(sampled_indices) >= n_target:
                    break
            if pool:
                next_active_rule_ids.append(rid)
        active_rule_ids = next_active_rule_ids

    return np.array(sampled_indices[:n_target], dtype=np.int64)

# =============================================================================
# EFFECTIVE NUMBER OF INDEPENDENT HYPOTHESES — Li & Ji (2005)
# =============================================================================
def effective_number_li_ji(corr_matrix: np.ndarray) -> float:

    # eigvalsh_only, MRRR driver: eigenvalues-only symmetric eigensolver,
    # numerically equivalent to np.linalg.eigvalsh but faster at this size
    # since no eigenvectors are computed.
    eigenvalues = eigvalsh(corr_matrix, driver="evr", check_finite=False)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    fractional_part = eigenvalues - np.floor(eigenvalues)
    contribution = np.where(eigenvalues >= 1.0, 1.0, fractional_part)

    return float(np.sum(contribution))

# =============================================================================
# HIGH-CORRELATION PAIR COUNT — direct pairwise redundancy check, complementary
# to M_eff (captures near-duplicate columns instead of diffuse correlation)
# =============================================================================
def count_high_correlation_pairs(corr_matrix: np.ndarray, threshold: float = HIGH_CORR_THRESHOLD) -> tuple:

    n_cols = corr_matrix.shape[0]
    upper_tri_idx = np.triu_indices(n_cols, k=1)
    upper_tri_values = corr_matrix[upper_tri_idx]

    n_total_pairs = upper_tri_values.shape[0]
    high_corr_pair_mask = np.abs(upper_tri_values) > threshold
    n_high_corr_pairs = int(np.count_nonzero(high_corr_pair_mask))
    high_corr_fraction = n_high_corr_pairs / n_total_pairs if n_total_pairs > 0 else 0.0

    redundant_col_mask = np.zeros(n_cols, dtype=bool)
    redundant_col_mask[upper_tri_idx[0][high_corr_pair_mask]] = True
    redundant_col_mask[upper_tri_idx[1][high_corr_pair_mask]] = True
    n_redundant_cols = int(redundant_col_mask.sum())
    redundant_col_fraction = n_redundant_cols / n_cols if n_cols > 0 else 0.0

    return n_high_corr_pairs, n_total_pairs, high_corr_fraction, n_redundant_cols, redundant_col_fraction

# =============================================================================
# SIGNAL MASK REDUNDANCY — pre-backtest, exact-duplicate detection based on
# which bars each rule's raw signal fires on (before TP/SL is applied).
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
    in parallel across rules — each rule's mask is independent."""

    symbols = list(ohlcv_arr.keys())

    with tqdm_joblib(tqdm(desc="SIGNAL MASK BUILD", total=len(all_rules), dynamic_ncols=True)):
        row_arrays = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_compute_signal_mask_row)(rule, ohlcv_arr, symbols)
            for rule in all_rules
        )

    mask_dense = np.vstack(row_arrays)
    return sparse.csr_matrix(mask_dense)


def count_high_jaccard_signal_pairs(
    mask_matrix: "sparse.csr_matrix",
    threshold: float = SIGNAL_JACCARD_THRESHOLD,
    chunk_size: int = SIGNAL_JACCARD_CHUNK_SIZE,
) -> tuple:
    """Jaccard similarity (active-bar overlap) between every pair of rule
    signal masks, row-chunked to avoid materializing the full n_rules x
    n_rules dense matrix at once."""

    n_rules = mask_matrix.shape[0]
    n_total_pairs = n_rules * (n_rules - 1) // 2
    if n_total_pairs == 0:
        return 0, 0, 0.0, 0, 0.0

    # boolean @ boolean saturates via OR instead of counting overlap — must
    # cast to a numeric dtype before the matmul, or every intersection
    # collapses to 0/1 regardless of true overlap size.
    mask_matrix_numeric = mask_matrix.astype(np.float32)
    row_sums = np.asarray(mask_matrix_numeric.sum(axis=1)).flatten()
    n_high_jaccard_pairs = 0
    redundant_rule_mask = np.zeros(n_rules, dtype=bool)

    for start in range(0, n_rules, chunk_size):
        end = min(start + chunk_size, n_rules)

        intersection_block = (mask_matrix_numeric[start:end] @ mask_matrix_numeric.T).toarray()
        union_block = row_sums[start:end, None] + row_sums[None, :] - intersection_block

        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard_block = np.where(union_block > 0, intersection_block / union_block, 0.0)

        # Restrict to j > global_i within this block to count each pair once.
        for local_i in range(end - start):
            global_i = start + local_i
            partner_offset = np.flatnonzero(jaccard_block[local_i, global_i + 1:] >= threshold)
            if partner_offset.size > 0:
                n_high_jaccard_pairs += int(partner_offset.size)
                redundant_rule_mask[global_i] = True
                redundant_rule_mask[global_i + 1 + partner_offset] = True

    n_redundant_rules = int(redundant_rule_mask.sum())
    redundant_rule_fraction = n_redundant_rules / n_rules
    high_jaccard_fraction = n_high_jaccard_pairs / n_total_pairs

    return n_high_jaccard_pairs, n_total_pairs, high_jaccard_fraction, n_redundant_rules, redundant_rule_fraction


def count_unique_signal_masks(mask_matrix: "sparse.csr_matrix") -> int:
    """
    Number of distinct signal masks, by EXACT positional match (same bars
    active, not just same count). Two rules can share the same row_sum
    while firing on different bars — this checks the real thing, not a
    proxy. Compares each row's active-column indices directly on the
    sparse structure, without densifying.
    """
    mask_matrix = mask_matrix.tocsr()
    mask_matrix.sort_indices()

    seen = set()
    for i in range(mask_matrix.shape[0]):
        row_cols = mask_matrix.indices[mask_matrix.indptr[i]:mask_matrix.indptr[i + 1]]
        seen.add(row_cols.tobytes())

    return len(seen)

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
    (
        n_high_jaccard_pairs,
        n_signal_pairs_total,
        high_jaccard_fraction,
        n_redundant_rules,
        redundant_rule_fraction,
    ) = count_high_jaccard_signal_pairs(signal_mask_matrix)
    timings["jaccard"] = time.time() - _t0

    _t0 = time.time()
    n_unique_signal_masks = count_unique_signal_masks(signal_mask_matrix)
    timings["unique_masks"] = time.time() - _t0

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  SIGNAL MASK REDUNDANCY (pre-backtest, Jaccard) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  rules (pre TP/SL grid)              : {len(all_rules):,}")
    logger.info(f"  rules involved in a redundant pair  : {n_redundant_rules:,} / {len(all_rules):,} │ {redundant_rule_fraction:.4%}")
    logger.info(f"  unique signal masks (exact match)   : {n_unique_signal_masks:,} / {len(all_rules):,} │ {n_unique_signal_masks / len(all_rules):.4%}")
    logger.info(f"{'─' * 70}\n")

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
    sample_idx  = stratified_sample_column_indices(col_names, N_SAMPLE_COLS, RANDOM_SEED)
    sample_arr2 = matrix_arr[:, sample_idx]
    n_sample_cols = sample_arr2.shape[1]

    corr_matrix = np.corrcoef(sample_arr2, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    m_eff_sample = effective_number_li_ji(corr_matrix)
    reduction_ratio = m_eff_sample / n_sample_cols
    m_eff_extrapolated = reduction_ratio * n_cols_total

    n_high_corr_pairs, n_total_pairs, high_corr_fraction, n_redundant_cols, redundant_col_fraction = count_high_correlation_pairs(corr_matrix)
    timings["sampling_and_meff"] = time.time() - _t0

    elapsed = int(time.time() - start)

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  REDUNDANCY DIAGNOSTIC (Li & Ji, 2005) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  total columns           : {n_cols_total}")
    logger.info(f"  sampled columns         : {n_sample_cols}")
    logger.info(f"  M_eff (sample)          : {m_eff_sample:.1f}")
    logger.info(f"  reduction ratio         : {reduction_ratio:.4%}")
    logger.info(f"  M_eff (extrapolated)    : {m_eff_extrapolated:.1f}")
    logger.info(f"  high-corr pairs (|ρ|>{HIGH_CORR_THRESHOLD}) : {n_high_corr_pairs:,} / {n_total_pairs:,}")
    logger.info(f"  high-corr pair fraction : {high_corr_fraction:.4%}")
    logger.info(f"  columns involved in a high-corr pair : {n_redundant_cols:,} / {n_sample_cols:,}")
    logger.info(f"  redundant column fraction : {redundant_col_fraction:.4%}")
    logger.info(f"  elapsed                 : {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")
    logger.info(f"{'─' * 70}\n")

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
    logger.info(f"  N_SAMPLE_COLS  : {N_SAMPLE_COLS}")
    logger.info(f"{'─' * 115}\n")

    for timeframe in TIMEFRAMES:
        run_redundancy_diagnostic_for_timeframe(timeframe)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")