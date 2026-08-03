import os
import sys
import hashlib
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("analyze_rule_duplicates")

from shared_batchs.symbols.universe import filter_symbols, select_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.setup.config_backtest import MIN_PRICE
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.rule_mining.rule_generator import generate_all_rules, MAX_DEPTH
from shared_batchs.utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory
from shared_config import VOLUME_COL

# =============================================================================
# CONFIG
# =============================================================================
TIMEFRAME   = "1H"
N_SYMBOLS   = 10
HASH_N_JOBS = -1
CHUNK_SIZE  = 200  # rules per joblib task — amortizes cloudpickle/process overhead

# =============================================================================
# SIGNAL HASHING
# =============================================================================
def _signal_hash(signal_fn, ohlcv_arr: dict) -> bytes:
    hasher = hashlib.blake2b(digest_size=16)
    for sym in sorted(ohlcv_arr.keys()):
        arr = ohlcv_arr[sym]
        signal = np.asarray(signal_fn(arr, live_trading=False))
        hasher.update(sym.encode("utf-8"))
        hasher.update(signal.astype(np.int8).tobytes())
    return hasher.digest()


def _hash_rule_chunk(rule_chunk: list, shm_metadata: dict) -> list:
    """Runs in a worker: reattach to shared ohlcv_arr, hash each rule's signal
    in the chunk, return (label, hash) pairs. Avoids pickling ohlcv_arr per task."""
    ohlcv_arr, shm_handles = arrays_from_shared_memory(shm_metadata)
    try:
        return [(label, _signal_hash(signal_fn, ohlcv_arr)) for label, signal_fn in rule_chunk]
    finally:
        for shm in shm_handles:
            shm.close()


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main() -> None:
    ohlcv_is = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        timeframe         = TIMEFRAME,
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )
    ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)

    arr_sample = next(iter(ohlcv_arr.values()))
    all_rules  = generate_all_rules({
        "open":  arr_sample["open"],
        "high":  arr_sample["high"],
        "low":   arr_sample["low"],
        "close": arr_sample["close"],
        VOLUME_COL: arr_sample[VOLUME_COL],
    }, max_depth=MAX_DEPTH)

    logger.info(f"Total candidate rules: {len(all_rules)}")

    rule_pairs = [(rule["label"], rule["signal_fn"]) for rule in all_rules]
    chunks     = list(_chunked(rule_pairs, CHUNK_SIZE))

    shm_list, shm_metadata = arrays_to_shared_memory(ohlcv_arr)
    try:
        with tqdm_joblib(tqdm(desc="Hashing rule signals", total=len(chunks), dynamic_ncols=True)):
            chunk_results = Parallel(n_jobs=HASH_N_JOBS, batch_size=1, pre_dispatch="all")(
                delayed(_hash_rule_chunk)(chunk, shm_metadata) for chunk in chunks
            )
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()

    groups = defaultdict(list)
    for chunk_result in chunk_results:
        for label, h in chunk_result:
            groups[h].append(label)

    n_rules  = len(all_rules)
    n_groups = len(groups)
    group_sizes = sorted((len(v) for v in groups.values()), reverse=True)

    size_hist = defaultdict(int)
    for size in group_sizes:
        if size == 1:
            size_hist["1 (unique)"] += 1
        elif size <= 5:
            size_hist["2-5"] += 1
        elif size <= 10:
            size_hist["6-10"] += 1
        else:
            size_hist["11+"] += 1

    logger.info("─" * 70)
    logger.info(f"TIMEFRAME            : {TIMEFRAME}")
    logger.info(f"Total rules          : {n_rules}")
    logger.info(f"Unique signal groups : {n_groups}")
    logger.info(f"Duplication ratio    : {n_rules / n_groups:.2f}x")
    logger.info(f"Redundant rules      : {n_rules - n_groups} ({(n_rules - n_groups) / n_rules * 100:.1f}%)")
    logger.info("Group size distribution:")
    for label, count in size_hist.items():
        logger.info(f"  groups with size {label:12s}: {count}")
    logger.info("─" * 70)

    logger.info("Top 10 largest duplicate groups (sample labels):")
    largest = sorted(groups.values(), key=len, reverse=True)[:10]
    for grp in largest:
        if len(grp) > 1:
            logger.info(f"  size={len(grp):4d} — e.g. {grp[0]}")


if __name__ == "__main__":
    main()