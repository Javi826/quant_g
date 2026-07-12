import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "signals")))

import numpy as np
from condition_bank import ConditionBank
# =============================================================================
# CONFIG
# =============================================================================

REDUNDANCY_CORR_THRESHOLD = 0.6

TIMEFRAME = "4H"
N_SYMBOLS = 10


# =============================================================================
# CORE LOGIC
# =============================================================================

def evaluate_all_specs_multi_symbol(ohlcv_data: dict) -> tuple:
    specs        = None
    masks_by_spec_idx = None

    for sym, arr in ohlcv_data.items():
        bank = ConditionBank(arr)
        if specs is None:
            specs = bank.build_condition_specs()
            masks_by_spec_idx = [[] for _ in specs]

        for i, spec in enumerate(specs):
            masks_by_spec_idx[i].append(bank.evaluate(spec))

    concatenated_masks = [
        np.concatenate(masks).astype(np.float64) for masks in masks_by_spec_idx
    ]
    return specs, concatenated_masks


def compute_spec_correlation_matrix(concatenated_masks: list) -> np.ndarray:
    n = len(concatenated_masks)
    corr_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            x = concatenated_masks[i]
            y = concatenated_masks[j]

            if np.std(x) == 0 or np.std(y) == 0:
                continue

            corr = np.corrcoef(x, y)[0, 1]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def _is_trivial_pair(spec_a: dict, spec_b: dict) -> bool:
    """True if spec_a and spec_b are the same condition except for the operator (>,<)."""
    if spec_a["type"] != spec_b["type"]:
        return False

    keys_to_compare = [k for k in spec_a.keys() if k not in ("op",)]
    return all(spec_a.get(k) == spec_b.get(k) for k in keys_to_compare)


def find_redundant_pairs(specs: list, corr_matrix: np.ndarray, bank: ConditionBank, threshold: float = REDUNDANCY_CORR_THRESHOLD) -> list:
    redundant_pairs = []
    n = len(specs)

    for i in range(n):
        for j in range(i + 1, n):
            if _is_trivial_pair(specs[i], specs[j]):
                continue

            corr = corr_matrix[i, j]
            if not np.isnan(corr) and abs(corr) > threshold:
                redundant_pairs.append({
                    "spec_a":       bank.describe(specs[i]),
                    "spec_b":       bank.describe(specs[j]),
                    "type_a":       specs[i]["type"],
                    "type_b":       specs[j]["type"],
                    "correlation":  round(float(corr), 3),
                })

    redundant_pairs.sort(key=lambda row: abs(row["correlation"]), reverse=True)
    return redundant_pairs


def summarize_redundancy_by_type_pair(redundant_pairs: list) -> dict:
    summary = {}
    for pair in redundant_pairs:
        key = tuple(sorted((pair["type_a"], pair["type_b"])))
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items(), key=lambda kv: kv[1], reverse=True))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batch")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batch_regime")))

    from shared_batchs.pipeline.universe import filter_symbols, select_universe
    from shared_batchs.backtesters.ZX_compute_BT import MIN_PRICE
    from shared_batch_regime.config_paths import DATA_FOLDER_IS, DATA_FOLDER_OOS1

    symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos1 = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        data_folder_oos   = DATA_FOLDER_OOS1,
        timeframe         = TIMEFRAME,
        n_symbols         = N_SYMBOLS,
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )

    specs, concatenated_masks = evaluate_all_specs_multi_symbol(ohlcv_is)
    corr_matrix               = compute_spec_correlation_matrix(concatenated_masks)

    sample_bank      = ConditionBank(next(iter(ohlcv_is.values())))
    redundant_pairs  = find_redundant_pairs(specs, corr_matrix, sample_bank)
    summary_by_types = summarize_redundancy_by_type_pair(redundant_pairs)

    print(f"Redundancy check — timeframe={TIMEFRAME}, n_symbols={N_SYMBOLS}, total_specs={len(specs)}")
    print(f"Redundant pairs found (|corr| > {REDUNDANCY_CORR_THRESHOLD}): {len(redundant_pairs)}\n")

    print("Redundant pairs (sorted by |correlation|):")
    for pair in redundant_pairs:
        print(f"  {pair['correlation']:+.3f}  {pair['spec_a']:<35} <-> {pair['spec_b']:<35} ({pair['type_a']} vs {pair['type_b']})")

    print("\nRedundancy summary by indicator-type pair:")
    for (type_a, type_b), count in summary_by_types.items():
        print(f"  {type_a:<16} vs {type_b:<16} : {count} redundant pairs")