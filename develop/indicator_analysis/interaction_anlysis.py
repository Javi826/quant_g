import itertools
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

FORWARD_N        = 10
N_SHUFFLES       = 20
NOISE_MULTIPLIER = 2.0
MIN_OBS          = 30
MIN_OBS_DISPLAY  = 500

TIMEFRAME = "4H"
N_SYMBOLS = 10

TOP_N_RESULTS = 30


# =============================================================================
# BLOCK 1 - FORWARD RETURN (same definition as indicator_power_analysis.py)
# =============================================================================

def compute_forward_return(close: np.ndarray, forward_n: int = FORWARD_N) -> np.ndarray:
    n = len(close)
    forward_return = np.full(n, np.nan, dtype=np.float64)
    forward_return[:n - forward_n] = (close[forward_n:] - close[:n - forward_n]) / close[:n - forward_n]
    return forward_return


def compute_forward_returns_multi_symbol(ohlcv_data: dict, forward_n: int = FORWARD_N) -> dict:
    forward_returns = {}
    for symbol, arr in ohlcv_data.items():
        close = np.ascontiguousarray(arr["close"], dtype=np.float64)
        forward_returns[symbol] = compute_forward_return(close, forward_n)
    return forward_returns


# =============================================================================
# BLOCK 2 - BINARIZE ALL INDICATORS, CONCATENATE ACROSS SYMBOLS
# =============================================================================

def compute_binary_masks_multi_symbol(
    ohlcv_data: dict,
    forward_returns: dict,
    candidate_indicators: dict,
    thresholds: dict,
) -> tuple:
    """Returns (masks, valids, fr_concat) where masks/valids are {name: concatenated bool array}."""
    per_symbol_masks  = {name: [] for name in candidate_indicators}
    per_symbol_valids = {name: [] for name in candidate_indicators}
    fr_list = []

    for symbol, arr in ohlcv_data.items():
        fr_list.append(forward_returns[symbol])
        for name, fn in candidate_indicators.items():
            values    = fn(arr)
            threshold = thresholds[name]
            valid     = ~np.isnan(values)
            mask      = valid & (values > threshold)
            per_symbol_masks[name].append(mask)
            per_symbol_valids[name].append(valid)

    fr_concat = np.concatenate(fr_list)
    masks     = {name: np.concatenate(chunks) for name, chunks in per_symbol_masks.items()}
    valids    = {name: np.concatenate(chunks) for name, chunks in per_symbol_valids.items()}

    fr_valid = ~np.isnan(fr_concat)
    for name in valids:
        valids[name] = valids[name] & fr_valid

    return masks, valids, fr_concat


# =============================================================================
# BLOCK 3 - SHUFFLE MATRIX (precomputed once, reused for every combo)
# =============================================================================

def build_shuffled_fr_matrix(fr: np.ndarray, fr_valid: np.ndarray, n_shuffles: int = N_SHUFFLES, seed: int = 42) -> np.ndarray:
    """(n_shuffles, n_rows) matrix: fr with its valid entries independently permuted per row."""
    rng = np.random.default_rng(seed)
    valid_idx = np.where(fr_valid)[0]

    shuffled_matrix = np.tile(fr, (n_shuffles, 1))
    for i in range(n_shuffles):
        shuffled_matrix[i, valid_idx] = rng.permutation(fr[valid_idx])

    return shuffled_matrix


# =============================================================================
# BLOCK 4 - EFFECT + NOISE TEST
# =============================================================================

def compute_effect_with_noise(
    mask: np.ndarray,
    valid: np.ndarray,
    fr: np.ndarray,
    shuffled_fr_matrix: np.ndarray,
    min_obs: int = MIN_OBS,
) -> dict:
    true_rows  = valid & mask
    false_rows = valid & ~mask
    n_obs_true = int(true_rows.sum())

    if n_obs_true < min_obs or false_rows.sum() < min_obs:
        return {"effect": np.nan, "n_obs": n_obs_true, "noise_std": np.nan, "is_above_noise": False, "score": 0.0}

    effect = float(np.mean(fr[true_rows]) - np.mean(fr[false_rows]))

    shuffled_true_means  = shuffled_fr_matrix[:, true_rows].mean(axis=1)
    shuffled_false_means = shuffled_fr_matrix[:, false_rows].mean(axis=1)
    noise_std = float(np.std(shuffled_true_means - shuffled_false_means))

    is_above_noise = abs(effect) > (NOISE_MULTIPLIER * noise_std) if noise_std > 0 else False
    score          = abs(effect) if is_above_noise else 0.0

    return {"effect": round(effect, 4), "n_obs": n_obs_true, "noise_std": round(noise_std, 4),
            "is_above_noise": is_above_noise, "score": round(score, 4)}


def compute_individual_effects(masks: dict, valids: dict, fr: np.ndarray, shuffled_fr_matrix: np.ndarray) -> dict:
    return {
        name: compute_effect_with_noise(masks[name], valids[name], fr, shuffled_fr_matrix)
        for name in masks
    }


# =============================================================================
# BLOCK 5 - PAIR INTERACTIONS
# =============================================================================

def compute_pair_interactions(masks: dict, valids: dict, fr: np.ndarray, shuffled_fr_matrix: np.ndarray, individual_effects: dict) -> list:
    results = []
    names   = list(masks.keys())

    for name_a, name_b in itertools.combinations(names, 2):
        combined_mask  = masks[name_a] & masks[name_b]
        combined_valid = valids[name_a] & valids[name_b]
        result_pair    = compute_effect_with_noise(combined_mask, combined_valid, fr, shuffled_fr_matrix)

        if np.isnan(result_pair["effect"]):
            continue

        ind_a = individual_effects[name_a]["effect"]
        ind_b = individual_effects[name_b]["effect"]
        best_individual = max(abs(ind_a) if not np.isnan(ind_a) else 0.0, abs(ind_b) if not np.isnan(ind_b) else 0.0)
        gain             = abs(result_pair["effect"]) - best_individual

        results.append({
            "combo":               f"{name_a} + {name_b}",
            "effect":              result_pair["effect"],
            "n_obs":               result_pair["n_obs"],
            "is_above_noise":      result_pair["is_above_noise"],
            "best_individual":     round(best_individual, 4),
            "gain_vs_individual":  round(gain, 4),
        })

    results.sort(key=lambda row: row["gain_vs_individual"], reverse=True)
    return results


# =============================================================================
# BLOCK 6 - TRIPLE INTERACTIONS
# =============================================================================

def compute_triple_interactions(masks: dict, valids: dict, fr: np.ndarray, shuffled_fr_matrix: np.ndarray, individual_effects: dict) -> list:
    results = []
    names   = list(masks.keys())

    for name_a, name_b, name_c in itertools.combinations(names, 3):
        combined_mask  = masks[name_a] & masks[name_b] & masks[name_c]
        combined_valid = valids[name_a] & valids[name_b] & valids[name_c]
        result_triple  = compute_effect_with_noise(combined_mask, combined_valid, fr, shuffled_fr_matrix)

        if np.isnan(result_triple["effect"]):
            continue

        pair_ab = compute_effect_with_noise(masks[name_a] & masks[name_b], valids[name_a] & valids[name_b], fr, shuffled_fr_matrix)
        pair_ac = compute_effect_with_noise(masks[name_a] & masks[name_c], valids[name_a] & valids[name_c], fr, shuffled_fr_matrix)
        pair_bc = compute_effect_with_noise(masks[name_b] & masks[name_c], valids[name_b] & valids[name_c], fr, shuffled_fr_matrix)
        pair_effects = [abs(p["effect"]) for p in (pair_ab, pair_ac, pair_bc) if not np.isnan(p["effect"])]

        if not pair_effects:
            continue
        best_pair = max(pair_effects)

        ind_effects = [individual_effects[n]["effect"] for n in (name_a, name_b, name_c)]
        ind_effects = [abs(e) for e in ind_effects if not np.isnan(e)]
        best_individual = max(ind_effects) if ind_effects else 0.0

        gain_vs_pair       = abs(result_triple["effect"]) - best_pair
        gain_vs_individual = abs(result_triple["effect"]) - best_individual

        results.append({
            "combo":               f"{name_a} + {name_b} + {name_c}",
            "effect":              result_triple["effect"],
            "n_obs":               result_triple["n_obs"],
            "is_above_noise":      result_triple["is_above_noise"],
            "best_pair":           round(best_pair, 4),
            "best_individual":     round(best_individual, 4),
            "gain_vs_pair":        round(gain_vs_pair, 4),
            "gain_vs_individual":  round(gain_vs_individual, 4),
        })

    results.sort(key=lambda row: row["gain_vs_pair"], reverse=True)
    return results


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_individual_table(individual_effects: dict, top_n: int = TOP_N_RESULTS, min_obs_display: int = MIN_OBS_DISPLAY) -> None:
    filtered = {name: res for name, res in individual_effects.items() if res["n_obs"] > min_obs_display}
    rows = sorted(
        filtered.items(),
        key=lambda kv: abs(kv[1]["effect"]) if not np.isnan(kv[1]["effect"]) else -1,
        reverse=True,
    )
    header = f"{'indicator_name':<25} {'effect_pct':>10} {'n_obs':>8} {'is_above_noise':>15}"
    print(header)
    print("-" * len(header))
    for name, res in rows[:top_n]:
        effect_str = f"{res['effect'] * 100:.4f}" if not np.isnan(res["effect"]) else "N/A"
        print(f"{name:<25} {effect_str:>10} {res['n_obs']:>8} {str(res['is_above_noise']):>15}")


def print_pair_table(pair_results: list, top_n: int = TOP_N_RESULTS, min_obs_display: int = MIN_OBS_DISPLAY) -> None:
    filtered = [row for row in pair_results if row["n_obs"] > min_obs_display]
    header = f"{'combo':<45} {'effect_pct':>10} {'n_obs':>8} {'is_above_noise':>15} {'best_ind_pct':>13} {'gain_pct':>10}"
    print(header)
    print("-" * len(header))
    for row in filtered[:top_n]:
        print(
            f"{row['combo']:<45} {row['effect'] * 100:>10.4f} {row['n_obs']:>8} {str(row['is_above_noise']):>15} "
            f"{row['best_individual'] * 100:>13.4f} {row['gain_vs_individual'] * 100:>10.4f}"
        )


def print_triple_table(triple_results: list, top_n: int = TOP_N_RESULTS, min_obs_display: int = MIN_OBS_DISPLAY) -> None:
    filtered = [row for row in triple_results if row["n_obs"] > min_obs_display]
    header = (
        f"{'combo':<65} {'effect_pct':>10} {'n_obs':>8} {'is_above_noise':>15} "
        f"{'best_pair_pct':>13} {'best_ind_pct':>13} {'gain_pair_pct':>14} {'gain_ind_pct':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in filtered[:top_n]:
        print(
            f"{row['combo']:<65} {row['effect'] * 100:>10.4f} {row['n_obs']:>8} {str(row['is_above_noise']):>15} "
            f"{row['best_pair'] * 100:>13.4f} {row['best_individual'] * 100:>13.4f} "
            f"{row['gain_vs_pair'] * 100:>14.4f} {row['gain_vs_individual'] * 100:>13.4f}"
        )


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

    from indicators import CANDIDATE_INDICATORS, INDICATOR_THRESHOLDS

    symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos1 = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        data_folder_oos   = DATA_FOLDER_OOS1,
        timeframe         = TIMEFRAME,
        n_symbols         = N_SYMBOLS,
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )

    forward_returns = compute_forward_returns_multi_symbol(ohlcv_is)
    masks, valids, fr = compute_binary_masks_multi_symbol(
        ohlcv_is, forward_returns, CANDIDATE_INDICATORS, INDICATOR_THRESHOLDS
    )

    fr_valid           = ~np.isnan(fr)
    shuffled_fr_matrix = build_shuffled_fr_matrix(fr, fr_valid)

    individual_effects = compute_individual_effects(masks, valids, fr, shuffled_fr_matrix)

    print(f"Interaction analysis — timeframe={TIMEFRAME}, n_symbols={N_SYMBOLS}\n")

    print("=== INDIVIDUAL EFFECTS ===")
    print_individual_table(individual_effects)

    print("\n=== PAIR INTERACTIONS (top gain vs best individual) ===")
    pair_results = compute_pair_interactions(masks, valids, fr, shuffled_fr_matrix, individual_effects)
    print_pair_table(pair_results)

    print("\n=== TRIPLE INTERACTIONS (top gain vs best pair) ===")
    triple_results = compute_triple_interactions(masks, valids, fr, shuffled_fr_matrix, individual_effects)
    print_triple_table(triple_results)