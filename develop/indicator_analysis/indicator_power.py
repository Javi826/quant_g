import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

# =============================================================================
# CONFIG
# =============================================================================

FORWARD_N                 = 25
N_SHUFFLES                = 50
REDUNDANCY_CORR_THRESHOLD = 0.8

TIMEFRAME = "4H"
N_SYMBOLS = 10

# =============================================================================
# BLOCK 1 - FORWARD RETURN
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
# BLOCK 2 - IC / MI PER SYMBOL
# =============================================================================


def compute_ic_mi_single_symbol(indicator_values: np.ndarray, forward_return: np.ndarray) -> dict:
    valid_mask = ~np.isnan(indicator_values) & ~np.isnan(forward_return)
    x = indicator_values[valid_mask]
    y = forward_return[valid_mask]

    if len(x) < 30:
        return {"ic": np.nan, "mi": np.nan, "n_obs": len(x)}

    ic, _ = spearmanr(x, y)
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]

    return {"ic": ic, "mi": mi, "n_obs": len(x)}


def compute_ic_mi_multi_symbol(indicator_by_symbol: dict, forward_returns: dict) -> dict:
    results = {}
    for symbol, indicator_values in indicator_by_symbol.items():
        forward_return = forward_returns[symbol]
        results[symbol] = compute_ic_mi_single_symbol(indicator_values, forward_return)
    return results

# =============================================================================
# BLOCK 3 - CROSS-SYMBOL AGGREGATION + NOISE BASELINE
# =============================================================================


def compute_shuffle_baseline(indicator_values: np.ndarray, forward_return: np.ndarray, n_shuffles: int = N_SHUFFLES, seed: int = 42) -> float:
    valid_mask = ~np.isnan(indicator_values) & ~np.isnan(forward_return)
    x = indicator_values[valid_mask]
    y = forward_return[valid_mask]

    if len(x) < 30:
        return np.nan

    rng = np.random.default_rng(seed)
    shuffled_ics = np.empty(n_shuffles)
    for i in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        ic, _ = spearmanr(x, y_shuffled)
        shuffled_ics[i] = ic

    return float(np.std(shuffled_ics))


def aggregate_cross_symbol(per_symbol_results: dict, indicator_by_symbol: dict, forward_returns: dict) -> dict:
    ics = np.array([r["ic"] for r in per_symbol_results.values() if not np.isnan(r["ic"])])

    if len(ics) == 0:
        return {"ic_mean": np.nan, "mi_mean": np.nan, "sign_consistency": np.nan, "noise_std": np.nan, "score": np.nan}

    mis = np.array([r["mi"] for r in per_symbol_results.values() if not np.isnan(r["mi"])])

    ic_mean          = float(np.mean(ics))
    mi_mean           = float(np.mean(mis)) if len(mis) > 0 else np.nan
    sign_consistency = float(np.mean(np.sign(ics) == np.sign(ic_mean)))

    noise_stds = []
    for symbol, indicator_values in indicator_by_symbol.items():
        noise_std = compute_shuffle_baseline(indicator_values, forward_returns[symbol])
        if not np.isnan(noise_std):
            noise_stds.append(noise_std)
    noise_std = float(np.mean(noise_stds)) if noise_stds else np.nan

    is_above_noise = abs(ic_mean) > (2 * noise_std) if not np.isnan(noise_std) else False
    score          = abs(ic_mean) * sign_consistency if is_above_noise else 0.0

    return {
        "ic_mean":          ic_mean,
        "mi_mean":          mi_mean,
        "sign_consistency": sign_consistency,
        "noise_std":        noise_std,
        "is_above_noise":   is_above_noise,
        "score":            score,
    }

# =============================================================================
# BLOCK 4 - RANK MULTIPLE CANDIDATE INDICATORS
# =============================================================================


def rank_indicators(indicators_by_name: dict, forward_returns: dict) -> list:
    ranking = []

    for indicator_name, indicator_by_symbol in indicators_by_name.items():
        per_symbol_results = compute_ic_mi_multi_symbol(indicator_by_symbol, forward_returns)
        aggregated_result  = aggregate_cross_symbol(per_symbol_results, indicator_by_symbol, forward_returns)

        ranking.append({
            "indicator_name":    indicator_name,
            "ic_mean":           aggregated_result["ic_mean"],
            "mi_mean":           aggregated_result["mi_mean"],
            "sign_consistency":  aggregated_result["sign_consistency"],
            "is_above_noise":    aggregated_result["is_above_noise"],
            "score":             aggregated_result["score"],
        })

    ranking.sort(key=lambda row: row["score"], reverse=True)
    return ranking


def print_ranking_table(ranking: list) -> None:
    header = f"{'indicator_name':<25} {'ic_mean':>10} {'mi_mean':>10} {'sign_consistency':>18} {'is_above_noise':>15} {'score':>10}"
    print(header)
    print("-" * len(header))
    for row in ranking:
        print(
            f"{row['indicator_name']:<25} "
            f"{row['ic_mean']:>10.4f} "
            f"{row['mi_mean']:>10.4f} "
            f"{row['sign_consistency']:>18.4f} "
            f"{str(row['is_above_noise']):>15} "
            f"{row['score']:>10.4f}"
        )

# =============================================================================
# BLOCK 5 - REDUNDANCY DETECTION
# =============================================================================


def _concat_indicator_across_symbols(indicator_by_symbol: dict) -> np.ndarray:
    return np.concatenate([values for values in indicator_by_symbol.values()])


def compute_correlation_matrix(indicators_by_name: dict, indicator_names: list) -> np.ndarray:
    concatenated = {
        name: _concat_indicator_across_symbols(indicators_by_name[name])
        for name in indicator_names
    }

    n_indicators = len(indicator_names)
    corr_matrix  = np.full((n_indicators, n_indicators), np.nan)

    for i in range(n_indicators):
        for j in range(i, n_indicators):
            x = concatenated[indicator_names[i]]
            y = concatenated[indicator_names[j]]
            valid_mask = ~np.isnan(x) & ~np.isnan(y)

            if valid_mask.sum() < 30:
                continue

            corr = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def remove_redundant_indicators(ranking: list, indicators_by_name: dict, corr_threshold: float = REDUNDANCY_CORR_THRESHOLD) -> dict:
    survivors_ranking = [row for row in ranking if row["is_above_noise"]]
    survivor_names    = [row["indicator_name"] for row in survivors_ranking]

    corr_matrix = compute_correlation_matrix(indicators_by_name, survivor_names)

    kept_names      = []
    discarded_names = []

    for i, name in enumerate(survivor_names):
        if name in discarded_names:
            continue
        kept_names.append(name)
        for j in range(i + 1, len(survivor_names)):
            other_name = survivor_names[j]
            if other_name in discarded_names:
                continue
            if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) > corr_threshold:
                discarded_names.append(other_name)

    return {
        "kept_names":      kept_names,
        "discarded_names": discarded_names,
        "corr_matrix":     corr_matrix,
        "survivor_names":  survivor_names,
    }

# =============================================================================
# BUILD INDICATORS FROM POOL
# =============================================================================


def build_indicators_by_name(ohlcv_data: dict, candidate_indicators: dict) -> dict:
    indicators_by_name = {}
    for indicator_name, indicator_fn in candidate_indicators.items():
        indicators_by_name[indicator_name] = {
            symbol: indicator_fn(arr) for symbol, arr in ohlcv_data.items()
        }
    return indicators_by_name

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

    from indicators import CANDIDATE_INDICATORS

    symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos1 = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        data_folder_oos   = DATA_FOLDER_OOS1,
        timeframe         = TIMEFRAME,
        n_symbols         = N_SYMBOLS,
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )

    forward_returns     = compute_forward_returns_multi_symbol(ohlcv_is)
    indicators_by_name  = build_indicators_by_name(ohlcv_is, CANDIDATE_INDICATORS)
    ranking              = rank_indicators(indicators_by_name, forward_returns)

    print(f"Indicator ranking — timeframe={TIMEFRAME}, n_symbols={N_SYMBOLS}\n")
    print_ranking_table(ranking)

    redundancy_result = remove_redundant_indicators(ranking, indicators_by_name)

    print(f"\nFinal indicators kept (non-redundant): {redundancy_result['kept_names']}")
    print(f"Discarded as redundant: {redundancy_result['discarded_names']}")