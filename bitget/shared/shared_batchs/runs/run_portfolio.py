#shared/shared_batchs/runs/run_best_wfo_portfolio.py
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.reporting import print_best_wfo_portfolio
from shared_batchs.utils.plotting import plot_wfo_portfolio
logger = logging.getLogger("BOT_batch.runs.run_best_wfo_portfolio")
# =============================================================================
# CONFIGURATION
# =============================================================================
#NET_GAIN_PCT | WEEKLY_PCT | WIN_RATE | CALMAR | R_SQUARED | MAX_DD_PCT
WFO_METRIC            = "R_SQUARED" 
WFO_N_SPLITS          = 4
WFO_SUBPERIOD_WEIGHTS = [0.10, 0.20, 0.20, 0.50]

# WFO_N_SPLITS          = 8
# WFO_SUBPERIOD_WEIGHTS = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20, 0.20]


MIN_STRATEGIES     = 3
MAX_STRATEGIES     = 6
TOP_N              = 2

REQUIRE_LONG_SHORT     = True
REQUIRE_ALL_TIMEFRAMES = True

# Metric extraction: (column_in_compute_metrics_output, higher_is_better)
_METRIC_MAP = {
    "NET_GAIN_PCT": ("Net_Gain_pct", True),
    "WEEKLY_PCT":   ("Weekly_pct",   True),
    "WIN_RATE":     ("Win_Rate",     True),
    "R_SQUARED":    ("R_Squared",    True),
    "MAX_DD_PCT":   ("Max_DD_pct",   False),  # lower is better → negated internally
    "CALMAR":       ("Calmar",       True),
}
# =============================================================================
# PRIVATE HELPERS — Validation
# =============================================================================

def _validate_config(n_splits: int, weights: list, metric: str) -> None:
    if len(weights) != n_splits:
        raise ValueError(
            f"WFO_SUBPERIOD_WEIGHTS has {len(weights)} entries but WFO_N_SPLITS={n_splits}. "
            "They must match."
        )
    if metric not in _METRIC_MAP:
        raise ValueError(
            f"Unknown WFO_METRIC='{metric}'. "
            f"Valid options: {list(_METRIC_MAP.keys())}"
        )
# =============================================================================
# PRIVATE HELPERS — Identity
# =============================================================================

def _is_long(strategy_id: str) -> bool:
    return "_long_" in strategy_id

def _is_short(strategy_id: str) -> bool:
    return "_short_" in strategy_id

def _get_timeframe(strategy_id: str) -> str:
    return strategy_id.split("_")[1]
# =============================================================================
# PRIVATE HELPERS — Splitting
# =============================================================================
def _split_trades_by_time(
    trades_list: list,
    n_splits: int,
) -> list:

    if not trades_list:
        return []

    all_times  = pd.concat([df["sell_time"] for _, df in trades_list], ignore_index=True)
    t_min      = pd.Timestamp(all_times.min())
    t_max      = pd.Timestamp(all_times.max())
    total_days = (t_max - t_min).days
    split_len  = total_days / n_splits

    result = []
    for i in range(n_splits):
        t_start = t_min + pd.Timedelta(days=i * split_len)
        t_end   = t_min + pd.Timedelta(days=(i + 1) * split_len)
        label   = f"S{i + 1}"

        subset = [
            (sid, df[(df["sell_time"] >= t_start) & (df["sell_time"] < t_end)])
            for sid, df in trades_list
        ]
        subset = [(sid, df) for sid, df in subset if len(df) > 0]

        if subset:
            result.append((label, t_start, t_end, subset))

    return result
# =============================================================================
# PRIVATE HELPERS — Metric extraction
# =============================================================================
def _extract_metric(m: dict, metric: str) -> float:
    if m.get("Net_Gain_pct", np.nan) <= 0:
        return np.nan  

    col, higher_is_better = _METRIC_MAP[metric]
    val = m.get(col, np.nan)
    return val if higher_is_better else -abs(val)  
# =============================================================================
# PRIVATE HELPERS — Combo scoring (raw metric per subperiod)
# =============================================================================
def _score_combo(
    combo: tuple,
    subperiods: list,
    initial_balance: float,
    metric: str,
) -> dict:

    scores = {}

    for label, _t_start, _t_end, split_trades in subperiods:
        combo_trades = [(sid, df) for sid, df in split_trades if sid in combo]
        if not combo_trades:
            scores[label] = np.nan
            continue

        tl            = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
        total_capital = initial_balance * len(combo_trades)
        m             = compute_metrics(tl, capital=total_capital, name="")
        val           = _extract_metric(m, metric)

        scores[label] = round(val, 3)

    return {"combo": combo, **scores}

# =============================================================================
# PRIVATE HELPERS — Rank-based scoring across subperiods
# =============================================================================
def _rank_combos_by_subperiod(
    raw_scores: list,
    subperiods: list,
) -> list:

    split_labels = [label for label, _, _, _ in subperiods]

    for label in split_labels:
        values      = [(i, r[label]) for i, r in enumerate(raw_scores)]
        n_combos    = len(values)
        worst_rank  = n_combos

        valid = [(i, v) for i, v in values if not np.isnan(v)]
        valid.sort(key=lambda x: x[1], reverse=True)  # higher metric value = better

        for rank, (i, _) in enumerate(valid, start=1):
            raw_scores[i][f"{label}_rank"] = rank

        ranked_indices = {i for i, _ in valid}
        for i, v in values:
            if i not in ranked_indices:
                raw_scores[i][f"{label}_rank"] = worst_rank

    return raw_scores

def _weighted_rank_score(
    entry: dict,
    subperiods: list,
    weights: list,
) -> float:
    """Weighted average of per-subperiod ranks (lower = better)."""
    split_labels = [label for label, _, _, _ in subperiods]
    return sum(entry[f"{label}_rank"] * w for label, w in zip(split_labels, weights))

def _generate_combos(
    all_ids: list,
    min_strategies: int,
    max_strategies: int,
    require_long_short: bool,
    require_all_timeframes: bool = False,
) -> list:
    """Generate strategy combinations, avoiding long-only/short-only combos upfront when required."""
    if not require_long_short:
        combos = [
            combo
            for size in range(min_strategies, max_strategies + 1)
            for combo in combinations(all_ids, size)
        ]
    else:
        longs  = [s for s in all_ids if _is_long(s)]
        shorts = [s for s in all_ids if _is_short(s)]

        combos = []
        for size in range(min_strategies, max_strategies + 1):
            min_longs = max(1, size - len(shorts))
            max_longs = min(size - 1, len(longs))
            for n_longs in range(min_longs, max_longs + 1):
                n_shorts = size - n_longs
                for long_combo in combinations(longs, n_longs):
                    for short_combo in combinations(shorts, n_shorts):
                        combos.append(long_combo + short_combo)

    if require_all_timeframes:
        required_tfs = {_get_timeframe(s) for s in all_ids}
        combos = [
            combo for combo in combos
            if required_tfs.issubset({_get_timeframe(s) for s in combo})
        ]

    return combos

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def find_best_portfolio_combination_wfo(
    validated_wfo_trades: list,
    initial_balance: float,
    metric: str                    = WFO_METRIC,
    n_splits: int                  = WFO_N_SPLITS,
    subperiod_weights: list        = WFO_SUBPERIOD_WEIGHTS,
    min_strategies: int            = MIN_STRATEGIES,
    max_strategies: int            = MAX_STRATEGIES,
    top_n: int                     = TOP_N,
    require_long_short: bool       = REQUIRE_LONG_SHORT,
    require_all_timeframes: bool   = REQUIRE_ALL_TIMEFRAMES,
    show_plots: bool               = False,
) -> list:

    _validate_config(n_splits, subperiod_weights, metric)

    if not validated_wfo_trades:
        logger.warning("No validated WFO trades — skipping best WFO portfolio search.")
        return []

    all_ids = list({sid for sid, _ in validated_wfo_trades})
    logger.info(f"\n{'='*115}")
    logger.info(f"  BEST WFO PORTFOLIO — {len(all_ids)} validated strategies | metric: {metric} | splits: {n_splits}")
    logger.info(f"{'='*115}")

    subperiods = _split_trades_by_time(validated_wfo_trades, n_splits)
    if not subperiods:
        logger.warning("No subperiods could be built — check WFO trades data.")
        return []

    if len(subperiods) != n_splits:
        logger.warning(
            f"Expected {n_splits} subperiods but got {len(subperiods)} "
            "(some buckets were empty). Weights may not align — check trade coverage."
        )

    logger.info(f"\n  Subperiods:")
    for i, (lbl, t_start, t_end, subset) in enumerate(subperiods):
        n_strats = len({sid for sid, _ in subset})
        logger.info(
            f"    {lbl}  {t_start.strftime('%Y-%m-%d')} → {t_end.strftime('%Y-%m-%d')}  "
            f"weight={subperiod_weights[i]:.2f}  strategies={n_strats}"
        )

    combos = _generate_combos(all_ids, min_strategies, max_strategies, require_long_short, require_all_timeframes)
    if not combos:
        logger.warning("No valid combinations found — check require_long_short or strategy count.")
        return []

    logger.info(f"\n  Evaluating {len(combos)} combo(s)...\n")

    raw_scores = Parallel(n_jobs=-1)(
        delayed(_score_combo)(combo, subperiods, initial_balance, metric)
        for combo in combos
    )

    raw_scores = _rank_combos_by_subperiod(raw_scores, subperiods)

    for entry in raw_scores:
        entry["weighted_rank_score"] = _weighted_rank_score(entry, subperiods, subperiod_weights)

    raw_scores.sort(key=lambda x: x["weighted_rank_score"])  # lower rank = better

    top = raw_scores[:top_n]
    print_best_wfo_portfolio(top, subperiods, validated_wfo_trades, initial_balance, metric, subperiod_weights)

    df_scored = pd.DataFrame([
        {"combo": r["combo"], "weighted_rank_score": r["weighted_rank_score"]}
        for r in raw_scores
    ])

    if top and show_plots:
        top_entry         = top[0]
        top_subp_scores   = {lbl: top_entry.get(lbl, np.nan) for lbl, _, _, _ in subperiods}
        plot_wfo_portfolio(
            combo             = top_entry["combo"],
            trades_list       = validated_wfo_trades,
            subperiods        = subperiods,
            subperiod_scores  = top_subp_scores,
            df_scored         = df_scored,
            initial_balance   = initial_balance,
            metric            = metric,
            weights           = subperiod_weights,
            title             = f"Best WFO Portfolio — {metric}",
            validated_trades  = validated_wfo_trades,
        )

    return top