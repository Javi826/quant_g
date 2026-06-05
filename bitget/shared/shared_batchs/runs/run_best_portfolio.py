#shared/shared_batchs/runs/run_best_portfolio.py
import os
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from shared_batchs.utils.batch_metrics import compute_metrics, _weekly_returns, _cvar, _neg_streak_stats
from shared_batchs.utils.reporting import _print_robustness_df
from shared_batchs.utils.plotting import plot_best_portfolio

logger = logging.getLogger("BOT_batch.runs.run_best_portfolio")

# =============================================================================
# CONFIGURATION
# =============================================================================
REQUIRE_LONG_SHORT = True
MIN_STRATEGIES     = 2
MAX_STRATEGIES     = 5
SELECTION_MODE     = "ranking"  # "weighted" | "ranking"

PERIOD_WEIGHTS = {
    "OOS1": 0.50,
    "OOS2": 0.25,
    "OOS3": 0.25,
}

# ascending=False → higher is better  |  ascending=True → lower is better
RANKING_CRITERIA = [
    ("Weekly_pct",False),
    ("Weekly_std",True),
]

# =============================================================================
# RANKING_CRITERIA = [
#     ("MaxNegStreak",True),
#     ("Weekly_pct",False),
# ]
# =============================================================================

TOP_N    = 2
N_SPLITS = 4# 1=annual, 2=semesters, 3=quadrimesters, 4=quarters, 6=bimesters, 12=months

# Label prefix per n_splits value
_SPLIT_LABELS = {1: "A", 2: "S", 3: "P", 4: "Q", 6: "B", 12: "M"}
_SPLIT_NAMES  = {1: "Annual", 2: "Semester", 3: "Quadrimester", 4: "Quarter", 6: "Bimester", 12: "Month"}



# =============================================================================
# PRIVATE HELPERS — Identity
# =============================================================================
def _parse_period_bounds(data_folder: str) -> tuple[str, str] | None:
    """Extract (start, end) date strings from data folder name pattern 'crypto_YYYY-MM_YYYY-MM_OOS'."""
    import re
    match = re.search(r'(\d{4}-\d{2})_(\d{4}-\d{2})', os.path.basename(data_folder))
    if not match:
        return None
    return (match.group(1) + "-01", match.group(2) + "-01")

def _is_long(strategy_id: str) -> bool:
    return "_long_" in strategy_id


def _is_short(strategy_id: str) -> bool:
    return "_short_" in strategy_id


def _has_long_and_short(strategy_ids: tuple) -> bool:
    return any(_is_long(s) for s in strategy_ids) and any(_is_short(s) for s in strategy_ids)


# =============================================================================
# PRIVATE HELPERS — Period splitting
# =============================================================================

def _split_into_subperiods(trades_list: list, n_splits: int, bounds: tuple | None = None) -> list[tuple[str, list]]:
    """
    Split a list of (strategy_id, trades_df) into n_splits equal time buckets
    based on sell_time. Returns list of (label, trades_list_subset).
    Drops empty subperiods.
    """
    if not trades_list:
        return []

    if bounds:
        t_min = pd.Timestamp(bounds[0])
        t_max = pd.Timestamp(bounds[1])
    else:
        all_times = pd.concat([df["sell_time"] for _, df in trades_list], ignore_index=True)
        t_min     = all_times.min()
        t_max     = all_times.max()
    total_days = (t_max - t_min).days
    
    split_len  = total_days / n_splits
    prefix     = _SPLIT_LABELS.get(n_splits, "P")


    result = []
    for i in range(n_splits):
        t_start = t_min + pd.Timedelta(days=i * split_len)
        t_end   = t_min + pd.Timedelta(days=(i + 1) * split_len)
        label   = f"{prefix}{i + 1}"

        subset = []
        for sid, df in trades_list:
            mask     = (df["sell_time"] >= t_start) & (df["sell_time"] < t_end)
            filtered = df[mask]
            if len(filtered) > 0:
                subset.append((sid, filtered))

        if subset:
            result.append((label, subset))

    return result


def _build_subperiod_index(
    trades_per_period: dict,
    n_splits: int,
    period_weights: dict,
    period_date_bounds: dict = {},
) -> list[tuple[str, str, list, float]]:
    subperiods = []
    for period, trades_list in trades_per_period.items():
        split_weight = period_weights.get(period, 0) / n_splits
        bounds = period_date_bounds.get(period)
        for split_label, split_trades in _split_into_subperiods(trades_list, n_splits, bounds):
            subperiods.append((period, split_label, split_trades, split_weight))
    return subperiods


# =============================================================================
# PRIVATE HELPERS — Metrics
# =============================================================================

def _compute_subperiod_metrics(
    combo: tuple,
    trades_list: list,
    initial_balance: float,
) -> dict | None:
    """
    Compute all ranking metrics for a combo in one subperiod.
    Returns None if no trades found for this combo.
    """
    combo_trades = [(sid, df) for sid, df in trades_list if sid in combo]
    if not combo_trades:
        return None

    all_tl        = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
    total_capital = initial_balance * len(combo_trades)
    m             = compute_metrics(all_tl, capital=total_capital, name="")
    pf            = m["Profit_Factor"]
    weekly        = _weekly_returns(combo_trades, initial_balance)
    
    if len(weekly) == 0:
        return None

    cvar10           = _cvar(weekly, pct=10)
    avg_neg, max_neg = _neg_streak_stats(weekly)
    #logger.info(f"  [DBG] {all_tl['sell_time'].min()} → {all_tl['sell_time'].max()}  n_weeks={len(weekly)}  wpct={round(float((weekly > 0).mean() * 100), 1)}")


    return {
        "NetGain%":     round(m["Net_Gain_pct"], 2),
        "MaxDD%":       round(m["Max_DD_pct"], 2),
        "R2":           round(m["R_Squared"], 3),
        "ProfitFactor": round(pf if pf != float("inf") else 0, 1),
        "CVaR10%":      round(cvar10, 2),
        "AvgNegStreak": round(avg_neg, 1),
        "MaxNegStreak": max_neg,
        "Weekly_pct":   round(float((weekly > 0).mean() * 100), 1),
        "Weekly_avg%":  round(float(weekly.mean()), 1),
        "MinWeekly%":   round(float(weekly.min()), 1),
    }


def _compute_period_metrics(
    combo: tuple,
    trades_list: list,
    initial_balance: float,
) -> dict | None:
    """
    Compute full robustness metrics for a combo in one OOS period.
    Returns None if no trades found.
    """
    combo_trades = [(sid, df) for sid, df in trades_list if sid in combo]
    if not combo_trades:
        return None

    all_tl        = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
    total_capital = initial_balance * len(combo_trades)
    m             = compute_metrics(all_tl, capital=total_capital, name="")
    pf            = m["Profit_Factor"]

    weekly           = _weekly_returns(combo_trades, initial_balance)
    cvar10           = _cvar(weekly, pct=10)
    avg_neg, max_neg = _neg_streak_stats(weekly)
    weekly_avg       = round(float(weekly.mean()), 1) if len(weekly) > 0 else np.nan
    weekly_min       = round(float(weekly.min()), 1)  if len(weekly) > 0 else np.nan
    weekly_pct       = round(float((weekly > 0).mean() * 100), 1)

    return {
        "NetGain%":     round(m["Net_Gain_pct"], 1),
        "MaxDD%":       round(m["Max_DD_pct"], 1),
        "R2":           round(m["R_Squared"], 2),
        "ProfitFactor": round(pf if pf != float("inf") else 0, 1),
        "CVaR10%":      round(cvar10, 2),
        "AvgNegStreak": round(avg_neg, 1),
        "MaxNegStreak": max_neg,
        "Weekly_pct":   weekly_pct,
        "Weekly_avg%":  weekly_avg,
        "MinWeekly%":   weekly_min,
    }


# =============================================================================
# PRIVATE HELPERS — Weighted scoring
# =============================================================================
def _compute_weighted_scores(
    combos: list[tuple],
    subperiods: list[tuple],
    initial_balance: float,
    ranking_criteria: list[tuple[str, bool]],
) -> pd.DataFrame:
    from joblib import Parallel, delayed

    metric_keys = [m for m, _ in ranking_criteria]
    primary_key = metric_keys[0]

    def _process_combo(combo: tuple) -> dict | None:
        row              = {"combo": combo}
        total_weight     = 0.0
        weighted_metrics = {k: 0.0 for k in metric_keys}
        subperiod_scores = {}
        for period, split_label, split_trades, split_weight in subperiods:
            m = _compute_subperiod_metrics(combo, split_trades, initial_balance)
            if m is None:
                continue
            key = f"{period}_{split_label}"
            subperiod_scores[key] = m.get(primary_key, np.nan)
            for k in metric_keys:
                if k in m and not np.isnan(m[k]):
                    weighted_metrics[k] += m[k] * split_weight
            total_weight += split_weight
        if total_weight == 0:
            return None
        for k in metric_keys:
            row[k] = round(weighted_metrics[k] / total_weight, 3)
        values = [v for v in subperiod_scores.values() if not np.isnan(v)]
        row["subperiod_scores"] = subperiod_scores
        row["subperiod_std"]    = round(float(np.std(values)), 2)  if values else np.nan
        row["subperiod_min"]    = round(float(np.min(values)), 1)  if values else np.nan
        row["subperiod_max"]    = round(float(np.max(values)), 1)  if values else np.nan
        row["Weekly_std"]       = row["subperiod_std"]
        return row

    results = Parallel(n_jobs=-1)(
        delayed(_process_combo)(combo) for combo in combos
    )

    records = [r for r in results if r is not None]

    if not records:
        return pd.DataFrame()

    df        = pd.DataFrame(records)
    sort_cols = [c for c, _ in ranking_criteria]
    sort_asc  = [a for _, a in ranking_criteria]
    df        = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    # --- DEBUG LOG: top combo subperiod breakdown ---
    top_row = df.iloc[0]
    top_combo = top_row["combo"]
    scores    = top_row.get("subperiod_scores", {})
    nan_keys  = [k for k, v in scores.items() if isinstance(v, float) and np.isnan(v)]
    none_keys = [k for k, v in scores.items() if v is None]

    logger.info(f"\n{'─'*115}")
    logger.info(f"  [WEIGHTED] Top combo subperiod breakdown — {primary_key}")
    logger.info(f"  Combo: {list(top_combo)}")
    logger.info(f"  {'Subperiod':<18} {'Value':>10} {'NaN/None':>10}")
    logger.info(f"  {'-'*40}")
    for key in sorted(scores.keys()):
        val    = scores[key]
        is_nan = isinstance(val, float) and np.isnan(val)
        flag   = "⚠️ NaN" if is_nan else ("⚠️ None" if val is None else "")
        val_str = "NaN" if is_nan else ("None" if val is None else f"{val:.1f}")
        logger.info(f"  {key:<18} {val_str:>10} {flag}")
    logger.info(f"  {'-'*40}")
    logger.info(f"  NaN subperiods  : {len(nan_keys)}  → {nan_keys}")
    logger.info(f"  None subperiods : {len(none_keys)} → {none_keys}")
    logger.info(f"  Weighted {primary_key:<12}: {top_row[primary_key]:.3f}")
    logger.info(f"  Subperiod std   : {top_row.get('subperiod_std', np.nan):.2f}")
    logger.info(f"  Subperiod min   : {top_row.get('subperiod_min', np.nan):.1f}")
    logger.info(f"  Subperiod max   : {top_row.get('subperiod_max', np.nan):.1f}")
    logger.info(f"{'─'*115}")

    return df

def _compute_ranking_scores(
    combos: list[tuple],
    subperiods: list[tuple],
    initial_balance: float,
    ranking_criteria: list[tuple[str, bool]],
) -> pd.DataFrame:
    from joblib import Parallel, delayed
 
    primary_key    = ranking_criteria[0][0]
    primary_asc    = ranking_criteria[0][1]
    subperiod_keys = [f"{p}_{q}" for p, q, _, _ in subperiods]
    weights        = {f"{p}_{q}": w for p, q, _, w in subperiods}
 
    def _score_combo(combo: tuple) -> dict | None:
        row = {"combo": combo}
        for period, split_label, split_trades, _ in subperiods:
            key = f"{period}_{split_label}"
            m   = _compute_subperiod_metrics(combo, split_trades, initial_balance)
            row[key] = m.get(primary_key, np.nan) if m is not None else np.nan
        return row
 
    results = Parallel(n_jobs=-1)(
        delayed(_score_combo)(combo) for combo in combos
    )
    records = [r for r in results if r is not None]
    if not records:
        return pd.DataFrame()
 
    df = pd.DataFrame(records)
 
    rank_cols = []
    for key in subperiod_keys:
        rank_col     = f"rank_{key}"
        col_vals     = df[key].fillna(-np.inf if not primary_asc else np.inf).values
        order        = np.argsort(col_vals)[::-1] if not primary_asc else np.argsort(col_vals)
        ranks        = np.empty(len(df), dtype=int)
        ranks[order] = np.arange(1, len(df) + 1)
        df[rank_col] = ranks
        rank_cols.append(rank_col)
 
    n_combos            = len(df)
    total_weight        = sum(weights[key] for key in subperiod_keys)
    df["weighted_rank"] = sum(
        (df[f"rank_{key}"] / n_combos) * weights[key] for key in subperiod_keys
    ) / total_weight
    df["rank_variance"] = df[[f"rank_{key}" for key in subperiod_keys]].var(axis=1)
 
    df["subperiod_scores"] = df.apply(
        lambda row: {key: row[key] for key in subperiod_keys}, axis=1
    )
    df["subperiod_std"] = df[subperiod_keys].std(axis=1).round(2)
    df["subperiod_min"] = df[subperiod_keys].min(axis=1).round(1)
    df["subperiod_max"] = df[subperiod_keys].max(axis=1).round(1)
    df["Weekly_std"]    = df["subperiod_std"]
 
    df[primary_key] = df[subperiod_keys].apply(
        lambda row: sum(row[key] * weights[key] for key in subperiod_keys
                        if not np.isnan(row[key])) / total_weight,
        axis=1
    ).round(3)
 
    df = df.sort_values(["weighted_rank", "rank_variance"]).reset_index(drop=True)
 
    # --- DEBUG LOG: top combo subperiod breakdown (before column selection) ---
    top_row   = df.iloc[0]
    top_combo = top_row["combo"]
    scores    = top_row.get("subperiod_scores", {})
    nan_keys  = [k for k, v in scores.items() if isinstance(v, float) and np.isnan(v)]
    none_keys = [k for k, v in scores.items() if v is None]
 
    logger.info(f"\n{'─'*115}")
    logger.info(f"  [RANKING] Top combo subperiod breakdown — {primary_key}")
    logger.info(f"  Combo: {list(top_combo)}")
    logger.info(f"  {'Subperiod':<18} {'Value':>10} {'Rank':>6} {'NaN/None':>10}")
    logger.info(f"  {'-'*48}")
    for key in sorted(scores.keys()):
        val      = scores[key]
        is_nan   = isinstance(val, float) and np.isnan(val)
        flag     = "⚠️ NaN" if is_nan else ("⚠️ None" if val is None else "")
        val_str  = "NaN" if is_nan else ("None" if val is None else f"{val:.1f}")
        rank_val = top_row.get(f"rank_{key}", "—")
        rank_str = f"{rank_val}" if isinstance(rank_val, (int, np.integer)) else "—"
        logger.info(f"  {key:<18} {val_str:>10} {rank_str:>6} {flag}")
    logger.info(f"  {'-'*48}")
    logger.info(f"  NaN subperiods    : {len(nan_keys)}  → {nan_keys}")
    logger.info(f"  None subperiods   : {len(none_keys)} → {none_keys}")
    logger.info(f"  Weighted rank     : {top_row['weighted_rank']:.2f}")
    logger.info(f"  Rank variance     : {top_row['rank_variance']:.1f}")
    logger.info(f"  {primary_key:<18}: {top_row[primary_key]:.3f}")
    logger.info(f"  Subperiod std     : {top_row.get('subperiod_std', np.nan):.2f}")
    logger.info(f"  Subperiod min     : {top_row.get('subperiod_min', np.nan):.1f}")
    logger.info(f"  Subperiod max     : {top_row.get('subperiod_max', np.nan):.1f}")
    logger.info(f"{'─'*115}")
 
    return df[["combo", "weighted_rank", "rank_variance", "subperiod_scores",
               "subperiod_std", "subperiod_min", "subperiod_max", "Weekly_std", primary_key]]
# =============================================================================
# PRINT
# =============================================================================

def _print_best_combinations(
    top: list,
    trades_per_period: dict,
    initial_balance: float,
    n_splits: int,
    ranking_criteria: list[tuple[str, bool]],
    validation_results: list = None,
    selection_mode: str = "weighted",
) -> None:
    W          = 115
    split_name = _SPLIT_NAMES.get(n_splits, f"{n_splits}-Split")
    metric_str = " | ".join(f"{m} ({'↑' if not asc else '↓'})" for m, asc in ranking_criteria)
    mode_str   = "Ranking (avg position)" if selection_mode == "ranking" else "Weighted metrics"

    logger.info(f"\n\033[94m{'='*W}\n  BEST PORTFOLIO COMBINATIONS — {split_name} splits | mode: {mode_str} | ranked by: {metric_str}\n{'='*W}\033[0m")
    primary_key = ranking_criteria[0][0]

    val_lookup = {}
    if validation_results:
        for v in validation_results:
            val_lookup[v["strategy_id"]] = v

    for rank, entry in enumerate(top, start=1):
        combo            = entry["combo"]
        metric_values    = {k: entry[k] for k, _ in ranking_criteria if k in entry}
        subperiod_scores = entry.get("subperiod_scores", {})
        std              = entry.get("subperiod_std", np.nan)
        mn               = entry.get("subperiod_min",  np.nan)
        mx               = entry.get("subperiod_max",  np.nan)
        avg_trades       = np.mean([len(df) for sid, df in trades_per_period.get("OOS1", []) if sid in combo])

        wpct_w        = metric_values.get(primary_key, 0)
        weighted_rank = entry.get("weighted_rank", np.nan)
        rank_variance = entry.get("rank_variance", np.nan)

        if selection_mode == "ranking":
            stats_str = f"avg_rank={weighted_rank:.3f}  rank_var={rank_variance:.1f}  std={std:.1f}  min={mn:.1f}  max={mx:.1f}  |  {primary_key}_w={wpct_w:.2f}"
        else:
            stats_str = f"{primary_key}_w={wpct_w:.2f}  std={std:.1f}  min={mn:.1f}  max={mx:.1f}  |  avg_rank={weighted_rank:.3f}"

        header_parts = [f"Strategies: {len(combo)}", f"AvgTrades={avg_trades:.0f}", stats_str]

        logger.info(f"\nBEST #{rank} — " + "  |  ".join(header_parts))
        logger.info(f"{'─'*W}")
        for s in sorted(combo, key=lambda s: int(s.split("_")[0])):
            icon = "🟢" if _is_long(s) else "🔴"
            logger.info(f"    {icon} {s}")

        if subperiod_scores:
            logger.debug(f"\n  {primary_key} per subperiod:")
            for key, val in subperiod_scores.items():
                logger.debug(f"    {key:<12} → {val:.1f}%")

        rows = []
        for period, trades_list in trades_per_period.items():
            m = _compute_period_metrics(combo, trades_list, initial_balance)
            if m is None:
                continue
            rows.append({"Period": period, **m})
        _print_robustness_df(rows, f"ROBUSTNESS TABLE — Combination #{rank}")

        if val_lookup:
            combo_strategies = sorted(combo, key=lambda s: int(s.split("_")[0]))
            combo_val        = [val_lookup[sid] for sid in combo_strategies if sid in val_lookup]

            if combo_val:
                worst_netgain = min(min(v.get("net_gain_pct", 0), v.get("net_gain_pct_oos2", 0), v.get("net_gain_pct_oos3", 0)) for v in combo_val)
                worst_dd      = min(min(v.get("dd_pct", 0), v.get("dd_pct_oos2", 0), v.get("dd_pct_oos3", 0)) for v in combo_val)
                worst_r2      = min(min(v.get("r2", 0), v.get("r2_oos2", 0), v.get("r2_oos3", 0)) for v in combo_val)

                logger.info(f"\n{'─'*W}")
                logger.info(f"  INDIVIDUAL STRATEGY THRESHOLDS — Combination #{rank}")
                logger.info(f"{'─'*W}")
                logger.info(f"  {'Strategy':<30} {'OOS':<6} {'NetGain%':>10} {'MaxDD%':>8} {'R2':>7}")
                logger.info(f"  {'-'*W}")
                for sid in combo_strategies:
                    if sid not in val_lookup:
                        continue
                    v    = val_lookup[sid]
                    icon = "🟢" if _is_long(sid) else "🔴"
                    logger.info(f"  {icon} {sid:<28} {'OOS1':<6} {v.get('net_gain_pct', 0):>9.1f}% {v.get('dd_pct', 0):>7.1f}% {v.get('r2', 0):>7.3f}")
                    logger.info(f"  {'':31} {'OOS2':<6} {v.get('net_gain_pct_oos2', 0):>9.1f}% {v.get('dd_pct_oos2', 0):>7.1f}% {v.get('r2_oos2', 0):>7.3f}")
                    logger.info(f"  {'':31} {'OOS3':<6} {v.get('net_gain_pct_oos3', 0):>9.1f}% {v.get('dd_pct_oos3', 0):>7.1f}% {v.get('r2_oos3', 0):>7.3f}")
                    logger.info(f"  {'-'*W}")
                logger.info(f"  {'WORST':<30} {'ALL':<6} {worst_netgain:>9.1f}% {worst_dd:>7.1f}% {worst_r2:>7.3f}")
                logger.info(f"{'─'*W}")

    logger.info(f"\n{'─'*W}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def find_best_portfolio_combination(
    validated_trades_oos1: list,
    validated_trades_oos2: list,
    validated_trades_oos3: list,
    initial_balance: float,
    min_strategies: int      = MIN_STRATEGIES,
    max_strategies: int      = MAX_STRATEGIES,
    period_weights: dict     = PERIOD_WEIGHTS,
    ranking_criteria: list   = RANKING_CRITERIA,
    top_n: int               = TOP_N,
    n_splits: int            = N_SPLITS,
    data_folder_oos1: str    = "",
    data_folder_oos2: str    = "",
    data_folder_oos3: str    = "",
    show_plots: bool         = False,
    require_long_short: bool = REQUIRE_LONG_SHORT,
    validation_results: list = None,
    selection_mode: str      = SELECTION_MODE,
    **kwargs,
) -> list:

    trades_per_period = {
        "OOS1": validated_trades_oos1,
        "OOS2": validated_trades_oos2,
        "OOS3": validated_trades_oos3,
    }

    all_ids = list({sid for sid, _ in validated_trades_oos1})
    if not all_ids:
        logger.warning("No validated strategies — skipping best portfolio search.")
        return []

    period_date_bounds = {
        "OOS1": _parse_period_bounds(data_folder_oos1),
        "OOS2": _parse_period_bounds(data_folder_oos2),
        "OOS3": _parse_period_bounds(data_folder_oos3),
    }
    subperiods = _build_subperiod_index(trades_per_period, n_splits, period_weights, period_date_bounds)
    if not subperiods:
        logger.warning("No subperiods could be built — check trades data.")
        return []

    split_name = _SPLIT_NAMES.get(n_splits, f"{n_splits}-Split")
    logger.info(f"\n  Subperiods: {len(subperiods)} {split_name.lower()}(s) across {len(trades_per_period)} OOS periods")

    combos = [
        combo
        for size in range(min_strategies, max_strategies + 1)
        for combo in combinations(all_ids, size)
        if not require_long_short or _has_long_and_short(combo)
    ]

    if not combos:
        logger.warning("No valid combinations found.")
        return []

    logger.info(f"  Evaluating {len(combos)} combo(s)...")
    #RANKING
    if selection_mode == "ranking":
        df_scored = _compute_ranking_scores(combos, subperiods, initial_balance, ranking_criteria)
    else:
       df_scored = _compute_weighted_scores(combos, subperiods, initial_balance, ranking_criteria)
       
    if df_scored.empty:
        logger.warning("No scores computed — check trades data.")
        return []
    
    # Distribution analysis — overfitting check
    primary_metric = ranking_criteria[0][0]
    best_val       = df_scored[primary_metric].iloc[0]
    mean_val       = df_scored[primary_metric].mean()
    std_val        = df_scored[primary_metric].std()
    gap            = round(best_val - mean_val, 2)
    zscore         = round((best_val - mean_val) / std_val, 2) if std_val > 0 else 0.0
    logger.info(f"  Distribution ({primary_metric}): best={best_val:.1f}  mean={mean_val:.1f}  std={std_val:.1f}  min={df_scored[primary_metric].min():.1f}  max={df_scored[primary_metric].max():.1f}  gap={gap:.2f}%  z-score={zscore}")

    top = df_scored.head(top_n).to_dict("records")
    _print_best_combinations(top, trades_per_period, initial_balance, n_splits, ranking_criteria, validation_results, selection_mode)
    
    plot_best_portfolio(
        combo             = top[0]["combo"],
        trades_per_period = trades_per_period,
        subperiod_scores  = top[0]["subperiod_scores"],
        subperiods        = subperiods,
        initial_balance   = initial_balance,
        data_folder_oos1  = data_folder_oos1,
        data_folder_oos2  = data_folder_oos2,
        data_folder_oos3  = data_folder_oos3,
        df_scored         = df_scored,        # añadir esto
        ranking_criteria  = ranking_criteria, # añadir esto
        show_plots        = show_plots,
    )


    return top
        