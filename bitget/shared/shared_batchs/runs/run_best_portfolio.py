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

PERIOD_WEIGHTS = {
    "OOS1": 0.50,
    "OOS2": 0.25,
    "OOS3": 0.25,
}

# ascending=False → higher is better  |  ascending=True → lower is better
RANKING_CRITERIA = [
    ("Weekly_pct", False),
    ("Weekly_std", True),
]

TOP_N    = 2
N_SPLITS = 4  # 1=annual, 2=semesters, 3=quadrimesters, 4=quarters, 6=bimesters, 12=months

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

def _split_into_subperiods(
    trades_list: list,
    n_splits: int,
    bounds: tuple | None = None,
) -> list[tuple[str, list]]:
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
        bounds       = period_date_bounds.get(period)
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
# PRIVATE HELPERS — Precomputed metrics table
# =============================================================================

def _build_metrics_table(
    combos: list[tuple],
    subperiods: list[tuple],
    initial_balance: float,
    primary_key: str,
) -> pd.DataFrame:
    """
    Compute primary metric for every combo × subperiod once.
    Returns a DataFrame with columns: combo, {period}_{label}, ...
    Parallelized over combos.
    """
    from joblib import Parallel, delayed

    subperiod_keys = [f"{p}_{q}" for p, q, _, _ in subperiods]

    def _score_combo(combo: tuple) -> dict:
        row = {"combo": combo}
        for period, split_label, split_trades, _ in subperiods:
            key = f"{period}_{split_label}"
            m   = _compute_subperiod_metrics(combo, split_trades, initial_balance)
            row[key] = m.get(primary_key, np.nan) if m is not None else np.nan
        return row

    results = Parallel(n_jobs=-1)(delayed(_score_combo)(combo) for combo in combos)
    df = pd.DataFrame(results)
    df["_combo_key"] = df["combo"].apply(lambda c: tuple(sorted(c)))
    df = df.sort_values("_combo_key").drop(columns="_combo_key").reset_index(drop=True)
    return df


# =============================================================================
# PRIVATE HELPERS — Scoring modes
# =============================================================================

def _apply_weighted_scoring(
    df_metrics: pd.DataFrame,
    subperiods: list[tuple],
    ranking_criteria: list[tuple[str, bool]],
) -> pd.DataFrame:
    """
    Aggregate precomputed primary metric as weighted average across subperiods.
    Secondary criteria are carried from df_metrics if present, else set to NaN.
    """
    subperiod_keys = [f"{p}_{q}" for p, q, _, _ in subperiods]
    weights        = {f"{p}_{q}": w for p, q, _, w in subperiods}
    primary_key    = ranking_criteria[0][0]
    total_weight   = sum(weights[key] for key in subperiod_keys)

    df = df_metrics.copy()

    df[primary_key] = df[subperiod_keys].apply(
        lambda row: sum(row[key] * weights[key] for key in subperiod_keys
                        if not np.isnan(row[key])) / total_weight,
        axis=1,
    ).round(3)

    df["subperiod_scores"] = df.apply(lambda row: {k: row[k] for k in subperiod_keys}, axis=1)
    df["subperiod_std"]    = df[subperiod_keys].std(axis=1).round(2)
    df["subperiod_min"]    = df[subperiod_keys].min(axis=1).round(1)
    df["subperiod_max"]    = df[subperiod_keys].max(axis=1).round(1)
    df["Weekly_std"]       = df["subperiod_std"]
    df["weighted_rank"]    = np.nan
    df["rank_variance"]    = np.nan

    sort_cols = [c for c, _ in ranking_criteria if c in df.columns]
    sort_asc  = [a for c, a in ranking_criteria if c in df.columns]
    df        = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    _log_top_combo_breakdown(df, [], primary_key, "weighted")

    return df


def _apply_ranking_scoring(
    df_metrics: pd.DataFrame,
    subperiods: list[tuple],
    ranking_criteria: list[tuple[str, bool]],
) -> pd.DataFrame:
    """
    Rank combos within each subperiod, then compute normalized weighted average rank.
    Lower weighted_rank = more consistently good combo.
    """
    subperiod_keys = [f"{p}_{q}" for p, q, _, _ in subperiods]
    weights        = {f"{p}_{q}": w for p, q, _, w in subperiods}
    primary_key    = ranking_criteria[0][0]
    primary_asc    = ranking_criteria[0][1]
    total_weight   = sum(weights[key] for key in subperiod_keys)

    df = df_metrics.copy()

    for key in subperiod_keys:
        rank_col     = f"rank_{key}"
        col_vals     = df[key].fillna(-np.inf if not primary_asc else np.inf).values
        order        = np.argsort(col_vals)[::-1] if not primary_asc else np.argsort(col_vals)
        ranks        = np.empty(len(df), dtype=int)
        ranks[order] = np.arange(1, len(df) + 1)
        df[rank_col] = ranks

    n_combos            = len(df)
    df["weighted_rank"] = sum(
        (df[f"rank_{key}"] / n_combos) * weights[key] for key in subperiod_keys
    ) / total_weight
    df["rank_variance"] = df[[f"rank_{key}" for key in subperiod_keys]].var(axis=1)

    df["subperiod_scores"] = df.apply(lambda row: {k: row[k] for k in subperiod_keys}, axis=1)
    df["subperiod_std"]    = df[subperiod_keys].std(axis=1).round(2)
    df["subperiod_min"]    = df[subperiod_keys].min(axis=1).round(1)
    df["subperiod_max"]    = df[subperiod_keys].max(axis=1).round(1)
    df["Weekly_std"]       = df["subperiod_std"]

    df[primary_key] = df[subperiod_keys].apply(
        lambda row: sum(row[key] * weights[key] for key in subperiod_keys
                        if not np.isnan(row[key])) / total_weight,
        axis=1,
    ).round(3)

    df = df.sort_values(["weighted_rank", "rank_variance"]).reset_index(drop=True)

    _log_top_combo_breakdown(df, subperiods, primary_key, "ranking")
    logger.debug(f"  [DEBUG] n_combos={n_combos}  top weighted_rank={df['weighted_rank'].iloc[0]:.4f}  top combo={df['combo'].iloc[0]}")
    return df[["combo", "weighted_rank", "rank_variance", "subperiod_scores",
               "subperiod_std", "subperiod_min", "subperiod_max", "Weekly_std", primary_key]]


# =============================================================================
# PRIVATE HELPERS — Debug logging
# =============================================================================

def _log_top_combo_breakdown(
    df: pd.DataFrame,
    subperiods: list[tuple],
    primary_key: str,
    mode: str,
) -> None:
    """Log subperiod breakdown for the top combo in a scored DataFrame."""
    top_row   = df.iloc[0]
    top_combo = top_row["combo"]
    scores    = top_row.get("subperiod_scores", {})
    nan_keys  = [k for k, v in scores.items() if isinstance(v, float) and np.isnan(v)]
    none_keys = [k for k, v in scores.items() if v is None]

    tag = f"[{mode.upper()}]"
    logger.debug(f"\n{'─'*115}")
    logger.debug(f"  {tag} Top combo subperiod breakdown — {primary_key}")
    logger.debug(f"  Combo: {list(top_combo)}")

    if mode == "ranking":
        logger.debug(f"  {'Subperiod':<18} {'Value':>10} {'Rank':>6} {'NaN/None':>10}")
        logger.debug(f"  {'-'*48}")
        for key in sorted(scores.keys()):
            val      = scores[key]
            is_nan   = isinstance(val, float) and np.isnan(val)
            flag     = "⚠️ NaN" if is_nan else ("⚠️ None" if val is None else "")
            val_str  = "NaN" if is_nan else ("None" if val is None else f"{val:.1f}")
            rank_val = top_row.get(f"rank_{key}", "—")
            rank_str = f"{rank_val}" if isinstance(rank_val, (int, np.integer)) else "—"
            logger.debug(f"  {key:<18} {val_str:>10} {rank_str:>6} {flag}")
        logger.debug(f"  {'-'*48}")
        logger.debug(f"  Weighted rank     : {top_row['weighted_rank']:.3f}")
        logger.debug(f"  Rank variance     : {top_row['rank_variance']:.1f}")
    else:
        logger.debug(f"  {'Subperiod':<18} {'Value':>10} {'NaN/None':>10}")
        logger.debug(f"  {'-'*40}")
        for key in sorted(scores.keys()):
            val     = scores[key]
            is_nan  = isinstance(val, float) and np.isnan(val)
            flag    = "⚠️ NaN" if is_nan else ("⚠️ None" if val is None else "")
            val_str = "NaN" if is_nan else ("None" if val is None else f"{val:.1f}")
            logger.debug(f"  {key:<18} {val_str:>10} {flag}")
        logger.debug(f"  {'-'*40}")

    logger.debug(f"  NaN subperiods    : {len(nan_keys)}  → {nan_keys}")
    logger.debug(f"  None subperiods   : {len(none_keys)} → {none_keys}")
    logger.debug(f"  {primary_key:<18}: {top_row[primary_key]:.3f}")
    logger.debug(f"  Subperiod std     : {top_row.get('subperiod_std', np.nan):.2f}")
    logger.debug(f"  Subperiod min     : {top_row.get('subperiod_min', np.nan):.1f}")
    logger.debug(f"  Subperiod max     : {top_row.get('subperiod_max', np.nan):.1f}")
    logger.debug(f"{'─'*115}")


# =============================================================================
# CONSENSUS
# =============================================================================

def _print_consensus(top_weighted: list, top_ranking: list) -> None:
    """Print combos that appear in both weighted and ranking top-N results."""
    W = 115

    weighted_combos = {frozenset(e["combo"]): (i + 1, e) for i, e in enumerate(top_weighted)}
    ranking_combos  = {frozenset(e["combo"]): (i + 1, e) for i, e in enumerate(top_ranking)}

    shared      = sorted(weighted_combos.keys() & ranking_combos.keys(),  key=lambda c: weighted_combos[c][0])

    logger.info(f"\n\033[94m{'='*W}\n  PORTFOLIO CONSENSUS\n{'='*W}\033[0m")

    if shared:
        for fs in shared:
            w_rank = weighted_combos[fs][0]
            r_rank = ranking_combos[fs][0]
            combo_str = "  |  ".join(sorted(fs, key=lambda s: int(s.split("_")[0])))
            logger.info(f" ⭐  {combo_str}   →   Weighted #{w_rank}  |  Ranking #{r_rank}")
    else:
        logger.info(f"  ⚠️  No combos in common between weighted and ranking top-{len(top_weighted)}")


    logger.info(f"{'─'*W}")


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
    mode: str = "weighted",
) -> None:
    W           = 115
    split_name  = _SPLIT_NAMES.get(n_splits, f"{n_splits}-Split")
    metric_str  = " | ".join(f"{m} ({'↑' if not asc else '↓'})" for m, asc in ranking_criteria)
    mode_str    = "Ranking (avg position)" if mode == "ranking" else "Weighted metrics"
    primary_key = ranking_criteria[0][0]

    logger.info(f"\n\033[94m{'='*W}\n  BEST PORTFOLIO COMBINATIONS — {split_name} splits | mode: {mode_str} | ranked by: {metric_str}\n{'='*W}\033[0m")

    val_lookup = {v["strategy_id"]: v for v in validation_results} if validation_results else {}

    for rank, entry in enumerate(top, start=1):
        combo            = entry["combo"]
        subperiod_scores = entry.get("subperiod_scores", {})
        std              = entry.get("subperiod_std",  np.nan)
        mn               = entry.get("subperiod_min",  np.nan)
        mx               = entry.get("subperiod_max",  np.nan)
        wpct_w           = entry.get(primary_key, 0)
        weighted_rank    = entry.get("weighted_rank",  np.nan)
        rank_variance    = entry.get("rank_variance",  np.nan)
        avg_trades       = np.mean([len(df) for sid, df in trades_per_period.get("OOS1", []) if sid in combo])

        if mode == "ranking":
            stats_str = f"avg_rank={weighted_rank:.3f}  rank_var={rank_variance:.1f}  std={std:.1f}  min={mn:.1f}  max={mx:.1f}  |  {primary_key}_w={wpct_w:.2f}"
        else:
            avg_rank_str = f"{weighted_rank:.3f}" if not (isinstance(weighted_rank, float) and np.isnan(weighted_rank)) else "—"
            stats_str    = f"{primary_key}_w={wpct_w:.2f}  std={std:.1f}  min={mn:.1f}  max={mx:.1f}  |  avg_rank={avg_rank_str}"

        logger.info(f"\nBEST #{rank} — Strategies: {len(combo)}  |  AvgTrades={avg_trades:.0f}  |  {stats_str}")
        logger.info(f"{'─'*W}")
        for s in sorted(combo, key=lambda s: int(s.split("_")[0])):
            icon = "🟢" if _is_long(s) else "🔴"
            logger.info(f"    {icon} {s}")

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

                logger.debug(f"\n{'─'*W}")
                logger.debug(f"  INDIVIDUAL STRATEGY THRESHOLDS — Combination #{rank}")
                logger.debug(f"{'─'*W}")
                logger.debug(f"  {'Strategy':<30} {'OOS':<6} {'NetGain%':>10} {'MaxDD%':>8} {'R2':>7}")
                logger.debug(f"  {'-'*W}")
                for sid in combo_strategies:
                    if sid not in val_lookup:
                        continue
                    v    = val_lookup[sid]
                    icon = "🟢" if _is_long(sid) else "🔴"
                    logger.debug(f"  {icon} {sid:<28} {'OOS1':<6} {v.get('net_gain_pct', 0):>9.1f}% {v.get('dd_pct', 0):>7.1f}% {v.get('r2', 0):>7.3f}")
                    logger.debug(f"  {'':31} {'OOS2':<6} {v.get('net_gain_pct_oos2', 0):>9.1f}% {v.get('dd_pct_oos2', 0):>7.1f}% {v.get('r2_oos2', 0):>7.3f}")
                    logger.debug(f"  {'':31} {'OOS3':<6} {v.get('net_gain_pct_oos3', 0):>9.1f}% {v.get('dd_pct_oos3', 0):>7.1f}% {v.get('r2_oos3', 0):>7.3f}")
                    logger.debug(f"  {'-'*W}")
                logger.debug(f"  {'WORST':<30} {'ALL':<6} {worst_netgain:>9.1f}% {worst_dd:>7.1f}% {worst_r2:>7.3f}")
                logger.debug(f"{'─'*W}")

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
    **kwargs,
) -> dict:
    """
    Find best portfolio combinations using both weighted and ranking modes.
    Metrics are computed once and reused for both scoring methods.

    Returns:
        dict with keys "weighted" and "ranking", each containing top_n combo records.
    """
    trades_per_period = {
        "OOS1": validated_trades_oos1,
        "OOS2": validated_trades_oos2,
        "OOS3": validated_trades_oos3,
    }

    all_ids = list({sid for sid, _ in validated_trades_oos1})
    if not all_ids:
        logger.warning("No validated strategies — skipping best portfolio search.")
        return {"weighted": [], "ranking": []}

    period_date_bounds = {
        "OOS1": _parse_period_bounds(data_folder_oos1),
        "OOS2": _parse_period_bounds(data_folder_oos2),
        "OOS3": _parse_period_bounds(data_folder_oos3),
    }
    subperiods = _build_subperiod_index(trades_per_period, n_splits, period_weights, period_date_bounds)
    if not subperiods:
        logger.warning("No subperiods could be built — check trades data.")
        return {"weighted": [], "ranking": []}

    split_name   = _SPLIT_NAMES.get(n_splits, f"{n_splits}-Split")
    primary_key  = ranking_criteria[0][0]
    logger.info(f"\n  Subperiods: {len(subperiods)} {split_name.lower()}(s) across {len(trades_per_period)} OOS periods")

    combos = [
        combo
        for size in range(min_strategies, max_strategies + 1)
        for combo in combinations(all_ids, size)
        if not require_long_short or _has_long_and_short(combo)
    ]
    if not combos:
        logger.warning("No valid combinations found.")
        return {"weighted": [], "ranking": []}

    logger.info(f"  Evaluating {len(combos)} combo(s)...")

    # --- Compute metrics once ---
    df_metrics = _build_metrics_table(combos, subperiods, initial_balance, primary_key)
    if df_metrics.empty:
        logger.warning("No scores computed — check trades data.")
        return {"weighted": [], "ranking": []}

    # --- Apply both scoring modes ---
    df_weighted = _apply_weighted_scoring(df_metrics, subperiods, ranking_criteria)
    df_ranking  = _apply_ranking_scoring(df_metrics, subperiods, ranking_criteria)

    results = {}
    for mode, df_scored in [("weighted", df_weighted), ("ranking", df_ranking)]:
        best_val   = df_scored[primary_key].iloc[0]
        mean_val   = df_scored[primary_key].mean()
        std_val    = df_scored[primary_key].std()
        gap        = round(best_val - mean_val, 2)
        zscore     = round((best_val - mean_val) / std_val, 2) if std_val > 0 else 0.0
        logger.info(
            f"  [{mode.upper()}] Distribution ({primary_key}): "
            f"best={best_val:.1f}  mean={mean_val:.1f}  std={std_val:.1f}  "
            f"min={df_scored[primary_key].min():.1f}  max={df_scored[primary_key].max():.1f}  "
            f"gap={gap:.2f}  z-score={zscore}"
        )

        top = df_scored.head(top_n).to_dict("records")
        _print_best_combinations(top, trades_per_period, initial_balance, n_splits, ranking_criteria, validation_results, mode)
        results[mode] = top

    _print_consensus(results["weighted"], results["ranking"])

    top_weighted = results["weighted"][0]
    top_ranking  = results["ranking"][0]
    consensus    = frozenset(top_weighted["combo"]) == frozenset(top_ranking["combo"])

    if consensus:
        plot_best_portfolio(
            combo             = top_weighted["combo"],
            trades_per_period = trades_per_period,
            subperiod_scores  = top_weighted["subperiod_scores"],
            subperiods        = subperiods,
            initial_balance   = initial_balance,
            data_folder_oos1  = data_folder_oos1,
            data_folder_oos2  = data_folder_oos2,
            data_folder_oos3  = data_folder_oos3,
            df_scored         = df_weighted,
            ranking_criteria  = ranking_criteria,
            show_plots        = show_plots,
            title             = "Best Portfolio — CONSENSUS",
        )
    else:
        plot_best_portfolio(
            combo             = top_weighted["combo"],
            trades_per_period = trades_per_period,
            subperiod_scores  = top_weighted["subperiod_scores"],
            subperiods        = subperiods,
            initial_balance   = initial_balance,
            data_folder_oos1  = data_folder_oos1,
            data_folder_oos2  = data_folder_oos2,
            data_folder_oos3  = data_folder_oos3,
            df_scored         = df_weighted,
            ranking_criteria  = ranking_criteria,
            show_plots        = show_plots,
            title             = "Best Portfolio — Weighted #1",
        )
        plot_best_portfolio(
            combo             = top_ranking["combo"],
            trades_per_period = trades_per_period,
            subperiod_scores  = top_ranking["subperiod_scores"],
            subperiods        = subperiods,
            initial_balance   = initial_balance,
            data_folder_oos1  = data_folder_oos1,
            data_folder_oos2  = data_folder_oos2,
            data_folder_oos3  = data_folder_oos3,
            df_scored         = df_ranking,
            ranking_criteria  = ranking_criteria,
            show_plots        = show_plots,
            title             = "Best Portfolio — Ranking #1",
        )

    return results