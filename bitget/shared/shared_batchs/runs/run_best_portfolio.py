# shared_batchs/runs/run_best_portfolio.py
import logging
import numpy as np
import pandas as pd
from itertools import combinations

from shared_batchs.utils.batch_metrics import compute_metrics, _weekly_returns, _cvar, _neg_streak_stats
from shared_batchs.utils.reporting import _build_robustness_rows, _print_robustness_df

logger = logging.getLogger("BOT_batch.runs.run_best_portfolio")


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_STRATEGIES = 2
MAX_STRATEGIES = 5

PERIOD_WEIGHTS = {
    "OOS1": 0.50,
    "OOS2": 0.25,
    "OOS3": 0.25,
}

# ascending=False → higher is better  |  ascending=True → lower is better
RANKING_CRITERIA = [
    ("Weekly_pct", False),
    ("NetGain%",   False),
]

# =============================================================================
# RANKING_CRITERIA = [
#     ("CVaR10%",False),
#     ("NetGain%",False),
# ]
# =============================================================================

TOP_N = 2


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _is_long(strategy_id: str) -> bool:
    return "_long_" in strategy_id


def _is_short(strategy_id: str) -> bool:
    return "_short_" in strategy_id


def _has_long_and_short(strategy_ids: tuple) -> bool:
    return any(_is_long(s) for s in strategy_ids) and any(_is_short(s) for s in strategy_ids)


def _compute_period_metrics(
    combo: tuple,
    trades_list: list,
    initial_balance: float,
) -> dict | None:
    """
    Compute full robustness metrics for a combo in one OOS period.
    Mirrors the logic of _build_robustness_rows. Returns None if no trades found.
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


def _compute_weighted_metrics(
    combo: tuple,
    trades_per_period: dict,
    initial_balance: float,
    period_weights: dict,
) -> dict | None:
    """
    Compute weighted metrics across OOS periods for a strategy combination.
    Returns None if any period has no trades for the combo.
    """
    metric_keys    = ["NetGain%", "MaxDD%", "R2", "ProfitFactor", "CVaR10%",
                      "AvgNegStreak", "MaxNegStreak", "Weekly_pct", "Weekly_avg%", "MinWeekly%"]
    period_metrics = {}

    for period, trades_list in trades_per_period.items():
        m = _compute_period_metrics(combo, trades_list, initial_balance)
        if m is None:
            continue
        period_metrics[period] = m

    if not period_metrics:
        return None

    weight_total = sum(period_weights.get(p, 0) for p in period_metrics)
    if weight_total == 0:
        return None

    weighted = {
        key: sum(
            period_metrics[p][key] * period_weights.get(p, 0)
            for p in period_metrics
            if period_metrics[p].get(key) is not None and not np.isnan(period_metrics[p][key])
        ) / weight_total
        for key in metric_keys
    }
    weighted["Weekly_pct_rounded"] = round(weighted["Weekly_pct"])
    weighted["period_metrics"]     = period_metrics

    # Arithmetic mean for display only — not used for ranking
    n = len(period_metrics)
    weighted["display_mean"] = {
        key: round(sum(period_metrics[p][key] for p in period_metrics) / n, 2)
        for key in metric_keys
    }
    return weighted


# =============================================================================
# PRINT
# =============================================================================

def _print_best_combinations(top: list, period_weights: dict) -> None:
    W = 115
    logger.info(f"\n{'─'*W}\n  BEST PORTFOLIO COMBINATIONS\n{'─'*W}")

    for rank, entry in enumerate(top, start=1):
        combo          = entry["combo"]
        period_metrics = entry["period_metrics"]

        logger.info(f"\n  #{rank}  ({len(combo)} strategies)")
        logger.info(f"  {'─'*W}")
        for s in combo:
            icon = "🟢" if _is_long(s) else "🔴"
            logger.info(f"    {icon} {s}")

        rows = []
        for period, m in period_metrics.items():
            rows.append({
                "Period":       period,
                "NetGain%":     m["NetGain%"],
                "MaxDD%":       m["MaxDD%"],
                "R2":           m["R2"],
                "ProfitFactor": m["ProfitFactor"],
                "CVaR10%":      m["CVaR10%"],
                "AvgNegStreak": m["AvgNegStreak"],
                "MaxNegStreak": m["MaxNegStreak"],
                "Weekly_pct":   m["Weekly_pct"],
                "Weekly_avg%":  m["Weekly_avg%"],
                "MinWeekly%":   m["MinWeekly%"],
            })
        _print_robustness_df(rows, f"ROBUSTNESS TABLE — Combination #{rank}")

    logger.info(f"\n{'─'*W}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def find_best_portfolio_combination(
    validated_trades_oos1: list,
    validated_trades_oos2: list,
    validated_trades_oos3: list,
    initial_balance: float,
    min_strategies: int    = MIN_STRATEGIES,
    max_strategies: int    = MAX_STRATEGIES,
    period_weights: dict   = PERIOD_WEIGHTS,
    ranking_criteria: list = RANKING_CRITERIA,
    top_n: int             = TOP_N,
) -> list:
    """
    Find the best combinations of validated strategies based on weighted robustness metrics.

    validated_trades_oosN : list of (strategy_id, trades_df) for each OOS period
    initial_balance       : capital per strategy
    min_strategies        : minimum strategies in a combination
    max_strategies        : maximum strategies in a combination
    period_weights        : dict {period_label: weight} — must sum to 1.0
    ranking_criteria      : list of (metric_key, ascending) — ascending=True → lower is better
    top_n                 : number of top combinations to display
    """
    trades_per_period = {
        "OOS1": validated_trades_oos1,
        "OOS2": validated_trades_oos2,
        "OOS3": validated_trades_oos3,
    }

    all_ids = list({sid for sid, _ in validated_trades_oos1})
    if not all_ids:
        logger.warning("⚠️  No validated strategies — skipping best portfolio search.")
        return []

    results = []
    for size in range(min_strategies, max_strategies + 1):
        for combo in combinations(all_ids, size):
            if not _has_long_and_short(combo):
                continue
            metrics = _compute_weighted_metrics(combo, trades_per_period, initial_balance, period_weights)
            if metrics is None:
                continue
            results.append({"combo": combo, **metrics})

    if not results:
        logger.warning("⚠️  No valid combinations found.")
        return []

    df_results = pd.DataFrame(results)
    sort_cols  = [c for c, _ in ranking_criteria]
    sort_asc   = [a for _, a in ranking_criteria]
    df_results = df_results.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    top = df_results.head(top_n).to_dict("records")
    _print_best_combinations(top, period_weights)
    return top