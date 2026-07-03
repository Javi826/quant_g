#shared/shared_batchs/runs/run_best_wfo_portfolio.py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import combinations
from joblib import Parallel, delayed

from shared_batchs.utils.batch_metrics import compute_metrics

logger = logging.getLogger("BOT_batch.runs.run_best_wfo_portfolio")

# =============================================================================
# CONFIGURATION
# =============================================================================

WFO_METRIC            = "R_SQUARED"         # NET_GAIN_PCT | WEEKLY_PCT | WIN_RATE | CALMAR | R_SQUARED | MAX_DD_PCT
WFO_N_SPLITS          = 4
WFO_SUBPERIOD_WEIGHTS = [0.10, 0.20, 0.20, 0.50]  # must match WFO_N_SPLITS, recency-weighted
#WFO_SUBPERIOD_WEIGHTS = [1] 

MIN_STRATEGIES     = 2
MAX_STRATEGIES     = 6
TOP_N              = 2
REQUIRE_LONG_SHORT = True

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


def _has_long_and_short(combo: tuple) -> bool:
    return any(_is_long(s) for s in combo) and any(_is_short(s) for s in combo)


# =============================================================================
# PRIVATE HELPERS — Splitting
# =============================================================================

def _split_trades_by_time(
    trades_list: list,
    n_splits: int,
) -> list:
    """
    Split (strategy_id, trades_df) list into n_splits equal time buckets by sell_time.
    Returns list of (label, t_start, t_end, subset_trades_list). Empty buckets are skipped.
    """
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
    """Extract scalar metric value from compute_metrics output. Always returns a value to maximize."""
    col, higher_is_better = _METRIC_MAP[metric]
    val = m.get(col, np.nan)
    return val if higher_is_better else -abs(val)  # penalize larger magnitude, regardless of sign # negate so we always maximize


# =============================================================================
# PRIVATE HELPERS — Combo scoring (raw metric per subperiod)
# =============================================================================

def _score_combo(
    combo: tuple,
    subperiods: list,
    initial_balance: float,
    metric: str,
) -> dict:
    """Compute raw metric value for one combo in each subperiod (NaN if no trades)."""
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
    """
    Convert raw per-subperiod metric values into per-subperiod ranks (1 = best).
    Combos with NaN in a subperiod get the worst rank in that subperiod (penalized).
    Returns the same list of dicts, augmented with '<label>_rank' entries.
    """
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


# =============================================================================
# PRIVATE HELPERS — Printing
# =============================================================================

def _print_results(
    top: list,
    subperiods: list,
    trades_list: list,
    initial_balance: float,
    metric: str,
    weights: list,
) -> None:
    W          = 115
    split_keys = [label for label, _, _, _ in subperiods]

    logger.info(f"\n{'='*W}")
    logger.info(f"  BEST WFO PORTFOLIO — metric: {metric} | splits: {len(subperiods)} | weights: {[round(w, 2) for w in weights]}")
    logger.info(f"{'='*W}")

    for rank, entry in enumerate(top, start=1):
        combo      = entry["combo"]
        score      = entry["weighted_rank_score"]
        avg_trades = np.mean([len(df) for sid, df in trades_list if sid in combo])

        logger.info(f"\nBEST #{rank} — Strategies: {len(combo)}  |  AvgTrades/strat={avg_trades:.0f}  |  WeightedRankScore={score:.2f}")
        logger.info(f"{'─'*W}")

        for s in sorted(combo, key=lambda s: int(s.split("_")[0])):
            icon = "🟢" if _is_long(s) else "🔴"
            logger.info(f"    {icon} {s}")

        logger.info(f"\n  {'Subperiod':<10} {'Weight':>8} {'Value':>10} {'Rank':>6}  {'Period'}")
        logger.info(f"  {'─'*65}")
        for i, (lbl, t_start, t_end, _) in enumerate(subperiods):
            val      = entry.get(lbl, np.nan)
            val_str  = f"{val:.3f}" if not np.isnan(val) else "N/A"
            rank_val = entry.get(f"{lbl}_rank", "-")
            logger.info(f"  {lbl:<10} {weights[i]:>8.2f} {val_str:>10} {rank_val:>6}  ({t_start.strftime('%Y-%m-%d')} → {t_end.strftime('%Y-%m-%d')})")
        logger.info(f"  {'─'*65}")
        logger.info(f"  {'WEIGHTED RANK':<10} {'':>8} {'':>10} {score:>6.2f}")

        combo_trades = [(sid, df) for sid, df in trades_list if sid in combo]
        if combo_trades:
            tl            = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
            total_capital = initial_balance * len(combo_trades)
            m             = compute_metrics(tl, capital=total_capital, name="")
            logger.info(
                f"\n  Full period ── "
                f"NetGain={m['Net_Gain_pct']:.1f}%  "
                f"DD={m['Max_DD_pct']:.1f}%  "
                f"WinRate={m['Win_Rate']:.1f}%  "
                f"R2={m['R_Squared']:.3f}  "
                f"PF={m['Profit_Factor']:.2f}  "
                f"Calmar={m['Calmar']:.2f}  "
                f"Weekly%={m['Weekly_pct']:.1f}%"
            )

            n_months        = max((pd.to_datetime(tl["sell_time"]).max() - pd.to_datetime(tl["sell_time"]).min()).days / 30.44, 1)
            avg_monthly_pct = round(m["Net_Gain_pct"] / n_months, 2)
            logger.info(f"  Monthly NetGain  ── {avg_monthly_pct:+.2f}% / month  ({n_months:.1f} months)")

    logger.info(f"\n{'─'*W}")


# =============================================================================
# PRIVATE HELPERS — Plotting
# =============================================================================

def _plot_wfo_portfolio(
    combo: tuple,
    trades_list: list,
    subperiods: list,
    subperiod_scores: dict,
    df_scored: pd.DataFrame,
    initial_balance: float,
    metric: str,
    weights: list,
    title: str,
    validated_trades: list = None,
) -> None:
    combo_trades = [(sid, df) for sid, df in trades_list if sid in combo]
    if not combo_trades:
        return

    tl            = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
    total_capital = initial_balance * len(combo_trades)
    m             = compute_metrics(tl, capital=total_capital, name="")

    eq     = total_capital + tl["profit"].cumsum().values
    eq_pct = (eq - total_capital) / total_capital * 100
    ts     = pd.to_datetime(tl["sell_time"]).values

    ts_val = eq_val_pct = m_val = None
    if validated_trades:
        tl_val            = pd.concat([df for _, df in validated_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
        total_capital_val = initial_balance * len(validated_trades)
        m_val             = compute_metrics(tl_val, capital=total_capital_val, name="")
        eq_val            = total_capital_val + tl_val["profit"].cumsum().values
        eq_val_pct        = (eq_val - total_capital_val) / total_capital_val * 100
        ts_val            = pd.to_datetime(tl_val["sell_time"]).values

    _BG          = "#F8F9FA"
    _COLORS_BAND = ["#EBF5FB", "#EAFAF1", "#FEF9E7", "#FDEDEC"]
    _COLOR_EQ    = "#2E86C1"
    _COLOR_VAL   = "#00897B"

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(_BG)
    combo_str = " | ".join(sorted(combo, key=lambda s: int(s.split("_")[0])))
    fig.suptitle(title or f"Best WFO Portfolio — {combo_str}", fontsize=10, fontweight="bold")

    # ── Panel 1: Equity curve ─────────────────────────────────────────────────
    ax1.set_facecolor(_BG)

    for i, (_, t_start, t_end, _) in enumerate(subperiods):
        ax1.axvspan(t_start, t_end, alpha=0.25, color=_COLORS_BAND[i % len(_COLORS_BAND)])
        ax1.axvline(t_start, color="#AAAAAA", linewidth=0.5, linestyle="--", alpha=0.4)

    legend_label = (
        f"Best combo  NetGain={m['Net_Gain_pct']:.1f}%  "
        f"DD={m['Max_DD_pct']:.1f}%  "
        f"R²={m['R_Squared']:.3f}"
    )
    ax1.plot(ts, eq_pct, color=_COLOR_EQ, linewidth=1.0, label=legend_label)

    if ts_val is not None:
        legend_label_val = (
            f"Validated   NetGain={m_val['Net_Gain_pct']:.1f}%  "
            f"DD={m_val['Max_DD_pct']:.1f}%  "
            f"R²={m_val['R_Squared']:.3f}"
        )
        ax1.plot(ts_val, eq_val_pct, color=_COLOR_VAL, linewidth=0.8, alpha=0.8, label=legend_label_val)

    ax1.axhline(0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.5)

    _legend = ax1.legend(loc="upper left", fontsize=8, framealpha=0.9,
                         facecolor="white", edgecolor="#AAAAAA")
    for _text in _legend.get_texts():
        _text.set_fontfamily("monospace")
    ax1.set_title("Equity Curve (WFO Test)", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Net Gain (%)", fontsize=9)
    _locator = mdates.MonthLocator(interval=2)
    ax1.xaxis.set_major_locator(_locator)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="both", labelsize=7)
    ax1.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, color="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def find_best_portfolio_combination_wfo(
    validated_wfo_trades: list,
    initial_balance: float,
    metric: str              = WFO_METRIC,
    n_splits: int            = WFO_N_SPLITS,
    subperiod_weights: list  = WFO_SUBPERIOD_WEIGHTS,
    min_strategies: int      = MIN_STRATEGIES,
    max_strategies: int      = MAX_STRATEGIES,
    top_n: int               = TOP_N,
    require_long_short: bool = REQUIRE_LONG_SHORT,
    show_plots: bool         = False,
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

    combos = [
        combo
        for size in range(min_strategies, max_strategies + 1)
        for combo in combinations(all_ids, size)
        if not require_long_short or _has_long_and_short(combo)
    ]
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
    _print_results(top, subperiods, validated_wfo_trades, initial_balance, metric, subperiod_weights)

    df_scored = pd.DataFrame([
        {"combo": r["combo"], "weighted_rank_score": r["weighted_rank_score"]}
        for r in raw_scores
    ])

    if top and show_plots:
        top_entry         = top[0]
        top_subp_scores   = {lbl: top_entry.get(lbl, np.nan) for lbl, _, _, _ in subperiods}
        _plot_wfo_portfolio(
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