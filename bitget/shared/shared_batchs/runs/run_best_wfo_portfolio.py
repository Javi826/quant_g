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

WFO_METRIC            = "NET_GAIN_PCT"         # NET_GAIN_PCT | WEEKLY_PCT | WIN_RATE | CALMAR | R_SQUARED | MAX_DD_PCT
WFO_N_SPLITS          = 4
WFO_SUBPERIOD_WEIGHTS = [0.10, 0.20, 0.30, 0.40]  # must match WFO_N_SPLITS, recency-weighted

MIN_STRATEGIES     = 2
MAX_STRATEGIES     = 5
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
    return val if higher_is_better else -val  # negate so we always maximize


# =============================================================================
# PRIVATE HELPERS — Combo scoring
# =============================================================================

def _score_combo(
    combo: tuple,
    subperiods: list,
    initial_balance: float,
    metric: str,
    weights: list,
) -> dict:
    """Compute weighted score for one combo across all subperiods."""
    scores     = {}
    valid_sum  = 0.0
    weight_sum = 0.0

    for i, (label, _t_start, _t_end, split_trades) in enumerate(subperiods):
        combo_trades = [(sid, df) for sid, df in split_trades if sid in combo]
        if not combo_trades:
            scores[label] = np.nan
            continue

        tl            = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
        total_capital = initial_balance * len(combo_trades)
        m             = compute_metrics(tl, capital=total_capital, name="")
        val           = _extract_metric(m, metric)

        scores[label]  = round(val, 3)
        valid_sum     += val * weights[i]
        weight_sum    += weights[i]

    weighted_score = valid_sum / weight_sum if weight_sum > 0 else np.nan
    return {"combo": combo, "weighted_score": weighted_score, **scores}


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
        score      = entry["weighted_score"]
        avg_trades = np.mean([len(df) for sid, df in trades_list if sid in combo])

        logger.info(f"\nBEST #{rank} — Strategies: {len(combo)}  |  AvgTrades/strat={avg_trades:.0f}  |  WeightedScore={score:.3f}")
        logger.info(f"{'─'*W}")

        for s in sorted(combo, key=lambda s: int(s.split("_")[0])):
            icon = "🟢" if _is_long(s) else "🔴"
            logger.info(f"    {icon} {s}")

        logger.info(f"\n  {'Subperiod':<10} {'Weight':>8} {'Score':>10}  {'Period'}")
        logger.info(f"  {'─'*55}")
        for i, (lbl, t_start, t_end, _) in enumerate(subperiods):
            val     = entry.get(lbl, np.nan)
            val_str = f"{val:.3f}" if not np.isnan(val) else "N/A"
            logger.info(f"  {lbl:<10} {weights[i]:>8.2f} {val_str:>10}  ({t_start.strftime('%Y-%m-%d')} → {t_end.strftime('%Y-%m-%d')})")
        logger.info(f"  {'─'*55}")
        logger.info(f"  {'WEIGHTED':<10} {'':>8} {score:>10.3f}")

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

    split_labels  = [label for label, _, _, _ in subperiods]
    split_scores  = [subperiod_scores.get(lbl, np.nan) for lbl in split_labels]
    split_mids    = [t_start + (t_end - t_start) / 2 for _, t_start, t_end, _ in subperiods]
    weighted_mean = float(np.nanmean(split_scores)) if split_scores else 0.0

    _BG          = "#F8F9FA"
    _COLORS_BAND = ["#EBF5FB", "#EAFAF1", "#FEF9E7", "#FDEDEC"]
    _COLOR_EQ    = "#2E86C1"
    _COLOR_MEAN  = "#E67E22"
    _COLOR_POS   = "#2E86C1"
    _COLOR_NEG   = "#C0392B"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [2, 1.2, 1]})
    fig.patch.set_facecolor(_BG)
    combo_str = " | ".join(sorted(combo, key=lambda s: int(s.split("_")[0])))
    fig.suptitle(title or f"Best WFO Portfolio — {combo_str}", fontsize=10, fontweight="bold")

    # ── Panel 1: Equity curve ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(_BG)

    for i, (_, t_start, t_end, _) in enumerate(subperiods):
        ax1.axvspan(t_start, t_end, alpha=0.25, color=_COLORS_BAND[i % len(_COLORS_BAND)])
        ax1.axvline(t_start, color="#AAAAAA", linewidth=0.5, linestyle="--", alpha=0.4)

    ax1.plot(ts, eq_pct, color=_COLOR_EQ, linewidth=1.0)
    ax1.axhline(0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.5)

    legend_label = (
        f"NetGain={m['Net_Gain_pct']:.1f}%  "
        f"DD={m['Max_DD_pct']:.1f}%  "
        f"R²={m['R_Squared']:.3f}"
    )
    ax1.legend([legend_label], loc="upper left", fontsize=8, framealpha=0.9,
               facecolor="white", edgecolor="#AAAAAA", fontfamily="monospace")
    ax1.set_title("Equity Curve (WFO Test)", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Net Gain (%)", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(axis="both", labelsize=7)
    ax1.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, color="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: Metric per subperiod bar chart ───────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(_BG)

    x_idx    = list(range(len(split_labels)))
    x_labels = [
        f"{lbl}\n{split_mids[i].strftime('%Y-%m')}"
        for i, lbl in enumerate(split_labels)
    ]
    bar_colors = [
        _COLOR_POS if (not np.isnan(v) and v >= 0) else _COLOR_NEG
        for v in split_scores
    ]

    ax2.bar(x_idx, split_scores, color=bar_colors, alpha=0.75, edgecolor="white")
    ax2.axhline(weighted_mean, color=_COLOR_MEAN, linewidth=1.0, linestyle="--",
                label=f"Weighted mean={weighted_mean:.2f}")

    for i, (xi, yi) in enumerate(zip(x_idx, split_scores)):
        if not np.isnan(yi):
            offset = abs(yi) * 0.03 if yi >= 0 else -abs(yi) * 0.08
            ax2.text(xi, yi + offset, f"w={weights[i]:.2f}", ha="center", fontsize=7, color="#444444")

    ax2.set_title(f"{metric} per Subperiod", fontsize=9, fontweight="bold")
    ax2.set_ylabel(metric, fontsize=9)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, color="#CCCCCC")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=8, framealpha=0.9, facecolor="white", edgecolor="#AAAAAA")

    # ── Panel 3: Distribution of all combos ───────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(_BG)

    all_scores = df_scored["weighted_score"].dropna().values
    best_score = float(df_scored["weighted_score"].iloc[0])
    mean_score = float(np.mean(all_scores))
    std_score  = float(np.std(all_scores))

    ax3.hist(all_scores, bins=20, color=_COLOR_POS, alpha=0.65, edgecolor="white", linewidth=0.5)
    ax3.axvline(best_score, color="#0D3B6E", linewidth=1.5, linestyle="-",
                label=f"Best={best_score:.2f}")
    ax3.axvline(mean_score, color=_COLOR_MEAN, linewidth=1.2, linestyle="--",
                label=f"Mean={mean_score:.2f}")
    ax3.axvspan(mean_score - std_score, mean_score + std_score,
                alpha=0.1, color=_COLOR_MEAN, label=f"±1σ={std_score:.2f}")

    ax3.set_title(f"Combo Distribution ({len(df_scored)} combos)", fontsize=9, fontweight="bold")
    ax3.set_xlabel(f"{metric} (weighted)", fontsize=8)
    ax3.set_ylabel("Count", fontsize=8)
    ax3.tick_params(axis="both", labelsize=7)
    ax3.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, color="#CCCCCC")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(fontsize=7, framealpha=0.9, facecolor="white", edgecolor="#AAAAAA")

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
    """
    Find best portfolio combination from WFO test trades of validated strategies.

    Splits the WFO test period into n_splits temporal subperiods, applies
    recency weights (subperiod_weights[-1] = most recent), and selects the
    combination that maximizes the weighted metric score.

    Args:
        validated_wfo_trades : list of (strategy_id, trades_df) — validated strategies only
        initial_balance      : capital per strategy
        metric               : metric to maximize — NET_GAIN_PCT | WEEKLY_PCT | WIN_RATE | CALMAR | R_SQUARED | MAX_DD_PCT
        n_splits             : number of equal temporal subperiods
        subperiod_weights    : weight per subperiod — len must equal n_splits
        min_strategies       : minimum strategies per combo
        max_strategies       : maximum strategies per combo
        top_n                : number of top combos to display
        require_long_short   : combo must include at least one long and one short strategy
        show_plots           : render equity curve, subperiod bars, and combo distribution

    Returns:
        list of top_n dicts with keys: combo, weighted_score, S1..SN
    """
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

    results = Parallel(n_jobs=-1)(
        delayed(_score_combo)(combo, subperiods, initial_balance, metric, subperiod_weights)
        for combo in combos
    )

    results.sort(
        key=lambda x: x["weighted_score"] if not np.isnan(x["weighted_score"]) else -np.inf,
        reverse=True,
    )

    top = results[:top_n]
    _print_results(top, subperiods, validated_wfo_trades, initial_balance, metric, subperiod_weights)

    df_scored = pd.DataFrame([
        {"combo": r["combo"], "weighted_score": r["weighted_score"]}
        for r in results
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
        )

    return top