#shared/shared_batch_regime/regime_reporting.py

import logging

logger = logging.getLogger(__name__)

from shared_batch_regime.regime_core import BINS, pct_improvement

OPT_METRIC = "Net_Gain_pct"

# =============================================================================
# COMBO STRATEGY TABLE — one row per strategy, baseline vs each bin
# =============================================================================

def print_combo_strategy_table(strategy_metrics: dict, label: str, combo_idx: int = 0, n_combos: int = 0) -> None:
    bin_w   = max(12, max(len(b) for b in BINS) + 2)
    delta_w = 8
    row_w   = 2 + 35 + 1 + 8 + len(BINS) * (2 + bin_w + 1 + delta_w + 1)
    sep     = f"  {'─' * row_w}"
    header  = (
        f"  {'STRATEGY':<35} {'BASELINE':>8}"
        + "".join(f"  {b.upper():<{bin_w}} {'Δ%':>{delta_w}}" for b in BINS)
    )
    idx_str = f"  [{combo_idx}/{n_combos}]" if combo_idx else ""
    logger.debug(f"\n{sep}")
    logger.debug(f"  {label}{idx_str}")
    logger.debug(sep)
    logger.debug(header)
    logger.debug(sep)

    for sid, data in sorted(strategy_metrics.items()):
        m        = data["metrics"]
        base_val = m.get(f"b_{OPT_METRIC}", 0.0)
        bin_cols = ""
        for b in BINS:
            val     = m.get(f"{b}_{OPT_METRIC}", 0.0)
            delta   = pct_improvement(val, base_val)
            color   = "\033[92m" if delta > 0 else "\033[91m"
            val_str = f"{val:>{bin_w}.1f}"
            dlt_str = f"{delta:>+{delta_w - 1}.1f}%"
            bin_cols += f"  {val_str} {color}{dlt_str}\033[0m"
        logger.debug(f"  {sid:<35} {base_val:>8.1f}{bin_cols}")
    logger.debug(sep)

# =============================================================================
# COMBO SUMMARY
# =============================================================================

def print_combo_summary(
    bin_counts: dict[str, int],
    n_neutral:  int,
    comb_p:     float,
    comb_dd:    float,
    base_p:     float,
    base_dd:    float,
    label:      str,
    combo_idx:  int = 0,
    n_combos:   int = 0,
) -> None:
    idx_str  = f"  [{combo_idx}/{n_combos}]" if combo_idx else ""
    comb_pct = pct_improvement(comb_p, base_p)
    color    = "\033[92m" if comb_pct > 0 else "\033[91m"
    cls_str  = "  ".join(f"{b.upper()}:{bin_counts.get(b, 0)}" for b in BINS)
    logger.debug(f"\n  COMBO SUMMARY{idx_str} — {label}")
    logger.debug(f"  Classifications — {cls_str}  NEUTRAL:{n_neutral}")
    logger.debug(f"  Baseline  profit={base_p:>10.1f}  avg_dd={base_dd:>6.1f}%")
    logger.debug(f"  Combined  profit={comb_p:>10.1f}  avg_dd={comb_dd:>6.1f}%  {color}Delta={comb_pct:>+6.1f}%\033[0m")

# =============================================================================
# RANKING
# =============================================================================

def print_ranking(ranking: list[dict]) -> None:
    cfg_w       = max(25, max(len(r["label"]) for r in ranking[:5]) + 2)
    bin_headers = "  ".join(f"{b.upper()[:8]:>8}" for b in BINS)
    header_line = (
        f"  {'#':>3}  {'COMBO':>5}  {'CFG':<{cfg_w}}  "
        f"{bin_headers}  {'NEUT':>5}  "
        f"{'BASELINE':>10} {'COMB_PROF':>10} {'COMB_Δ%':>8} {'W_DELTA%':>9}  "
        f"{'BASE_DD%':>8} {'COMB_DD%':>8}"
    )
    total_w = len(header_line) - 2
    logger.info(f"\n\n{'='*total_w}")
    logger.info(f"  FINAL RANKING — ALL COMBOS BY WEIGHTED DELTA VS BASELINE")
    logger.info(f"{'='*total_w}")
    logger.info(header_line)
    logger.info(f"  {'─'*total_w}")
    for i, row in enumerate(ranking[:5], 1):
        pct     = pct_improvement(row["combined_profit"], row["baseline_profit"])
        w_delta = row.get("weighted_delta", 0.0)
        cc      = "\033[92m" if pct > 0 else "\033[91m"
        wc      = "\033[92m" if w_delta > 0 else "\033[91m"
        ddc     = "\033[92m" if row["combined_dd"] > row["baseline_dd"] else "\033[91m"
        rs      = "\033[0m"
        bin_cols = "  ".join(f"{row['bin_counts'].get(b, 0):>8}" for b in BINS)
        logger.info(
            f"  {i:>3}  {row['combo_idx']:>5}  {row['label']:<{cfg_w}}  "
            f"{bin_cols}  {row['n_neutral']:>5}  "
            f"{row['baseline_profit']:>10.1f} {cc}{row['combined_profit']:>10.1f}{rs} "
            f"{cc}{pct:>+7.1f}%{rs} {wc}{w_delta:>+8.1f}%{rs}  "
            f"{row['baseline_dd']:>7.1f}% {ddc}{row['combined_dd']:>7.1f}%{rs}"
        )
    logger.info(f"  {'─'*total_w}\n")

# =============================================================================
# CLASSIFICATION SUMMARY
# =============================================================================

def print_classification_summary(strategy_results: dict, excluded_ids: list[str] | None = None) -> None:
    print(f"\n{'='*120}")
    print(f"  STRATEGY CLASSIFICATION SUMMARY")
    print(f"{'='*120}")
    print(f"  {'STRATEGY':<35} {'DIR':<6} {'BIN'}")
    print(f"  {'─'*70}")
    _color_cycle = ["\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[95m"]
    bin_colors   = {b: _color_cycle[i % len(_color_cycle)] for i, b in enumerate(BINS)}
    bin_colors["neutral"] = "\033[90m"
    excluded_set = set(excluded_ids or [])
    all_entries  = {
        **{sid: (data.get("is_long"), data.get("classification", []), False) for sid, data in strategy_results.items()},
        **{sid: ("long" in sid, [], True) for sid in excluded_set},
    }
    for sid, (is_long, cls, excluded) in sorted(all_entries.items()):
        direction = "LONG" if is_long else "SHORT"
        if excluded:
            print(f"  {sid:<35} {direction:<6} \033[90mNEUTRAL (excluded)\033[0m")
        else:
            label = " + ".join(c.upper() for c in cls) if cls else "NEUTRAL"
            color = bin_colors.get(cls[0], "\033[90m") if cls else "\033[90m"
            print(f"  {sid:<35} {direction:<6} {color}{label}\033[0m")
    print(f"  {'─'*70}\n")