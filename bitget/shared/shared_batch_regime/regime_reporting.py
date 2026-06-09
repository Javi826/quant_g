#shared/shared_batch_regime/regime_reporting.py

import logging

logger = logging.getLogger(__name__)
BINS: list[str] = ["uptrend", "dwtrend"]

def pct_improvement(val: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (val - base) / abs(base) * 100

# =============================================================================
# COMBO PERIOD TABLE
# =============================================================================

def print_combo_period_table(results: dict, strategies: list[dict], period_key: str, label: str) -> dict:
    logger.debug(f"\n  {'─'*120}")
    logger.debug(f"  {label}  |  PERIOD: {period_key}")
    logger.debug(f"  {'─'*120}")
    logger.debug(
        f"  {'STRATEGY':<35} {'B_PROF':>8}"
        + "  ".join(f"  {b.upper()[:10]:>12} {'Δ%':>6}" for b in BINS)
        + f"  {'UP%':>7}"
    )
    logger.debug(f"  {'─'*120}")

    sys_b   = 0.0
    sys_bin = {b: 0.0 for b in BINS}
    dd_b    = []
    dd_bin  = {b: [] for b in BINS}
    up_pcts = []

    for s in strategies:
        sid = s['id']
        if sid not in results or period_key not in results[sid]:
            continue
        if not isinstance(results[sid][period_key], dict):
            continue
        d = results[sid][period_key]

        bin_cols = ""
        for b in BINS:
            delta = pct_improvement(d[f"{b}_prof"], d['b_prof'])
            color = "\033[92m" if delta > 0 else "\033[91m"
            bin_cols += f"  {d[f'{b}_prof']:>12.1f} {color}{delta:>+5.1f}%\033[0m"

        logger.debug(f"  {sid:<35} {d['b_prof']:>8.1f}{bin_cols}  {d['uptrend_pct']:>6.1f}%")

        sys_b += d['b_prof']
        dd_b.append(d['b_dd'])
        up_pcts.append(d['uptrend_pct'])
        for b in BINS:
            sys_bin[b] += d[f"{b}_prof"]
            dd_bin[b].append(d[f"{b}_dd"])

    logger.debug(f"  {'─'*120}")
    sys_cols = ""
    avg_up   = sum(up_pcts) / len(up_pcts) if up_pcts else 0.0
    for b in BINS:
        delta = pct_improvement(sys_bin[b], sys_b)
        color = "\033[92m" if delta > 0 else "\033[91m"
        sys_cols += f"  {sys_bin[b]:>12.1f} {color}{delta:>+5.1f}%\033[0m"
    logger.debug(f"  {'SYSTEM TOTAL':<35} {sys_b:>8.1f}{sys_cols}  {avg_up:>6.1f}%")

    return {
        'sys_b':       sys_b,
        'avg_dd_b':    sum(dd_b) / len(dd_b) if dd_b else 0.0,
        'avg_up_pct':  avg_up,
        **{f"sys_{b}":    sys_bin[b]                             for b in BINS},
        **{f"pct_{b}":    pct_improvement(sys_bin[b], sys_b)     for b in BINS},
        **{f"avg_dd_{b}": sum(dd_bin[b]) / len(dd_bin[b]) if dd_bin[b] else 0.0 for b in BINS},
    }


# =============================================================================
# COMBO SUMMARY
# =============================================================================

def print_combo_summary(
    period_summaries: dict,
    bin_counts:       dict[str, int],
    n_neutral:        int,
    comb_p:           float,
    comb_dd:          float,
    base_p:           float,
    base_dd:          float,
    label:            str,
) -> None:
    logger.info(f"\n  COMBO SUMMARY — {label}")
    header = f"  {'PERIOD':<8} {'B_PROF':>10}" + "".join(f"  {b.upper():>12} {'Δ%':>7}" for b in BINS) + f"  {'UP%':>7}"
    logger.info(header)
    logger.info(f"  {'─'*90}")
    for pk, s in period_summaries.items():
        row = f"  {pk:<8} {s['sys_b']:>10.1f}"
        for b in BINS:
            color = "\033[92m" if s[f'pct_{b}'] > 0 else "\033[91m"
            row  += f"  {s[f'sys_{b}']:>12.1f} {color}{s[f'pct_{b}']:>+6.1f}%\033[0m"
        row += f"  {s['avg_up_pct']:>6.1f}%"
        logger.info(row)
    logger.info(f"  {'─'*90}")
    comb_pct = pct_improvement(comb_p, base_p)
    color    = "\033[92m" if comb_pct > 0 else "\033[91m"
    cls_str  = "  ".join(f"{b.upper()}:{bin_counts.get(b, 0)}" for b in BINS)
    logger.info(f"  Classifications — {cls_str}  NEUTRAL:{n_neutral}")
    logger.info(f"  Baseline  profit={base_p:>10.1f}  avg_dd={base_dd:>6.1f}%")
    logger.info(f"  Combined  profit={comb_p:>10.1f}  avg_dd={comb_dd:>6.1f}%  {color}Delta={comb_pct:>+6.1f}%\033[0m")


# =============================================================================
# RANKING
# =============================================================================

def print_ranking(ranking: list[dict]) -> None:
    bin_headers = "  ".join(f"{b.upper()[:8]:>8}" for b in BINS)
    header_line = (
        f"  {'#':>3}  {'COMBO':>5}  {'MA_W':>5}  "
        f"{bin_headers}  {'NEUT':>5}  "
        f"{'BASELINE':>10} {'COMB_PROF':>10} {'COMB_Δ%':>8} {'W_DELTA%':>9}  "
        f"{'BASE_DD%':>8} {'COMB_DD%':>8}"
    )
    total_w = len(header_line) - 2
    logger.info(f"\n\n{'='*total_w}")
    logger.info(f"  FINAL RANKING — ALL COMBOS BY WEIGHTED DELTA VS BASELINE  [MA UPTREND MODE]")
    logger.info(f"{'='*total_w}")
    logger.info(header_line)
    logger.info(f"  {'─'*total_w}")
    for i, row in enumerate(ranking[:5], 1):
        pct     = pct_improvement(row['combined_profit'], row['baseline_profit'])
        w_delta = row.get('weighted_delta', 0.0)
        cc      = "\033[92m" if pct > 0 else "\033[91m"
        wc      = "\033[92m" if w_delta > 0 else "\033[91m"
        ddc     = "\033[92m" if row['combined_dd'] > row['baseline_dd'] else "\033[91m"
        rs      = "\033[0m"
        bin_cols = "  ".join(f"{row['bin_counts'].get(b, 0):>8}" for b in BINS)
        logger.info(
            f"  {i:>3}  {row['combo_idx']:>5}  {row['ma_window']:>5}  "
            f"{bin_cols}  {row['n_neutral']:>5}  "
            f"{row['baseline_profit']:>10.1f} {cc}{row['combined_profit']:>10.1f}{rs} "
            f"{cc}{pct:>+7.1f}%{rs} {wc}{w_delta:>+8.1f}%{rs}  "
            f"{row['baseline_dd']:>7.1f}% {ddc}{row['combined_dd']:>7.1f}%{rs}"
        )
    logger.info(f"  {'─'*total_w}\n")


# =============================================================================
# CLASSIFICATION SUMMARY
# =============================================================================

def print_classification_summary(strategy_results: dict) -> None:
    print(f"\n{'='*120}")
    print(f"  STRATEGY CLASSIFICATION SUMMARY  [MA UPTREND MODE]")
    print(f"{'='*120}")
    print(f"  {'STRATEGY':<35} {'DIR':<6} {'BIN'}")
    print(f"  {'─'*70}")
    bin_colors = {
        "uptrend":   "\033[92m",
        "dwtrend": "\033[91m",
        "neutral":   "\033[90m",
    }
    for sid, data in sorted(strategy_results.items()):
        direction = "LONG" if data.get('is_long') else "SHORT"
        cls       = data.get('classification', 'neutral')
        color     = bin_colors.get(cls, "")
        print(f"  {sid:<35} {direction:<6} {color}{cls.upper()}\033[0m")
    print(f"  {'─'*70}\n")