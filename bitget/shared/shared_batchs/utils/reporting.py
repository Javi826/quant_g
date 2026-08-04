# shared_batchs/utils/reporting.py
import logging
import numpy as np
import pandas as pd
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.plotting import plot_multiverse_synthetic_vs_historical
logger = logging.getLogger("BOT_batch.utils.reporting")

# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_metrics_table(metrics_list: list, title: str) -> None:
    df          = pd.DataFrame(metrics_list)
    df["Curve"] = df["Curve"].astype(str)
    max_len     = df["Curve"].str.len().max()
    df["Curve"] = df["Curve"].apply(lambda x: x.ljust(max_len))
    logger.debug(f"\n{title}\n{df.to_string(index=False)}")


def print_portfolio_metrics_table(
    strategy_trades: list,
    label: str,
    initial_balance: float,
) -> None:
    """Print individual + combined metrics table for a list of (strategy_id, trade_log)."""
    named        = {sid: df for sid, df in strategy_trades}
    metrics_list = [compute_metrics(df, capital=initial_balance, name=sid) for sid, df in named.items()]

    if len(named) > 1:
        combined_tl      = pd.concat(list(named.values()), ignore_index=True).sort_values("buy_time").reset_index(drop=True)
        combined_capital = initial_balance * len(named)
        metrics_list.append(compute_metrics(combined_tl, capital=combined_capital, name="Combined"))

    print_metrics_table(metrics_list, f"📊 METRICS TABLE — {label}")


def print_all_curves_table(
    strategy_trades: list,
    label: str,
    initial_balance: float,
) -> None:
    """Print metrics table for all curves plus long/short aggregates and a combined row."""
    named = {sid: df for sid, df in strategy_trades}
    rows  = [compute_metrics(df, capital=initial_balance, name=sid) for sid, df in named.items()]

    long_trades  = [(sid, df) for sid, df in named.items() if "_long_"  in sid]
    short_trades = [(sid, df) for sid, df in named.items() if "_short_" in sid]

    if long_trades:
        long_tl  = pd.concat([df for _, df in long_trades], ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
        rows.append(compute_metrics(long_tl, capital=initial_balance * len(long_trades), name="── Longs"))

    if short_trades:
        short_tl = pd.concat([df for _, df in short_trades], ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
        rows.append(compute_metrics(short_tl, capital=initial_balance * len(short_trades), name="── Shorts"))

    all_tl  = pd.concat(list(named.values()), ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
    rows.append(compute_metrics(all_tl, capital=initial_balance * len(named), name="── Combined"))

    cols   = ["Curve", "Net_Gain_pct", "Max_DD_pct", "Win_Rate", "R_Squared", "Profit_Factor", "Profit_abs", "Profit_pctT", "Weekly_pct"]
    df_out = pd.DataFrame(rows)

    strategy_rows         = df_out[~df_out["Curve"].str.strip().str.startswith("──")]
    total_profit          = strategy_rows["Profit_abs"].sum()
    df_out["Profit_pctT"] = df_out["Profit_abs"].apply(
        lambda x: round(x / total_profit * 100, 1) if total_profit != 0 else np.nan
    )

    df_out = df_out[cols].copy()
    df_out["Net_Gain_pct"]  = df_out["Net_Gain_pct"].round(1)
    df_out["Max_DD_pct"]    = df_out["Max_DD_pct"].round(1)
    df_out["Win_Rate"]      = df_out["Win_Rate"].round(1)
    df_out["R_Squared"]     = df_out["R_Squared"].round(2)
    df_out["Profit_Factor"] = df_out["Profit_Factor"].round(2)
    df_out["Profit_pctT"]   = df_out["Profit_pctT"].round(0)
    df_out["Weekly_pct"]    = df_out["Weekly_pct"].round(0)

    max_len         = df_out["Curve"].str.len().max()
    df_out["Curve"] = df_out["Curve"].apply(lambda x: x.ljust(max_len))
    df_out["Profit_abs"] = df_out["Profit_abs"].apply(
        lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )

    longs_idx = df_out[df_out["Curve"].str.strip() == "── Longs"].index
    if len(longs_idx) > 0:
        sep_agg = pd.DataFrame({col: ["-----"] for col in cols})
        sep_agg["Curve"] = "-" * max_len
        df_out = pd.concat([df_out.iloc[:longs_idx[0]], sep_agg, df_out.iloc[longs_idx[0]:]], ignore_index=True)

    combined_idx     = df_out[df_out["Curve"].str.strip() == "── Combined"].index[0]
    sep_row          = pd.DataFrame({col: ["─" * max(len(str(df_out[col].iloc[0])), 9)] for col in cols})
    sep_row["Curve"] = "─" * max_len
    df_out           = pd.concat([df_out.iloc[:combined_idx], sep_row, df_out.iloc[combined_idx:]], ignore_index=True)

    n     = len(named)
    lines = [
        f"\n{'─'*115}\n📊 ALL CURVES COMBINED ({n}) — {label}\n{'─'*115}\n",
        df_out.to_string(index=False),
    ]
    logger.info("\n".join(lines))


# =============================================================================
# WFO SUMMARY
# =============================================================================

def print_wfo_summary(wfo_results: list, validation_results: list = None) -> None:
    """Print fused WFO approval + strategy metrics table."""
    if not wfo_results:
        return
    n_pass        = sum(1 for w in wfo_results if "PASS" in w["verdict"])
    mean_net_gain = round(np.mean([w["net_gain"] for w in wfo_results]), 1)
    mean_max_dd   = round(np.mean([w["max_dd"] for w in wfo_results]), 1)
    val_map     = {v["strategy_id"]: v for v in validation_results} if validation_results else {}
    has_metrics = bool(val_map)
    header = (
        f"  {'Strategy':<27} {'Verdict':<10} {'NetGain%':>9} {'DD%':>7}"
        + (f"  {'NetGain%':>9} {'DD%':>7} {'WinRate%':>9} {'R2':>7} {'Trades':>7}" if has_metrics else "")
    )
    sep = (
        f"  {'-'*27} {'-'*10} {'-'*9} {'-'*7}"
        + (f"  {'-'*9} {'-'*7} {'-'*9} {'-'*7} {'-'*7}" if has_metrics else "")
    )
    lines = [
        f"\n{'─'*115}",
        f"  WFO SUMMARY — Pass: {n_pass}/{len(wfo_results)} | MeanNetGain: {mean_net_gain}% | MeanDD: {mean_max_dd}%",
        f"{'─'*115}",
        header,
        sep,
    ]
    for w in wfo_results:
        sid  = w["strategy_id"]
        line = f"  {sid:<27} {w['verdict']:<10} {w['net_gain']:>8.1f}% {w['max_dd']:>6.1f}%"
        if has_metrics and sid in val_map:
            v        = val_map[sid]
            n_trades = v.get("tn_trades", 0)
            line += (
                f"  {v['net_gain_pct']:>8.2f}%"
                f" {v['dd_pct']:>6.2f}%"
                f" {v['win_ratio']:>8.1f}%"
                f" {v['r2']:>7.3f}"
                f" {n_trades:>7}"
            )
        lines.append(line)
    lines.append(f" {'─'*115}")
    logger.info("\n".join(lines))

def print_best_wfo_portfolio(
    top: list,
    subperiods: list,
    trades_list: list,
    initial_balance: float,
    metric: str,
    weights: list,
    n_qualified: int,
) -> None:
    W          = 115
    split_keys = [label for label, _, _, _ in subperiods]
    logger.info(f"\n{'='*W}")
    logger.info(f"  BEST WFO PORTFOLIO — metric: {metric} | splits: {len(subperiods)}")
    logger.info(f"{'='*W}")
    for rank, entry in enumerate(top, start=1):
        combo      = entry["combo"]
        score      = entry["weighted_rank_score"]
        avg_trades = np.mean([len(df) for sid, df in trades_list if sid in combo])
        percentile = score / n_qualified * 100
        logger.info(f"\nBEST #{rank} — Strategies: {len(combo)}  |  AvgTrades/strat={avg_trades:.0f}  |  WeightedRankScore={score:.2f}  |  Top {percentile:.1f}%")
        logger.info(f"{'─'*W}")
        for s in sorted(combo, key=lambda s: int(s.split("_")[0])):
            icon = "🟢" if "_long_" in s else "🔴"
            logger.info(f"    {icon} {s}")
        logger.debug(f"\n  {'Subperiod':<10} {'Weight':>8} {'Value':>10} {'Rank':>6}  {'Period'}")
        logger.debug(f"  {'─'*65}")
        for i, (lbl, t_start, t_end, _) in enumerate(subperiods):
            val      = entry.get(lbl, np.nan)
            val_str  = f"{val:.3f}" if not np.isnan(val) else "N/A"
            rank_val = entry.get(f"{lbl}_rank", "-")
            logger.debug(f"  {lbl:<10} {weights[i]:>8.2f} {val_str:>10} {rank_val:>6}  ({t_start.strftime('%Y-%m-%d')} → {t_end.strftime('%Y-%m-%d')})")
        logger.debug(f"  {'─'*65}")
        logger.debug(f"  {'WEIGHTED RANK':<10} {'':>8} {'':>10} {score:>6.2f}")

        combo_trades = [(sid, df) for sid, df in trades_list if sid in combo]
        if combo_trades:
            tl            = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
            total_capital = initial_balance * len(combo_trades)
            m             = compute_metrics(tl, capital=total_capital, name="")

            last_label, last_t_start, last_t_end, _ = subperiods[-1]
            tl_last = tl[(tl["sell_time"] >= last_t_start) & (tl["sell_time"] < last_t_end)]
            m_last  = compute_metrics(tl_last, capital=total_capital, name="") if len(tl_last) > 0 else None

            _cols = ["NetGain", "DD", "WinRate", "R2", "PF", "Calmar", "Weekly%", "MaxWeeksToRecovery"]
            logger.info(f"\n  {'Period':<16} {' '.join(f'{c:>10}' for c in _cols)}")
            logger.info(f"  {'─'*16} {'─'*(10*len(_cols) + len(_cols) - 1)}")

            def _row(label: str, mm: dict) -> str:
                vals = [
                    f"{mm['Net_Gain_pct']:.1f}%",
                    f"{mm['Max_DD_pct']:.1f}%",
                    f"{mm['Win_Rate']:.1f}%",
                    f"{mm['R_Squared']:.3f}",
                    f"{mm['Profit_Factor']:.2f}",
                    f"{mm['Calmar']:.2f}",
                    f"{mm['Weekly_pct']:.1f}%",
                    f"{mm['Max_Weeks_to_Recovery']}",
                ]
                return f"  {label:<16} " + " ".join(f"{v:>10}" for v in vals)

            logger.info(_row("Full period", m))
            if m_last is not None:
                logger.info(_row(f"Last split ({last_label})", m_last))
            else:
                logger.info(f"  {f'Last split ({last_label})':<16} {'N/A':>10}")

            n_months        = max((pd.to_datetime(tl["sell_time"]).max() - pd.to_datetime(tl["sell_time"]).min()).days / 30.44, 1)
            avg_monthly_pct = round(m["Net_Gain_pct"] / n_months, 2)
            logger.info(f"\n  Monthly NetGain  ── {avg_monthly_pct:+.2f}% / month  ({n_months:.1f} months)")
    logger.info(f"\n{'─'*W}")

def _short_id(rule_id: str) -> str:
    parts = rule_id.split("_")
    return "_".join(parts[:3])

def print_rule_mining_ranking(all_raw_results: list, candidate_ids: list, stage_label: str, survivor_ids: list = None, debug: bool = False) -> None:
    rows = [r for r in all_raw_results if r["rule_id"] in set(candidate_ids)]
    rows.sort(key=lambda r: r["rule_id"])

    show_status  = survivor_ids is not None
    survivor_set = set(survivor_ids) if show_status else None
    log_fn       = logger.debug if debug else logger.info

    id_width    = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2
    label_width = max((len(r["label"]) for r in rows), default=8) + 2

    count_str = f"{len(survivor_ids)} / {len(candidate_ids)} passed" if show_status else f"{len(rows)} / {len(candidate_ids)} tested"

    log_fn(f"\n{'─' * 170}")
    log_fn(f"  RULE MINING RESULTS — {stage_label} ── {count_str}")
    log_fn(f"{'─' * 170}")

    status_header = f"  {'STATUS':<8}" if show_status else ""
    log_fn(
        f"{'ID':<{id_width}}{'NET_GAIN%':<12}{'MAX_DD%':<10}{'PF':<8}{'CALMAR':<8}{'R2':<8}"
        f"{'DSR':<8}{'WFR':<8}{'MC_RUIN':<9}{'MV_PVAL':<9}{'TRADES':<8}{'RULE':<{label_width}}{status_header}"
    )
    log_fn(f"{'─' * 170}")
    for r in rows:
        status_cell = f"  {('✅' if r['rule_id'] in survivor_set else '❌'):<8}" if show_status else ""
        log_fn(
            f"{_short_id(r['rule_id']):<{id_width}}{r['net_gain']:<12.1f}{r['max_dd']:<10.1f}"
            f"{r['profit_factor']:<8.2f}{r['calmar']:<8.2f}{r['r_squared']:<8.3f}"
            f"{r.get('dsr', 0.0):<8.3f}{r['wfr']:<8.2f}{r.get('montecarlo_prob_ruin', 0.0):<9.1f}"
            f"{r.get('multiverse_p_value', 0.0):<9.3f}"
            f"{r['n_trades']:<8}{r['label']:<{label_width}}{status_cell}"
        )

    log_fn(f"{'─' * 170}\n")

def print_rule_mining_min_by_group(all_raw_results: list, highlight_ids: list) -> None:
    rows = [r for r in all_raw_results if r["rule_id"] in set(highlight_ids)]
    if not rows:
        return
    threshold_metrics = ["net_gain", "max_dd", "r_squared", "dsr", "wfr"]
    groups = {}
    for r in rows:
        key = (r["timeframe"], r["side"])
        groups.setdefault(key, []).append(r)
    group_stats = {}
    for key, group_rows in groups.items():
        group_stats[key] = {
            m: (min(r[m] for r in group_rows), max(r[m] for r in group_rows))
            for m in threshold_metrics
        }
    logger.info(f"\n{'─' * 140}")
    logger.info(f"  MIN/MAX METRICS BY TIMEFRAME + SIDE ── {len(groups)} group(s)")
    logger.info(f"{'─' * 140}")
    logger.info(
        f"{'TIMEFRAME':<12}{'SIDE':<8}{'N':<6}"
        f"{'NET_GAIN% min/max':<22}{'MAX_DD% min/max':<20}{'R2 min/max':<16}"
        f"{'DSR min/max':<16}{'WFR min/max':<16}"
    )
    logger.info(f"{'─' * 140}")
    for (tf, side), group_rows in sorted(groups.items()):
        s = group_stats[(tf, side)]
        logger.info(
            f"{tf:<12}{side:<8}{len(group_rows):<6}"
            f"{f'{s['net_gain'][0]:.1f} / {s['net_gain'][1]:.1f}':<22}"
            f"{f'{s['max_dd'][0]:.1f} / {s['max_dd'][1]:.1f}':<20}"
            f"{f'{s['r_squared'][0]:.3f} / {s['r_squared'][1]:.3f}':<16}"
            f"{f'{s['dsr'][0]:.3f} / {s['dsr'][1]:.3f}':<16}"
            f"{f'{s['wfr'][0]:.2f} / {s['wfr'][1]:.2f}':<16}"
        )
    logger.info(f"{'─' * 140}")
    # ALL-SAFE: worst-case across the whole table -> every rule shown passes all conditions at once.
    all_safe_net_gain = min(s["net_gain"][0]   for s in group_stats.values())
    all_safe_max_dd   = max(abs(s["max_dd"][0]) for s in group_stats.values())
    all_safe_r2       = min(s["r_squared"][0]  for s in group_stats.values())
    all_safe_dsr      = min(s["dsr"][0]        for s in group_stats.values())
    all_safe_wfr      = min(s["wfr"][0]        for s in group_stats.values())

    # JOINT-SAFE: worst-case across each group's best (anchor) row -> guarantees >=1 survivor per group.
    anchors = {key: max(group_rows, key=lambda r: r["net_gain"]) for key, group_rows in groups.items()}
    safe_net_gain = min(a["net_gain"]  for a in anchors.values())
    safe_max_dd   = max(abs(a["max_dd"]) for a in anchors.values())
    safe_r2       = min(a["r_squared"] for a in anchors.values())
    safe_dsr      = min(a["dsr"] for a in anchors.values())
    safe_wfr      = min(a["wfr"] for a in anchors.values())

    logger.debug("\n  Anchor row per group (highest NET_GAIN, used to derive joint-safe thresholds):")
    for (tf, side), a in sorted(anchors.items()):
        logger.debug(f"    {tf:<10}{side:<8}{a['rule_id']}")

    label_all_safe   = "ALL - SAFE THRESHOLDS (guaranteed ALL rows in the table)"
    label_joint_safe = "ONE - SAFE THRESHOLDS (guaranteed ≥1 survivor per group)"
    label_width      = max(len(label_all_safe), len(label_joint_safe))

    logger.info(
        f"\n  {label_all_safe.ljust(label_width)} ── "
        f"NET_GAIN>={all_safe_net_gain:.1f}  MAX_DD<={all_safe_max_dd:.1f}  R2>={all_safe_r2:.3f}  "
        f"DSR>={all_safe_dsr:.3f}  WFR>={all_safe_wfr:.2f}"
    )
    logger.info(
        f"  {label_joint_safe.ljust(label_width)} ── "
        f"NET_GAIN>={safe_net_gain:.1f}  MAX_DD<={safe_max_dd:.1f}  R2>={safe_r2:.3f}  "
        f"DSR>={safe_dsr:.3f}  WFR>={safe_wfr:.2f}"
    )
    logger.info(f"{'─' * 140}\n")

# =============================================================================
# DSR — debug-only reporting (moved from pipeline/dsr.py)
# =============================================================================

def _dsr_train_period_str(r: dict) -> str:
    combo_daily_profit = r.get("combo_daily_profit") or {}
    best_combo_id       = r.get("best_combo_id")
    if best_combo_id is None or best_combo_id not in combo_daily_profit:
        return "n/a"

    daily_profit = combo_daily_profit[best_combo_id]
    if daily_profit is None or daily_profit.empty:
        return "n/a"

    start = daily_profit.index.min()
    end   = daily_profit.index.max()
    return f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"

def print_dsr_train_metrics(raw_by_id: dict, dsr_by_id: dict, sr_by_id: dict, candidate_ids: set, passed_ids: set, sr0: float) -> None:

    rows = [raw_by_id[rid] for rid in candidate_ids if rid in raw_by_id]
    rows.sort(key=lambda r: dsr_by_id.get(r["rule_id"], 0.0), reverse=True)

    if not rows:
        return

    id_width     = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2
    label_width  = max((len(r.get("label", "")) for r in rows), default=8) + 2
    combo_width  = max((len(r.get("best_combo_id", "") or "") for r in rows), default=8) + 2
    period_width = max((len(_dsr_train_period_str(r)) for r in rows), default=8) + 2

    logger.debug(f"\n{'─' * 200}")
    logger.debug(f"  DSR TRAIN METRICS (full-period grid search) ── SR0={sr0:.4f} ── {len(rows)} candidates")
    logger.debug(f"{'─' * 200}")
    logger.debug(
        f"{'ID':<{id_width}}{'SIDE':<6}{'NET_GAIN_TR':<13}{'MAX_DD_TR':<11}{'SR_ANN':<10}{'SR_UNANN':<11}"
        f"{'SKEW_TR':<10}{'KURT_TR':<10}{'N_DAYS_TR':<11}{'DSR':<9}{'BEST_COMBO':<{combo_width}}"
        f"{'TRAIN_PERIOD':<{period_width}}{'RULE':<{label_width}}{'STATUS':<8}"
    )
    logger.debug(f"{'─' * 200}")

    for r in rows:
        rule_id = r["rule_id"]
        status  = "✅" if rule_id in passed_ids else "❌"
        logger.debug(
            f"{_short_id(rule_id):<{id_width}}{r.get('side', ''):<6}"
            f"{r.get('net_gain_train', float('nan')):<13.1f}{r.get('max_dd_train', float('nan')):<11.1f}"
            f"{r.get('sharpe_train', float('nan')):<10.4f}{sr_by_id.get(rule_id, float('nan')):<11.4f}"
            f"{r.get('skew_train', float('nan')):<10.4f}{r.get('kurtosis_train', float('nan')):<10.4f}"
            f"{r.get('n_days_train', 0):<11}{dsr_by_id.get(rule_id, 0.0):<9.4f}"
            f"{(r.get('best_combo_id', '') or 'n/a'):<{combo_width}}"
            f"{_dsr_train_period_str(r):<{period_width}}"
            f"{r.get('label', ''):<{label_width}}{status:<8}"
        )
    logger.debug(f"{'─' * 200}\n")

# =============================================================================
# MULTIVERSE — debug-only reporting (moved from pipeline/multiverse.py)
# =============================================================================

def print_multiverse_drift_analysis(ohlcv_data: dict, paths: dict) -> None:
    rows = []
    for sym, df_hist in ohlcv_data.items():
        arr_paths = paths.get(sym)
        if arr_paths is None or arr_paths.shape[0] == 0:
            continue

        hist_close = df_hist["close"].to_numpy(dtype=np.float64)
        hist_n_bars        = len(hist_close)
        hist_total_ret_pct = float((hist_close[-1] / hist_close[0] - 1.0) * 100.0)

        synth_close = arr_paths[:, :, 3].astype(np.float64)
        synth_n_bars           = arr_paths.shape[1]
        synth_total_ret_pct    = (synth_close[:, -1] / synth_close[:, 0] - 1.0) * 100.0
        synth_pct_paths_positive = float(np.mean(synth_total_ret_pct > 0) * 100.0)

        rows.append({
            "symbol":                   sym,
            "hist_n_bars":              hist_n_bars,
            "synth_n_bars":             synth_n_bars,
            "hist_total_ret_pct":       hist_total_ret_pct,
            "synth_pct_paths_positive": synth_pct_paths_positive,
        })

    if not rows:
        logger.warning("MULTIVERSE DRIFT ANALYSIS ── no valid symbols to analyze")
        return

    df_drift = pd.DataFrame(rows)
    summary  = df_drift.drop(columns=["symbol"]).mean()
    df_drift = pd.concat(
        [df_drift, pd.DataFrame([{"symbol": "MEAN", **summary.to_dict()}])],
        ignore_index=True,
    )

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    logger.info(f"\n{'─' * 115}")
    logger.info("  MULTIVERSE DRIFT ANALYSIS ── historical vs MCPT permuted paths")
    logger.info(f"{'─' * 115}")
    logger.info(f"\n{df_drift.to_string(index=False)}")
    logger.info(f"{'─' * 115}\n")

def print_multiverse_debug_summary(
    per_path_results: list,
    rules: list,
    p_value_by_id: dict,
    approved_by_id: dict,
    n_paths: int,
    block_size: int,
) -> None:
    for r in rules:
        rid = r["rule_id"]
        rule_results = [res[rid] for res in per_path_results]

        n_valid = sum(1 for res in rule_results if res[0] is not None)
        if n_valid == 0:
            continue

        n_no_trades   = sum(1 for res in rule_results if res[0] is not None and not res[2])
        n_with_trades = sum(1 for res in rule_results if res[0] is not None and res[2])
        pct_no_trades = n_no_trades / n_valid * 100.0
        logger.debug(
            f"MULTIVERSE DEBUG ── {rid} ── no_trades_paths={n_no_trades}/{n_valid} ({pct_no_trades:.1f}%) "
            f"with_trades_paths={n_with_trades}/{n_valid}"
        )
        if n_with_trades > 0:
            with_trades_profits = [res[1] for res in rule_results if res[0] is not None and res[2]]
            logger.debug(
                f"MULTIVERSE DEBUG ── {rid} ── with_trades profit_sum stats: "
                f"min={min(with_trades_profits):.2f} max={max(with_trades_profits):.2f} "
                f"mean={float(np.mean(with_trades_profits)):.2f}"
            )

        real_profit = float(r["wfo_test_trades"]["profit"].sum())
        p_value     = p_value_by_id[rid]
        approved    = approved_by_id[rid]
        logger.debug(
            f"MULTIVERSE ── {rid} ── n_paths={n_paths} block_size={block_size} valid_universes={n_valid} "
            f"real_profit={real_profit:.2f} p_value={p_value:.4f} -> {'PASS' if approved else 'FAIL'}"
        )

def report_multiverse_debug(
    ohlcv_data: dict,
    paths: dict,
    per_path_results: list,
    rules: list,
    p_value_by_id: dict,
    approved_by_id: dict,
    n_paths: int,
    block_size: int,
) -> None:
    """Debug-only orchestrator: drift analysis table + synthetic-vs-historical plots + per-rule summary."""
    print_multiverse_drift_analysis(ohlcv_data, paths)
    plot_multiverse_synthetic_vs_historical(ohlcv_data, paths)
    print_multiverse_debug_summary(per_path_results, rules, p_value_by_id, approved_by_id, n_paths, block_size)