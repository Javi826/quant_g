# shared_batchs/utils/reporting.py
import logging
import numpy as np
import pandas as pd
from shared_batchs.utils.batch_metrics import compute_metrics
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
            icon = "🟢" if "_long_" in s else "🔴"
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