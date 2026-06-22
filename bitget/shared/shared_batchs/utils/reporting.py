# shared_batchs/utils/reporting.py
import logging
import numpy as np
import pandas as pd
from shared_batchs.utils.batch_metrics import compute_metrics, _weekly_returns, _cvar, _neg_streak_stats
logger = logging.getLogger("BOT_batch.utils.reporting")


# =============================================================================
# PRIVATE HELPERS — ROBUSTNESS TABLE
# =============================================================================

def _build_robustness_rows(
    strategy_trades_per_period: list,
    initial_balance: float,
) -> list:
    """Compute robustness metrics for each OOS period. Shared by all robustness print functions."""
    rows = []
    for label, strategy_trades in strategy_trades_per_period:
        if not strategy_trades:
            continue

        all_tl        = pd.concat([df for _, df in strategy_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
        total_capital = initial_balance * len(strategy_trades)
        m             = compute_metrics(all_tl, capital=total_capital, name="")
        pf            = m["Profit_Factor"]

        weekly           = _weekly_returns(strategy_trades, initial_balance)
        cvar10           = _cvar(weekly, pct=10)
        avg_neg, max_neg = _neg_streak_stats(weekly)
        weekly_avg       = round(float(weekly.mean()), 1) if len(weekly) > 0 else np.nan

        rows.append({
            "Period":       label,
            "NetGain%":     round(m["Net_Gain_pct"], 1),
            "MaxDD%":       round(m["Max_DD_pct"], 1),
            "R2":           round(m["R_Squared"], 2),
            "ProfitFactor": round(pf if pf != float("inf") else 0, 1),
            "CVaR10%":      round(cvar10, 2),
            "AvgNegStreak": round(avg_neg, 1),
            "MaxNegStreak": max_neg,
            "Weekly_pct":   round(float((weekly > 0).mean() * 100), 1),
            "Weekly_avg%":  weekly_avg,
            "MinWeekly%":   round(float(weekly.min()), 1) if len(weekly) > 0 else np.nan,
        })
    return rows

def _print_robustness_df(rows: list, title: str) -> None:
    """Render robustness rows as a formatted table. Shared by all robustness print functions."""
    if not rows:
        logger.info("  No data for robustness table.")
        return

    df                = pd.DataFrame(rows)
    sep_row           = {col: "─" * 8 for col in df.columns}
    sep_row["Period"] = "─" * 6
    mean_row          = df.drop(columns="Period").mean().round(2).to_dict()
    mean_row["Period"] = "MEAN"
    df                = pd.concat([df, pd.DataFrame([sep_row, mean_row])], ignore_index=True)

    lines = [
        f"\n{'─'*115}",
        f"  {title}",
        f"{'─'*115}",
        df.to_string(index=False),
        f"{'─'*115}",
    ]
    logger.info("\n".join(lines))
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
        f"\n{'─'*115}\n📊 ALL CURVES COMBINED OOS1 ({n}) — {label}\n{'─'*115}\n",
        df_out.to_string(index=False),
    ]
    logger.info("\n".join(lines))


# =============================================================================
# ROBUSTNESS TABLES
# =============================================================================

def print_robustness_table(
    strategy_trades_per_period: list,
    initial_balance: float,
) -> None:
    """Print robustness table with one row per OOS period (combined portfolio metrics)."""
    rows = _build_robustness_rows(strategy_trades_per_period, initial_balance)
    _print_robustness_df(rows, "ROBUSTNESS TABLE — Validated Combined Portfolio")


def print_best_r2_robustness_table(
    combo_ids: list,
    strategy_trades_per_period: list,
    initial_balance: float,
) -> None:
    """Evaluate a fixed strategy combination (selected on IS) across OOS periods."""
    filtered_per_period = [
        (label, [(sid, df) for sid, df in trades if sid in combo_ids])
        for label, trades in strategy_trades_per_period
    ]
    rows = _build_robustness_rows(filtered_per_period, initial_balance)
    _print_robustness_df(rows, "ROBUSTNESS TABLE — Validated Combined Portfolio")


# =============================================================================
# STRATEGIES SUMMARY
# =============================================================================

def print_strategies_summary(validation_results: list) -> None:
    """Print validation summary table for all strategies."""
    if not validation_results:
        return
    lines = [
        f"\n{'─'*115}",
        f"  STRATEGIES SUMMARY",
        f"{'─'*115}",
        f"  {'Strategy':<27} {'Verdict':<14} {'Round':<16} {'NetGain%':>10} {'DD%':>8} {'WinRate%':>10} {'R2':>7} {'ProbNeg%':>10}",
        f"  {'-'*115}",
    ]
    for v in validation_results:
        lines.append(
            f"  {v['strategy_id']:<27} {v['verdict']:<14} {v['round']:<16} "
            f"{v['net_gain_pct']:>9.2f}% {v['dd_pct']:>7.2f}% {v['win_ratio']:>9.1f}% "
            f"{v['r2']:>7.3f} {v['prob_neg_pct']:>9.2f}%"
        )
    lines.append(f" {'─'*115}")
    logger.info("\n".join(lines))
    
# =============================================================================
# WFO SUMMARY
# =============================================================================

def print_wfo_summary(wfo_results: list) -> None:
    """Print WFO approval summary table for all strategies."""
    if not wfo_results:
        return
    n_pass         = sum(1 for w in wfo_results if "PASS" in w["verdict"])
    mean_win_rate  = round(np.mean([w["win_rate"]       for w in wfo_results]) * 100, 1)
    mean_criterion = round(np.mean([w["mean_criterion"] for w in wfo_results]), 2)

    lines = [
        f"\n{'─'*115}",
        f"  WFO SUMMARY — Pass: {n_pass}/{len(wfo_results)} | MeanWinRate: {mean_win_rate}% | MeanCriterion: {mean_criterion}",
        f"{'─'*115}",
    ]
    for w in wfo_results:
        lines.append(
            f"  {w['strategy_id']:<27} {w['verdict']:<14} "
            f"{w['win_rate']*100:>9.1f}% {w['mean_criterion']:>15.2f}"
        )
    lines.append(f" {'─'*115}")
    logger.info("\n".join(lines))