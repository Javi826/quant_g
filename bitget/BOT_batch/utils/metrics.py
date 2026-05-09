#BOT_batch/utils/metrics.py
import logging
import numpy as np
import pandas as pd
from itertools import combinations as _combinations
from sklearn.linear_model import LinearRegression

from backtesters.ZX_compute_BT import INITIAL_BALANCE

logger = logging.getLogger("BOT_batch.utils.metrics")


# =============================================================================
# PRIVATE HELPERS — WEEKLY STATS
# =============================================================================

def _weekly_returns(strategy_trades: list, capital_per_strategy: float) -> pd.Series:
    if not strategy_trades:
        return pd.Series(dtype=float)
    all_tl = pd.concat(
        [df for _, df in strategy_trades], ignore_index=True
    ).sort_values("sell_time").reset_index(drop=True)
    total_capital   = capital_per_strategy * len(strategy_trades)
    all_tl["_date"] = pd.to_datetime(all_tl["sell_time"]).dt.normalize()
    daily           = all_tl.groupby("_date")["profit"].sum()
    date_range      = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
    daily           = daily.reindex(date_range, fill_value=0.0)
    eq              = total_capital + daily.cumsum()
    eq_series       = pd.Series(eq.values, index=date_range)
    return eq_series.resample("W").last().pct_change().dropna() * 100


def _neg_streak_stats(weekly: pd.Series) -> tuple:
    if len(weekly) == 0:
        return np.nan, np.nan
    streaks, current = [], 0
    for val in weekly:
        if val < 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    if not streaks:
        return 0.0, 0
    return round(float(np.mean(streaks)), 2), int(np.max(streaks))


def _cvar(weekly: pd.Series, pct: int = 10) -> float:
    if len(weekly) == 0:
        return np.nan
    threshold = np.percentile(weekly, pct)
    tail      = weekly[weekly <= threshold]
    return round(float(tail.mean()), 2) if len(tail) > 0 else np.nan


# =============================================================================
# COMPUTE METRICS
# =============================================================================

def compute_metrics(trade_log: pd.DataFrame, capital: float, name: str = "Equity") -> dict:
    tl      = trade_log.sort_values("sell_time").reset_index(drop=True)
    profits = tl["profit"].values

    win_rate = round((profits > 0).mean() * 100, 1)

    gains  = profits[profits > 0].sum()
    losses = -profits[profits < 0].sum()
    pf     = round(float(gains / losses), 3) if losses > 0 else np.inf

    tl["_date"]  = pd.to_datetime(tl["sell_time"]).dt.normalize()
    daily_profit = tl.groupby("_date")["profit"].sum()
    date_range   = pd.date_range(
        start=daily_profit.index.min(), end=daily_profit.index.max(), freq="1D"
    )
    daily_profit = daily_profit.reindex(date_range, fill_value=0.0)
    eq           = capital + daily_profit.cumsum().values
    eq_series    = pd.Series(eq, index=date_range)

    cm       = np.maximum.accumulate(eq)
    max_dd   = ((eq - cm) / cm * 100).min()
    net_gain = (eq[-1] - capital) / capital * 100
    profit_abs = round(float(eq[-1] - capital), 2)

    daily_returns = eq_series.pct_change().dropna()
    weekly        = eq_series.resample("W").last().pct_change().dropna()
    weekly_pct    = (weekly > 0).mean() * 100

    sharpe = (round(float(profits.mean() / profits.std() * np.sqrt(252)), 3)
              if profits.std() > 0 else np.nan)

    if "buy_time" in tl.columns and "sell_time" in tl.columns:
        duration_d = round(float(
            (pd.to_datetime(tl["sell_time"]) - pd.to_datetime(tl["buy_time"]))
            .dt.total_seconds().mean() / 86400
        ), 2)
    else:
        duration_d = np.nan

    X  = np.arange(len(eq)).reshape(-1, 1)
    y  = eq.reshape(-1, 1)
    r2 = round(LinearRegression().fit(X, y).score(X, y), 3)

    return {
        "Curve":         name,
        "Net_Gain_pct":  round(float(net_gain), 2),
        "Max_DD_pct":    round(float(max_dd), 2),
        "Win_Rate":      win_rate,
        "R_Squared":     r2,
        "Profit_Factor": pf,
        "Profit_abs":    profit_abs,
        "Sharpe":        sharpe,
        "Duration_d":    duration_d,
        "Weekly_pct":    round(float(weekly_pct), 2),
    }


def calc_r2_from_equity_hist(equity_hist: dict) -> float:
    """Compute R² of equity curve vs straight line from sim_balance_history dict."""
    if not equity_hist or len(equity_hist.get("balance", [])) < 2:
        return np.nan
    y = np.array(equity_hist["balance"]).reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    return round(LinearRegression().fit(X, y).score(X, y), 3)


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

    color = "\033[94m" if label == "Regime 0+1 — Validated only" else ""
    reset = "\033[0m" if color else ""
    n     = len(named)
    lines = [
        f"\n{color}{'─'*115}\n📊 ALL CURVES COMBINED OOS1 ({n}) — {label}\n{'─'*115}{reset}\n",
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

    if not rows:
        logger.info("  No data for robustness table.")
        return

    df                 = pd.DataFrame(rows)
    sep_row            = {col: "─" * 8 for col in df.columns}
    sep_row["Period"]  = "─" * 6
    mean_row           = df.drop(columns="Period").mean().round(2).to_dict()
    mean_row["Period"] = "MEAN"
    df                 = pd.concat([df, pd.DataFrame([sep_row, mean_row])], ignore_index=True)

    lines = [
        f"\n\033[94m{'─'*115}",
        f"  ROBUSTNESS TABLE — Validated Combined Portfolio",
        f"{'─'*115}\033[0m",
        df.to_string(index=False),
        f"{'─'*115}",
    ]
    logger.info("\n".join(lines))


def print_best_r2_robustness_table(
    combo_ids: list,
    strategy_trades_per_period: list,
    initial_balance: float,
) -> None:
    """Evaluate a fixed strategy combination (selected on IS) across OOS periods."""
    rows = []
    for label, strategy_trades in strategy_trades_per_period:
        filtered = [(sid, df) for sid, df in strategy_trades if sid in combo_ids]
        if not filtered:
            continue

        all_tl        = pd.concat([df for _, df in filtered], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
        total_capital = initial_balance * len(filtered)
        m             = compute_metrics(all_tl, capital=total_capital, name="")
        pf            = m["Profit_Factor"]

        weekly           = _weekly_returns(filtered, initial_balance)
        cvar10           = _cvar(weekly, pct=10)
        avg_neg, max_neg = _neg_streak_stats(weekly)

        rows.append({
            "Period":       label,
            "NetGain%":     m["Net_Gain_pct"],
            "MaxDD%":       m["Max_DD_pct"],
            "R2":           m["R_Squared"],
            "ProfitFactor": pf if pf != float("inf") else np.nan,
            "CVaR10%":      cvar10,
            "AvgNegStreak": avg_neg,
            "MaxNegStreak": max_neg,
            "Weekly_pct":   round(float((weekly > 0).mean() * 100), 2),
        })

    if not rows:
        logger.info("  No data for best R² robustness table.")
        return

    df       = pd.DataFrame(rows)
    sep_row  = {col: "─" * 8 for col in df.columns}
    sep_row["Period"] = "─" * 6
    mean_row = df.drop(columns="Period").mean().round(1).to_dict()
    mean_row["Period"] = "MEAN"
    mean_row["R2"]     = round(mean_row["R2"], 2)
    mean_row["CVaR10%"] = round(mean_row["CVaR10%"], 2)
    df = pd.concat([df, pd.DataFrame([sep_row, mean_row])], ignore_index=True)

    lines = [
        f"\n\033[94m{'─'*115}",
        f"  ROBUSTNESS TABLE — Validated Combined Portfolio",
        f"{'─'*115}\033[0m",
        df.to_string(index=False),
        f"{'─'*115}",
    ]
    logger.info("\n".join(lines))


# =============================================================================
# BEST R² COMBINATION
# =============================================================================

def find_best_r2_combination_ids(
    strategy_trades_is: list,
    initial_balance: float,
    precomputed_metrics: dict = None,
) -> list:
    """Find the strategy combination with highest R² on IS trade logs."""
    named   = {sid: df for sid, df in strategy_trades_is}
    metrics = precomputed_metrics or {
        sid: compute_metrics(df, capital=initial_balance, name=sid)
        for sid, df in named.items()
    }

    best_r2    = -1.0
    best_combo = list(named.keys())

    for r in range(1, len(named) + 1):
        for combo in _combinations(named.keys(), r):
            if len(combo) == 1:
                r2 = metrics.get(combo[0], {}).get("R_Squared", -1.0)
            else:
                combo_tl = pd.concat(
                    [named[sid] for sid in combo], ignore_index=True
                ).sort_values("sell_time").reset_index(drop=True)
                r2 = compute_metrics(combo_tl, capital=initial_balance * len(combo), name="")["R_Squared"]

            if r2 > best_r2:
                best_r2    = r2
                best_combo = list(combo)

    logger.debug(f"Best R² combination on IS: {best_combo} — R²={best_r2:.3f}")
    return best_combo


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
        f"  {'Strategy':<27} {'Verdict':<14} {'Round':<16} {'NetGain%':>10} {'DD%':>8} {'WinRate%':>10} {'R2':>7} {'ProbNeg%':>10} {'MCRegime%':>11}",
        f"  {'-'*115}",
    ]
    for v in validation_results:
        lines.append(
            f"  {v['strategy_id']:<27} {v['verdict']:<14} {v['round']:<16} "
            f"{v['net_gain_pct']:>9.2f}% {v['dd_pct']:>7.2f}% {v['win_ratio']:>9.1f}% "
            f"{v['r2']:>7.3f} {v['prob_neg_pct']:>9.2f}% {v.get('mc_regime_pct', 0.0):>10.1f}%"
        )
    lines.append(f" {'─'*115}")
    logger.info("\n".join(lines))