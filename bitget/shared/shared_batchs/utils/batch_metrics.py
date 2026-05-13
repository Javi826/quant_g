# shared_batchs/utils/batch_metrics.py
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("BOT_batch.utils.batch_metrics")


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
    return eq_series.resample("W").last().diff().dropna() / total_capital * 100


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