# shared_batchs/utils/batch_metrics.py
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
logger = logging.getLogger("BOT_batch.utils.batch_metrics")


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

    calmar = round(float(net_gain / abs(max_dd)), 3) if max_dd < 0 else np.nan

    daily_returns = eq_series.pct_change().dropna()
    weekly        = eq_series.resample("W").last().pct_change().dropna()
    weekly_pct    = (weekly > 0).mean() * 100

    sharpe = (round(float(profits.mean() / profits.std() * np.sqrt(252)), 3)
              if profits.std() > 0 else np.nan)
    
    running_max_w   = weekly.add(1).cumprod().cummax()
    equity_w        = weekly.add(1).cumprod()
    is_underwater   = equity_w < running_max_w
    recovery_weeks  = 0
    max_weeks_to_recovery = 0
    for underwater in is_underwater:
        if underwater:
            recovery_weeks += 1
            max_weeks_to_recovery = max(max_weeks_to_recovery, recovery_weeks)
        else:
            recovery_weeks = 0

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
        "Calmar":        calmar,
        "Profit_abs":    profit_abs,
        "Sharpe":        sharpe,
        "Duration_d":    duration_d,
        "Weekly_pct":    round(float(weekly_pct), 2),
        "Max_Weeks_to_Recovery": int(max_weeks_to_recovery),
    }
