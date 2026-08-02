# shared_batchs/utils/batch_metrics.py
import logging
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
logger = logging.getLogger("BOT_batch.utils.batch_metrics")
# =============================================================================
# METRICS CONFIG
# =============================================================================
SHARPE_ABS_CAP = 50.0  # annualized Sharpe values beyond this are treated as invalid/degenerate

# =============================================================================
# R_SQUARED
# =============================================================================
def _r_squared_linear_trend(y: np.ndarray) -> float:

    n = len(y)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev  = x - x_mean
    y_dev  = y - y_mean

    ss_xx = np.dot(x_dev, x_dev)
    if ss_xx == 0.0:
        # Single point (n == 1): the fitted line passes through it exactly.
        return 1.0

    b = np.dot(x_dev, y_dev) / ss_xx
    a = y_mean - b * x_mean
    y_pred = a + b * x

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum(y_dev ** 2)

    if ss_tot == 0.0:
        return 1.0 if ss_res < 1e-12 else 0.0

    return float(1.0 - ss_res / ss_tot)

# =============================================================================
# SHARPE
# =============================================================================
def sharpe_from_daily_values(daily_values: np.ndarray) -> float:
    daily_std = daily_values.std()
    sharpe = (round(float(daily_values.mean() / daily_std * np.sqrt(365)), 3)
              if daily_std > 0 else np.nan)
    if sharpe is not None and np.isfinite(sharpe) and abs(sharpe) > SHARPE_ABS_CAP:
        sharpe = np.nan
    return sharpe

# =============================================================================
# COMPUTE METRICS
# =============================================================================

def compute_metrics(
    trade_log: pd.DataFrame,
    capital: float,
    name: str = "Equity",
    include_weekly: bool = True,
    include_skew_kurtosis: bool = True,
    include_r2: bool = True,
) -> dict:
    tl            = trade_log
    profits       = tl["profit"].values
    win_rate      = round((profits > 0).mean() * 100, 1)
    gains         = profits[profits > 0].sum()
    losses        = -profits[profits < 0].sum()
    pf            = round(float(gains / losses), 3) if losses > 0 else np.inf
    sell_days     = tl["sell_time"].values.astype("datetime64[D]")
    start_day     = sell_days.min()
    end_day       = sell_days.max()
    n_days        = int((end_day - start_day).astype("int64")) + 1
    day_offset    = (sell_days - start_day).astype("int64")
    daily_values  = np.bincount(day_offset, weights=profits, minlength=n_days)
    date_index    = pd.DatetimeIndex(start_day + np.arange(n_days, dtype=np.int64))
    eq            = capital + np.cumsum(daily_values)
    cm            = np.maximum.accumulate(eq)
    max_dd        = ((eq - cm) / cm * 100).min()
    net_gain      = (eq[-1] - capital) / capital * 100
    profit_abs    = round(float(eq[-1] - capital), 2)
    calmar        = round(float(net_gain / abs(max_dd)), 3) if max_dd < 0 else np.nan
    if include_weekly:
        eq_series     = pd.Series(eq, index=date_index)
        weekly        = eq_series.resample("W").last().pct_change().dropna()
        weekly_pct    = (weekly > 0).mean() * 100
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
    else:
        weekly_pct            = np.nan
        max_weeks_to_recovery = 0

    sharpe = sharpe_from_daily_values(daily_values)

    if include_skew_kurtosis:
        skew_daily = float(skew(daily_values)) if n_days > 2 else np.nan
        kurt_daily = float(kurtosis(daily_values, fisher=False)) if n_days > 2 else np.nan
    else:
        skew_daily = np.nan
        kurt_daily = np.nan
    if "buy_time" in tl.columns and "sell_time" in tl.columns:
        duration_d = round(float(
            (tl["sell_time"] - tl["buy_time"]).dt.total_seconds().mean() / 86400
        ), 2)
    else:
        duration_d = np.nan
    if include_r2:
        r2 = round(_r_squared_linear_trend(eq), 3)
    else:
        r2 = np.nan
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
        "Skew":          skew_daily,
        "Kurtosis":      kurt_daily,
        "N_days":        n_days,
        "Duration_d":    duration_d,
        "Weekly_pct":    round(float(weekly_pct), 2),
        "Max_Weeks_to_Recovery": int(max_weeks_to_recovery),
    }