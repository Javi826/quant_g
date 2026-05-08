#BOT_batch/utils/plotting.py
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from shared_config import REGIME_REFERENCE_SYMBOL
from utils.metrics import compute_metrics

logger = logging.getLogger("BOT_batch.utils.plotting")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _load_btc(data_folder: str, t_start, t_end) -> tuple:
    """Load and normalize BTC 1D data for a given time range."""
    btc_file = os.path.join(data_folder, f"{REGIME_REFERENCE_SYMBOL}_1Dutc.parquet")
    btc_df   = pd.read_parquet(btc_file)
    btc_df.columns = btc_df.columns.str.lower()
    btc_df["ts"]   = pd.to_datetime(btc_df["timestamp"] if "timestamp" in btc_df.columns else btc_df.index)
    btc_df         = btc_df.sort_values("ts").reset_index(drop=True)

    btc_sub = btc_df[(btc_df["ts"] >= t_start) & (btc_df["ts"] <= t_end)]
    if len(btc_sub) > 0:
        btc_ref = btc_sub["close"].iloc[0]
        return btc_sub["ts"].values, (btc_sub["close"].values / btc_ref - 1) * 100
    return None, None


def _render_comparison_plot(
    ts_base, eq_base, m_base,
    ts_r01, eq_r01, m_r01,
    btc_ts, btc_pct,
    title: str,
) -> None:
    """Core rendering function for equity curve comparison plots."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    if ts_r01 is not None and btc_ts is not None:
        btc_aligned = np.interp(
            pd.to_datetime(ts_r01).astype(np.int64) / 1e9,
            pd.to_datetime(btc_ts).astype(np.int64) / 1e9,
            btc_pct,
        )
        above = eq_r01 >= btc_aligned
        below = eq_r01 < btc_aligned
        ax.fill_between(ts_r01, eq_r01, 0, where=above, alpha=0.35, color="#00897B", interpolate=True)
        ax.fill_between(ts_r01, eq_r01, 0, where=below, alpha=0.35, color="#C62828", interpolate=True)

    lbl_base = (f"Baseline    NetGain={m_base['Net_Gain_pct']:>6.1f}%  "
                f"DD={m_base['Max_DD_pct']:>6.1f}%  R²={m_base['R_Squared']:.3f}")
    ax.plot(ts_base, eq_base, color="#2E86C1", linewidth=0.8, label=lbl_base)

    if ts_r01 is not None:
        lbl_r01 = (f"Regime 0+1  NetGain={m_r01['Net_Gain_pct']:>6.1f}%  "
                   f"DD={m_r01['Max_DD_pct']:>6.1f}%  R²={m_r01['R_Squared']:.3f}")
        ax.plot(ts_r01, eq_r01, color="#00897B", linewidth=1.4, label=lbl_r01)

    if btc_ts is not None:
        ax.plot(btc_ts, btc_pct, color="#FF8C00", linewidth=0.9,
                linestyle="--", alpha=0.6, label="_BTC")

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Net Gain (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.8, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()

    legend = ax.legend(
        loc="upper left", fontsize=10, framealpha=0.95,
        facecolor="white", edgecolor="#AAAAAA", fancybox=False,
        borderpad=0.8, labelspacing=0.6, handlelength=2.5,
    )
    for text in legend.get_texts():
        text.set_fontfamily("monospace")

    plt.tight_layout()
    plt.show()


# =============================================================================
# PUBLIC PLOT FUNCTIONS
# =============================================================================

def plot_filter_comparison(
    strategy_id: str,
    trades_df_baseline: pd.DataFrame,
    trades_df_r01,
    data_folder: str,
    initial_balance: float,
) -> None:
    """Plot equity curves for a single strategy: baseline vs regime 0+1 vs BTC."""
    def _equity_pct(tl, t_start):
        tl  = tl.sort_values("sell_time").reset_index(drop=True)
        eq  = initial_balance + tl["profit"].cumsum().values
        pct = (eq - initial_balance) / initial_balance * 100
        m   = compute_metrics(tl, capital=initial_balance, name="")
        ts  = pd.to_datetime(tl["sell_time"]).values
        ts  = np.concatenate([[np.datetime64(t_start)], ts])
        pct = np.concatenate([[0.0], pct])
        return ts, pct, m

    t_start = pd.Timestamp(pd.to_datetime(trades_df_baseline["sell_time"]).min())
    t_end   = pd.Timestamp(pd.to_datetime(trades_df_baseline["sell_time"]).max())

    ts_base, eq_base, m_base = _equity_pct(trades_df_baseline, t_start)
    ts_r01, eq_r01, m_r01 = (
        _equity_pct(trades_df_r01, t_start)
        if trades_df_r01 is not None and len(trades_df_r01) > 0
        else (None, None, None)
    )
    btc_ts, btc_pct = _load_btc(data_folder, t_start, t_end)

    _render_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, btc_ts, btc_pct, strategy_id)


def plot_portfolio_comparison(
    strategy_trades_baseline: list,
    strategy_trades_regime01: list,
    data_folder: str,
    initial_balance: float,
    title: str = "Portfolio",
) -> None:
    """Plot combined portfolio equity curves: baseline vs regime 0+1 vs BTC."""
    if not strategy_trades_baseline:
        return

    def _combined_equity_pct(strategy_trades, capital_per_strategy):
        all_tl = pd.concat(
            [df for _, df in strategy_trades], ignore_index=True
        ).sort_values("buy_time").reset_index(drop=True)
        total_capital = capital_per_strategy * len(strategy_trades)
        eq    = total_capital + all_tl["profit"].cumsum().values
        pct   = (eq - total_capital) / total_capital * 100
        ts    = pd.to_datetime(all_tl["buy_time"]).values
        m     = compute_metrics(all_tl, capital=total_capital, name="")
        t_start = pd.Timestamp(all_tl["buy_time"].min())
        ts  = np.concatenate([[np.datetime64(t_start)], ts])
        pct = np.concatenate([[0.0], pct])
        return ts, pct, m, t_start

    ts_base, eq_base, m_base, t_start_base = _combined_equity_pct(strategy_trades_baseline, initial_balance)
    ts_r01, eq_r01, m_r01, t_start_r01 = (
        _combined_equity_pct(strategy_trades_regime01, initial_balance)
        if strategy_trades_regime01 else (None, None, None, None)
    )

    t_start = min(t_start_base, t_start_r01) if t_start_r01 else t_start_base
    t_end   = pd.Timestamp(pd.to_datetime(ts_base).max())
    btc_ts, btc_pct = _load_btc(data_folder, t_start, t_end)

    _render_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, btc_ts, btc_pct, title)