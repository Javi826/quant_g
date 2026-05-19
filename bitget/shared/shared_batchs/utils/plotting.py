#shared_batch/utils/plotting.py
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shared_batchs.regime.regime_config import REGIME_REFERENCE
from shared_batchs.utils.batch_metrics import compute_metrics
logger = logging.getLogger("BOT_batch.utils.plotting")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _load_reference(data_folder: str, t_start, t_end) -> tuple:
    """Load and normalize BTC 1D data for a given time range."""
    btc_file = os.path.join(data_folder, f"{REGIME_REFERENCE}_1Dutc.parquet")
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
    ref_ts, ref_pct,
    title: str,
) -> None:
    """Core rendering function for equity curve comparison plots."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    if ts_r01 is not None and ref_ts is not None:
        btc_aligned = np.interp(
            pd.to_datetime(ts_r01).astype(np.int64) / 1e9,
            pd.to_datetime(ref_ts).astype(np.int64) / 1e9,
            ref_pct,
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

    if ref_ts is not None:
        ax.plot(ref_ts, ref_pct, color="#FF8C00", linewidth=0.9,
                linestyle="--", alpha=0.6, label="_REF")

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
    ref_ts, ref_pct = _load_reference(data_folder, t_start, t_end)

    _render_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, ref_ts, ref_pct, strategy_id)


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
    ref_ts, ref_pct = _load_reference(data_folder, t_start, t_end)

    _render_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, ref_ts, ref_pct, title)
    
def plot_best_portfolio(
    combo: tuple,
    trades_per_period: dict,
    subperiod_scores: dict,
    subperiods: list,
    initial_balance: float,
    data_folder_oos1: str,
    data_folder_oos2: str,
    data_folder_oos3: str,
    show_plots: bool = False,
) -> None:
    """
    Radar chart of normalized metrics (NetGain%, Weekly_pct, MaxDD% inv)
    across OOS1/2/3 for the best portfolio combo.
    Fixed reference ranges ensure meaningful visual comparison.
    """
    if not show_plots:
        return

    colors_per_period = {"OOS1": "#00897B", "OOS2": "#2E86C1", "OOS3": "#8E44AD"}

    # -------------------------------------------------------------------------
    # Compute metrics per OOS period
    # -------------------------------------------------------------------------
    period_metrics = {}
    for period, trades_list in trades_per_period.items():
        combo_trades = [(sid, df) for sid, df in trades_list if sid in combo]
        if not combo_trades:
            continue
        tl            = pd.concat([df for _, df in combo_trades], ignore_index=True).sort_values("sell_time").reset_index(drop=True)
        total_capital = initial_balance * len(combo)
        period_metrics[period] = compute_metrics(tl, capital=total_capital, name="")

    if not period_metrics:
        return

    # -------------------------------------------------------------------------
    # Radar data — fixed reference ranges, higher = better on all axes
    # -------------------------------------------------------------------------
    radar_labels  = ["NetGain%", "Weekly_pct", "MaxDD%\n(inv)"]
    n_axes        = len(radar_labels)
    _RADAR_RANGES = [
        (0,  300),  # NetGain%
        (50, 100),  # Weekly_pct
        (0,  10),   # MaxDD% inv
    ]

    def _get_radar_values(m: dict) -> list:
        return [
            m.get("Net_Gain_pct", 0),
            m.get("Weekly_pct",   0),
            -m.get("Max_DD_pct",  0),
        ]

    def _normalize(values: list) -> list:
        return [
            float(np.clip((v - lo) / (hi - lo), 0, 1))
            for v, (lo, hi) in zip(values, _RADAR_RANGES)
        ]

    radar_raw  = {p: _get_radar_values(m) for p, m in period_metrics.items()}
    radar_norm = {p: _normalize(v) for p, v in radar_raw.items()}

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for period, norm_vals in radar_norm.items():
        vals  = norm_vals + norm_vals[:1]
        color = colors_per_period.get(period, "#555555")
        raw   = radar_raw[period]
        label = f"{period}  NetGain={raw[0]:.1f}%  Wpct={raw[1]:.1f}%  DD={-raw[2]:.1f}%"
        ax.plot(angles, vals, color=color, linewidth=1.6, label=label)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#888888")
    ax.set_title(
        f"Best Portfolio — {' | '.join(sorted(combo, key=lambda s: int(s.split('_')[0])))}\nNormalized Metrics per OOS",
        fontsize=10, fontweight="bold", pad=20
    )
    ax.legend(
            loc="lower right", bbox_to_anchor=(1.6, -0.1),
            fontsize=8, framealpha=0.9, facecolor="white", edgecolor="#AAAAAA",
        )
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax.spines["polar"].set_visible(False)

    plt.tight_layout()
    plt.show()