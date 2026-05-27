#shared_batch/utils/plotting.py
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shared_batchs.regime.regime_GE_module import REGIME_REFERENCE
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
    
def plot_netgain_dd(equity_hist, initial_capital, data_folder, title="Net Gain % y DD", reference_symbol=None):
    from sklearn.linear_model import LinearRegression

    timestamps = pd.to_datetime(equity_hist['timestamp'])
    balances = np.array(equity_hist['balance'])

    net_gain_pct = (balances - initial_capital) / initial_capital * 100
    cumulative_max = np.maximum.accumulate(balances)
    dd_pct = (balances - cumulative_max) / cumulative_max * 100

    X = np.arange(len(balances)).reshape(-1, 1)
    y = balances.reshape(-1, 1)
    r2 = LinearRegression().fit(X, y).score(X, y)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if reference_symbol is not None:
        ref_file = os.path.join(data_folder, f"{reference_symbol}_4H.parquet")
        ref_df = pd.read_parquet(ref_file)
        if 'timestamp' not in ref_df.columns:
            if isinstance(ref_df.index, pd.DatetimeIndex):
                ref_df = ref_df.reset_index().rename(columns={'index': 'timestamp'})
            else:
                raise ValueError("Reference parquet has no 'timestamp' column or datetime index.")
        ref_df = ref_df[['timestamp', 'close']]
        ref_df['timestamp'] = pd.to_datetime(ref_df['timestamp'])
        ref_df['ref_net_gain_pct'] = (ref_df['close'] / ref_df['close'].iloc[0] - 1) * 100
        ref_aligned = np.interp(
            timestamps.astype(np.int64) / 10**9,
            ref_df['timestamp'].astype(np.int64) / 10**9,
            ref_df['ref_net_gain_pct']
        )
        above_ref = net_gain_pct >= ref_aligned
        below_ref = net_gain_pct < ref_aligned
        ax1.fill_between(timestamps, net_gain_pct, 0, where=above_ref, alpha=0.2, color='green', interpolate=True)
        ax1.fill_between(timestamps, net_gain_pct, 0, where=below_ref, alpha=0.2, color='red', interpolate=True)
        ax1.plot(ref_df['timestamp'], ref_df['ref_net_gain_pct'],
                 color='darkorange', linewidth=0.6, linestyle='--', label=f'{reference_symbol} %')
        final_ref = ref_df['ref_net_gain_pct'].iloc[-1]
    else:
        above_zero = net_gain_pct >= 0
        ax1.fill_between(timestamps, net_gain_pct, 0, where=above_zero, alpha=0.2, color='green', interpolate=True)
        ax1.fill_between(timestamps, net_gain_pct, 0, where=~above_zero, alpha=0.2, color='red', interpolate=True)
        final_ref = None

    ax1.plot(timestamps, net_gain_pct, color='blue', linewidth=1.2, label='Net Gain %')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net_Gain_pct", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(timestamps, dd_pct, color='lightcoral', linewidth=0.1, label='DD %')
    ax2.set_ylabel("Drawdown", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    final_net_gain = net_gain_pct[-1]
    max_dd = dd_pct.min()

    textstr = (
        f'Net Gain STR : {final_net_gain:.2f}%\n'
        + (f'Net Gain REF : {final_ref:.2f}%\n' if final_ref is not None else '')
        + f'Max DD       : {max_dd:.2f}%\n'
        + f'R²           : {r2:.3f}'
    )

    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title)
    fig.autofmt_xdate()
    ax1.grid(True, linestyle='--', alpha=0.6)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    plt.show() 

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
    df_scored: pd.DataFrame = None,
    ranking_criteria: list  = None,
    show_plots: bool = False,
) -> None:
    """
    4-panel plot for the best portfolio combo:
      1. Top-left:     Radar chart — normalized metrics per OOS
      2. Top-right:    Weekly_pct line per subperiod — consecutive
      3. Bottom-left:  Histogram — distribution of all combos by primary metric
      4. Bottom-right: Distribution summary table
    """
    if not show_plots:
        return

    colors_per_period = {"OOS1": "#0D3B6E", "OOS2": "#1B4D2E", "OOS3": "#7D1F00"}

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
    # Radar data
    # -------------------------------------------------------------------------
    radar_labels = ["NetGain%", "Weekly_pct", "MaxDD%\n(inv)"]
    n_axes       = len(radar_labels)

    def _get_radar_values(m: dict) -> list:
        return [
            m.get("Net_Gain_pct", 0),
            m.get("Weekly_pct",   0),
            m.get("Max_DD_pct",   0),
        ]

    radar_raw = {p: _get_radar_values(m) for p, m in period_metrics.items()}
    all_vals  = np.array(list(radar_raw.values()))
    max_gain  = all_vals[:, 0].max()
    max_wpct  = all_vals[:, 1].max()
    min_dd    = all_vals[:, 2].min()

    radar_norm = {
        p: [
            float(vals[0] / max_gain) if max_gain > 0 else 0.0,
            float(vals[1] / max_wpct) if max_wpct > 0 else 0.0,
            float(vals[2] / min_dd)   if min_dd   < 0 else 0.0,
        ]
        for p, vals in radar_raw.items()
    }
    for p in radar_norm:
        radar_norm[p][2] = 1.0 - radar_norm[p][2] + min(v[2] for v in radar_norm.values())

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    # -------------------------------------------------------------------------
    # Build consecutive line data
    # -------------------------------------------------------------------------
    all_points = []
    for period, split_label, split_trades, _ in subperiods:
        key = f"{period}_{split_label}"
        if key not in subperiod_scores:
            continue
        combo_trades = [(sid, df) for sid, df in split_trades if sid in combo]
        if not combo_trades:
            continue
        all_times = pd.concat([df["sell_time"] for _, df in combo_trades], ignore_index=True)
        t_mid     = pd.Timestamp(all_times.min() + (all_times.max() - all_times.min()) / 2)
        all_points.append((t_mid, subperiod_scores[key], period))

    all_points.sort(key=lambda x: x[0])
    x_idx         = list(range(len(all_points)))
    x_labels      = [p[0].strftime("%Y-%m") for p in all_points]
    y_vals        = [p[1] for p in all_points]
    pt_periods    = [p[2] for p in all_points]
    weighted_mean = np.mean(y_vals) if y_vals else 0

    # -------------------------------------------------------------------------
    # Distribution stats
    # -------------------------------------------------------------------------
    dist_stats = {}
    if df_scored is not None and ranking_criteria:
        primary_metric = ranking_criteria[0][0]
        n_total        = len(df_scored)
        best_val       = df_scored[primary_metric].iloc[0]
        mean_val       = df_scored[primary_metric].mean()
        std_val        = df_scored[primary_metric].std()
        min_val        = df_scored[primary_metric].min()
        max_val        = df_scored[primary_metric].max()
        gap            = round(best_val - mean_val, 2)
        zscore         = round((best_val - mean_val) / std_val, 2) if std_val > 0 else 0.0
        dist_stats     = {
            "metric":  primary_metric,
            "best":    best_val,
            "mean":    mean_val,
            "std":     std_val,
            "min":     min_val,
            "max":     max_val,
            "gap":     gap,
            "zscore":  zscore,
            "n_total": n_total,
            "values":  df_scored[primary_metric].values,
        }
    # -------------------------------------------------------------------------
    # Figure — 2x2
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={"width_ratios": [1, 1.4]},
                             subplot_kw={"polar": False})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"Best Portfolio — {' | '.join(sorted(combo, key=lambda s: int(s.split('_')[0])))}",
        fontsize=10, fontweight="bold"
    )

    # Replace top-left with polar
    axes[0, 0].remove()
    ax1 = fig.add_subplot(2, 2, 1, polar=True)
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    for ax in [ax2, ax3, ax4]:
        ax.set_facecolor("#F8F9FA")

    # ── Panel 1: Radar chart ──────────────────────────────────────────────────
    ax1.set_facecolor("#F8F9FA")
    ax1.set_yticks([0.6, 1.0])
    ax1.set_yticklabels(["60%", "100%"], fontsize=6, color="#444444")
    ax1.yaxis.grid(True, color="black", linewidth=0.4, linestyle="-", alpha=0.2)
    ax1.xaxis.grid(True, color="black", linewidth=0.4, linestyle="-", alpha=0.2)

    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(theta_circle, [1.0] * 200, color="#333333", linewidth=0.6, linestyle="-", alpha=0.4, zorder=4)
    ax1.plot(theta_circle, [0.6] * 200, color="#C0392B", linewidth=0.9, linestyle="--", alpha=0.8, zorder=5)

    for period, norm_vals in radar_norm.items():
        vals  = norm_vals + norm_vals[:1]
        color = colors_per_period.get(period, "#555555")
        raw   = radar_raw[period]
        label = f"{period}  NetGain={raw[0]:.1f}%  Wpct={raw[1]:.1f}%  DD={raw[2]:.1f}%"
        ax1.plot(angles, vals, color=color, linewidth=0.8, label=label)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(radar_labels, fontsize=9)
    ax1.set_title("Normalized Metrics per OOS", fontsize=9, fontweight="bold", pad=15)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.2, 0.1), fontsize=7,
               framealpha=0.9, facecolor="white", edgecolor="#AAAAAA")
    ax1.spines["polar"].set_color("black")
    ax1.spines["polar"].set_linewidth(0.4)

    # ── Panel 2: Weekly_pct line ──────────────────────────────────────────────
    def _flush_segment(ax, sx, sy, sc, wm):
        if len(sx) > 1:
            ax.plot(sx, sy, color=sc, linewidth=1.0, alpha=0.85)
            ax.fill_between(sx, sy, wm, alpha=0.05, color=sc)
        if sx:
            ax.scatter(sx, sy, color=sc, s=16, zorder=5, alpha=0.9)

    prev_period = None
    seg_x, seg_y, seg_c = [], [], None

    for xi, yi, pi in zip(x_idx, y_vals, pt_periods):
        color = colors_per_period.get(pi, "#555555")
        if pi != prev_period:
            if seg_x:
                seg_x.append(xi)
                seg_y.append(yi)
                _flush_segment(ax2, seg_x, seg_y, seg_c, weighted_mean)
            seg_x       = [xi]
            seg_y       = [yi]
            seg_c       = color
            prev_period = pi
            ax2.axvline(xi - 0.3, color=color, linewidth=0.5, linestyle="--", alpha=0.4)
            ax2.text(xi - 0.1, yi + 1, pi, fontsize=7, color=color, fontweight="bold")
        else:
            seg_x.append(xi)
            seg_y.append(yi)

    _flush_segment(ax2, seg_x, seg_y, seg_c, weighted_mean)
    ax2.axhline(weighted_mean, color="#E67E22", linewidth=0.9, linestyle="--",
                label=f"Weighted mean={weighted_mean:.1f}%")
    ax2.set_title("Weekly_pct per Subperiod", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Weekly_pct (%)", fontsize=9)
    ax2.set_ylim(max(0, min(y_vals) - 10), 105)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
    ax2.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, color="#CCCCCC")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.9, facecolor="white", edgecolor="#AAAAAA")

    # ── Panel 3: Histogram — combo distribution ───────────────────────────────
    if dist_stats:
        vals_all = dist_stats["values"]
        ax3.hist(vals_all, bins=20, color="#2E86C1", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax3.axvline(dist_stats["best"], color="#0D3B6E", linewidth=1.5, linestyle="-",
                    label=f"Best={dist_stats['best']:.1f}%")
        ax3.axvline(dist_stats["mean"], color="#E67E22", linewidth=1.2, linestyle="--",
                    label=f"Mean={dist_stats['mean']:.1f}%")
        ax3.axvspan(dist_stats["mean"] - dist_stats["std"],
                    dist_stats["mean"] + dist_stats["std"],
                    alpha=0.1, color="#E67E22", label=f"±1 std={dist_stats['std']:.1f}%")
        ax3.set_title(f"Combo Distribution — {dist_stats['metric']} ({dist_stats['n_total']} combos)",
                      fontsize=9, fontweight="bold")
        ax3.set_xlabel(f"{dist_stats['metric']} (%)", fontsize=8)
        ax3.set_ylabel("Count", fontsize=8)
        ax3.tick_params(axis="both", labelsize=7)
        ax3.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, color="#CCCCCC")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.legend(fontsize=7, framealpha=0.9, facecolor="white", edgecolor="#AAAAAA")

    # ── Panel 4: Distribution summary table ───────────────────────────────────
    if dist_stats:
        ax4.axis("off")
        rows = [
            ["Metric",      dist_stats["metric"]],
            ["N combos",    str(dist_stats["n_total"])],
            ["Best",        f"{dist_stats['best']:.2f}%"],
            ["Mean",        f"{dist_stats['mean']:.2f}%"],
            ["Std",         f"{dist_stats['std']:.2f}%"],
            ["Min",         f"{dist_stats['min']:.2f}%"],
            ["Max",         f"{dist_stats['max']:.2f}%"],
            ["Gap (best-mean)", f"{dist_stats['gap']:.2f}%"],
            ["Z-score",     f"{dist_stats['zscore']:.2f}"],
        ]
        table = ax4.table(
            cellText=rows,
            colLabels=["Parameter", "Value"],
            cellLoc="center",
            loc="center",
            bbox=[0.1, 0.1, 0.8, 0.85],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            if row == 0:
                cell.set_facecolor("#2E86C1")
                cell.set_text_props(color="white", fontweight="bold")
            elif row in [8, 9]:  # percentile and zscore rows
                zscore_val = dist_stats["zscore"]
                color = "#FDECEA" if zscore_val > 3 else "#E8F5E9" if zscore_val < 1.5 else "#FFF8E1"
                cell.set_facecolor(color)
            else:
                cell.set_facecolor("#F8F9FA" if row % 2 == 0 else "white")
        ax4.set_title("Distribution Summary", fontsize=9, fontweight="bold")

    from matplotlib.lines import Line2D
    line = Line2D([0.43, 0.43], [0.02, 0.92], transform=fig.transFigure,
                  color="#222222", linewidth=1.5, linestyle="--")
    fig.add_artist(line)

    plt.tight_layout()
    plt.show()
    
