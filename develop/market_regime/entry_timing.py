#develop/market_regime/entry_timing.py
"""
Temporal analysis of system performance by:
- Hour of day (buy_time)
- Day of week
- Day + Hour compound (top 10)
- Month of year
- Day + Hour combinations below threshold
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_market_regime")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

# =============================================================================
# CONFIGURATION
# =============================================================================

TRADES_FOLDER      = os.path.join(os.path.dirname(__file__), "..", "brief_trades")
SPLIT_MODE         = "expanding"
SPLIT_BASE         = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)

PERIOD_LABELS = [
 #   ("IS",   "is_regime"),
    ("OOS1", "oos1_regime"),
# =============================================================================
#     ("OOS2", "oos2_regime"),
#     ("OOS3", "oos3_regime"),
# =============================================================================
]

INITIAL_CAPITAL    = 800
MIN_TRADES         = 100
TOP_N_COMPOUND     = 10
WIN_RATE_THRESHOLD = 60  # combinations below this win rate are flagged

# =============================================================================
# DATA LOADING
# =============================================================================

def load_trades() -> pd.DataFrame:
    all_trades = []
    for period_label, trades_label in PERIOD_LABELS:
        files = sorted(glob(str(Path(TRADES_FOLDER) / f"trades_{trades_label}_*.csv")))
        for filepath in files:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower().str.strip()
            df['buy_time']  = pd.to_datetime(df['buy_time'])
            df['sell_time'] = pd.to_datetime(df['sell_time'])
            df['period']    = period_label
            all_trades.append(df)
    if not all_trades:
        return pd.DataFrame()
    return pd.concat(all_trades, ignore_index=True).sort_values('buy_time').reset_index(drop=True)

# =============================================================================
# CONSOLE TABLES
# =============================================================================

def print_table(df: pd.DataFrame, col: str, title: str, pct_all: float):
    OOS_PERIODS       = ['OOS1', 'OOS2', 'OOS3']
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

    oos_means = {p: (df[df['period'] == p]['profit'] > 0).mean() * 100 for p in OOS_PERIODS}

    print(f"  {title}")
    print(f"  {'VALUE':<12} {'N':>6} {'WIN%':>6}" +
          "".join(f"  {p:>6}" for p in OOS_PERIODS))
    print(f"  {'-'*46}")

    categories = df[col].cat.categories if hasattr(df[col], 'cat') else sorted(df[col].unique())
    for val in categories:
        grp = df[df[col] == val]
        if len(grp) < MIN_TRADES:
            continue

        pct_p = (grp['profit'] > 0).mean() * 100
        color = GREEN if pct_p >= pct_all else RED
        row   = f"  {str(val):<12} {len(grp):>6} {color}{pct_p:>5.0f}%{RESET}"

        for period in OOS_PERIODS:
            sub = grp[grp['period'] == period]
            if len(sub) < MIN_TRADES:
                row += f"  {'—':>6}"
            else:
                pct_oos = (sub['profit'] > 0).mean() * 100
                c       = GREEN if pct_oos >= oos_means[period] else RED
                row    += f"  {c}{pct_oos:>5.0f}%{RESET}"

        print(row)

    row_all = f"  {'ALL':<12} {len(df):>6} {pct_all:>5.0f}%"
    for period in OOS_PERIODS:
        row_all += f"  {oos_means[period]:>5.0f}%"
    print(row_all + "\n")


def print_compound_table(df: pd.DataFrame, pct_all: float):
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

    print(f"  3. TOP {TOP_N_COMPOUND} DAY + HOUR COMBINATIONS")
    print(f"  {'DAY':<12} {'HOUR':>6} {'N':>6} {'WIN%':>6}")
    print(f"  {'-'*34}")

    grouped = (
        df.groupby(['dow', 'hour'], observed=True)['profit']
        .agg(n='count', win_rate=lambda x: (x > 0).mean() * 100)
        .reset_index()
    )
    grouped = grouped[grouped['n'] >= MIN_TRADES]
    top     = grouped.sort_values('win_rate', ascending=False).head(TOP_N_COMPOUND)

    for _, row in top.iterrows():
        color = GREEN if row['win_rate'] >= pct_all else RED
        print(f"  {str(row['dow']):<12} {int(row['hour']):>6}h {int(row['n']):>6} "
              f"{color}{row['win_rate']:>5.0f}%{RESET}")
    print()


def print_below_threshold(df: pd.DataFrame):
    RED, RESET = "\033[91m", "\033[0m"

    print(f"  5. DAY + HOUR COMBINATIONS BELOW {WIN_RATE_THRESHOLD:.0f}% WIN RATE")
    print(f"  {'DAY':<12} {'HOUR':>6} {'N':>6} {'WIN%':>6}")
    print(f"  {'-'*34}")

    grouped = (
        df.groupby(['dow', 'hour'], observed=True)['profit']
        .agg(n='count', win_rate=lambda x: (x > 0).mean() * 100)
        .reset_index()
    )
    below = (
        grouped[(grouped['n'] >= MIN_TRADES) & (grouped['win_rate'] < WIN_RATE_THRESHOLD)]
        .sort_values('win_rate')
    )

    if below.empty:
        print(f"  No combinations found below {WIN_RATE_THRESHOLD:.0f}%\n")
        return

    for _, row in below.iterrows():
        print(f"  {str(row['dow']):<12} {int(row['hour']):>6}h {int(row['n']):>6} "
              f"{RED}{row['win_rate']:>5.0f}%{RESET}")
    print()

# =============================================================================
# PLOTS
# =============================================================================

def _base_plot_style(ax, title: str, xlabel: str):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Win Rate (%)", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333355")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(axis="y", color="#333355", linestyle="--", linewidth=0.6, alpha=0.7)


def plot_win_rate(df: pd.DataFrame, col: str, title: str, xlabel: str,
                  pct_all: float, x_labels=None):
    categories = list(df[col].cat.categories) if hasattr(df[col], 'cat') else sorted(df[col].unique())

    values, labels, counts = [], [], []
    for val in categories:
        grp = df[df[col] == val]
        if len(grp) < MIN_TRADES:
            continue
        values.append((grp['profit'] > 0).mean() * 100)
        labels.append(str(val) if x_labels is None else x_labels[val] if isinstance(x_labels, dict) else val)
        counts.append(len(grp))

    if not values:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f23")
    _base_plot_style(ax, title, xlabel)

    colors = ["#2ecc71" if v >= pct_all else "#e74c3c" for v in values]
    bars   = ax.bar(range(len(values)), values, color=colors, alpha=0.85, width=0.65, zorder=3)

    ax.axhline(pct_all, color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Average: {pct_all:.1f}%", zorder=4)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.0f}%", ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 8 else 0,
                       ha="right" if len(labels) > 8 else "center", color="#cccccc", fontsize=9)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_compound_top(df: pd.DataFrame, pct_all: float):
    grouped = (
        df.groupby(['dow', 'hour'], observed=True)['profit']
        .agg(n='count', win_rate=lambda x: (x > 0).mean() * 100)
        .reset_index()
    )
    grouped = grouped[grouped['n'] >= MIN_TRADES]
    top     = grouped.sort_values('win_rate', ascending=False).head(TOP_N_COMPOUND)

    if top.empty:
        return

    labels = [f"{row['dow'][:3]} {int(row['hour'])}h" for _, row in top.iterrows()]
    values = top['win_rate'].tolist()
    colors = ["#2ecc71" if v >= pct_all else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f23")
    _base_plot_style(ax, f"Top {TOP_N_COMPOUND} Day + Hour Combinations by Win Rate", "Day + Hour")

    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.85, width=0.65, zorder=3)
    ax.axhline(pct_all, color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Average: {pct_all:.1f}%", zorder=4)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.0f}%", ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color="#cccccc", fontsize=9)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_below_threshold(df: pd.DataFrame):
    grouped = (
        df.groupby(['dow', 'hour'], observed=True)['profit']
        .agg(n='count', win_rate=lambda x: (x > 0).mean() * 100)
        .reset_index()
    )
    below = (
        grouped[(grouped['n'] >= MIN_TRADES) & (grouped['win_rate'] < WIN_RATE_THRESHOLD)]
        .sort_values('win_rate')
    )

    if below.empty:
        return

    labels = [f"{row['dow'][:3]} {int(row['hour'])}h" for _, row in below.iterrows()]
    values = below['win_rate'].tolist()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f23")
    _base_plot_style(ax, f"Day + Hour Combinations Below {WIN_RATE_THRESHOLD:.0f}% Win Rate", "Day + Hour")

    bars = ax.bar(range(len(values)), values, color="#e74c3c", alpha=0.85, width=0.65, zorder=3)
    ax.axhline(WIN_RATE_THRESHOLD, color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Threshold: {WIN_RATE_THRESHOLD:.0f}%", zorder=4)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.0f}%", ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color="#cccccc", fontsize=9)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_temporal_analysis():
    print(f"\n{'='*70}")
    print(f"  TEMPORAL ANALYSIS — OOS aggregated (OOS1 + OOS2 + OOS3)")
    print(f"{'='*70}\n")

    trades_df = load_trades()
    if trades_df.empty:
        print("  No trades found — aborting")
        return

    oos = trades_df[trades_df['period'].isin(['OOS1', 'OOS2', 'OOS3'])].copy()
    print(f"  Total OOS trades: {len(oos)}\n")

    oos['hour']  = oos['buy_time'].dt.hour
    oos['dow']   = pd.Categorical(
        oos['buy_time'].dt.day_name(),
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True
    )
    oos['month'] = oos['buy_time'].dt.month

    pct_all = (oos['profit'] > 0).mean() * 100

    print_table(oos, 'hour',  '1. HOUR OF DAY (UTC)', pct_all)
    print_table(oos, 'dow',   '2. DAY OF WEEK',       pct_all)
    print_compound_table(oos, pct_all)
    print_table(oos, 'month', '4. MONTH OF YEAR',     pct_all)
    print_below_threshold(oos)

    print(f"{'='*70}\n")

    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',  6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    plot_win_rate(oos, 'hour',  "Win Rate by Hour of Day (UTC)", "Hour (UTC)", pct_all)
    plot_win_rate(oos, 'dow',   "Win Rate by Day of Week",       "Day",        pct_all)
    plot_compound_top(oos, pct_all)
    plot_win_rate(oos, 'month', "Win Rate by Month of Year",     "Month",      pct_all,
                  x_labels=month_names)
    plot_below_threshold(oos)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_temporal_analysis()