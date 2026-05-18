# develop/market_regime/regime_distribution.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))
from shared.shared_batch_develop.market_regime.regime_analysis import get_macro_direction, calc_all_metrics_at_time, classify_trade_by_family
# =============================================================================
# CONFIGURATION
# =============================================================================

BTC_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline",
    "data", "04_split", "expanding", "IS", "rwa_2025-01_2026-05_IS", "BTCUSDT_1Dutc.parquet"
)

PERIOD_MODE   = "year"      # "year" | "semester"
ANALYSIS_MODE = "family"    # "direction" | "family" | "regime"
START_YEAR    = 2021        # data loaded from this year (warmup for lookback)
DISPLAY_YEAR  = 2022        # periods shown in table and chart

# Regime0 — macro BTC direction
BTC_MA_PERIOD = 5
LONG_TH       = 1.00
SHORT_TH      = 1.00

# Regime1 — family classification
FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.6)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging':  {}
}

HURST_WINDOW  = 100
ER_WINDOW     = 14
ATR_WINDOW    = 14
PE_WINDOW     = 50
PE_ORDER      = 3
LOOKBACK_BARS = 100

ALL_BINS_REGIME    = [
    "trending_uptrend", "trending_dwtrend",
    "volatile_uptrend", "volatile_dwtrend",
    "ranging_uptrend",  "ranging_dwtrend",
]
ALL_BINS_FAMILY    = ["trending", "volatile", "ranging"]
ALL_BINS_DIRECTION = ["uptrend", "dwtrend"]

def get_active_bins() -> list[str]:
    if ANALYSIS_MODE == "regime":
        return ALL_BINS_REGIME
    elif ANALYSIS_MODE == "family":
        return ALL_BINS_FAMILY
    elif ANALYSIS_MODE == "direction":
        return ALL_BINS_DIRECTION
    raise ValueError(f"Unknown ANALYSIS_MODE: {ANALYSIS_MODE}")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_btc() -> pd.DataFrame:
    filepath = Path(BTC_FILE)
    if not filepath.exists():
        raise FileNotFoundError(f"BTC file not found: {filepath}")

    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
    df = df.sort_values('ts').reset_index(drop=True)
    return df


# =============================================================================
# PERIOD SLICING
# =============================================================================

def build_periods(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """
    Returns list of (label, subset_df) for each year or semester from START_YEAR.
    """
    periods = []
    max_year = df['ts'].dt.year.max()

    for year in range(START_YEAR, max_year + 1):
        if PERIOD_MODE == "year":
            subset = df[df['ts'].dt.year == year].copy()
            if not subset.empty:
                periods.append((str(year), subset))

        elif PERIOD_MODE == "semester":
            s1 = df[(df['ts'].dt.year == year) & (df['ts'].dt.month <= 6)].copy()
            s2 = df[(df['ts'].dt.year == year) & (df['ts'].dt.month >= 7)].copy()
            if not s1.empty:
                periods.append((f"{year}-H1", s1))
            if not s2.empty:
                periods.append((f"{year}-H2", s2))

    return periods


# =============================================================================
# BIN CLASSIFICATION
# =============================================================================

def classify_bar(row: pd.Series, btc_df: pd.DataFrame) -> str:
    """Classify a single BTC daily bar according to ANALYSIS_MODE."""
    direction = get_macro_direction(
        btc_1d_df  = btc_df,
        trade_time = row['ts'],
        ma_period  = BTC_MA_PERIOD,
        long_th    = LONG_TH,
        short_th   = SHORT_TH,
    )

    if ANALYSIS_MODE == "direction":
        return direction if direction in ('uptrend', 'dwtrend') else 'unknown'

    if direction not in ('uptrend', 'dwtrend'):
        return 'unknown'

    metrics = calc_all_metrics_at_time(
        btc_df       = btc_df,
        buy_time     = row['ts'],
        lookback     = LOOKBACK_BARS,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    )

    family = classify_trade_by_family(metrics, FAMILIES) if metrics else 'unknown'

    if family == 'unknown':
        return 'unknown'

    if ANALYSIS_MODE == "family":
        return family

    return f"{family}_{direction}"  # regime


# =============================================================================
# PERIOD ANALYSIS
# =============================================================================

def analyze_period(label: str, subset: pd.DataFrame, btc_df: pd.DataFrame) -> dict:
    """
    Classify every bar in the period and compute bin statistics.

    Returns dict with:
        label, n_days, btc_return_pct, bin_days, bin_pct
    """
    bins_assigned = []

    for _, row in subset.iterrows():
        bin_key = classify_bar(row, btc_df)
        bins_assigned.append(bin_key)

    subset = subset.copy()
    subset['bin'] = bins_assigned

    valid     = subset[subset['bin'] != 'unknown']
    n_valid   = len(valid)
    n_total   = len(subset)
    n_unknown = n_total - n_valid

    active_bins = get_active_bins()
    bin_days = {b: int((valid['bin'] == b).sum()) for b in active_bins}
    bin_pct  = {b: int(round(bin_days[b] / n_valid * 100)) if n_valid > 0 else 0 for b in active_bins}

    # BTC return for the period (first close → last close)
    btc_return = None
    if len(subset) >= 2:
        price_start = subset['close'].iloc[0]
        price_end   = subset['close'].iloc[-1]
        if price_start and price_start != 0:
            btc_return = round((price_end / price_start - 1) * 100, 2)

    return {
        'label':      label,
        'n_days':     n_total,
        'n_valid':    n_valid,
        'n_unknown':  n_unknown,
        'btc_return': btc_return,
        'bin_days':   bin_days,
        'bin_pct':    bin_pct,
    }


# =============================================================================
# PLOTTING
# =============================================================================

BIN_COLORS = {
    "trending_uptrend":  "#2ecc71",
    "trending_dwtrend":  "#27ae60",
    "volatile_uptrend":  "#e74c3c",
    "volatile_dwtrend":  "#c0392b",
    "ranging_uptrend":   "#3498db",
    "ranging_dwtrend":   "#2980b9",
    "trending":          "#2ecc71",
    "volatile":          "#e74c3c",
    "ranging":           "#3498db",
    "uptrend":           "#2ecc71",
    "dwtrend":           "#e74c3c",
}

def plot_regime_distribution(results: list[dict]) -> None:
    """Stacked bar chart — regime distribution per period."""
    active_bins = get_active_bins()
    labels      = [r['label'] for r in results]
    n           = len(labels)
    x           = np.arange(n)
    bar_w       = 0.6

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))

    bottoms = np.zeros(n)
    for b in active_bins:
        values = np.array([r['bin_pct'][b] for r in results], dtype=float)
        ax.bar(x, values, bar_w, bottom=bottoms,
               color=BIN_COLORS[b], label=b, edgecolor='white', linewidth=0.4)

        for i, (val, bot) in enumerate(zip(values, bottoms)):
            if val >= 6:
                ax.text(x[i], bot + val / 2, f"{int(val)}%",
                        ha='center', va='center', fontsize=7.5,
                        color='white', fontweight='bold')
        bottoms += values

    for i, r in enumerate(results):
        if r['btc_return'] is not None:
            color = "#2ecc71" if r['btc_return'] >= 0 else "#e74c3c"
            ax.text(x[i], 102, f"{r['btc_return']:+.0f}%",
                    ha='center', va='bottom', fontsize=8,
                    color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of days", fontsize=9)
    ax.set_title(f"BTC Regime Distribution — {ANALYSIS_MODE.upper()}  per {PERIOD_MODE.upper()}  "
                 f"(MA{BTC_MA_PERIOD}, numbers above = BTC return)",
                 fontsize=10, pad=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# PRINTING
# =============================================================================

def print_results(results: list[dict]) -> None:
    """Print formatted regime distribution table."""
    active_bins = get_active_bins()
    col_w       = 18
    label_w     = 12
    sep         = "─"
    total_w     = label_w + 6 + 10 + len(active_bins) * col_w

    header_bins = "".join(f"{b:>{col_w}}" for b in active_bins)
    print(f"\n\033[94m{'─' * total_w}\033[0m")
    print(f"\033[94m  BTC REGIME DISTRIBUTION — {ANALYSIS_MODE.upper()}  per {PERIOD_MODE.upper()}\033[0m")
    print(f"\033[94m{'─' * total_w}\033[0m")
    print(f"\n{'PERIOD':<{label_w}} {'DAYS':>6} {'BTC%':>8}" + header_bins)
    print(sep * total_w)

    for r in results:
        btc_ret_str = f"{r['btc_return']:+.1f}%" if r['btc_return'] is not None else "  N/A"
        row = f"{r['label']:<{label_w}} {r['n_valid']:>6} {btc_ret_str:>8}"
        for b in active_bins:
            row += f"{str(r['bin_pct'][b]) + '%':>{col_w}}"
        print(row)

    print(sep * total_w)

    total_unknown = sum(r['n_unknown'] for r in results)
    if total_unknown > 0:
        print(f"\n  ⚠️  {total_unknown} bars excluded (unknown regime — insufficient lookback)")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print(f"BTC REGIME ANALYZER — {ANALYSIS_MODE.upper()}  mode={PERIOD_MODE.upper()}  MA={BTC_MA_PERIOD}  from {DISPLAY_YEAR}")
    print("=" * 80)

    # Load BTC
    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc()
    btc_df = btc_df[btc_df['ts'].dt.year >= START_YEAR].reset_index(drop=True)
    print(f"✅ {len(btc_df)} daily bars loaded  "
          f"({btc_df['ts'].iloc[0].date()} → {btc_df['ts'].iloc[-1].date()})")

    # Build periods
    periods = build_periods(btc_df)
    print(f"📅 {len(periods)} periods to analyze\n")

    # Analyze
    results = []
    for label, subset in periods:
        print(f"  Analyzing {label}... ({len(subset)} bars)", end="", flush=True)
        result = analyze_period(label, subset, btc_df)
        results.append(result)
        print(f"  ✅  unknown={result['n_unknown']}")

    # Filter display results
    display_results = [r for r in results if int(r['label'][:4]) >= DISPLAY_YEAR]

    # Print
    print_results(display_results)

    # Plot
    plot_regime_distribution(display_results)


if __name__ == "__main__":
    main()