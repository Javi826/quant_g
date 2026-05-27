# develop/market_regime/analyze_autocor_regime.py
"""
Estimates the optimal block_size for MC regime permutation test.

Computes the real regime bin series for a given data folder and timeframe,
then analyzes the autocorrelation structure to recommend a block_size.

Usage: run directly, adjust CONFIGURATION section.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared.shared_batch_develop.market_regime.regime_analysis import (
    load_reference_symbol_for_timeframe,
    get_macro_direction,
    classify_trade_by_family,
)
from shared_batchs.regime.regime_filter import build_metrics_cache, REGIME_FAMILY_SOURCE
from shared_batchs.regime.regime_config import (
    REGIME_REFERENCE,
    REGIME_LOOKBACK_BARS,
    REGIME0_MA_PERIOD as R0_MA_PERIOD,
    REGIME0_LONG_TH   as R0_LONG_TH,
    REGIME0_SHORT_TH  as R0_SHORT_TH,
)
from shared.shared_trading_batch.config_trading_batch import (
    REGIME_FAMILIES  as FAMILIES,
    REGIME_ATR_WINDOW as ATR_WINDOW,
    REGIME_PE_WINDOW  as PE_WINDOW,
    REGIME_PE_ORDER   as PE_ORDER,
    REGIME_HURST_WINDOW as HURST_WINDOW,
    REGIME_ER_WINDOW    as ER_WINDOW,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
SPLIT_MODE  = "expanding"
SPLIT_BASE  = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER = os.path.join(SPLIT_BASE, "OOS", "crypto_2025-05_2026-05_OOS")
TIMEFRAME   = "1H"

# "combined"  → family × direction (6 bins) — used for block_size recommendation
# "direction" → direction only (uptrend/dwtrend) — used to estimate direction regime duration
ANALYSIS_MODE = "direction"


# =============================================================================
# BUILD BIN SERIES
# =============================================================================

def build_bin_series(data_folder: str, timeframe: str) -> list:
    """Build regime bin series from reference symbol data."""
    cache     = {}
    ref_1d_df = load_reference_symbol_for_timeframe(data_folder, REGIME_REFERENCE, "1Dutc", cache)
    ref_tf_df = load_reference_symbol_for_timeframe(data_folder, REGIME_REFERENCE, timeframe, cache) \
                if REGIME_FAMILY_SOURCE == "strategy" else ref_1d_df

    if ANALYSIS_MODE == "direction":
        # Direction only — no family metrics needed
        ref_1d_df_sorted = ref_1d_df.sort_values("ts") if "ts" in ref_1d_df.columns else ref_1d_df
        timestamps = sorted(ref_1d_df_sorted["ts"].unique()) if "ts" in ref_1d_df_sorted.columns else []

        # Use metrics_cache timestamps as anchor (same as combined mode)
        metrics_cache = build_metrics_cache(
            ref_df       = ref_tf_df,
            lookback     = REGIME_LOOKBACK_BARS,
            hurst_window = HURST_WINDOW,
            er_window    = ER_WINDOW,
            atr_window   = ATR_WINDOW,
            pe_window    = PE_WINDOW,
            pe_order     = PE_ORDER,
        )
        timestamps = sorted(metrics_cache.keys())
        bins = []
        for ts_pd in timestamps:
            direction = get_macro_direction(
                ref_1d_df  = ref_1d_df,
                trade_time = ts_pd,
                ma_period  = R0_MA_PERIOD,
                long_th    = R0_LONG_TH,
                short_th   = R0_SHORT_TH,
            )
            bins.append(direction if direction in ("uptrend", "dwtrend") else None)
        return bins

    # combined mode (default)
    metrics_cache = build_metrics_cache(
        ref_df       = ref_tf_df,
        lookback     = REGIME_LOOKBACK_BARS,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    )
    timestamps = sorted(metrics_cache.keys())
    bins       = []
    for ts_pd in timestamps:
        metrics   = metrics_cache.get(ts_pd)
        family    = classify_trade_by_family(metrics, FAMILIES) if metrics else None
        direction = get_macro_direction(
            ref_1d_df  = ref_1d_df,
            trade_time = ts_pd,
            ma_period  = R0_MA_PERIOD,
            long_th    = R0_LONG_TH,
            short_th   = R0_SHORT_TH,
        )
        if family and family != "unknown" and direction in ("uptrend", "dwtrend"):
            bins.append(f"{family}_{direction}")
        else:
            bins.append(None)
    return bins


# =============================================================================
# AUTOCORRELATION ANALYSIS
# =============================================================================

def analyze_autocorr(bins: list) -> None:
    """Analyze regime bin series autocorrelation and recommend block_size."""
    total = len(bins)
    known = [b for b in bins if b is not None]

    # --- Bin distribution ---
    all_bins = sorted(set(known))
    print(f"\n{'='*80}")
    print(f"  REGIME BIN SERIES — {DATA_FOLDER.split('/')[-1]}  |  {TIMEFRAME}")
    print(f"{'='*80}")
    print(f"  Total timestamps : {total}")
    print(f"  Known bins       : {len(known)} ({len(known)/total*100:.1f}%)")
    print(f"  Unknown (None)   : {total - len(known)}")
    print(f"\n  {'BIN':<30} {'COUNT':>8} {'PCT':>8}")
    print(f"  {'-'*50}")
    for b in all_bins:
        n = known.count(b)
        print(f"  {b:<30} {n:>8} {n/len(known)*100:>7.1f}%")

    # --- Run length analysis ---
    runs = []
    current_bin  = bins[0]
    current_len  = 1
    for b in bins[1:]:
        if b == current_bin:
            current_len += 1
        else:
            runs.append((current_bin, current_len))
            current_bin = b
            current_len = 1
    runs.append((current_bin, current_len))

    run_lengths = [r[1] for r in runs]
    known_runs  = [r[1] for r in runs if r[0] is not None]

    print(f"\n{'─'*80}")
    print(f"  RUN LENGTH ANALYSIS (consecutive bins with same regime)")
    print(f"{'─'*80}")
    print(f"  Total runs       : {len(runs)}")
    print(f"  Known runs       : {len(known_runs)}")
    print(f"\n  {'METRIC':<30} {'ALL':>10} {'KNOWN ONLY':>12}")
    print(f"  {'-'*55}")
    print(f"  {'Mean run length':<30} {np.mean(run_lengths):>10.1f} {np.mean(known_runs):>12.1f}")
    print(f"  {'Median run length':<30} {np.median(run_lengths):>10.1f} {np.median(known_runs):>12.1f}")
    print(f"  {'P75 run length':<30} {np.percentile(run_lengths, 75):>10.1f} {np.percentile(known_runs, 75):>12.1f}")
    print(f"  {'P90 run length':<30} {np.percentile(run_lengths, 90):>10.1f} {np.percentile(known_runs, 90):>12.1f}")
    print(f"  {'Max run length':<30} {max(run_lengths):>10.0f} {max(known_runs):>12.0f}")

    # --- Per-bin run lengths ---
    print(f"\n{'─'*80}")
    print(f"  MEAN RUN LENGTH PER BIN")
    print(f"{'─'*80}")
    print(f"  {'BIN':<30} {'MEAN_RUN':>10} {'MEDIAN_RUN':>12} {'MAX_RUN':>10}")
    print(f"  {'-'*65}")
    for b in all_bins:
        b_runs = [r[1] for r in runs if r[0] == b]
        if b_runs:
            print(f"  {b:<30} {np.mean(b_runs):>10.1f} {np.median(b_runs):>12.1f} {max(b_runs):>10.0f}")

    # --- Recommendation ---
    mean_known  = np.mean(known_runs)
    p75_known   = np.percentile(known_runs, 75)

    print(f"\n{'='*80}")
    print(f"  BLOCK_SIZE RECOMMENDATION")
    print(f"{'='*80}")
    print(f"  Mean run length (known bins) : {mean_known:.1f} bins")
    print(f"  P75  run length (known bins) : {p75_known:.1f} bins")
    print(f"\n  Suggested values:")
    print(f"    Conservative (mean)  : block_size = {int(round(mean_known))}")
    print(f"    Moderate     (P75)   : block_size = {int(round(p75_known))}")
    print(f"    Aggressive   (2×P75) : block_size = {int(round(p75_known * 2))}")
    print(f"{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Building bin series for {TIMEFRAME} — {DATA_FOLDER.split('/')[-1]} — mode={ANALYSIS_MODE} ...")
    bins = build_bin_series(DATA_FOLDER, TIMEFRAME)
    analyze_autocorr(bins)