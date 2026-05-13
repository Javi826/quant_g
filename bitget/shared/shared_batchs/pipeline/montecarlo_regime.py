#shared_batch/pipeline/montecarlo_regime.py
import contextlib
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from shared.shared_trading_batch.config_trading_batch import REGIME_ATR_WINDOW as ATR_WINDOW, REGIME_PE_WINDOW as PE_WINDOW, REGIME_PE_ORDER as PE_ORDER
from shared.shared_trading_batch.config_trading_batch import REGIME_FAMILIES as FAMILIES, REGIME_HURST_WINDOW as HURST_WINDOW, REGIME_ER_WINDOW as ER_WINDOW
from shared_batch.regime.regime_config import REGIME0_MA_PERIOD as R0_MA_PERIOD, REGIME0_LONG_TH as R0_LONG_TH, REGIME0_SHORT_TH as R0_SHORT_TH
from shared.shared_batch_develop.market_regime.regime_analysis import load_btc_for_timeframe, get_btc_macro_direction
from shared.shared_batch_develop.market_regime.regime_analysis import classify_trade_by_family, calc_all_metrics
from shared_batch.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batch.utils.torque import prepare_ohlcv_arrays
from shared_batch.utils.metrics import compute_metrics
from shared_batch.regime.regime_filter import analyze_regime_is, run_oos_backtest_with_regime
logger = logging.getLogger("BOT_batch.pipeline.montecarlo_regime")

# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _build_bin_series(
    timestamps: np.ndarray,
    btc_1d_df: pd.DataFrame,
    metrics_cache: dict,
) -> list:
    """
    Build a list of regime bins for each timestamp.
    Returns a list of bin strings e.g. 'trending_uptrend', or None if unknown.
    """
    bins = []
    for ts in timestamps:
        ts_pd   = pd.Timestamp(ts)
        metrics = metrics_cache.get(ts_pd)
        family  = classify_trade_by_family(metrics, FAMILIES) if metrics else None
        direction = get_btc_macro_direction(
            btc_1d_df  = btc_1d_df,
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


def _apply_regime_filter(
    signal_arr: np.ndarray,
    timestamps: np.ndarray,
    bin_series: list,
    bins_to_filter: set,
) -> np.ndarray:
    """
    Zero out signals whose timestamp maps to a filtered bin.
    Returns a new signal array.
    """
    filtered = signal_arr.copy()
    ts_to_bin = {int(pd.Timestamp(ts).value): b for ts, b in zip(timestamps, bin_series)}
    for i, ts in enumerate(timestamps):
        key = int(pd.Timestamp(ts).value)
        bin_ = ts_to_bin.get(key)
        if bin_ in bins_to_filter:
            filtered[i] = 0
    return filtered


def _run_single_permutation(
    perm_idx: int,
    ohlcv_arrays: dict,
    signal_arrays: dict,
    timestamps_per_sym: dict,
    bin_series: list,
    all_timestamps: np.ndarray,
    bins_to_filter: set,
    best_params: dict,
    order_amount: int,
    seed: int,
) -> dict:
    """Run one permutation: shuffle bin series, filter signals, backtest."""
    rng            = np.random.default_rng(seed + perm_idx)
    shuffled_bins  = bin_series.copy()
    rng.shuffle(shuffled_bins)

    # Map shuffled bins back to timestamps
    ts_to_bin = {int(pd.Timestamp(ts).value): b for ts, b in zip(all_timestamps, shuffled_bins)}

    ohlcv_permuted = {}
    for sym, arr in ohlcv_arrays.items():
        sig      = signal_arrays[sym].copy()
        sym_ts   = timestamps_per_sym[sym]
        for i, ts in enumerate(sym_ts):
            key  = int(pd.Timestamp(ts).value)
            bin_ = ts_to_bin.get(key)
            if bin_ in bins_to_filter:
                sig[i] = 0
        ohlcv_permuted[sym] = {**arr, "signal": sig}

    result   = run_grid_backtest(
        ohlcv_permuted,
        sell_after   = best_params["SELL_AFTER"],
        tp_pct       = best_params["TP_PCT"],
        sl_pct       = best_params["SL_PCT"],
        order_amount = order_amount,
    )
    trade_log            = result["__PORTFOLIO__"]["trade_log"].copy()
    trade_log.columns    = trade_log.columns.str.lower().str.strip()
    trade_log["buy_time"] = pd.to_datetime(trade_log["buy_time"])

    metrics = compute_metrics(trade_log, capital=INITIAL_BALANCE, name="") if len(trade_log) > 0 else None
    return {
        "perm_idx":    perm_idx,
        "n_trades":    len(trade_log),
        "net_gain_pct": metrics["Net_Gain_pct"]    if metrics else 0.0,
        "max_dd_pct":   metrics["Max_DD_pct"]      if metrics else 0.0,
        "r2":           metrics["R_Squared"]        if metrics else 0.0,
        "win_rate":     metrics["Win_Rate"]         if metrics else 0.0,
    }


# =============================================================================
# DEBUG
# =============================================================================

# =============================================================================
# def debug_regime_signals(
#     ohlcv_data: dict,
#     signal_fn: callable,
#     signal_params: dict,
#     bins_to_filter: set,
#     data_folder: str,
#     timeframe: str,
#     symbols: list = ("ETHUSDT", "SOLUSDT"),
#     n_rows: int = 10,
#     perm_seed: int = 42,
# ) -> None:
#     """
#     Print a side-by-side comparison of real vs permuted regime bins and their
#     effect on signals for the given symbols.
# 
#     Shows only rows where signal_baseline != 0 (actual signals), top n_rows each.
# 
#     Args:
#         ohlcv_data      : raw ohlcv dict {symbol: df}
#         signal_fn       : signal generation function
#         signal_params   : signal parameters dict
#         bins_to_filter  : regime bins to filter (from IS analysis)
#         data_folder     : data folder for BTC loading
#         timeframe       : timeframe string
#         symbols         : symbols to inspect
#         n_rows          : number of signal rows to print per symbol
#         perm_seed       : seed for the example permutation
#     """
#     ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_data)
# 
#     # Build signal arrays
#     signal_arrays      = {}
#     timestamps_per_sym = {}
#     for sym, arr in ohlcv_arrays.items():
#         signal_arrays[sym]      = np.asarray(signal_fn(arr, **signal_params, live_trading=False))
#         timestamps_per_sym[sym] = arr["ts"]
# 
#     # Build BTC regime data
#     btc_cache = {}
#     btc_1d_df = load_btc_for_timeframe(data_folder, "1Dutc", btc_cache)
#     btc_tf_df = load_btc_for_timeframe(data_folder, timeframe, btc_cache) \
#                 if REGIME_FAMILY_SOURCE == "strategy" else btc_1d_df
# 
#     metrics_cache = build_metrics_cache(
#         btc_df       = btc_tf_df,
#         lookback     = REGIME_LOOKBACK_BARS,
#         hurst_window = HURST_WINDOW,
#         er_window    = ER_WINDOW,
#         atr_window   = ATR_WINDOW,
#         pe_window    = PE_WINDOW,
#         pe_order     = PE_ORDER,
#     )
# 
#     all_timestamps = np.unique(
#         np.concatenate([arr["ts"] for arr in ohlcv_arrays.values()])
#     )
#     bin_series_real = _build_bin_series(all_timestamps, btc_1d_df, metrics_cache)
# 
#     print(f"\n  Cache first key  : {list(metrics_cache.keys())[0]}")
#     print(f"  Cache last key   : {list(metrics_cache.keys())[-1]}")
#     print(f"  Cache total keys : {len(metrics_cache)}")
#     print(f"  Timestamps total : {len(all_timestamps)}")
#     print(f"  Bins unknown     : {bin_series_real.count(None)}")
#     print(f"  Bins known       : {len([b for b in bin_series_real if b])}")
# 
#     # Build one example permutation
#     rng              = np.random.default_rng(perm_seed)
#     bin_series_perm  = bin_series_real.copy()
#     rng.shuffle(bin_series_perm)
# 
#     ts_to_bin_real = {int(pd.Timestamp(ts).value): b for ts, b in zip(all_timestamps, bin_series_real)}
#     ts_to_bin_perm = {int(pd.Timestamp(ts).value): b for ts, b in zip(all_timestamps, bin_series_perm)}
# 
#     # Distribution check
#     all_bins   = sorted(set(b for b in bin_series_real if b))
#     total_real = len([b for b in bin_series_real if b])
#     print(f"\n  {'BIN':<30} {'REAL_N':>8} {'REAL_%':>8} {'PERM_N':>8} {'PERM_%':>8} {'MATCH':>7}")
#     print(f"  {'-'*75}")
#     for bin_ in all_bins:
#         n_real   = bin_series_real.count(bin_)
#         n_perm   = bin_series_perm.count(bin_)
#         pct_real = n_real / total_real * 100
#         pct_perm = n_perm / total_real * 100
#         match    = "✅" if n_real == n_perm else "❌"
#         print(f"  {bin_:<30} {n_real:>8} {pct_real:>7.1f}% {n_perm:>8} {pct_perm:>7.1f}% {match:>7}")
#     print(f"  {'-'*75}")
#     n_none_real = bin_series_real.count(None)
#     n_none_perm = bin_series_perm.count(None)
#     print(f"  {'unknown':<30} {n_none_real:>8} {'':>8} {n_none_perm:>8} {'':>8} {'✅' if n_none_real == n_none_perm else '❌':>7}")
# 
#     print(f"\n{'='*100}")
#     print(f"  DEBUG — Regime signals comparison  |  bins_to_filter: {bins_to_filter}")
#     print(f"{'='*100}")
# 
#     for sym in symbols:
#         if sym not in ohlcv_arrays:
#             print(f"\n  ⚠️  {sym} not found in ohlcv_data")
#             continue
# 
#         arr        = ohlcv_arrays[sym]
#         sig_base   = signal_arrays[sym]
#         sym_ts     = timestamps_per_sym[sym]
# 
#         rows = []
#         for i, ts in enumerate(sym_ts):
#             if sig_base[i] == 0:
#                 continue
#             key       = int(pd.Timestamp(ts).value)
#             bin_real  = ts_to_bin_real.get(key)
#             bin_perm  = ts_to_bin_perm.get(key)
#             sig_real  = 0 if bin_real in bins_to_filter else sig_base[i]
#             sig_perm  = 0 if bin_perm in bins_to_filter else sig_base[i]
#             rows.append({
#                 "timestamp":      pd.Timestamp(ts),
#                 "signal_base":    int(sig_base[i]),
#                 "bin_real":       bin_real or "unknown",
#                 "signal_regime":  int(sig_real),
#                 "bin_permuted":   bin_perm  or "unknown",
#                 "signal_permuted": int(sig_perm),
#                 "changed":        "⚡" if sig_real != sig_perm else "",
#             })
#             if len(rows) >= n_rows:
#                 break
# 
#         print(f"\n  Symbol: {sym}  ({len(rows)} signal rows shown)\n")
#         print(f"  {'TIMESTAMP':<22} {'SIG_BASE':>9} {'BIN_REAL':<25} {'SIG_REGIME':>11} {'BIN_PERMUTED':<25} {'SIG_PERM':>9} {'':>4}")
#         print(f"  {'-'*98}")
#         for r in rows:
#             print(
#                 f"  {str(r['timestamp']):<22} "
#                 f"{r['signal_base']:>9} "
#                 f"{r['bin_real']:<25} "
#                 f"{r['signal_regime']:>11} "
#                 f"{r['bin_permuted']:<25} "
#                 f"{r['signal_permuted']:>9} "
#                 f"{r['changed']:>4}"
#             )
# 
#     print(f"\n{'='*100}\n")
# =============================================================================


# =============================================================================
# PUBLIC API
# =============================================================================

def run_mc_regime_robustness(
    ohlcv_data: dict,
    signal_fn: callable,
    signal_params: dict,
    best_params: dict,
    order_amount: int,
    bins_to_filter: set,
    data_folder: str,
    timeframe: str,
    n_permutations: int = 1000,
    netgain_th: float   = 0.0,
    n_jobs: int         = -1,
    show_progress: bool = False,
    seed: int           = 42,
) -> tuple[float, pd.DataFrame]:
    """
    Monte Carlo regime robustness test via bin-series permutation.

    Generates signals baseline, computes the real regime bin series for the
    period, shuffles it N times, and measures in what fraction of alternative
    regime histories the strategy remains profitable.

    Args:
        ohlcv_data      : raw ohlcv dict {symbol: df}
        signal_fn       : signal generation function
        signal_params   : signal parameters dict
        best_params     : best params dict (SELL_AFTER, TP_PCT, SL_PCT, ...)
        order_amount    : order amount
        bins_to_filter  : regime bins to filter (from IS analysis)
        data_folder     : data folder for BTC loading
        timeframe       : timeframe string
        n_permutations  : number of bin-series permutations
        netgain_th      : net gain threshold to consider a permutation positive
        n_jobs          : joblib parallelism (-1 = all cores)
        show_progress   : show tqdm progress bar
        seed            : base random seed

    Returns:
        tuple: (robustness_score, df_results)
            robustness_score : float — % permutations with NetGain > netgain_th
            df_results       : DataFrame with per-permutation metrics
    """
    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_data)

    # --- Build baseline signal arrays per symbol ---
    signal_arrays      = {}
    timestamps_per_sym = {}
    for sym, arr in ohlcv_arrays.items():
        signal_arrays[sym]      = np.asarray(signal_fn(arr, **signal_params, live_trading=False))
        timestamps_per_sym[sym] = arr["ts"]

    # --- Build unified timestamp axis & regime bin series ---
    btc_cache = {}
    btc_1d_df = load_btc_for_timeframe(data_folder, "1Dutc", btc_cache)
    btc_tf_df = load_btc_for_timeframe(data_folder, timeframe, btc_cache) \
                if REGIME_FAMILY_SOURCE == "strategy" else btc_1d_df

    metrics_cache = build_metrics_cache(
        btc_df       = btc_tf_df,
        lookback     = REGIME_LOOKBACK_BARS,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    )

    all_timestamps = np.unique(
        np.concatenate([arr["ts"] for arr in ohlcv_arrays.values()])
    )
    bin_series = _build_bin_series(all_timestamps, btc_1d_df, metrics_cache)

    logger.debug(
        f"MC Regime — bin distribution: "
        + " | ".join(
            f"{b}={bin_series.count(b)}"
            for b in sorted(set(b for b in bin_series if b))
        )
    )

    # --- Run permutations in parallel ---
    with (
        tqdm_joblib(tqdm(total=n_permutations, desc="🔄 MC Regime permutations"))
        if show_progress else contextlib.nullcontext()
    ):
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_permutation)(
                perm_idx           = i,
                ohlcv_arrays       = ohlcv_arrays,
                signal_arrays      = signal_arrays,
                timestamps_per_sym = timestamps_per_sym,
                bin_series         = bin_series,
                all_timestamps     = all_timestamps,
                bins_to_filter     = bins_to_filter,
                best_params        = best_params,
                order_amount       = order_amount,
                seed               = seed,
            )
            for i in range(n_permutations)
        )

    df_results       = pd.DataFrame(results_list)
    robustness_score = float((df_results["net_gain_pct"] > netgain_th).mean() * 100)

    logger.debug(
        f"MC Regime Robustness ── {n_permutations} permutations "
        f"| Robustness={robustness_score:.1f}% "
        f"| NetGain p50={df_results['net_gain_pct'].median():.2f}% "
        f"| NetGain p5={df_results['net_gain_pct'].quantile(0.05):.2f}%"
    )

    return robustness_score, df_results