#shared/BOT_regime/regime_engine.py
import os
import logging
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.engines.wfo_WF import WARMUP_BARS
from shared_config import VOLUME_COL
from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE, MIN_PRICE
from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.pipeline.wfo import run_wfo_is
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batch_regime.config_paths import BITGET_ROOT, DATA_FOLDER_IS, DATA_FOLDER_OOS1
from shared_batch_regime.regime_core import BINS, REGIME_TIMEFRAME, combo_label
from shared_batch_regime.regime_core import precompute_indicators
from shared_batch_regime.regime_core import load_ohlcv_raw
from shared_batch_regime.regime_core import  classify_signal_regimes
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

LONG_KEYWORD = "long"

# =============================================================================
# CONFIG LOADER — full param_grid (needed to re-run WFO), same pattern as main_batch_E1.py
# =============================================================================

def load_strategies_config(strategies_set_name: str) -> list[dict]:
    strategies_files_dir = os.path.join(BITGET_ROOT, f"BOT_batch_{strategies_set_name}", "strategies_files")

    batch_name   = f"strategies_BT_{strategies_set_name}_batch"
    batch_path   = os.path.join(strategies_files_dir, f"{batch_name}.py")
    spec         = spec_from_file_location(batch_name, batch_path)
    batch_module = module_from_spec(spec)
    spec.loader.exec_module(batch_module)
    strategies_batch = batch_module.STRATEGIES

    loop_name   = f"strategies_loop_{strategies_set_name}_09"
    loop_path   = os.path.join(strategies_files_dir, f"{loop_name}.py")
    spec        = spec_from_file_location(loop_name, loop_path)
    loop_module = module_from_spec(spec)
    spec.loader.exec_module(loop_module)
    loop_map = {s["id"]: s for s in loop_module.STRATEGIES_LOOP}

    strategies = []
    for s in strategies_batch:
        loop = loop_map.get(s["id"])
        if not loop:
            logger.warning(f"⚠️  {s['id']} not found in strategies_loop — skipping.")
            continue

        merged      = {**s, **loop}
        signal_key  = "_".join(merged["name"].split("_")[:-1])
        if signal_key not in SIGNAL_REGISTRY:
            continue

        registry = SIGNAL_REGISTRY[signal_key]
        strategies.append({
            "id":                 merged["id"],
            "timeframe":          merged["timeframe"],
            "n_symbols":          merged["N_SYMBOLS"],
            "order_amount":       merged["ORDER_AMOUNT"],
            "param_grid":         merged["param_grid"],
            "param_names":        list(merged["param_grid"].keys()),
            "lists_for_grid":     [merged["param_grid"][k] for k in merged["param_grid"].keys()],
            "signal_fn":          registry["fn"],
            "signal_params_keys": registry["params"],
            "is_long":            LONG_KEYWORD in merged["id"],
        })
    return strategies

# =============================================================================
# UNIVERSE LOADING
# =============================================================================

def load_ohlcv_is(strategy: dict) -> dict:
    _, _, ohlcv_is, _ = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        data_folder_oos   = DATA_FOLDER_OOS1,
        timeframe         = strategy["timeframe"],
        n_symbols         = strategy["n_symbols"],
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )
    return ohlcv_is

# =============================================================================
# INDICATOR CACHE (daily regime indicator, precomputed once per symbol)
# =============================================================================

def build_indicator_cache(ohlcv_data: dict, indicator_cfg: dict, data_folder: str) -> dict:
    cache = {}
    for sym in sorted(ohlcv_data.keys()):
        df = load_ohlcv_raw(sym, data_folder)
        if not df.empty:
            cache[sym] = precompute_indicators(df, indicator_cfg)
    return cache

# =============================================================================
# BASELINE WFO — no regime filtering
# =============================================================================

def run_baseline_wfo(strategy: dict, ohlcv_is: dict, dtype, n_jobs: int = -1, show_progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run WFO once, no regime filtering. Returns (wfo_test_trades, df_results)."""
    _, _, _, _, _, wfo_test_trades, df_results = run_wfo_is(
        ohlcv_data          = ohlcv_is,
        param_names         = strategy["param_names"],
        lists_for_grid      = strategy["lists_for_grid"],
        signal_fn           = strategy["signal_fn"],
        signal_params_keys  = strategy["signal_params_keys"],
        order_amount        = strategy["order_amount"],
        timeframe           = strategy["timeframe"],
        net_gain_th         = 0,
        dd_th               = 100,
        dtype               = dtype,
        n_jobs              = n_jobs,
        show_progress       = show_progress,
        n_symbols           = strategy["n_symbols"],
    )
    return wfo_test_trades, df_results

# =============================================================================
# PER-WINDOW BIN SPLIT + BACKTEST
# =============================================================================

def _split_signals_by_bin(
    sym: str,
    arr: dict,
    signals: np.ndarray,
    indicator_cache: dict,
    indicator_cfg: dict,
) -> dict[str, np.ndarray]:
    """Split one symbol's signal array into one zero-filled array per bin."""
    bin_signals = {b: np.zeros_like(signals) for b in BINS}

    sym_cache = indicator_cache.get(sym)
    regimes   = classify_signal_regimes(signals, arr, sym_cache, indicator_cfg)

    for idx, regime in regimes.items():
        bin_signals[regime][idx] = signals[idx]

    return bin_signals


def _make_collect_all_bins_fn(
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
    indicator_cache: dict,
    indicator_cfg: dict,
    window_bin_trades: list,
) -> callable:

    def _collect_all_bins_fn(params: dict, base_arrays_test: dict) -> pd.DataFrame:
        bin_arrays = {b: {} for b in BINS}

        for sym, arr in base_arrays_test.items():
            sig_kwargs = {k: params[k.upper()] for k in signal_params_keys if k.upper() in params}
            signals    = signal_fn(arr, **sig_kwargs, live_trading=False)
            bin_signals = _split_signals_by_bin(sym, arr, signals, indicator_cache, indicator_cfg)
            for b in BINS:
                bin_arrays[b][sym] = {**arr, "signal": np.asarray(bin_signals[b], dtype=dtype)}

        for b in BINS:
            results = run_grid_backtest(
                bin_arrays[b],
                sell_after   = params["SELL_AFTER"],
                tp_pct       = params["TP_PCT"],
                sl_pct       = params["SL_PCT"],
                order_amount = order_amount,
            )
            trades = results["__PORTFOLIO__"]["trade_log"].copy()
            if not trades.empty:
                trades.columns    = trades.columns.str.lower().str.strip()
                trades["buy_time"] = pd.to_datetime(trades["buy_time"])
            window_bin_trades.append({"bin": b, "trades": trades})

        return pd.DataFrame()

    return _collect_all_bins_fn

# =============================================================================
# COMBO WFO — regime filtering per bin
# =============================================================================

def run_combo_from_baseline(
    strategy: dict,
    ohlcv_is: dict,
    df_results: pd.DataFrame,
    indicator_cache: dict,
    indicator_cfg: dict,
    dtype,
    order_amount: int,
) -> dict[str, pd.DataFrame]:

    ohlcv_arr   = prepare_ohlcv_arrays(ohlcv_is)
    param_names = strategy["param_names"]
    bin_trades: dict[str, list] = {b: [] for b in BINS}

    window_rows = df_results.iloc[:-1]  # drop the trailing MODE/MEAN/EMA summary row

    for _, row in window_rows.iterrows():
        test_start_ts = np.datetime64(pd.Timestamp(row["_test_start_ts"]))
        test_end_ts   = np.datetime64(pd.Timestamp(row["_test_end_ts"]))
        test_syms     = row["ts_syms"]
        best_params   = {k: row[k] for k in param_names}

        bin_arrays = {b: {} for b in BINS}
        for sym in test_syms:
            arr_dict = ohlcv_arr[sym]
            t0       = int(np.searchsorted(arr_dict["ts"], test_start_ts, side="left"))
            t1       = int(np.searchsorted(arr_dict["ts"], test_end_ts,   side="right"))
            warm_start = max(0, t0 - WARMUP_BARS)

            arr = {
                "ts":        arr_dict["ts"][warm_start:t1],
                "open":      arr_dict["open"][warm_start:t1],
                "high":      arr_dict["high"][warm_start:t1],
                "low":       arr_dict["low"][warm_start:t1],
                "close":     arr_dict["close"][warm_start:t1],
                VOLUME_COL:  arr_dict.get(VOLUME_COL, arr_dict["close"] * 0)[warm_start:t1],
                "low_time":  arr_dict["low_time"][warm_start:t1],
                "high_time": arr_dict["high_time"][warm_start:t1],
            }

            sig_kwargs = {k: best_params[k.upper()] for k in strategy["signal_params_keys"] if k.upper() in best_params}
            signals    = strategy["signal_fn"](arr, **sig_kwargs, live_trading=False)
            signals    = np.asarray(signals, dtype=dtype)

            bin_signals = _split_signals_by_bin(sym, arr, signals, indicator_cache, indicator_cfg)
            for b in BINS:
                bin_arrays[b][sym] = {**arr, "signal": np.asarray(bin_signals[b], dtype=dtype)}

        for b in BINS:
            results = run_grid_backtest(
                bin_arrays[b],
                sell_after   = best_params["SELL_AFTER"],
                tp_pct       = best_params["TP_PCT"],
                sl_pct       = best_params["SL_PCT"],
                order_amount = order_amount,
            )
            trades = results["__PORTFOLIO__"]["trade_log"].copy()
            if not trades.empty:
                trades.columns     = trades.columns.str.lower().str.strip()
                trades["buy_time"] = pd.to_datetime(trades["buy_time"])
                trades              = trades[trades["buy_time"] >= pd.Timestamp(test_start_ts)]
                bin_trades[b].append(trades)

    result = {
        b: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        for b, dfs in bin_trades.items()
    }
    counts_str = " | ".join(f"{b}={len(result[b])}" for b in BINS)
    logger.info(f"  run_combo_from_baseline [{strategy['id']}] {combo_label(indicator_cfg)} — {counts_str}")

    return result

# =============================================================================
# METRICS — baseline + per-bin, using compute_metrics naming
# =============================================================================

def compute_metrics_per_bin(bin_trades: dict[str, pd.DataFrame], baseline_trades: pd.DataFrame) -> dict:

    metrics: dict = {}

    if baseline_trades is not None and not baseline_trades.empty:
        m = compute_metrics(baseline_trades, capital=INITIAL_BALANCE, name="baseline")
        for k, v in m.items():
            if k != "Curve":
                metrics[f"b_{k}"] = v
        _verdict = "🟢 PASS" if m["Net_Gain_pct"] > 0 else "🔴 FAIL"
        logger.info(
            f"STAGE 1 ── WFO results [baseline ] ── {_verdict} NetGain={m['Net_Gain_pct']:.1f}% DD={m['Max_DD_pct']:.1f}% "
            f"WinRate%={m['Win_Rate']:.1f}% R2={m['R_Squared']:.3f} PF={m['Profit_Factor']:.2f} "
            f"Calmar={m['Calmar']:.2f} Trades={len(baseline_trades)}"
        )
    else:
        for k in ("Net_Gain_pct", "Max_DD_pct", "Win_Rate", "R_Squared", "Profit_Factor", "Calmar"):
            metrics[f"b_{k}"] = 0.0

    for b in BINS:
        trades = bin_trades.get(b)
        if trades is not None and not trades.empty:
            m = compute_metrics(trades, capital=INITIAL_BALANCE, name=b)
            for k, v in m.items():
                if k != "Curve":
                    metrics[f"{b}_{k}"] = v
            metrics[f"{b}_n_trades"] = len(trades)
            _verdict = "🟢 PASS" if m["Net_Gain_pct"] > 0 else "🔴 FAIL"
            logger.info(
                f"STAGE 1 ── WFO results [{b:<9}] ── {_verdict} NetGain={m['Net_Gain_pct']:.1f}% DD={m['Max_DD_pct']:.1f}% "
                f"WinRate%={m['Win_Rate']:.1f}% R2={m['R_Squared']:.3f} PF={m['Profit_Factor']:.2f} "
                f"Calmar={m['Calmar']:.2f} Trades={len(trades)}"
            )
        else:
            for k in ("Net_Gain_pct", "Max_DD_pct", "Win_Rate", "R_Squared", "Profit_Factor", "Calmar"):
                metrics[f"{b}_{k}"] = 0.0
            metrics[f"{b}_n_trades"] = 0

    return metrics

# =============================================================================
# CLASSIFICATION — partitioning helper (split mode only)
# =============================================================================

def _split_trades_by_buy_time(
    baseline_trades: pd.DataFrame,
    bin_trades:      dict[str, pd.DataFrame],
    n_splits:        int,
) -> list[tuple[pd.DataFrame, dict[str, pd.DataFrame]]]:
    """
    Split baseline and bin trades into n_splits equal time buckets based on the
    baseline's buy_time range. Same date boundaries are applied to baseline and
    every bin, so each partition compares like-for-like periods.
    Returns a list of (baseline_subset, {bin: subset}) per partition.
    """
    if baseline_trades is None or baseline_trades.empty:
        return []

    t_min      = pd.Timestamp(baseline_trades["buy_time"].min())
    t_max      = pd.Timestamp(baseline_trades["buy_time"].max())
    total_days = (t_max - t_min).days
    split_len  = total_days / n_splits

    partitions = []
    for i in range(n_splits):
        t_start = t_min + pd.Timedelta(days=i * split_len)
        t_end   = t_min + pd.Timedelta(days=(i + 1) * split_len)

        baseline_subset = baseline_trades[
            (baseline_trades["buy_time"] >= t_start) & (baseline_trades["buy_time"] < t_end)
        ]
        bin_subset = {
            b: trades[(trades["buy_time"] >= t_start) & (trades["buy_time"] < t_end)]
            for b, trades in bin_trades.items()
            if trades is not None and not trades.empty
        }
        partitions.append((baseline_subset, bin_subset))

    return partitions

# =============================================================================
# CLASSIFICATION — integro mode (aggregate over the full period)
# =============================================================================

def classify_strategy_integro(combo_metrics: dict, optimize_metric: str = "Net_Gain_pct") -> list[str]:
    """Classify a strategy's winning bin using the full-period aggregate metrics."""

    baseline_val = combo_metrics.get(f"b_{optimize_metric}", 0.0)

    winning_bins = [
        b for b in BINS
        if combo_metrics.get(f"{b}_{optimize_metric}", 0.0) > baseline_val
    ]

    if not winning_bins:
        return []

    best_bin = max(winning_bins, key=lambda b: combo_metrics.get(f"{b}_{optimize_metric}", 0.0))
    return [best_bin]

# =============================================================================
# CLASSIFICATION — split mode (must win in every valid time partition)
# =============================================================================

def classify_strategy_split(
    baseline_trades: pd.DataFrame,
    bin_trades:      dict[str, pd.DataFrame],
    combo_metrics:   dict,
    n_splits:        int,
    optimize_metric: str = "Net_Gain_pct",
) -> list[str]:
    """
    Classify a strategy's winning bin by requiring the bin to beat the baseline
    in every time partition that has trades for both. Partitions with no trades
    for the bin or the baseline are skipped (neither confirm nor disqualify).
    """
    partitions = _split_trades_by_buy_time(baseline_trades, bin_trades, n_splits)

    winning_bins = []
    for b in BINS:
        valid_partitions = 0
        wins              = 0

        for baseline_subset, bin_subset in partitions:
            bin_subset_trades = bin_subset.get(b)

            if baseline_subset is None or baseline_subset.empty:
                continue
            if bin_subset_trades is None or bin_subset_trades.empty:
                continue

            base_m = compute_metrics(baseline_subset, capital=INITIAL_BALANCE, name="baseline")
            bin_m  = compute_metrics(bin_subset_trades, capital=INITIAL_BALANCE, name=b)

            valid_partitions += 1
            if bin_m[optimize_metric] > base_m[optimize_metric]:
                wins += 1

        if valid_partitions > 0 and wins == valid_partitions:
            winning_bins.append(b)

    if not winning_bins:
        return []

    best_bin = max(winning_bins, key=lambda b: combo_metrics.get(f"{b}_{optimize_metric}", 0.0))
    return [best_bin]

# =============================================================================
# PERSISTENCE
# =============================================================================

def save_bins(
    strategy_results:    dict,
    indicator_cfg:       dict,
    output_path:         str,
    strategies_set_name: str = "E1",
    all_strategies:      list[dict] | None = None,
    optimize_metric:     str = "",
) -> None:
    from datetime import datetime
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    header_lines = [
        '"""',
        f"regime_bins_{strategies_set_name}.py — auto-generated regime classification. Do not edit manually.",
        f"Generated by main_regime.py (WFO mode) on {REGIME_TIMEFRAME}",
        f"Auto-generated on {generated_at} UTC.",
        '"""',
        "",
        f"INDICATOR_CFG = {indicator_cfg}",
        "",
    ]
    if optimize_metric:
        header_lines.append(f'OPTIMIZE_METRIC = "{optimize_metric}"')
    header_lines += ["", "REGIME_BINS = {"]

    all_ids = {s["id"] for s in all_strategies} if all_strategies else set()
    missing = all_ids - set(strategy_results.keys())

    all_entries: dict[str, list[str]] = {
        sid: data.get("classification", [])
        for sid, data in strategy_results.items()
    }
    for sid in missing:
        all_entries[sid] = []

    bin_lines = [
        f'    "{sid}": {cls},{"  # excluded from calibration" if sid in missing else ""}'
        for sid, cls in sorted(all_entries.items())
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines + bin_lines + ["}"]) + "\n")
    print(f"\n  ✅ Bins saved to: {output_path}")