#shared_batchs/runs/run_deploy.py
import importlib.util
import itertools
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.pipeline.wfo import WFO_WINDOW_CONFIG, _build_ohlcv_with_signal, _compute_metric
from shared_batchs.engines.wfo_WF import WARMUP_BARS, _select_window_symbols
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_year
from shared_config import VOLUME_COL

logger = logging.getLogger("BOT_batch.runs.run_deploy")

# =============================================================================
# REGIME CLASSIFICATION MAP
# =============================================================================

_REGIME_MAP = {
    "uptrend": {"regime_uptrend": 1, "regime_dwtrend": 0, "regime_neutral": 0},
    "dwtrend": {"regime_uptrend": 0, "regime_dwtrend": 1, "regime_neutral": 0},
    "neutral": {"regime_uptrend": 1, "regime_dwtrend": 1, "regime_neutral": 1},
}


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _load_module(path: str, name: str):
    """Load a .py file as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fmt_val(val) -> str:
    """Format a Python value for writing into a .py file."""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _resolve_classification(bins_to_filter) -> str:
    """Resolve bins_to_filter (set, list, or str) to a _REGIME_MAP key."""
    if isinstance(bins_to_filter, (set, list)):
        bins_list = list(bins_to_filter)
        return bins_list[0] if len(bins_list) == 1 else "neutral"
    if isinstance(bins_to_filter, str):
        return bins_to_filter
    return "neutral"


def _slice_deploy_window(
    ohlcv_is: dict,
    timeframe: str,
) -> tuple:

    _wfo_cfg = WFO_WINDOW_CONFIG.get(timeframe)
    if _wfo_cfg is None:
        raise ValueError(f"No WFO window config for timeframe: {timeframe}")

    bars_per_month   = get_bars_per_year(timeframe) / 12
    length_train_set = int(_wfo_cfg["train_months"] * bars_per_month)

    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_is)
    ref_sym      = max(ohlcv_arrays, key=lambda k: len(ohlcv_arrays[k]["ts"]))
    ref_ts       = ohlcv_arrays[ref_sym]["ts"]
    start_idx    = max(0, len(ref_ts) - length_train_set)
    train_start_ts = ref_ts[start_idx]
    train_end_ts   = ref_ts[-1]

    ohlcv_window = {}
    for sym, arr in ohlcv_arrays.items():
        sym_ts     = arr["ts"]
        t0         = int(np.searchsorted(sym_ts, train_start_ts, side="left"))
        warm_start = max(0, t0 - WARMUP_BARS)
        if t0 >= len(sym_ts):
            continue
        ohlcv_window[sym] = {
            "ts":        sym_ts[warm_start:],
            "open":      arr["open"][warm_start:],
            "high":      arr["high"][warm_start:],
            "low":       arr["low"][warm_start:],
            "close":     arr["close"][warm_start:],
            VOLUME_COL:  arr.get(VOLUME_COL, arr["close"] * 0)[warm_start:],
            "low_time":  arr["low_time"][warm_start:],
            "high_time": arr["high_time"][warm_start:],
        }

    return ohlcv_window, train_start_ts, train_end_ts


def _select_deploy_symbols(
    ohlcv_window: dict,
    n_symbols: int,
    train_start_ts,
) -> dict:
    """Select top n_symbols by average volume within the deploy train window."""
    candidate_indices = {}
    for sym, arr in ohlcv_window.items():
        sym_ts = arr["ts"]
        t0     = int(np.searchsorted(sym_ts, train_start_ts, side="left"))
        t1     = len(sym_ts)
        if t1 > t0:
            candidate_indices[sym] = (t0, t1, t0, t1)

    selected = _select_window_symbols(candidate_indices, ohlcv_window, n_symbols)
    return {sym: ohlcv_window[sym] for sym in selected}


def _run_deploy_grid(
    ohlcv_selected: dict,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
    n_jobs: int,
) -> dict:
    """Run parallel grid search on deploy window and return best params."""
    dict_combinations = [
        dict(zip(param_names, comb))
        for comb in itertools.product(*lists_for_grid)
    ]

    def _evaluate(params):
        arrays  = _build_ohlcv_with_signal(ohlcv_selected, signal_fn, signal_params_keys, params, dtype)
        results = run_grid_backtest(
            arrays,
            sell_after   = params["SELL_AFTER"],
            tp_pct       = params["TP_PCT"],
            sl_pct       = params["SL_PCT"],
            order_amount = order_amount,
        )
        return _compute_metric(results), params

    results        = Parallel(n_jobs=n_jobs)(delayed(_evaluate)(p) for p in dict_combinations)
    _, best_params = max(results, key=lambda x: x[0])
    return best_params


def _save_deploy_symbols(
    strategy_id: str,
    deploy_symbols: list,
    timeframe: str,
    symbols_live_folder: str,
) -> bool:
    """
    Save deploy symbols to symbols_live/ folder.
    Returns True if the symbol list changed compared to the previous run.
    """
    os.makedirs(symbols_live_folder, exist_ok=True)
    path = os.path.join(symbols_live_folder, f"symbols_live_{strategy_id}_{timeframe}.csv")

    if os.path.exists(path):
        prev_symbols    = pd.read_csv(path, header=None)[0].tolist()
        symbols_changed = prev_symbols != list(deploy_symbols)
    else:
        symbols_changed = True

    pd.DataFrame(deploy_symbols).to_csv(path, index=False, header=False)
    logger.debug(f"symbols_live saved → {path}")

    return symbols_changed


def _save_deploy_batch(
    strategies_batch_path: str,
    module_name: str,
    output_path: str,
    deploy_map: dict,
    regime_bins_path: str,
    strategy_ids_to_run: list,
) -> None:

    if not os.path.exists(strategies_batch_path):
        logger.warning(f"⚠️  {os.path.basename(strategies_batch_path)} not found — skipping.")
        return

    strategies = _load_module(strategies_batch_path, module_name).STRATEGIES
    if strategy_ids_to_run:
        strategies = [s for s in strategies if s["id"] in strategy_ids_to_run]

    # Load regime bins
    regime_bins = {}
    if os.path.exists(regime_bins_path):
        regime_bins = getattr(_load_module(regime_bins_path, "regime_bins"), "REGIME_BINS", {})

    from shared_batchs.registry.signal_registry import SIGNAL_PARAM_KEYS

    # Build train window summary for the docstring header — one entry per timeframe
    _seen_tfs   = {}
    _window_lines = []
    for _entry in deploy_map.values():
        _tf = _entry.get("timeframe")
        _ts = _entry.get("train_start_ts")
        _te = _entry.get("train_end_ts")
        _tm = _entry.get("test_months")
        if _tf is None or _tf in _seen_tfs or _ts is None or _te is None or _tm is None:
            continue
        _seen_tfs[_tf] = True
        _trm    = _entry.get("train_months")
        _ts_str = pd.Timestamp(_ts).strftime("%Y-%m-%d")
        _te_str = pd.Timestamp(_te).strftime("%Y-%m-%d")
        _nt_str = (pd.Timestamp(_te) + pd.DateOffset(months=int(_tm))).strftime("%Y-%m-%d")
        _trm_str = f"{int(_trm)}m train" if _trm is not None else "?m train"
        _tm_str  = f"+{int(_tm)}m"
        _window_lines.append(f'  {_tf:<12}: {_ts_str} → {_te_str}  ({_trm_str})  |  next train: {_nt_str}  ({_tm_str})')

    lines = [
        '"""',
        'Trading Strategies Configuration — Deploy',
        '',
        f'Auto-generated by run_deploy on {datetime.now().strftime("%Y-%m-%d %H:%M")}. Do not edit manually.',
        'Copy to BOT_trading/config/strategies_00/E1.py to deploy.',
    ]
    if _window_lines:
        lines += ['', 'Train windows:'] + _window_lines
    lines += ['"""']
    lines += ['', 'STRATEGIES = [']

    for s in strategies:
        sid    = s["id"]
        entry  = deploy_map.get(sid, {})
        bp     = entry.get("params", {})
        active = entry.get("approved", False)

        # Resolve regime from bins
        bins           = regime_bins.get(sid, [])
        classification = bins[0] if len(bins) == 1 else "neutral"
        regime_values  = _REGIME_MAP.get(classification, _REGIME_MAP["neutral"])

        updated = dict(s)
        for k, val in bp.items():
            updated[k.lower()] = val

        lines.append("    {")
        lines.append(f'        "id": "{sid}",')
        lines.append(f'        "name": "{updated["name"]}",')
        lines.append(f'        "timeframe": "{updated["timeframe"]}",')
        lines.append(f'        "active": {active},')
        lines.append(f'        "direction": "{updated["direction"]}",')
        for bin_key, val in regime_values.items():
            lines.append(f'        "{bin_key}": {val},')
        lines.append(f'        "sell_after_ncandles": {updated.get("sell_after_ncandles", 0)},')
        lines.append(f'        "order_amount": {updated.get("order_amount_prod", updated.get("order_amount", 200))},')
        for k in SIGNAL_PARAM_KEYS:
            if k in updated:
                lines.append(f'        "{k}": {_fmt_val(updated[k])},')
        for k in ("tp_pct", "sl_pct"):
            if k in updated:
                lines.append(f'        "{k}": {_fmt_val(updated[k])},')
        lines.append("    },")

    lines.append("]")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"DEPLOY ── strategies PR batch saved → {output_path}")


# =============================================================================
# PUBLIC API
# =============================================================================

def run_deploy_train(
    strategy_id: str,
    ohlcv_is: dict,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    timeframe: str,
    n_symbols: int,
    approved_wfo: bool,
    dtype,
    n_jobs: int,
    symbols_live_folder: str,
    output_path: str,
    strategies_batch_path: str,
    module_name: str,
    regime_bins_path: str,
    deploy_map: dict,
    strategy_ids_to_run: list,
) -> bool:


    logger.info(f"DEPLOY  ── {strategy_id} ── running deploy train window")

    _wfo_cfg    = WFO_WINDOW_CONFIG.get(timeframe, {})
    test_months = _wfo_cfg.get("test_months")

    ohlcv_window, train_start_ts, train_end_ts = _slice_deploy_window(ohlcv_is, timeframe)
    ohlcv_selected = _select_deploy_symbols(ohlcv_window, n_symbols, train_start_ts)
    deploy_symbols = list(ohlcv_selected.keys())
    logger.info(
        f"DEPLOY  ── {strategy_id} ── {len(deploy_symbols)} symbols | "
        f"from {pd.Timestamp(train_start_ts)} to {pd.Timestamp(train_end_ts)}"
    )

    deploy_params = _run_deploy_grid(
        ohlcv_selected     = ohlcv_selected,
        param_names        = param_names,
        lists_for_grid     = lists_for_grid,
        signal_fn          = signal_fn,
        signal_params_keys = signal_params_keys,
        order_amount       = order_amount,
        dtype              = dtype,
        n_jobs             = n_jobs,
    )

    params_str = " | ".join(f"{k}={v}" for k, v in deploy_params.items() if k != "SELL_AFTER")
    logger.info(f"DEPLOY  ── {strategy_id} ── {'🟢 active' if approved_wfo else '🔴 inactive'} | {params_str}")

    # Accumulate into shared deploy_map
    deploy_map[strategy_id] = {
        "params":         deploy_params,
        "approved":       approved_wfo,
        "train_start_ts": train_start_ts,
        "train_end_ts":   train_end_ts,
        "train_months":   _wfo_cfg.get("train_months"),
        "test_months":    test_months,
        "timeframe":      timeframe,
    }

    # Save symbols (detects change vs previous run)
    symbols_changed = _save_deploy_symbols(strategy_id, deploy_symbols, timeframe, symbols_live_folder)

    # Write PR batch (overwrites each time — final call has all strategies)
    _save_deploy_batch(
        strategies_batch_path = strategies_batch_path,
        module_name           = module_name,
        output_path           = output_path,
        deploy_map            = deploy_map,
        regime_bins_path      = regime_bins_path,
        strategy_ids_to_run   = strategy_ids_to_run,
    )

    return symbols_changed