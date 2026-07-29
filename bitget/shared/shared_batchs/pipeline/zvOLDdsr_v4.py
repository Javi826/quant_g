#shared_batchs/pipeline/dsr_v4.py
import time
import itertools
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE, run_backtest_from_prepared
from shared_batchs.utils.batch_metrics import compute_metrics

logger = logging.getLogger("BOT_batch.pipeline.dsr")

# =============================================================================
# DSR EXECUTION CONFIG
# =============================================================================
EULER_GAMMA         = 0.5772156649015328606  # Euler-Mascheroni constant
SHARPE_PERIODS_YEAR = 365.0                  # must match the annualization factor in compute_metrics (sqrt(365))
DSR_N_JOBS          = -1                     # safe to parallelize fully: this Parallel runs as its own phase, sequential relative to WFO — no nesting
DSR_MIN_TRADES      = 100   
DSR_MAX_SHARPE_ANN  = 10.0                   # combos with unrealistically high annualized Sharpe are rejected (near-zero variance artifact)
M_TO_T_WARN_RATIO   = 2.0                    # warn if M (columns) exceeds this multiple of T (days) — ill-conditioned correlation matrix (paper Appendix 3)                 # combos with unrealistically high annualized Sharpe are rejected (near-zero variance artifact)                 # combos with fewer trades are rejected (near-zero variance inflates Sharpe artificially)
# =============================================================================
# FULL-PERIOD GRID SEARCH — selection-bias metrics (single pass, no WFO windows)
# =============================================================================

def _combo_id(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))


def _daily_profit_from_trades(trade_log: pd.DataFrame) -> pd.Series:
    tl = trade_log.copy()
    tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
    return tl.groupby("_date")["profit"].sum()


def _prepare_static_bundle(ohlcv_arr: dict) -> dict:
    """Local replica of the signal-independent part of ZX_compute_BT.pyx's
    prepare_data (Cython, compiled to .so — cannot be edited without a rebuild
    step outside our control): per-symbol static fields, the padded 2D
    price/time arrays (n_symbols x max_len) the compiled backtest core reads
    via typed memoryviews, and symbol/timestamp bookkeeping. Computed once per
    OHLCV universe and reused across every rule of the same timeframe.

    IMPORTANT: must be kept in sync with the static portion of
    ZX_compute_BT.pyx's own prepare_data if that implementation ever changes."""

    if not ohlcv_arr:
        return {
            "symbols":            [],
            "sym_ids":            {},
            "sym_data_static":    {},
            "ts_int_arrays":      {},
            "close_arrays":       {},
            "all_timestamps_int": np.array([], dtype=np.int64),
            "all_timestamps_dt":  np.array([], dtype='datetime64[ns]'),
            "max_len":            0,
            "open_2d":            None,
            "close_2d":           None,
            "high_2d":            None,
            "low_2d":             None,
            "high_time_2d":       None,
            "low_time_2d":        None,
            "ts_int_2d":          None,
            "sym_len":            np.array([], dtype=np.int64),
        }

    symbols = list(ohlcv_arr.keys())

    sym_data_static  = {}
    ts_int_arrays    = {}
    close_arrays     = {}
    all_ts_int_lists = []

    for sym in symbols:
        data = ohlcv_arr[sym]

        ts = data['ts']
        if ts.dtype.kind != 'M':
            ts = ts.astype('datetime64[ns]')

        ts_int     = ts.view('int64')
        close_view = data['close']
        n          = len(ts)

        sym_data_static[sym] = {
            'ts':        ts,
            'ts_int':    ts_int,
            'open':      data['open'],
            'close':     close_view,
            'high':      data['high'],
            'low':       data['low'],
            'len':       n,
            'high_time': data['high_time'],
            'low_time':  data['low_time'],
        }

        ts_int_arrays[sym] = ts_int
        close_arrays[sym]  = close_view
        all_ts_int_lists.append(ts_int)

    n_syms  = len(symbols)
    # NOTE: sym_ids uses sorted(symbols), independent of ohlcv_arr's insertion
    # order — matches ZX_compute_BT.pyx's own prepare_data exactly, since the
    # 2D arrays are indexed by this mapping (sid) and must line up.
    sym_ids = {s: i for i, s in enumerate(sorted(symbols))}

    max_len = max(sym_data_static[s]['len'] for s in symbols)

    open_2d      = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    close_2d     = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    high_2d      = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    low_2d       = np.full((n_syms, max_len), np.nan, dtype=np.float64)
    high_time_2d = np.full((n_syms, max_len), 0,      dtype=np.int64)
    low_time_2d  = np.full((n_syms, max_len), 0,      dtype=np.int64)
    ts_int_2d    = np.full((n_syms, max_len), 0,      dtype=np.int64)
    sym_len      = np.zeros(n_syms, dtype=np.int64)

    for sym in symbols:
        sid = sym_ids[sym]
        d   = sym_data_static[sym]
        n   = d['len']

        sym_len[sid]          = n
        open_2d[sid, :n]      = d['open'].astype(np.float64)
        close_2d[sid, :n]     = d['close'].astype(np.float64)
        high_2d[sid, :n]      = d['high'].astype(np.float64)
        low_2d[sid, :n]       = d['low'].astype(np.float64)
        high_time_2d[sid, :n] = d['high_time'].astype(np.int64)
        low_time_2d[sid, :n]  = d['low_time'].astype(np.int64)
        ts_int_2d[sid, :n]    = d['ts_int'].astype(np.int64)

    all_timestamps_int = np.unique(np.concatenate(all_ts_int_lists))
    all_timestamps_dt  = all_timestamps_int.view('datetime64[ns]')

    return {
        "symbols":            symbols,
        "sym_ids":            sym_ids,
        "sym_data_static":    sym_data_static,
        "ts_int_arrays":      ts_int_arrays,
        "close_arrays":       close_arrays,
        "all_timestamps_int": all_timestamps_int,
        "all_timestamps_dt":  all_timestamps_dt,
        "max_len":            max_len,
        "open_2d":            open_2d,
        "close_2d":           close_2d,
        "high_2d":            high_2d,
        "low_2d":             low_2d,
        "high_time_2d":       high_time_2d,
        "low_time_2d":        low_time_2d,
        "ts_int_2d":          ts_int_2d,
        "sym_len":            sym_len,
    }


def _build_dynamic_bundle_local(static_bundle: dict, signal_arrays: dict) -> tuple:
    """Local replica of the signal-dependent part of ZX_compute_BT.pyx's
    prepare_data — builds sym_data (static fields + 'signal'), signal_2d, and
    signal_events (the (ts, sym_id, idx) array the compiled core scans via
    binary search). See _prepare_static_bundle docstring for why this lives
    here instead of in ZX_compute_BT.pyx."""

    symbols = static_bundle["symbols"]
    sym_ids = static_bundle["sym_ids"]
    n_syms  = len(symbols)
    max_len = static_bundle["max_len"]

    sym_data     = {}
    signal_2d    = np.zeros((n_syms, max_len), dtype=np.int64)
    event_chunks = []

    for sym in symbols:
        static = static_bundle["sym_data_static"][sym]
        n      = static['len']
        sig    = np.asarray(signal_arrays[sym][:n])

        sym_data[sym] = {**static, 'signal': sig}

        sid = sym_ids[sym]
        signal_2d[sid, :n] = sig.astype(np.int64)

        sig_idxs = np.flatnonzero(sig)
        sig_idxs = sig_idxs[sig_idxs < n]
        if sig_idxs.size:
            ts_ints = static['ts_int'][sig_idxs]
            chunk   = np.empty((sig_idxs.size, 3), dtype=np.int64)
            chunk[:, 0] = ts_ints
            chunk[:, 1] = sid
            chunk[:, 2] = sig_idxs
            event_chunks.append(chunk)

    if event_chunks:
        signal_events = np.concatenate(event_chunks, axis=0)
        # Same key order as ZX_compute_BT.pyx: primary key = timestamp (col 0),
        # secondary key = symbol id (col 1) — np.lexsort takes keys last-first.
        order         = np.lexsort((signal_events[:, 1], signal_events[:, 0]))
        signal_events = signal_events[order]
    else:
        signal_events = np.empty((0, 3), dtype=np.int64)

    return sym_data, signal_2d, signal_events


def prepare_full_period_data(signal_arrays: dict, static_bundle: dict) -> tuple:
    """Combines already-computed signal arrays with the timeframe's precomputed
    static bundle into the same 8-tuple structure ZX_compute_BT.pyx's compiled
    run_backtest_from_prepared expects (sym_data, {}, all_timestamps_int,
    all_timestamps_dt, sym_ids, ts_int_arrays, close_arrays, arrays)."""

    sym_data, signal_2d, signal_events = _build_dynamic_bundle_local(static_bundle, signal_arrays)

    arrays = (
        static_bundle["open_2d"],
        static_bundle["close_2d"],
        static_bundle["high_2d"],
        static_bundle["low_2d"],
        static_bundle["high_time_2d"],
        static_bundle["low_time_2d"],
        static_bundle["ts_int_2d"],
        signal_2d,
        static_bundle["sym_len"],
        signal_events,
        static_bundle["all_timestamps_int"],
    )

    return (
        sym_data,
        {},
        static_bundle["all_timestamps_int"],
        static_bundle["all_timestamps_dt"],
        static_bundle["sym_ids"],
        static_bundle["ts_int_arrays"],
        static_bundle["close_arrays"],
        arrays,
    )


def _evaluate_combo_sharpe(params: dict, prepared_data, order_amount: int) -> tuple:
    results = run_backtest_from_prepared(
        prepared_data,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    trade_log = results["__PORTFOLIO__"]["trade_log"]
    if trade_log is None or trade_log.empty or len(trade_log) < DSR_MIN_TRADES:
        return -np.inf, params, None, None

    trade_log             = trade_log.copy()
    trade_log.columns     = trade_log.columns.str.lower().str.strip()
    trade_log["buy_time"] = pd.to_datetime(trade_log["buy_time"])

    m      = compute_metrics(trade_log, capital=INITIAL_BALANCE, name="")
    sharpe = m["Sharpe"] if np.isfinite(m["Sharpe"]) else -np.inf
    if sharpe > DSR_MAX_SHARPE_ANN:
        return -np.inf, params, None, None

    daily_profit = _daily_profit_from_trades(trade_log)
    return sharpe, params, m, daily_profit


def _run_full_period_for_rule(
    rule_id: str,
    ohlcv_arr: dict,
    signal_fn: callable,
    param_grid: dict,
    order_amount: int,
    dtype,
    static_bundle: dict,
) -> tuple:

    keys   = list(param_grid.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*[param_grid[k] for k in keys])]

    signal_arrays = {
        sym: np.asarray(signal_fn(arr, live_trading=False), dtype=dtype)
        for sym, arr in ohlcv_arr.items()
    }

    # Cheap upper bound: no combo can ever produce more trades than raw signal
    # firings (each firing yields at most one trade attempt, which may still
    # be skipped for lack of free cash). If that upper bound is already below
    # DSR_MIN_TRADES, every combo in the grid is guaranteed to fail the same
    # check _evaluate_combo_sharpe applies after the fact — skip building the
    # prepared backtest data and running the grid of backtests entirely.
    max_possible_trades = sum(int(np.count_nonzero(sig)) for sig in signal_arrays.values())

    if max_possible_trades < DSR_MIN_TRADES:
        winner_metrics = {
            "sharpe_train":   np.nan,
            "skew_train":     np.nan,
            "kurtosis_train": np.nan,
            "n_days_train":   0,
            "net_gain_train": np.nan,
            "max_dd_train":   np.nan,
        }
        return rule_id, {**winner_metrics, "combo_daily_profit": {}, "best_combo_id": _combo_id(combos[0])}

    prepared_data = prepare_full_period_data(signal_arrays, static_bundle)

    rows = [_evaluate_combo_sharpe(params, prepared_data, order_amount) for params in combos]

    combo_daily_profit = {
        _combo_id(params): daily_profit
        for _sharpe, params, _m, daily_profit in rows
        if daily_profit is not None and len(daily_profit) > 1
    }
    best_sharpe, best_params, best_metrics, _best_daily = max(rows, key=lambda x: x[0])
    best_combo_id = _combo_id(best_params)

    if best_metrics is None:
        winner_metrics = {
            "sharpe_train":   np.nan,
            "skew_train":     np.nan,
            "kurtosis_train": np.nan,
            "n_days_train":   0,
            "net_gain_train": np.nan,
            "max_dd_train":   np.nan,
        }
    else:
        winner_metrics = {
            "sharpe_train":   best_metrics["Sharpe"],
            "skew_train":     best_metrics["Skew"],
            "kurtosis_train": best_metrics["Kurtosis"],
            "n_days_train":   best_metrics["N_days"],
            "net_gain_train": best_metrics["Net_Gain_pct"],
            "max_dd_train":   best_metrics["Max_DD_pct"],
        }

    return rule_id, {**winner_metrics, "combo_daily_profit": combo_daily_profit, "best_combo_id": best_combo_id}

def run_full_period_search(rules: list, param_grid: dict, order_amount: int, dtype, static_bundle: dict, progress_label: str = "") -> dict:

    desc = f"DSR FULL-PERIOD SEARCH {progress_label}".strip()
   
    with tqdm_joblib(tqdm(desc=desc, total=len(rules), dynamic_ncols=True)):
        results = Parallel(n_jobs=DSR_N_JOBS)(
            delayed(_run_full_period_for_rule)(
                r["rule_id"], r["ohlcv_arr"], r["signal_fn"], param_grid, order_amount, dtype, static_bundle,
            )
            for r in rules
        )

    return dict(results)

# =============================================================================
# PRIVATE HELPERS — N_eff estimation (streaming Gram accumulation, eigenvalue method)
# =============================================================================
def _build_flat_daily_matrixDDD(all_raw_results: list) -> pd.DataFrame | None:

    series_by_col = {}
    for r in all_raw_results:
        combo_profit = r.get("combo_daily_profit") or {}
        for combo_id, s in combo_profit.items():
            series_by_col[f"{r['rule_id']}__{combo_id}"] = s

    if len(series_by_col) < 2:
        return None

    matrix = pd.concat(series_by_col, axis=1)
    matrix = matrix.sort_index().fillna(0.0)

    return matrix

def _iter_daily_profit_columns(all_raw_results: list):
    """Yields (col_name, daily_profit_series) for every rule x combo pair —
    the same set of columns a dense flat matrix would have had, but consumed
    one at a time instead of all held in memory simultaneously."""
    for r in all_raw_results:
        combo_profit = r.get("combo_daily_profit") or {}
        for combo_id, s in combo_profit.items():
            yield f"{r['rule_id']}__{combo_id}", s


def _common_date_axis(all_raw_results: list) -> np.ndarray | None:
    """Union of all dates across every column — same axis a dense flat matrix
    would have used to align/reindex each column. Requires touching each
    column's index once (cheap: dates only, not the profit values)."""

    date_arrays = [s.index.to_numpy() for _col_name, s in _iter_daily_profit_columns(all_raw_results)]
    if len(date_arrays) < 2:
        return None
    return np.unique(np.concatenate(date_arrays))


def _eigenvalues_desc(square_array: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(square_array)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return np.sort(eigenvalues)[::-1]


def _participation_ratio(eigenvalues: np.ndarray, n_const: int) -> float:
    sum_eig    = eigenvalues.sum() + n_const
    sum_eig_sq = np.sum(eigenvalues ** 2) + n_const
    if sum_eig_sq <= 0:
        return 1.0
    return float((sum_eig ** 2) / sum_eig_sq)


BATCH_SIZE_N_EFF = 2000  # standardized columns accumulated before each BLAS matmul —
                         # bounds RAM to T x BATCH_SIZE_N_EFF while keeping each matmul
                         # large enough for BLAS to run efficiently (vs. one matmul per column)

def _estimate_n_eff_eigen_streaming(all_raw_results: list, all_dates: np.ndarray, batch_size: int = BATCH_SIZE_N_EFF) -> float:
    """Computes the same participation-ratio N_eff as a dense-matrix eigenvalue
    approach, but by accumulating the T x T Gram matrix in batches of columns
    instead of ever materializing the full T x M dense matrix (M = rules x
    combos). M can reach the hundreds of thousands at scale — the dense matrix
    is the actual RAM driver; the Gram matrix (T x T) stays fixed-size and
    small (T = number of days) regardless of how many rules/combos exist.
    Batching (instead of one column at a time) lets each partial matmul run
    through BLAS, which is what makes this fast in practice."""

    t_days  = all_dates.shape[0]
    gram    = np.zeros((t_days, t_days), dtype=np.float64)
    n_const = 0
    n_valid = 0
    batch_cols = []

    for _col_name, s in _iter_daily_profit_columns(all_raw_results):
        col = np.zeros(t_days, dtype=np.float64)
        row_idx = np.searchsorted(all_dates, s.index.to_numpy())
        col[row_idx] = s.to_numpy(dtype=np.float64)

        std = col.std(ddof=1)
        if std <= 0:
            n_const += 1
            continue

        batch_cols.append((col - col.mean()) / std)
        n_valid += 1

        if len(batch_cols) >= batch_size:
            x_batch = np.column_stack(batch_cols)
            gram   += x_batch @ x_batch.T
            batch_cols = []

    if batch_cols:
        x_batch = np.column_stack(batch_cols)
        gram   += x_batch @ x_batch.T

    if n_valid == 0:
        return float(n_const) if n_const > 0 else 1.0

    gram /= (t_days - 1)
    eigenvalues = _eigenvalues_desc(gram)
    return _participation_ratio(eigenvalues, n_const)


def estimate_n_eff_flat(all_raw_results: list) -> float | None:

    all_dates = _common_date_axis(all_raw_results)
    if all_dates is None:
        return None
    return _estimate_n_eff_eigen_streaming(all_raw_results, all_dates)


# =============================================================================
# PRIVATE HELPERS — DSR formula (paper Eq. 1-2)
# =============================================================================
def _unannualize_sharpe(sharpe_annualized: float, periods_per_year: float = SHARPE_PERIODS_YEAR) -> float:
    if sharpe_annualized is None or not np.isfinite(sharpe_annualized):
        return np.nan
    return float(sharpe_annualized / np.sqrt(periods_per_year))


def _expected_max_sharpe(var_sr: float, n_trials: float) -> float:
    """Eq. 1 — expected maximum Sharpe ratio under N independent trials, assuming null skill."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z_n  = norm.ppf(1.0 - 1.0 / n_trials)
    z_ne = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    term = (1.0 - EULER_GAMMA) * z_n + EULER_GAMMA * z_ne
    return float(np.sqrt(var_sr) * term)


def _deflated_sharpe_ratio(sr: float, sr0: float, t_obs: int, skew_r: float, kurt_r: float) -> float:
    """Eq. 2. sr and sr0 must both be UNANNUALIZED. kurt_r is raw kurtosis (fisher=False)."""
    if t_obs <= 1 or not np.isfinite(sr):
        return 0.0
    moment_term = 1.0 - skew_r * sr + ((kurt_r - 1.0) / 4.0) * (sr ** 2)
    if moment_term <= 0:
        return 0.0
    numerator = (sr - sr0) * np.sqrt(t_obs - 1)
    return float(norm.cdf(numerator / np.sqrt(moment_term)))


# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_dsr_approval(dsr_value: float, dsr_th: float) -> bool:
    return dsr_value >= dsr_th


def _short_id(rule_id: str) -> str:
    return "_".join(rule_id.split("_")[:3])


def _train_period_str(r: dict) -> str:
    combo_daily_profit = r.get("combo_daily_profit") or {}
    best_combo_id       = r.get("best_combo_id")
    if best_combo_id is None or best_combo_id not in combo_daily_profit:
        return "n/a"

    daily_profit = combo_daily_profit[best_combo_id]
    if daily_profit is None or daily_profit.empty:
        return "n/a"

    start = daily_profit.index.min()
    end   = daily_profit.index.max()
    return f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"


def _print_train_metrics_table(raw_by_id: dict, dsr_by_id: dict, sr_by_id: dict, candidate_ids: set, passed_ids: set, sr0: float) -> None:

    rows = [raw_by_id[rid] for rid in candidate_ids if rid in raw_by_id]
    rows.sort(key=lambda r: dsr_by_id.get(r["rule_id"], 0.0), reverse=True)

    if not rows:
        return

    id_width     = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2
    label_width  = max((len(r.get("label", "")) for r in rows), default=8) + 2
    combo_width  = max((len(r.get("best_combo_id", "") or "") for r in rows), default=8) + 2
    period_width = max((len(_train_period_str(r)) for r in rows), default=8) + 2

    logger.debug(f"\n{'─' * 200}")
    logger.debug(f"  DSR TRAIN METRICS (full-period grid search) ── SR0={sr0:.4f} ── {len(rows)} candidates")
    logger.debug(f"{'─' * 200}")
    logger.debug(
        f"{'ID':<{id_width}}{'SIDE':<6}{'NET_GAIN_TR':<13}{'MAX_DD_TR':<11}{'SR_ANN':<10}{'SR_UNANN':<11}"
        f"{'SKEW_TR':<10}{'KURT_TR':<10}{'N_DAYS_TR':<11}{'DSR':<9}{'BEST_COMBO':<{combo_width}}"
        f"{'TRAIN_PERIOD':<{period_width}}{'RULE':<{label_width}}{'STATUS':<8}"
    )
    logger.debug(f"{'─' * 200}")

    for r in rows:
        rule_id = r["rule_id"]
        status  = "✅" if rule_id in passed_ids else "❌"
        logger.debug(
            f"{_short_id(rule_id):<{id_width}}{r.get('side', ''):<6}"
            f"{r.get('net_gain_train', float('nan')):<13.1f}{r.get('max_dd_train', float('nan')):<11.1f}"
            f"{r.get('sharpe_train', float('nan')):<10.4f}{sr_by_id.get(rule_id, float('nan')):<11.4f}"
            f"{r.get('skew_train', float('nan')):<10.4f}{r.get('kurtosis_train', float('nan')):<10.4f}"
            f"{r.get('n_days_train', 0):<11}{dsr_by_id.get(rule_id, 0.0):<9.4f}"
            f"{(r.get('best_combo_id', '') or 'n/a'):<{combo_width}}"
            f"{_train_period_str(r):<{period_width}}"
            f"{r.get('label', ''):<{label_width}}{status:<8}"
        )
    logger.debug(f"{'─' * 200}\n")
# =============================================================================
# CORE DSR CALCULATION (across a set of candidate trials — typically one timeframe)
# =============================================================================
def _compute_dsr(all_raw_results: list, dsr_th: float, n_combos: int) -> dict:

    total_candidates = len(all_raw_results)
    n_bruto           = total_candidates * max(n_combos, 1)

    n_eff = estimate_n_eff_flat(all_raw_results)

    n_bruto_str    = f"{n_bruto:,}".replace(",", ".")
    m_str          = f"{total_candidates:,}".replace(",", ".")
    n_eff_str      = f"{n_eff:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if n_eff is not None else "n/a (insufficient data)"

    logger.info(
        f"DSR ── N_bruto={n_bruto_str} (M={m_str} x n_combos={n_combos})  "
        f"N_eff={n_eff_str}"
    )

    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    if n_eff is None:
        logger.debug("DSR ── N_eff unavailable — setting DSR=0.0 for all rules (no rules pass).")
        dsr_by_id = {rule_id: 0.0 for rule_id in raw_by_id}
        return {
            "passed_dsr_ids": [],
            "dsr_by_rule_id": dsr_by_id,
            "n_eff":          None,
            "n_bruto":        n_bruto,
            "sr0":            np.nan,
        }

    sr_by_id = {
        rule_id: _unannualize_sharpe(r.get("sharpe_train", np.nan))
        for rule_id, r in raw_by_id.items()
    }
    sr_array = np.array(list(sr_by_id.values()), dtype=np.float64)
    sr_array = sr_array[np.isfinite(sr_array)]
    var_sr   = float(np.var(sr_array, ddof=1)) if sr_array.size > 1 else 0.0

    sr0 = _expected_max_sharpe(var_sr, n_eff)

    logger.debug(
        f"DSR ── SR0 terms ── total_candidates={total_candidates} n_combos={n_combos} "
        f"n_eff={n_eff:.4f} n_sr={sr_array.size} var_sr={var_sr:.6f} -> SR0={sr0:.4f}"
    )

    dsr_by_id = {}
    for rule_id, r in raw_by_id.items():
        t_days = int(r.get("n_days_train", 0))
        skew_r = float(r.get("skew_train", np.nan))
        kurt_r = float(r.get("kurtosis_train", np.nan))

        if not (np.isfinite(skew_r) and np.isfinite(kurt_r)):
            dsr_by_id[rule_id] = 0.0
            continue

        dsr_by_id[rule_id] = _deflated_sharpe_ratio(sr_by_id[rule_id], sr0, t_days, skew_r, kurt_r)

    passed_dsr_ids = [rid for rid, dsr_val in dsr_by_id.items() if _evaluate_dsr_approval(dsr_val, dsr_th)]

    if logger.isEnabledFor(logging.DEBUG):
        _print_train_metrics_table(raw_by_id, dsr_by_id, sr_by_id, set(passed_dsr_ids), set(passed_dsr_ids), sr0)

    logger.debug(
        f"DSR ── M={total_candidates} n_combos={n_combos} N_bruto={n_bruto} N_eff={n_eff:.4f} SR0={sr0:.3f} "
        f"-> {len(passed_dsr_ids)}/{total_candidates} significant at th={dsr_th}"
    )

    return {
        "passed_dsr_ids": passed_dsr_ids,
        "dsr_by_rule_id": dsr_by_id,
        "n_eff":          n_eff,
        "n_bruto":        n_bruto,
        "sr0":            sr0,
    }


# =============================================================================
# PIPE DSR — one timeframe at a time
# =============================================================================
def _empty_dsr_fields() -> dict:
    """Placeholder DSR fields for rules that were never evaluated (pipe disabled)."""
    return {
        "passed_dsr":         True,
        "dsr":                0.0,
        "sharpe_train":       None,
        "skew_train":         None,
        "kurtosis_train":     None,
        "n_days_train":       None,
        "net_gain_train":     None,
        "max_dd_train":       None,
        "combo_daily_profit": None,
        "best_combo_id":      None,
    }


def pipe_dsr(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    dtype,
    dsr_th: float,
    enabled: bool = True,
    timeframe: str = "",
) -> list:

    start = time.time()

    if not enabled:
        logger.info(f"DSR ── {timeframe} ── disabled — passing all {len(rules)} rules through untouched")
        return [{**r, **_empty_dsr_fields()} for r in rules]

    n_combos = 1
    for _values in param_grid.values():
        n_combos *= len(_values)

    _check_m_vs_t_ratio(ohlcv_arr, n_rules=len(rules), n_combos=n_combos, timeframe=timeframe)

    # Static part (prices/timestamps) is identical for every rule of this
    # timeframe — computed once here instead of once per rule. Built locally
    # (see _prepare_static_bundle) since the compiled backtester module cannot
    # be edited to expose this split without a rebuild step.
    static_bundle = _prepare_static_bundle(ohlcv_arr)

    rules_for_search = [
        {"rule_id": r["rule_id"], "ohlcv_arr": ohlcv_arr, "signal_fn": r["signal_fn"]}
        for r in rules
    ]
    full_period_by_rule = run_full_period_search(
        rules          = rules_for_search,
        param_grid     = param_grid,
        order_amount   = order_amount,
        dtype          = dtype,
        static_bundle  = static_bundle,
        progress_label = timeframe,
    )

    raw_for_dsr = [
        {**r, **full_period_by_rule[r["rule_id"]]}
        for r in rules
    ]
    dsr_result     = _compute_dsr(raw_for_dsr, dsr_th=dsr_th, n_combos=n_combos)
    passed_dsr_ids = set(dsr_result["passed_dsr_ids"])
    dsr_by_id      = dsr_result["dsr_by_rule_id"]

    logger.info(f"DSR ── {timeframe} ── {len(passed_dsr_ids)}/{len(rules)} rules pass")
    #debug_plot_approved_dsr_daily_profit(raw_for_dsr, passed_dsr_ids)  # DEBUG — remove after use

    results = []
    for r in rules:
        rid    = r["rule_id"]
        fp     = full_period_by_rule[rid]
        passed = rid in passed_dsr_ids
        results.append({
            **r,
            "passed_dsr":         passed,
            "dsr":                dsr_by_id.get(rid, 0.0),
            "sharpe_train":       fp["sharpe_train"],
            "skew_train":         fp["skew_train"],
            "kurtosis_train":     fp["kurtosis_train"],
            "n_days_train":       fp["n_days_train"],
            "net_gain_train":     fp["net_gain_train"],
            "max_dd_train":       fp["max_dd_train"],
            # freed for non-survivors: combo_daily_profit is no longer needed
            # once _compute_dsr (N_eff/DSR) has already used it above — keeping
            # it for ~9000/9280 rejected rules per timeframe was the main RAM driver.
            "combo_daily_profit": fp["combo_daily_profit"] if passed else None,
            "best_combo_id":      fp["best_combo_id"] if passed else None,
        })

    elapsed = int(time.time() - start)
    logger.info(f"DSR ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return results

def _check_m_vs_t_ratio(ohlcv_arr: dict, n_rules: int, n_combos: int, timeframe: str) -> None:

    any_sym  = next(iter(ohlcv_arr.values()))
    ts_range = pd.to_datetime(any_sym["ts"])
    t_days   = max((ts_range.max() - ts_range.min()).days, 1)
    m_bruto  = n_rules * max(n_combos, 1)

    if m_bruto > M_TO_T_WARN_RATIO * t_days:
        logger.debug(
            f"DSR ── {timeframe} ── M/T check ⚠️ ── M_bruto={m_bruto} (rules={n_rules} x combos={n_combos}) "
            f"vs T={t_days} days (ratio={m_bruto / t_days:.2f}x, warn_th={M_TO_T_WARN_RATIO}x) "
        )
    else:
        logger.debug(
            f"DSR ── {timeframe} ── M/T check ✅ ── M_bruto={m_bruto} (rules={n_rules} x combos={n_combos}) "
            f"vs T={t_days} days (ratio={m_bruto / t_days:.2f}x, warn_th={M_TO_T_WARN_RATIO}x)"
        )

# =============================================================================
# DEBUG ONLY — remove after use
# =============================================================================
def debug_plot_approved_dsr_daily_profit(results: list, passed_dsr_ids: set) -> None:
    for r in results:
        rule_id = r["rule_id"]
        if rule_id not in passed_dsr_ids:
            continue

        combo_daily_profit = r.get("combo_daily_profit") or {}
        if not combo_daily_profit:
            continue

        best_combo_id = r.get("best_combo_id")
        if best_combo_id is None or best_combo_id not in combo_daily_profit:
            continue
        daily_profit = combo_daily_profit[best_combo_id]

        values = daily_profit.values[np.isfinite(daily_profit.values)]
        if values.size == 0 or np.ptp(values) < 1e-6:
            logger.warning(
                f"DSR DEBUG STATS ── {rule_id} ── skipped, degenerate combo={best_combo_id} "
                f"n_days={values.size} n_nonzero={(values != 0).sum()} "
                f"min={values.min() if values.size else 'n/a'} max={values.max() if values.size else 'n/a'}"
            )
            continue

        nonzero_values = values[values != 0.0]
        n_days         = values.size
        n_nonzero      = nonzero_values.size
        top5           = np.sort(np.abs(nonzero_values))[-5:][::-1] if nonzero_values.size else np.array([])

        logger.warning(
            f"DSR DEBUG STATS ── {rule_id} — {best_combo_id} ── "
            f"n_days={n_days} n_nonzero={n_nonzero} ({n_nonzero / n_days:.1%}) "
            f"min={values.min():.2f} max={values.max():.2f} "
            f"mean_nonzero={nonzero_values.mean():.2f} std_nonzero={nonzero_values.std():.2f} "
            f"top5_abs={np.round(top5, 2).tolist()}"
        )