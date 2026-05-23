#bitget/develop/market_regime/regime03_bin_search.py
"""
Regime optimization — exhaustive bin combination search.

For each strategy/period:
  1. Pre-classify every signal by regime bin (once)
  2. Run baseline backtest (once)
  3. Evaluate all 64 bin combinations (2^6) — each requires one backtest
  4. Pick the combination that maximizes improvement on train periods
  5. Validate on held-out period

REGIME_SOURCE controls how family+direction are computed:
  'btc_strategy_tf' — BTC in same TF as strategy (default)
  'btc_daily'       — BTC 1Dutc
  'symbol_strategy_tf' — each symbol's own OHLCV in same TF
  'symbol_daily'    — each symbol's own 1Dutc

Modes:
  'search_by_tf'  — find best bin combo per timeframe on TRAIN_KEYS, validate on VALIDATE_KEY
  'analyze_by_tf' — use fixed bins per TF, persist
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from itertools import product, combinations
from importlib.util import spec_from_file_location, module_from_spec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.pipeline.universe import filter_symbols
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.regime.regime_module import build_metrics_cache, build_direction_cache, classify_trade_by_family, load_reference_symbol_for_timeframe

# =============================================================================
# CONFIGURATION
# =============================================================================

SPLIT_MODE = "expanding"
SPLIT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split_OLD", SPLIT_MODE)

PERIODS = {
    "IS":   os.path.join(SPLIT_BASE, "IS",  "crypto_2024-01_2025-05_IS"),
    "OOS1": os.path.join(SPLIT_BASE, "OOS", "crypto_2025-05_2026-05_OOS"),
    "OOS2": os.path.join(SPLIT_BASE, "OOS", "crypto_2022-01_2023-01_OOS"),
    "OOS3": os.path.join(SPLIT_BASE, "OOS", "crypto_2023-01_2024-01_OOS"),
}

STRATEGIES_SET_NAME  = "E1"
STRATEGIES_CSV_PATH  = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1", "strategies_E1", "strategies_E1.csv")
SYMBOLS_LIVE_FOLDER  = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1", "strategies_E1", "symbols_live")
BINS_OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), f"regime_bins_03_{STRATEGIES_SET_NAME}.py")
from importlib.util import spec_from_file_location, module_from_spec
STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_01"
STRATEGIES_LOOP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1",
    "strategies_files", f"files_{STRATEGIES_SET_NAME}",
    f"{STRATEGIES_LOOP_NAME}.py"
)

TRAIN_KEYS   = ["OOS2", "OOS3"]
TRAIN_KEYS   = ["IS"]
VALIDATE_KEY = "OOS1"

# =============================================================================
# MODE
# 'search_by_tf' — blacklist: find best bin combo to filter (exhaustive 64 combos)
# 'analyze_by_tf' — use FIXED_BINS_BY_STRATEGY, persist
# =============================================================================
MODE = 'search_by_tf'

# Fixed bins per strategy for MODE='analyze_by_tf'
FIXED_BINS_BY_STRATEGY: dict[str, set] = {}

# =============================================================================
# REGIME SOURCE
# 'btc_strategy_tf'    — BTC in same TF as strategy (default)
# 'btc_daily'          — BTC 1Dutc
# 'symbol_strategy_tf' — each symbol's own OHLCV in same TF
# 'symbol_daily'       — each symbol's own 1Dutc
# =============================================================================
REGIME_SOURCE = 'symbol_strategy_tf'

# Regime fixed params
MA_PERIOD     = 20
ER_WINDOW     = 14
ATR_WINDOW    = 14
LOOKBACK_BARS = 50
ORDER_AMOUNT  = 80
ER_TH         = 0.4
ATR_TH        = 2.0

# Minimum signals per bin to be considered valid
MIN_SIGNALS_PER_BIN = 50

ALL_BINS = [
    'trending_uptrend', 'trending_dwtrend',
    'ranging_uptrend',  'ranging_dwtrend',
    'volatile_uptrend', 'volatile_dwtrend',
]

# All 64 non-empty bin combinations (2^6 - 1 + empty)
ALL_BIN_COMBOS = [set()] + [
    set(combo)
    for r in range(1, len(ALL_BINS) + 1)
    for combo in combinations(ALL_BINS, r)
]

# =============================================================================
# HELPERS
# =============================================================================

_BIN_ABBREV = {
    'trending_uptrend': 'trd_up', 'trending_dwtrend': 'trd_dw',
    'ranging_uptrend':  'rng_up', 'ranging_dwtrend':  'rng_dw',
    'volatile_uptrend': 'vol_up', 'volatile_dwtrend': 'vol_dw',
}

def _abbrev_bins(bins: set) -> str:
    if not bins:
        return '—'
    return ', '.join(_BIN_ABBREV.get(b, b) for b in sorted(bins))


def _pct_improvement(profit_filtered: float, profit_baseline: float) -> float:
    if profit_baseline == 0:
        return 0.0
    return (profit_filtered - profit_baseline) / abs(profit_baseline) * 100


def build_families() -> dict:
    return {
        'trending': {'efficiency_ratio': ('>', ER_TH)},
        'volatile': {'atr_pct': ('>', ATR_TH)},
        'ranging':  {},
    }


def _parse_signal_key(strategy_name: str) -> str:
    return "_".join(strategy_name.split("_")[:-1])


def _is_daily_source() -> bool:
    return REGIME_SOURCE in ('btc_daily', 'symbol_daily')


def _is_symbol_source() -> bool:
    return REGIME_SOURCE in ('symbol_strategy_tf', 'symbol_daily')


# =============================================================================
# CONFIG LOADERS
# =============================================================================

def load_strategies_config() -> list[dict]:
    spec   = spec_from_file_location(STRATEGIES_LOOP_NAME, STRATEGIES_LOOP_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    loop_map = {s["id"]: s for s in module.STRATEGIES_LOOP}

    strategies = []
    for entry in module.STRATEGIES_LOOP:
        strategy_id   = entry["id"]
        strategy_name = strategy_id  # id == name pattern in loop
        signal_key    = "_".join(strategy_id.split("_")[1:-1])  # strip number prefix and TF suffix

        # Resolve signal key — try with and without number prefix
        if signal_key not in SIGNAL_REGISTRY:
            signal_key = "_".join(strategy_id.split("_")[:-1])
        if signal_key not in SIGNAL_REGISTRY:
            print(f"  ⚠️  '{signal_key}' not in SIGNAL_REGISTRY — skipping {strategy_id}")
            continue

        registry   = SIGNAL_REGISTRY[signal_key]
        signal_fn  = registry["fn"]
        param_keys = registry["params"]

        param_grid    = entry["param_grid"]
        best_params   = {k.upper(): v[0] for k, v in param_grid.items()}
        signal_params = {k: best_params[k.upper()] for k in param_keys if k.upper() in best_params}
        timeframe     = strategy_id.split("_")[-1]

        strategies.append({
            "id":            strategy_id,
            "name":          strategy_id,
            "timeframe":     timeframe,
            "signal_fn":     signal_fn,
            "signal_params": signal_params,
            "best_params":   best_params,
        })

    return strategies


def load_symbols(strategy_id: str, timeframe: str) -> list[str]:
    filename = f"symbols_live_{strategy_id}_{timeframe}.csv"
    filepath = os.path.join(SYMBOLS_LIVE_FOLDER, filename)
    if not os.path.exists(filepath):
        return []
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


# =============================================================================
# REGIME REF LOADING — cached globally by (data_folder, timeframe)
# =============================================================================

_regime_ref_cache: dict = {}

def _get_regime_ref(data_folder: str, timeframe: str, sym: str | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Returns (ref_df, metrics_cache) for the given source mode.
    Cached globally to avoid recomputing across strategies sharing same symbols/period.
    """
    if _is_symbol_source():
        tf_key = '1Dutc' if _is_daily_source() else timeframe
        key    = (data_folder, tf_key, sym)
    else:
        tf_key = '1Dutc' if _is_daily_source() else timeframe
        key    = (data_folder, tf_key, 'BTCUSDT')

    if key in _regime_ref_cache:
        return _regime_ref_cache[key]

    ref_cache = {}
    if _is_symbol_source():
        ref_df = load_reference_symbol_for_timeframe(data_folder, sym, tf_key, ref_cache)
    else:
        ref_df = load_reference_symbol_for_timeframe(data_folder, 'BTCUSDT', tf_key, ref_cache)

    metrics_cache = build_metrics_cache(
        ref_df       = ref_df,
        lookback     = LOOKBACK_BARS,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
    )
    _regime_ref_cache[key] = (ref_df, metrics_cache)
    return ref_df, metrics_cache


# =============================================================================
# SIGNAL CLASSIFICATION — pre-classify all signals by bin (once per strategy/period)
# =============================================================================

def _classify_signals(
    ohlcv_arrays: dict,
    signal_fn,
    signal_params: dict,
    data_folder:   str,
    timeframe:     str,
    families:      dict,
) -> dict[str, dict]:
    """
    For each symbol, generate signals and classify each signal into a regime bin.
    Returns {sym: {'signals': np.ndarray, 'signal_bins': dict{idx: bin_key}, 'arr': arr}}
    Pre-computed once — reused across all 64 bin combinations.
    """
    result = {}
    for sym, arr in ohlcv_arrays.items():
        signals     = signal_fn(arr, **signal_params, live_trading=False)
        signal_idxs = np.nonzero(signals)[0]

        ref_df, metrics_cache = _get_regime_ref(data_folder, timeframe, sym)

        trade_times     = pd.Series(pd.to_datetime(arr['ts'][signal_idxs]))
        direction_cache = build_direction_cache(
            ref_df, MA_PERIOD, trade_times,
            is_daily=_is_daily_source(),
        )

        signal_bins: dict[int, str] = {}
        for idx in signal_idxs:
            t            = pd.Timestamp(arr['ts'][idx])
            direction, _ = direction_cache.get(t, ('unknown', None))
            metrics      = metrics_cache.get(t)
            family       = classify_trade_by_family(metrics, families) if metrics else 'unknown'
            if family != 'unknown' and direction not in ('unknown', 'neutral'):
                signal_bins[int(idx)] = f"{family}_{direction}"

        result[sym] = {
            'signals':     signals,
            'signal_bins': signal_bins,
            'arr':         arr,
        }

    # Filter out bins with fewer than MIN_SIGNALS_PER_BIN signals across all symbols
    bin_counts: dict[str, int] = {}
    for sym_data in result.values():
        for bin_key in sym_data['signal_bins'].values():
            bin_counts[bin_key] = bin_counts.get(bin_key, 0) + 1

    valid_bins = {b for b, n in bin_counts.items() if n >= MIN_SIGNALS_PER_BIN}

    for sym_data in result.values():
        sym_data['signal_bins'] = {
            idx: b for idx, b in sym_data['signal_bins'].items() if b in valid_bins
        }

    return result


def _run_backtest_raw(ohlcv_arrays: dict, signal_fn, signal_params: dict, best_params: dict) -> tuple[pd.DataFrame, float]:
    """Run baseline backtest. Returns (trades_df, profit)."""
    arrays = {}
    for sym, arr in ohlcv_arrays.items():
        signals     = signal_fn(arr, **signal_params, live_trading=False)
        arrays[sym] = {**arr, 'signal': signals}

    result             = run_grid_backtest(
        arrays,
        sell_after   = best_params['SELL_AFTER'],
        tp_pct       = best_params['TP_PCT'],
        sl_pct       = best_params['SL_PCT'],
        order_amount = ORDER_AMOUNT,
    )
    trades             = result['__PORTFOLIO__']['trade_log'].copy()
    trades.columns     = trades.columns.str.lower().str.strip()
    trades['buy_time'] = pd.to_datetime(trades['buy_time'])
    return trades, trades['profit'].sum() if not trades.empty else 0.0


def _run_backtest_with_bins(
    classified:     dict[str, dict],
    bins_to_filter: set,
    best_params:    dict,
) -> dict:
    """
    Apply bin filter to pre-classified signals and run backtest.
    Returns dict with profit, win_rate, n_trades, max_dd.
    """
    arrays = {}
    for sym, data in classified.items():
        signals = data['signals'].copy()
        if bins_to_filter:
            for idx, bin_key in data['signal_bins'].items():
                if bin_key in bins_to_filter:
                    signals[idx] = 0
        arrays[sym] = {**data['arr'], 'signal': signals}

    result = run_grid_backtest(
        arrays,
        sell_after   = best_params['SELL_AFTER'],
        tp_pct       = best_params['TP_PCT'],
        sl_pct       = best_params['SL_PCT'],
        order_amount = ORDER_AMOUNT,
    )
    trades = result['__PORTFOLIO__']['trade_log']

    if len(trades) == 0:
        return {'profit': 0.0, 'win_rate': 0.0, 'n_trades': 0, 'max_dd': 0.0}

    profits  = trades['profit'] if hasattr(trades, 'columns') else pd.Series(trades['profit'])
    n        = len(profits)
    win_rate = float((profits > 0).mean() * 100)
    profit   = float(profits.sum())
    equity   = INITIAL_BALANCE + profits.cumsum()
    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min() * 100)

    return {'profit': profit, 'win_rate': win_rate, 'n_trades': n, 'max_dd': max_dd}


# =============================================================================
# PERIOD DATA LOADER
# =============================================================================

def _load_period_data(strategy: dict, period_key: str, families: dict) -> dict | None:
    data_folder = PERIODS[period_key]
    symbols     = load_symbols(strategy["id"], strategy["timeframe"])
    if not symbols:
        return None

    ohlcv_data, _ = filter_symbols(
        symbols, min_vol_usdt=0, timeframe=strategy["timeframe"],
        data_folder=data_folder, min_price=None, vol_window=50,
        my_symbols=True, custom_symbols=symbols,
    )
    if not ohlcv_data:
        return None

    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_data)

    classified = _classify_signals(
        ohlcv_arrays  = ohlcv_arrays,
        signal_fn     = strategy['signal_fn'],
        signal_params = strategy['signal_params'],
        data_folder   = data_folder,
        timeframe     = strategy['timeframe'],
        families      = families,
    )

    _, profit_baseline = _run_backtest_raw(
        ohlcv_arrays  = ohlcv_arrays,
        signal_fn     = strategy['signal_fn'],
        signal_params = strategy['signal_params'],
        best_params   = strategy['best_params'],
    )

    baseline_metrics = _run_backtest_with_bins(
        classified     = classified,
        bins_to_filter = set(),
        best_params    = strategy['best_params'],
    )

    return {
        'classified':        classified,
        'ohlcv_arrays':      ohlcv_arrays,
        'best_params':       strategy['best_params'],
        'profit_baseline':   profit_baseline,
        'wr_baseline':       baseline_metrics['win_rate'],
        'n_trades_baseline': baseline_metrics['n_trades'],
        'dd_baseline':       baseline_metrics['max_dd'],
    }


# =============================================================================
# BIN SEARCH — evaluate all 64 combinations on train periods
# =============================================================================

def _find_best_bins(
    strategy:    dict,
    period_data: dict,
) -> tuple[set, float]:
    """
    Evaluate all 64 bin combinations across train periods.
    Optimizes for profit improvement. Returns (best_bins, best_profit_improvement).
    """
    best_bins       = set()
    best_profit     = 0.0
    profit_baseline = sum(d['profit_baseline'] for d in period_data.values())

    for bins_combo in ALL_BIN_COMBOS:
        total_trades = 0
        profit_total = 0.0
        for data in period_data.values():
            m             = _run_backtest_with_bins(data['classified'], bins_combo, data['best_params'])
            total_trades += m['n_trades']
            profit_total += m['profit']

        if total_trades < MIN_SIGNALS_PER_BIN:
            continue

        if profit_total > best_profit:
            best_profit = profit_total
            best_bins   = bins_combo

    return best_bins, best_profit - profit_baseline


# =============================================================================
# SEARCH BY TIMEFRAME
# =============================================================================

def run_search_by_tf(train_keys: list[str], val_key: str) -> None:
    _t0      = time.time()
    families = build_families()

    print(f"\n{'='*100}")
    print(f"  REGIME BIN SEARCH — Train: {' + '.join(train_keys)}  →  Validate: {val_key}")
    print(f"  Source: {REGIME_SOURCE} | MA{MA_PERIOD} | ER>{ER_TH} | ATR>{ATR_TH}")
    print(f"  Bin combinations: {len(ALL_BIN_COMBOS)}")
    print(f"{'='*100}\n")

    strategies_all = load_strategies_config()
    if not strategies_all:
        print("  No active strategies — aborting.")
        return

    by_tf: dict[str, list] = {}
    for s in strategies_all:
        by_tf.setdefault(s['timeframe'], []).append(s)

    bins_per_strategy:      dict[str, set]   = {}
    train_pct_per_strategy: dict[str, float] = {}

    for tf, strategies in sorted(by_tf.items()):
        print(f"\n{'─'*100}")
        print(f"  TF: {tf} | {len(strategies)} strategies | {len(ALL_BIN_COMBOS)} combos × {len(train_keys)} periods")
        print(f"{'─'*100}")

        for strategy in strategies:
            sid         = strategy['id']
            period_data = {}

            for period_key in train_keys:
                data = _load_period_data(strategy, period_key, families)
                if data:
                    period_data[period_key] = data

            if not period_data:
                bins_per_strategy[sid]      = set()
                train_pct_per_strategy[sid] = 0.0
                continue

            best_bins, best_pct         = _find_best_bins(strategy, period_data)
            bins_per_strategy[sid]      = best_bins
            train_pct_per_strategy[sid] = best_pct

            color = "\033[92m" if best_pct > 0 else "\033[91m" if best_pct < 0 else ""
            reset = "\033[0m"
            print(f"  {sid:<35} {color}{best_pct:>+7.2f}%{reset}  bins: {_abbrev_bins(best_bins)}")

    # Validate on held-out period
    oos1_pct_per_strategy = _run_validation_period(strategies_all, val_key, bins_per_strategy, families)

    # Consistency table — train vs OOS1
    _print_consistency_table(strategies_all, train_pct_per_strategy, oos1_pct_per_strategy, bins_per_strategy)

    # Persist bins
    _save_bins(bins_per_strategy, train_keys)

    elapsed = int(time.time() - _t0)
    print(f"\n  ⏱  Completed in {elapsed//60}m {elapsed%60}s\n")


# =============================================================================
# ANALYZE — fixed bins
# =============================================================================

def run_analyze_by_tf(train_keys: list[str], val_key: str, fixed_bins: dict[str, set]) -> None:
    _t0      = time.time()
    families = build_families()

    print(f"\n{'='*100}")
    print(f"  ANALYZE — Train: {' + '.join(train_keys)}  →  Validate: {val_key}")
    print(f"  Source: {REGIME_SOURCE} | MA{MA_PERIOD} | ER>{ER_TH} | ATR>{ATR_TH}")
    print(f"{'='*100}\n")

    strategies_all = load_strategies_config()
    _run_validation_period(strategies_all, val_key, fixed_bins, families)
    _save_bins(fixed_bins, train_keys)

    elapsed = int(time.time() - _t0)
    print(f"\n  ⏱  Completed in {elapsed//60}m {elapsed%60}s\n")


# =============================================================================
# VALIDATION
# =============================================================================

def _run_validation_period(
    strategies:        list[dict],
    val_key:           str,
    bins_per_strategy: dict[str, set],
    families:          dict,
) -> dict[str, float]:
    print(f"\n{'='*100}")
    print(f"  VALIDATION RESULTS — {val_key} | Source: {REGIME_SOURCE}")
    print(f"{'='*100}")
    print(f"  {'STRATEGY':<35} {'B_WR%':>7} {'F_WR%':>7} {'ΔWR':>7} {'B_PROF':>8} {'F_PROF':>8} {'Δ%':>7} {'B_DD%':>7} {'F_DD%':>7}  {'BINS'}")
    print(f"  {'─'*115}")

    sys_b_profit         = 0.0
    sys_f_profit         = 0.0
    pct_imp_per_strategy : dict[str, float] = {}
    rows                 = []

    for strategy in strategies:
        sid  = strategy['id']
        data = _load_period_data(strategy, val_key, families)
        if not data:
            continue

        bins = bins_per_strategy.get(sid, set())
        m_b  = {
            'profit':   data['profit_baseline'],
            'win_rate': data['wr_baseline'],
            'n_trades': data['n_trades_baseline'],
            'max_dd':   data['dd_baseline'],
        }
        m_f = _run_backtest_with_bins(data['classified'], bins, data['best_params']) if bins else m_b

        dwr   = m_f['win_rate'] - m_b['win_rate']
        dpct  = _pct_improvement(m_f['profit'], m_b['profit'])
        color = "\033[92m" if dpct > 0 else "\033[91m" if dpct < 0 else ""
        reset = "\033[0m"

        print(f"  {sid:<35} {m_b['win_rate']:>6.1f}% {m_f['win_rate']:>6.1f}% "
              f"{dwr:>+6.1f}% {m_b['profit']:>8.1f} {m_f['profit']:>8.1f} "
              f"{color}{dpct:>+6.1f}%{reset} {m_b['max_dd']:>6.1f}% {m_f['max_dd']:>6.1f}%  {_abbrev_bins(bins)}")

        pct_imp_per_strategy[sid] = dpct
        sys_b_profit             += m_b['profit']
        sys_f_profit             += m_f['profit']
        rows.append({
            'b_wr':   m_b['win_rate'],
            'f_wr':   m_f['win_rate'],
            'dwr':    dwr,
            'b_dd':   m_b['max_dd'],
            'f_dd':   m_f['max_dd'],
        })

    # Summary row
    sys_pct   = _pct_improvement(sys_f_profit, sys_b_profit)
    color     = "\033[92m" if sys_pct > 0 else "\033[91m"
    reset     = "\033[0m"

    if rows:
        avg_b_wr  = sum(r['b_wr'] for r in rows) / len(rows)
        avg_f_wr  = sum(r['f_wr'] for r in rows) / len(rows)
        avg_dwr   = sum(r['dwr']  for r in rows) / len(rows)
        avg_b_dd  = sum(r['b_dd'] for r in rows) / len(rows)
        avg_f_dd  = sum(r['f_dd'] for r in rows) / len(rows)
        dwr_color = "\033[92m" if avg_dwr > 0 else "\033[91m"
        print(f"  {'─'*115}")
        print(f"  {'SYSTEM TOTAL':<35} {avg_b_wr:>6.1f}% {avg_f_wr:>6.1f}% "
              f"{dwr_color}{avg_dwr:>+6.1f}%{reset} {sys_b_profit:>8.1f} {sys_f_profit:>8.1f} "
              f"{color}{sys_pct:>+6.1f}%{reset} {avg_b_dd:>6.1f}% {avg_f_dd:>6.1f}%")
    else:
        print(f"  {'─'*115}")
        print(f"  {'SYSTEM TOTAL':<35} {'':>7} {'':>7} {'':>7} {sys_b_profit:>8.1f} {sys_f_profit:>8.1f} "
              f"{color}{sys_pct:>+6.1f}%{reset}")

    print(f"  {'─'*115}\n")
    return pct_imp_per_strategy


# =============================================================================
# CONSISTENCY TABLE
# =============================================================================

def _print_consistency_table(
    strategies:        list[dict],
    train_pct:         dict[str, float],
    oos1_pct:          dict[str, float],
    bins_per_strategy: dict[str, set],
) -> None:
    print(f"\n{'='*100}")
    print(f"  CONSISTENCY — Train vs OOS1")
    print(f"{'='*100}")
    print(f"  {'STRATEGY':<35} {'TRAIN_Δ%':>10} {'OOS1_Δ%':>10} {'OK':>5}  {'BINS'}")
    print(f"  {'─'*80}")
    consistent   = 0
    inconsistent = 0
    for strategy in strategies:
        sid   = strategy['id']
        t_pct = train_pct.get(sid, 0.0)
        o_pct = oos1_pct.get(sid, 0.0)
        bins  = bins_per_strategy.get(sid, set())
        ok    = t_pct > 0 and o_pct > 0
        icon  = "✅" if ok else "❌"
        color = "\033[92m" if ok else "\033[91m"
        reset = "\033[0m"
        if ok:
            consistent += 1
        else:
            inconsistent += 1
        print(f"  {sid:<35} {color}{t_pct:>+9.2f}%{reset} {color}{o_pct:>+9.2f}%{reset} {icon}  {_abbrev_bins(bins)}")
    print(f"  {'─'*80}")
    print(f"  Consistent: {consistent} | Inconsistent: {inconsistent}\n")

# =============================================================================
# PERSIST
# =============================================================================

def _save_bins(bins_per_strategy: dict[str, set], train_keys: list[str]) -> None:
    lines = [
        f"# Auto-generated by regime03_bin_search.py",
        f"# Source: {REGIME_SOURCE} | MA{MA_PERIOD} | ER>{ER_TH} | ATR>{ATR_TH}",
        f"# Train: {' + '.join(train_keys)}",
        f"",
        f"REGIME_SOURCE    = '{REGIME_SOURCE}'",
        f"REGIME_MA_PERIOD = {MA_PERIOD}",
        f"REGIME_ER_TH     = {ER_TH}",
        f"REGIME_ATR_TH    = {ATR_TH}",
        f"",
        f"REGIME_BINS = {{",
    ]
    for sid, bins in sorted(bins_per_strategy.items()):
        bins_str = "{" + ", ".join(f'"{b}"' for b in sorted(bins)) + "}"
        lines.append(f'    "{sid}": {bins_str},')
    lines.append("}")

    with open(BINS_OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✅ Bins saved to: {BINS_OUTPUT_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if MODE == 'search_by_tf':
        run_search_by_tf(TRAIN_KEYS, VALIDATE_KEY)

    elif MODE == 'analyze_by_tf':
        run_analyze_by_tf(TRAIN_KEYS, VALIDATE_KEY, FIXED_BINS_BY_STRATEGY)