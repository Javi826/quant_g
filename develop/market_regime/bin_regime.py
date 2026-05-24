"""
Regime cross-period bin search — finds best bin combo per train period,
requires consistency across periods, validates on OOS1.

MATCH_MODE = 'exact'        — combo must be identical in MIN_PERIODS_MATCH periods
MATCH_MODE = 'intersection' — uses bins common to all winning combos

For each strategy:
  1. Find best bin combo per train period independently
  2. Apply MATCH_MODE to derive final bins
  3. Validate on OOS1
  4. Persist to regime_bins_06_{SET}.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.pipeline.universe import filter_symbols
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.regime.regime_module import build_metrics_cache, build_direction_cache, classify_trade_by_family, load_reference_symbol_for_timeframe
from importlib.util import spec_from_file_location, module_from_spec

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
BINS_OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), f"regime_bins_06_{STRATEGIES_SET_NAME}.py")

STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_01"
STRATEGIES_LOOP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1",
    "strategies_files", f"files_{STRATEGIES_SET_NAME}",
    f"{STRATEGIES_LOOP_NAME}.py"
)

TRAIN_KEYS        = ["IS", "OOS2", "OOS3"]
VALIDATE_KEY      = "OOS1"

# =============================================================================
# MATCH MODE
# 'exact'        — combo must be identical in at least MIN_PERIODS_MATCH periods
# 'intersection' — bins common to ALL winning combos across train periods
# =============================================================================
MATCH_MODE        = 'exact'
MIN_PERIODS_MATCH = 2

# =============================================================================
# REGIME SOURCE
# =============================================================================
REGIME_SOURCE = 'symbol_strategy_tf'

MA_PERIOD     = 20
ER_WINDOW     = 14
ATR_WINDOW    = 14
LOOKBACK_BARS = 50
ORDER_AMOUNT  = 80
ER_TH         = 0.4
ATR_TH        = 2.0

MIN_SIGNALS_PER_BIN = 50

ALL_BINS = [
    'trending_uptrend', 'trending_dwtrend',
    'ranging_uptrend',  'ranging_dwtrend',
    'volatile_uptrend', 'volatile_dwtrend',
]

ALL_BIN_COMBOS = [set()] + [
    set(combo)
    for r in range(1, len(ALL_BINS) + 1)
    for combo in combinations(ALL_BINS, r)
]

_BIN_ABBREV = {
    'trending_uptrend': 'trd_up', 'trending_dwtrend': 'trd_dw',
    'ranging_uptrend':  'rng_up', 'ranging_dwtrend':  'rng_dw',
    'volatile_uptrend': 'vol_up', 'volatile_dwtrend': 'vol_dw',
}

# =============================================================================
# HELPERS
# =============================================================================

def _abbrev_bins(bins: set) -> str:
    if not bins:
        return '—'
    return ', '.join(_BIN_ABBREV.get(b, b) for b in sorted(bins))


def build_families() -> dict:
    return {
        'trending': {'efficiency_ratio': ('>', ER_TH)},
        'volatile': {'atr_pct': ('>', ATR_TH)},
        'ranging':  {},
    }


def _is_daily_source() -> bool:
    return REGIME_SOURCE in ('btc_daily', 'symbol_daily')


def _is_symbol_source() -> bool:
    return REGIME_SOURCE in ('symbol_strategy_tf', 'symbol_daily')


def _pct_improvement(profit_filtered: float, profit_baseline: float) -> float:
    if profit_baseline == 0:
        return 0.0
    return (profit_filtered - profit_baseline) / abs(profit_baseline) * 100

# =============================================================================
# CONFIG LOADERS
# =============================================================================

def load_strategies_config() -> list[dict]:
    spec   = spec_from_file_location(STRATEGIES_LOOP_NAME, STRATEGIES_LOOP_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    strategies = []
    for entry in module.STRATEGIES_LOOP:
        strategy_id = entry["id"]
        signal_key  = "_".join(strategy_id.split("_")[1:-1])

        if signal_key not in SIGNAL_REGISTRY:
            signal_key = "_".join(strategy_id.split("_")[:-1])
        if signal_key not in SIGNAL_REGISTRY:
            print(f"  ⚠️  '{signal_key}' not in SIGNAL_REGISTRY — skipping {strategy_id}")
            continue

        registry      = SIGNAL_REGISTRY[signal_key]
        signal_fn     = registry["fn"]
        param_keys    = registry["params"]
        param_grid    = entry["param_grid"]
        best_params   = {k.upper(): v[0] for k, v in param_grid.items()}
        signal_params = {k: best_params[k.upper()] for k in param_keys if k.upper() in best_params}
        timeframe     = strategy_id.split("_")[-1]

        strategies.append({
            "id":            strategy_id,
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
# REGIME REF CACHE
# =============================================================================

_regime_ref_cache: dict = {}

def _get_regime_ref(data_folder: str, timeframe: str, sym: str) -> tuple[pd.DataFrame, dict]:
    tf_key = '1Dutc' if _is_daily_source() else timeframe
    key    = (data_folder, tf_key, sym if _is_symbol_source() else 'BTCUSDT')

    if key in _regime_ref_cache:
        return _regime_ref_cache[key]

    ref_cache = {}
    ref_sym   = sym if _is_symbol_source() else 'BTCUSDT'
    ref_df    = load_reference_symbol_for_timeframe(data_folder, ref_sym, tf_key, ref_cache)
    metrics_cache = build_metrics_cache(
        ref_df=ref_df, lookback=LOOKBACK_BARS, er_window=ER_WINDOW, atr_window=ATR_WINDOW,
    )
    _regime_ref_cache[key] = (ref_df, metrics_cache)
    return ref_df, metrics_cache


# =============================================================================
# SIGNAL CLASSIFICATION
# =============================================================================

def _classify_signals(
    ohlcv_arrays:  dict,
    signal_fn,
    signal_params: dict,
    data_folder:   str,
    timeframe:     str,
    families:      dict,
) -> dict[str, dict]:
    result = {}
    for sym, arr in ohlcv_arrays.items():
        signals     = signal_fn(arr, **signal_params, live_trading=False)
        signal_idxs = np.nonzero(signals)[0]

        ref_df, metrics_cache = _get_regime_ref(data_folder, timeframe, sym)
        trade_times           = pd.Series(pd.to_datetime(arr['ts'][signal_idxs]))
        direction_cache       = build_direction_cache(
            ref_df, MA_PERIOD, trade_times, is_daily=_is_daily_source(),
        )

        signal_bins: dict[int, str] = {}
        for idx in signal_idxs:
            t            = pd.Timestamp(arr['ts'][idx])
            direction, _ = direction_cache.get(t, ('unknown', None))
            metrics      = metrics_cache.get(t)
            family       = classify_trade_by_family(metrics, families) if metrics else 'unknown'
            if family != 'unknown' and direction not in ('unknown', 'neutral'):
                signal_bins[int(idx)] = f"{family}_{direction}"

        result[sym] = {'signals': signals, 'signal_bins': signal_bins, 'arr': arr}

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


# =============================================================================
# BACKTEST
# =============================================================================

def _run_backtest_with_bins(
    classified:     dict[str, dict],
    bins_to_filter: set,
    best_params:    dict,
) -> dict:
    arrays = {}
    for sym, data in classified.items():
        signals = data['signals'].copy()
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

    profits  = trades['profit']
    profit   = float(profits.sum())
    win_rate = float((profits > 0).mean() * 100)
    n        = len(profits)
    equity   = INITIAL_BALANCE + profits.cumsum()
    max_dd   = float(((equity - equity.cummax()) / equity.cummax()).min() * 100)
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

    ohlcv_arrays     = prepare_ohlcv_arrays(ohlcv_data)
    classified       = _classify_signals(
        ohlcv_arrays  = ohlcv_arrays,
        signal_fn     = strategy['signal_fn'],
        signal_params = strategy['signal_params'],
        data_folder   = data_folder,
        timeframe     = strategy['timeframe'],
        families      = families,
    )
    baseline_metrics = _run_backtest_with_bins(classified, set(), strategy['best_params'])

    return {
        'classified':      classified,
        'best_params':     strategy['best_params'],
        'profit_baseline': baseline_metrics['profit'],
        'wr_baseline':     baseline_metrics['win_rate'],
        'n_trades_baseline': baseline_metrics['n_trades'],
        'dd_baseline':     baseline_metrics['max_dd'],
    }


# =============================================================================
# BEST COMBO PER PERIOD
# =============================================================================

def _find_best_combo(period_data: dict) -> tuple[set, float]:
    """Find the bin combo that maximizes profit for a single period."""
    best_bins   = set()
    best_profit = 0.0

    for bins_combo in ALL_BIN_COMBOS:
        m = _run_backtest_with_bins(period_data['classified'], bins_combo, period_data['best_params'])
        if m['n_trades'] < MIN_SIGNALS_PER_BIN:
            continue
        if m['profit'] > best_profit:
            best_profit = m['profit']
            best_bins   = bins_combo

    return best_bins, best_profit


# =============================================================================
# DERIVE FINAL BINS FROM MATCH MODE
# =============================================================================

def _derive_final_bins(
    best_combos:       dict[str, set],
    match_mode:        str,
    min_periods_match: int,
) -> set:
    """
    'exact'        — return combo that appears in at least min_periods_match periods.
                     If tie, pick combo with most periods. If none qualifies, return empty.
    'intersection' — return bins common to all winning combos.
    """
    if match_mode == 'intersection':
        if not best_combos:
            return set()
        combos = list(best_combos.values())
        common = combos[0].copy()
        for c in combos[1:]:
            common &= c
        return common

    # exact mode — count how many periods each combo appears in
    combo_counts: dict[frozenset, int] = {}
    for combo in best_combos.values():
        key = frozenset(combo)
        combo_counts[key] = combo_counts.get(key, 0) + 1

    best_key   = None
    best_count = 0
    for key, count in combo_counts.items():
        if count > best_count:
            best_count = count
            best_key   = key

    if best_key is not None and best_count >= min_periods_match:
        return set(best_key)
    return set()


# =============================================================================
# MAIN RUN
# =============================================================================

def run_cross_period_search(train_keys: list[str], val_key: str) -> None:
    _t0      = time.time()
    families = build_families()

    print(f"\n{'='*100}")
    print(f"  REGIME CROSS-PERIOD BIN SEARCH — Train: {' + '.join(train_keys)}  →  Validate: {val_key}")
    print(f"  Source: {REGIME_SOURCE} | MA{MA_PERIOD} | ER>{ER_TH} | ATR>{ATR_TH}")
    print(f"  Match mode: {MATCH_MODE}" + (f" | min_periods={MIN_PERIODS_MATCH}" if MATCH_MODE == 'exact' else ""))
    print(f"  Bin combinations: {len(ALL_BIN_COMBOS)}")
    print(f"{'='*100}\n")

    strategies_all = load_strategies_config()
    if not strategies_all:
        print("  No strategies found — aborting.")
        return

    bins_per_strategy:       dict[str, set]   = {}
    train_pct_per_strategy:  dict[str, float] = {}

    by_tf: dict[str, list] = {}
    for s in strategies_all:
        by_tf.setdefault(s['timeframe'], []).append(s)

    for tf, strategies in sorted(by_tf.items()):
        print(f"\n{'─'*100}")
        print(f"  TF: {tf} | {len(strategies)} strategies | {len(ALL_BIN_COMBOS)} combos × {len(train_keys)} periods")
        print(f"{'─'*100}")

        for strategy in strategies:
            sid         = strategy['id']
            best_combos : dict[str, set]   = {}
            best_profits: dict[str, float] = {}

            for period_key in train_keys:
                data = _load_period_data(strategy, period_key, families)
                if not data:
                    continue
                best_bins, best_profit      = _find_best_combo(data)
                best_combos[period_key]     = best_bins
                best_profits[period_key]    = best_profit

            if not best_combos:
                bins_per_strategy[sid]      = set()
                train_pct_per_strategy[sid] = 0.0
                continue

            final_bins             = _derive_final_bins(best_combos, MATCH_MODE, MIN_PERIODS_MATCH)
            bins_per_strategy[sid] = final_bins

            # Per-period results
            period_str = "  ".join(
                f"{p}={_abbrev_bins(best_combos.get(p, set()))}" for p in train_keys
            )
            consistent = len(final_bins) > 0
            color      = "\033[92m" if consistent else "\033[91m"
            reset      = "\033[0m"
            icon       = "✅" if consistent else "❌"

            avg_pct = 0.0
            if best_profits:
                avg_pct = sum(best_profits.values()) / len(best_profits)
            train_pct_per_strategy[sid] = avg_pct

            print(f"  {sid:<35} {color}{icon}{reset}  final={_abbrev_bins(final_bins)}")
            for p in train_keys:
                combo = best_combos.get(p)
                if combo is not None:
                    print(f"    {p:<8} best={_abbrev_bins(combo)}")

    # Validation on OOS1
    oos1_pct_per_strategy = _run_validation_period(strategies_all, val_key, bins_per_strategy, families)

    # Consistency table
    _print_consistency_table(strategies_all, train_pct_per_strategy, oos1_pct_per_strategy, bins_per_strategy)

    # Persist
    _save_bins(bins_per_strategy, train_keys)

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

    sys_b_profit          = 0.0
    sys_f_profit          = 0.0
    pct_imp_per_strategy  : dict[str, float] = {}
    rows                  = []

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
        m_f  = _run_backtest_with_bins(data['classified'], bins, data['best_params']) if bins else m_b

        dwr   = m_f['win_rate'] - m_b['win_rate']
        dpct  = _pct_improvement(m_f['profit'], m_b['profit'])
        color = "\033[92m" if dpct > 0 else "\033[91m" if dpct < 0 else ""
        reset = "\033[0m"

        print(f"  {sid:<35} {m_b['win_rate']:>6.1f}% {m_f['win_rate']:>6.1f}% "
              f"{dwr:>+6.1f}% {m_b['profit']:>8.1f} {m_f['profit']:>8.1f} "
              f"{color}{dpct:>+6.1f}%{reset} {m_b['max_dd']:>6.1f}% {m_f['max_dd']:>6.1f}%  {_abbrev_bins(bins)}")

        pct_imp_per_strategy[sid]  = dpct
        sys_b_profit              += m_b['profit']
        sys_f_profit              += m_f['profit']
        rows.append({'b_wr': m_b['win_rate'], 'f_wr': m_f['win_rate'], 'dwr': dwr,
                     'b_dd': m_b['max_dd'],   'f_dd': m_f['max_dd']})

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
    print(f"  {'STRATEGY':<35} {'TRAIN_AVG':>10} {'OOS1_Δ%':>10} {'OK':>5}  {'BINS'}")
    print(f"  {'─'*80}")
    consistent   = 0
    inconsistent = 0
    for strategy in strategies:
        sid   = strategy['id']
        t_pct = train_pct.get(sid, 0.0)
        o_pct = oos1_pct.get(sid, 0.0)
        bins  = bins_per_strategy.get(sid, set())
        ok    = len(bins) > 0 and o_pct > 0
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
        f"# Auto-generated by regime06_cross_period_search.py",
        f"# Source: {REGIME_SOURCE} | MA{MA_PERIOD} | ER>{ER_TH} | ATR>{ATR_TH}",
        f"# Train: {' + '.join(train_keys)}",
        f"# Match mode: {MATCH_MODE}" + (f" | min_periods={MIN_PERIODS_MATCH}" if MATCH_MODE == 'exact' else ""),
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
    run_cross_period_search(TRAIN_KEYS, VALIDATE_KEY)