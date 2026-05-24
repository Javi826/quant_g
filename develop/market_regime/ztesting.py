"""
Regime combo analysis — compare profit of two specific bin combinations for a strategy.

Configure STRATEGY_ID, COMBO_A, COMBO_B and PERIOD to analyze any strategy.
"""
import os
import sys
import numpy as np
import pandas as pd

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

STRATEGY_ID = "01_reversal_long_15m"

COMBO_A = {'ranging_dwtrend', 'ranging_uptrend', 'volatile_dwtrend'}
COMBO_B = {'ranging_dwtrend', 'ranging_uptrend', 'volatile_dwtrend', 'trending_dwtrend'}
COMBO_C = {'ranging_dwtrend', 'ranging_uptrend', 'trending_dwtrend'}

PERIOD = "IS"

STRATEGIES_SET_NAME  = "E1"
SYMBOLS_LIVE_FOLDER  = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1", "strategies_E1", "symbols_live")
STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_01"
STRATEGIES_LOOP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1",
    "strategies_files", f"files_{STRATEGIES_SET_NAME}",
    f"{STRATEGIES_LOOP_NAME}.py"
)

SPLIT_MODE = "expanding"
SPLIT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split_OLD", SPLIT_MODE)

PERIODS = {
    "IS":   os.path.join(SPLIT_BASE, "IS",  "crypto_2024-01_2025-05_IS"),
    "OOS1": os.path.join(SPLIT_BASE, "OOS", "crypto_2025-05_2026-05_OOS"),
    "OOS2": os.path.join(SPLIT_BASE, "OOS", "crypto_2022-01_2023-01_OOS"),
    "OOS3": os.path.join(SPLIT_BASE, "OOS", "crypto_2023-01_2024-01_OOS"),
}

REGIME_SOURCE = 'symbol_strategy_tf'
MA_PERIOD     = 20
ER_WINDOW     = 14
ATR_WINDOW    = 14
LOOKBACK_BARS = 50
ORDER_AMOUNT  = 80
ER_TH         = 0.4
ATR_TH        = 2.0

MIN_SIGNALS_PER_BIN = 10

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


# =============================================================================
# LOADERS
# =============================================================================

def load_strategy_config(strategy_id: str) -> dict | None:
    spec   = spec_from_file_location(STRATEGIES_LOOP_NAME, STRATEGIES_LOOP_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    for entry in module.STRATEGIES_LOOP:
        if entry["id"] != strategy_id:
            continue
        signal_key = "_".join(strategy_id.split("_")[1:-1])
        if signal_key not in SIGNAL_REGISTRY:
            signal_key = "_".join(strategy_id.split("_")[:-1])
        if signal_key not in SIGNAL_REGISTRY:
            print(f"  ⚠️  '{signal_key}' not in SIGNAL_REGISTRY")
            return None

        registry      = SIGNAL_REGISTRY[signal_key]
        param_grid    = entry["param_grid"]
        best_params   = {k.upper(): v[0] for k, v in param_grid.items()}
        signal_params = {k: best_params[k.upper()] for k in registry["params"] if k.upper() in best_params}

        return {
            "id":            strategy_id,
            "timeframe":     strategy_id.split("_")[-1],
            "signal_fn":     registry["fn"],
            "signal_params": signal_params,
            "best_params":   best_params,
        }
    print(f"  ⚠️  {strategy_id} not found in strategies loop")
    return None


def load_symbols(strategy_id: str, timeframe: str) -> list[str]:
    filename = f"symbols_live_{strategy_id}_{timeframe}.csv"
    filepath = os.path.join(SYMBOLS_LIVE_FOLDER, filename)
    if not os.path.exists(filepath):
        return []
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


# =============================================================================
# REGIME REF
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
# CLASSIFY SIGNALS
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
# BACKTEST WITH BINS
# =============================================================================

def _run_backtest(
    classified:     dict[str, dict],
    bins_to_filter: set,
    best_params:    dict,
    label:          str,
) -> dict:
    arrays = {}
    for sym, data in classified.items():
        signals = data['signals'].copy()
        for idx, bin_key in data['signal_bins'].items():
            if bin_key in bins_to_filter:
                signals[idx] = 0
        arrays[sym] = {**data['arr'], 'signal': signals}

    result   = run_grid_backtest(
        arrays,
        sell_after   = best_params['SELL_AFTER'],
        tp_pct       = best_params['TP_PCT'],
        sl_pct       = best_params['SL_PCT'],
        order_amount = ORDER_AMOUNT,
    )
    trades   = result['__PORTFOLIO__']['trade_log']
    n        = len(trades)
    profit   = float(trades['profit'].sum()) if n > 0 else 0.0
    win_rate = float((trades['profit'] > 0).mean() * 100) if n > 0 else 0.0
    equity   = INITIAL_BALANCE + trades['profit'].cumsum() if n > 0 else None
    max_dd   = float(((equity - equity.cummax()) / equity.cummax()).min() * 100) if equity is not None else 0.0

    return {'label': label, 'bins': bins_to_filter, 'profit': profit, 'win_rate': win_rate, 'n_trades': n, 'max_dd': max_dd}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    families = build_families()

    print(f"\n{'='*80}")
    print(f"  COMBO ANALYSIS — {STRATEGY_ID} | Period: {PERIOD}")
    print(f"  Source: {REGIME_SOURCE} | MA{MA_PERIOD} | ER>{ER_TH} | ATR>{ATR_TH}")
    print(f"{'='*80}\n")

    strategy = load_strategy_config(STRATEGY_ID)
    if not strategy:
        sys.exit(1)

    symbols = load_symbols(STRATEGY_ID, strategy["timeframe"])
    if not symbols:
        print(f"  ⚠️  No symbols found for {STRATEGY_ID}")
        sys.exit(1)

    data_folder   = PERIODS[PERIOD]
    ohlcv_data, _ = filter_symbols(
        symbols, min_vol_usdt=0, timeframe=strategy["timeframe"],
        data_folder=data_folder, min_price=None, vol_window=50,
        my_symbols=True, custom_symbols=symbols,
    )
    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_data)
    classified   = _classify_signals(
        ohlcv_arrays  = ohlcv_arrays,
        signal_fn     = strategy['signal_fn'],
        signal_params = strategy['signal_params'],
        data_folder   = data_folder,
        timeframe     = strategy['timeframe'],
        families      = families,
    )

    results = [
        _run_backtest(classified, set(),    strategy['best_params'], "Baseline"),
        _run_backtest(classified, COMBO_A,  strategy['best_params'], f"Combo A — regime03 winner ({_abbrev_bins(COMBO_A)})"),
        _run_backtest(classified, COMBO_B,  strategy['best_params'], f"Combo B — regime03 + trd_dw ({_abbrev_bins(COMBO_B)})"),
        _run_backtest(classified, COMBO_C,  strategy['best_params'], f"Combo C — regime04 toxic only ({_abbrev_bins(COMBO_C)})"),
    ]

    print(f"  {'COMBO':<50} {'PROFIT':>10} {'WIN_RATE':>10} {'N_TRADES':>10} {'MAX_DD':>10}")
    print(f"  {'─'*90}")
    for r in results:
        print(f"  {r['label']:<50} {r['profit']:>10.1f} {r['win_rate']:>9.1f}% {r['n_trades']:>10} {r['max_dd']:>9.1f}%")

    diff_ba = results[1]['profit'] - results[3]['profit']
    print(f"\n  Profit diff (A - C): {diff_ba:>+.1f}  ({'regime03 better' if diff_ba > 0 else 'regime04 better'})")
    print()