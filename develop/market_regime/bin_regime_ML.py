#develop/market_regime/bin_regime_ML.py
"""
Simplified ML regime filter.

Pipeline:
  1. Train XGBoost model on IS (BTC 1D)
  2. For each period (OOS1, OOS2, OOS3):
     - Long  signals pass only when model predicts uptrend  (prob_up  >= ML_THRESHOLD_UP)
     - Short signals pass only when model predicts dwtrend  (prob_down >= ML_THRESHOLD_DOWN)
  3. Compare baseline vs filtered per period
  4. Persist REGIME_BINS
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batchs")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.pipeline.universe import filter_symbols
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from importlib.util import spec_from_file_location, module_from_spec

# =============================================================================
# CONFIGURATION
# =============================================================================

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split_OLD", "expanding")

PERIODS = {
    "IS":   os.path.join(_BASE, "IS",  "crypto_2024-01_2025-05_IS"),
    "OOS1": os.path.join(_BASE, "OOS", "crypto_2025-05_2026-05_OOS"),
    "OOS2": os.path.join(_BASE, "OOS", "crypto_2022-01_2023-01_OOS"),
    "OOS3": os.path.join(_BASE, "OOS", "crypto_2023-01_2024-01_OOS"),
}

STRATEGIES_SET_NAME  = "E1"
SYMBOLS_LIVE_FOLDER  = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1", "strategies_E1", "symbols_live")
BINS_OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), f"regime_bins_06_{STRATEGIES_SET_NAME}_ML_simple.py")

STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_01"
STRATEGIES_LOOP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch_E1",
    "strategies_files", f"{STRATEGIES_LOOP_NAME}.py"
)

MODEL_TRAIN_KEY = "IS"
EVAL_KEYS       = ["OOS2", "OOS3", "OOS1"]

# =============================================================================
# ML CONFIGURATION
# =============================================================================

BTC_SYMBOL    = "BTCUSDT"
BTC_TIMEFRAME = "1Dutc"

ML_THRESHOLD_UP   = 0.55   # prob_up  >= this → uptrend  → longs pass
ML_THRESHOLD_DOWN = 0.50   # prob_down >= this → dwtrend → shorts pass

ML_TOP_N_FEATURES = 15

# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

ORDER_AMOUNT        = 80
MIN_SIGNALS_PER_BIN = 50

# Strategy name contains 'long' or 'short' — used to determine direction
LONG_KEYWORD  = "long"
SHORT_KEYWORD = "short"

# Set to e.g. ["4H"] to run only that timeframe — empty list runs all
DEBUG_TF_FILTER: list[str] = []

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = logging.DEBUG
logging.basicConfig(format="%(message)s", level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================

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
        is_long       = LONG_KEYWORD in strategy_id

        strategies.append({
            "id":            strategy_id,
            "timeframe":     timeframe,
            "signal_fn":     signal_fn,
            "signal_params": signal_params,
            "best_params":   best_params,
            "is_long":       is_long,
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
# BTC DATA LOADING
# =============================================================================

def _load_btc_data(period_keys: list[str]) -> pd.DataFrame:
    frames = []
    for period_key in period_keys:
        folder = PERIODS[period_key]
        candidates = [
            os.path.join(folder, f"{BTC_SYMBOL}_{BTC_TIMEFRAME}.parquet"),
            os.path.join(folder, f"{BTC_SYMBOL}_{BTC_TIMEFRAME}.csv"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            continue
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No BTC {BTC_TIMEFRAME} data found.")

    df = pd.concat(frames)
    df.columns = [c.lower().strip() for c in df.columns]

    if df.index.name and df.index.name.lower() in ("timestamp", "ts", "date", "time"):
        df.index.name = "ts"
        df = df.reset_index()

    if "volume_base" in df.columns and "volume_quote" in df.columns:
        df.drop(columns=["volume_base"], inplace=True)

    rename_map = {
        "timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts",
        "volume_quote": "volume", "vol": "volume",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")

    df.dropna(subset=["ts"], inplace=True)
    df.sort_values("ts", inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _build_features(df: pd.DataFrame, with_label: bool = True) -> tuple[pd.DataFrame, list]:
    d = df.copy()

    d["ret_1d"]  = d["close"].pct_change(1)
    d["ret_2d"]  = d["close"].pct_change(2)
    d["ret_3d"]  = d["close"].pct_change(3)
    d["ret_5d"]  = d["close"].pct_change(5)
    d["ret_10d"] = d["close"].pct_change(10)

    for w in [5, 10, 20, 50, 100, 200]:
        d[f"ma_{w}"]       = d["close"].rolling(w).mean()
        d[f"ma_ratio_{w}"] = d["close"] / d[f"ma_{w}"]

    d["ma_5_20_cross"]  = d["ma_5"]  / d["ma_20"]
    d["ma_20_50_cross"] = d["ma_20"] / d["ma_50"]
    d["atr_14"]         = _atr(d, 14)
    d["atr_pct_14"]     = d["atr_14"] / d["close"]
    d["std_5d"]         = d["ret_1d"].rolling(5).std()
    d["std_20d"]        = d["ret_1d"].rolling(20).std()
    d["rsi_7"]          = _rsi(d["close"], 7)
    d["rsi_14"]         = _rsi(d["close"], 14)
    d["rsi_21"]         = _rsi(d["close"], 21)
    d["vol_ma_20"]      = d["volume"].rolling(20).mean()
    d["vol_ratio"]      = d["volume"] / d["vol_ma_20"]
    d["vol_ret_1d"]     = d["volume"].pct_change(1)
    d["body_pct"]       = (d["close"] - d["open"]).abs() / d["open"]
    d["upper_wick_pct"] = (d["high"] - d[["open", "close"]].max(axis=1)) / d["open"]
    d["lower_wick_pct"] = (d[["open", "close"]].min(axis=1) - d["low"]) / d["open"]
    d["is_bullish"]     = (d["close"] > d["open"]).astype(int)
    d["hl_range"]       = d["high"] - d["low"]
    d["close_position"] = (d["close"] - d["low"]) / d["hl_range"].replace(0, np.nan)

    if with_label:
        d["close_next"] = d["close"].shift(-1)
        d["label"]      = (d["close_next"] > d["close"]).astype(float)
        d.loc[d["close_next"].isna(), "label"] = float("nan")
        d.drop(columns=["close_next"], inplace=True)

    feature_cols = [
        c for c in d.columns
        if c not in ("ts", "label", "open", "high", "low", "close", "volume",
                     "volume_quote", "low_time", "high_time", "_period")
    ]

    d.dropna(inplace=True)
    if with_label:
        d["label"] = d["label"].astype(int)
    d.reset_index(drop=True, inplace=True)

    return d, feature_cols


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def _select_features(df: pd.DataFrame, feature_cols: list) -> list:
    from sklearn.feature_selection import mutual_info_classif

    X       = df[feature_cols]
    y       = df["label"]
    pearson = X.corrwith(y).abs().rename("pearson")
    mi      = pd.Series(mutual_info_classif(X, y, random_state=42), index=feature_cols, name="mi")
    ranking = pd.concat([pearson, mi], axis=1)
    ranking["score"] = ranking["pearson"].rank() + ranking["mi"].rank()
    ranking.sort_values("score", ascending=False, inplace=True)
    return ranking.head(ML_TOP_N_FEATURES).index.tolist()


# =============================================================================
# MODEL TRAINING
# =============================================================================

def _train_model(df: pd.DataFrame, feature_cols: list):
    from xgboost import XGBClassifier

    X                = df[feature_cols].values
    y                = df["label"].values
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X, y)
    return model


# =============================================================================
# ML MODEL BOOTSTRAP
# =============================================================================

def build_ml_model() -> tuple:
    """Train on MODEL_TRAIN_KEY, build feature lookup from all periods."""
    logger.info(f"  Building ML model | train: {MODEL_TRAIN_KEY}...")

    btc_train_df              = _load_btc_data([MODEL_TRAIN_KEY])
    btc_features_df, all_cols = _build_features(btc_train_df, with_label=True)
    selected_cols             = _select_features(btc_features_df, all_cols)
    model                     = _train_model(btc_features_df, selected_cols)

    # Feature lookup from all periods for inference
    btc_all_df             = _load_btc_data(list(PERIODS.keys()))
    btc_all_features, _    = _build_features(btc_all_df, with_label=False)
    feature_lookup         = btc_all_features[["ts"] + selected_cols].copy()
    feature_lookup["ts"]   = feature_lookup["ts"].dt.normalize()
    feature_lookup.set_index("ts", inplace=True)

    logger.info(f"  ✅ Model trained | features: {selected_cols}")
    logger.info(f"  Thresholds — UP: {ML_THRESHOLD_UP} | DOWN: {ML_THRESHOLD_DOWN}")
    logger.info(f"  Model train range : {btc_features_df['ts'].min().date()} → {btc_features_df['ts'].max().date()}")
    logger.info(f"  Feature lookup    : {feature_lookup.index.min().date()} → {feature_lookup.index.max().date()}")
    return model, feature_lookup, selected_cols


# =============================================================================
# BATCH DIRECTION PRE-COMPUTATION
# =============================================================================

def precompute_directions(model, feature_lookup: pd.DataFrame) -> dict[pd.Timestamp, str | None]:
    """
    Run predict_proba once over the entire feature_lookup in batch.
    Returns a dict mapping each date (normalized UTC) → direction string or None.
    This replaces per-signal calls to _predict_direction entirely.
    """
    X          = feature_lookup.values.astype(float)
    probs_up   = model.predict_proba(X)[:, 1]
    probs_down = 1.0 - probs_up

    directions: dict[pd.Timestamp, str | None] = {}
    for date, prob_up, prob_down in zip(feature_lookup.index, probs_up, probs_down):
        if prob_up >= ML_THRESHOLD_UP:
            directions[date] = "uptrend"
        elif prob_down >= ML_THRESHOLD_DOWN:
            directions[date] = "dwtrend"
        else:
            directions[date] = None

    logger.info(f"  ✅ Direction lookup pre-computed | {len(directions)} dates")
    return directions


# =============================================================================
# SIGNAL DIRECTION LOOKUP
# =============================================================================

def _lookup_direction(
    direction_map: dict[pd.Timestamp, str | None],
    signal_ts: pd.Timestamp,
) -> str | None:
    """Return pre-computed direction for signal at T using BTC candle T-1."""
    ts = pd.Timestamp(signal_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    prev_date = ts.normalize() - pd.Timedelta(days=1)
    direction = direction_map.get(prev_date)

    logger.debug(f"  [DEBUG] signal={ts.date()} | prev={prev_date.date()} | → {direction}")
    return direction


# =============================================================================
# BACKTEST
# =============================================================================

def _run_backtest(ohlcv_arrays: dict, best_params: dict) -> dict:
    result = run_grid_backtest(
        ohlcv_arrays,
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
# PERIOD EVALUATION
# =============================================================================

def _evaluate_period(
    strategy:       dict,
    period_key:     str,
    direction_map:  dict[pd.Timestamp, str | None],
) -> dict | None:
    import time as _time
    _t0 = _time.time()

    data_folder = PERIODS[period_key]
    symbols     = load_symbols(strategy["id"], strategy["timeframe"])
    if not symbols:
        return None

    _t1 = _time.time()
    ohlcv_data, _ = filter_symbols(
        symbols, min_vol_usdt=0, timeframe=strategy["timeframe"],
        data_folder=data_folder, min_price=None, vol_window=50,
        my_symbols=True, custom_symbols=symbols,
    )
    _t_filter = _time.time() - _t1
    if not ohlcv_data:
        return None

    _t2 = _time.time()
    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_data)
    _t_prepare = _time.time() - _t2

    is_long      = strategy["is_long"]

    baseline_arrays  = {}
    filtered_arrays  = {}
    n_total = n_filtered = 0

    _t3 = _time.time()
    for sym, arr in ohlcv_arrays.items():
        signals     = strategy['signal_fn'](arr, **strategy['signal_params'], live_trading=False)
        signal_idxs = np.nonzero(signals)[0]

        filtered_signals = signals.copy()
        for idx in signal_idxs:
            ts        = pd.Timestamp(arr['ts'][idx])
            direction = _lookup_direction(direction_map, ts)
            n_total  += 1
            if is_long and direction != 'uptrend':
                filtered_signals[idx] = 0
                n_filtered += 1
            elif not is_long and direction != 'dwtrend':
                filtered_signals[idx] = 0
                n_filtered += 1

        baseline_arrays[sym] = {**arr, 'signal': signals}
        filtered_arrays[sym] = {**arr, 'signal': filtered_signals}

    _t_classify = _time.time() - _t3

    _t4 = _time.time()
    m_b = _run_backtest(baseline_arrays, strategy['best_params'])
    _t_bt_base = _time.time() - _t4

    _t5 = _time.time()
    m_f = _run_backtest(filtered_arrays, strategy['best_params'])
    _t_bt_filt = _time.time() - _t5

    _t_total = _time.time() - _t0
    logger.debug(
        f"  [TIMER] {strategy['id']} {period_key} | "
        f"filter_symbols={_t_filter:.2f}s | prepare={_t_prepare:.2f}s | "
        f"classify={_t_classify:.2f}s | bt_base={_t_bt_base:.2f}s | bt_filt={_t_bt_filt:.2f}s | "
        f"total={_t_total:.2f}s"
    )

    return {
        'baseline': m_b,
        'filtered': m_f,
        'n_total':    n_total,
        'n_filtered': n_filtered,
    }


# =============================================================================
# MAIN RUN
# =============================================================================

def run(eval_keys: list[str]) -> None:
    _t0 = time.time()

    print(f"\n{'='*110}")
    print(f"  REGIME ML SIMPLE — Model train: {MODEL_TRAIN_KEY}  |  Eval: {' + '.join(eval_keys)}")
    print(f"  Thresholds — UP: {ML_THRESHOLD_UP} | DOWN: {ML_THRESHOLD_DOWN}")
    print(f"{'='*110}\n")

    model, feature_lookup, _ = build_ml_model()

    # Pre-compute all directions in a single batch predict_proba call
    direction_map = precompute_directions(model, feature_lookup)

    strategies_all = load_strategies_config()
    if not strategies_all:
        print("  No strategies found — aborting.")
        return

    bins_per_strategy: dict[str, str | None] = {}

    for period_key in eval_keys:
        print(f"\n{'='*110}")
        print(f"  PERIOD: {period_key}")
        print(f"{'='*110}")
        print(f"  {'STRATEGY':<35} {'B_WR%':>7} {'F_WR%':>7} {'ΔWR':>7} {'B_PROF':>8} {'F_PROF':>8} {'Δ%':>7} {'B_DD%':>7} {'F_DD%':>7} {'FILTERED':>9}")
        print(f"  {'─'*115}")

        sys_b = sys_f = 0.0
        rows  = []

        for strategy in strategies_all:
            if DEBUG_TF_FILTER and strategy['timeframe'] not in DEBUG_TF_FILTER:
                continue

            sid    = strategy['id']
            result = _evaluate_period(strategy, period_key, direction_map)
            if not result:
                continue

            m_b   = result['baseline']
            m_f   = result['filtered']
            dwr   = m_f['win_rate'] - m_b['win_rate']
            dpct  = _pct_improvement(m_f['profit'], m_b['profit'])
            pct_filtered = result['n_filtered'] / max(result['n_total'], 1) * 100
            color = "\033[92m" if dpct > 0 else "\033[91m" if dpct < 0 else ""
            reset = "\033[0m"

            print(f"  {sid:<35} {m_b['win_rate']:>6.1f}% {m_f['win_rate']:>6.1f}% "
                  f"{dwr:>+6.1f}% {m_b['profit']:>8.1f} {m_f['profit']:>8.1f} "
                  f"{color}{dpct:>+6.1f}%{reset} {m_b['max_dd']:>6.1f}% {m_f['max_dd']:>6.1f}% "
                  f"{pct_filtered:>8.1f}%")

            sys_b += m_b['profit']
            sys_f += m_f['profit']
            rows.append({'b_wr': m_b['win_rate'], 'f_wr': m_f['win_rate'], 'dwr': dwr,
                         'b_dd': m_b['max_dd'],   'f_dd': m_f['max_dd']})

            if period_key == "OOS1":
                bins_per_strategy[sid] = 'uptrend' if strategy['is_long'] else 'dwtrend'

        sys_pct   = _pct_improvement(sys_f, sys_b)
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
                  f"{dwr_color}{avg_dwr:>+6.1f}%{reset} {sys_b:>8.1f} {sys_f:>8.1f} "
                  f"{color}{sys_pct:>+6.1f}%{reset} {avg_b_dd:>6.1f}% {avg_f_dd:>6.1f}%")

    _save_bins(bins_per_strategy)

    elapsed = int(time.time() - _t0)
    print(f"\n  ⏱  Completed in {elapsed//60}m {elapsed%60}s\n")


# =============================================================================
# PERSIST
# =============================================================================

def _save_bins(bins_per_strategy: dict) -> None:
    lines = [
        f"# Auto-generated by bin_regime_ML_simple.py",
        f"# Model trained on: {MODEL_TRAIN_KEY}",
        f"# Thresholds — UP: {ML_THRESHOLD_UP} | DOWN: {ML_THRESHOLD_DOWN}",
        f"",
        f"ML_THRESHOLD_UP   = {ML_THRESHOLD_UP}",
        f"ML_THRESHOLD_DOWN = {ML_THRESHOLD_DOWN}",
        f"",
        f"REGIME_BINS = {{",
    ]
    for sid, bin_key in sorted(bins_per_strategy.items()):
        lines.append(f'    "{sid}": "{bin_key}",')
    lines.append("}")

    with open(BINS_OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  ✅ Bins saved to: {BINS_OUTPUT_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run(EVAL_KEYS)