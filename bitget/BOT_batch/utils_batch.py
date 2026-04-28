#BOT_batch/batch_utils.py
import logging
import importlib.util
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from shared_market_regime.regime_common import load_btc_for_timeframe, calc_all_metrics_at_time,calculate_max_dd_pct
from shared_market_regime.regime_common import classify_trade_by_family, get_btc_macro_direction
from itertools import combinations as _combinations
from backtesters.ZX_compute_BT import INITIAL_BALANCE
from signals.add_signals_parity      import parity_long, parity_short
from signals.add_signals_reversal    import reversal_long, reversal_short
from signals.add_signals_flag        import flag_long, flag_short
from signals.add_signals_orderblocks import orderblocks_long, orderblocks_short

logger = logging.getLogger("BOT_batch.batch_utils")
SIGNAL_REGISTRY = {
    "parity_long":       {"fn": parity_long,       "params": ["lookback", "tolerance", "ma_period"]},
    "parity_short":      {"fn": parity_short,      "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_long":     {"fn": reversal_long,     "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_short":    {"fn": reversal_short,    "params": ["lookback", "tolerance", "ma_period"]},
    "flag_long":         {"fn": flag_long,         "params": ["lookback", "impulse", "flag", "ma_period"]},
    "flag_short":        {"fn": flag_short,        "params": ["lookback", "impulse", "flag", "ma_period"]},
    "orderblocks_long":  {"fn": orderblocks_long,  "params": ["lookback", "tolerance", "impulse"]},
    "orderblocks_short": {"fn": orderblocks_short, "params": ["lookback", "tolerance", "impulse"]},
}
# =============================================================================
# MODULE CONSTANTS
# =============================================================================
PARAM_KEYS        = {"lookback", "tolerance", "ma_period", "tp_pct", "sl_pct", "impulse", "flag"}
SIGNAL_PARAM_KEYS = tuple({p for entry in SIGNAL_REGISTRY.values() for p in entry["params"]})


# =============================================================================
# HELPER — ANALIZING REGIME
# =============================================================================

def analyze_regime_is(
    trade_log_is: pd.DataFrame,
    timeframe: str,
    data_folder_is: str,
    families: dict,
    regime_min_trades: int = 2,
    regime_lookback: int = 100,
    family_source: str = 'strategy',
    hurst_window: int = 100,
    er_window: int = 14,
    atr_window: int = 14,
    pe_window: int = 50,
    pe_order: int = 3,
    ma_period: int = 5,
    long_th: float = 1.0,
    short_th: float = 1.0,
    strategy_direction: str = 'long',
    force_direction_filter: bool = False,
    metrics_cache: dict = None,
) -> set:
    """
    Analyze regime on IS trades to determine bins to filter for OOS.
    Evaluates all 6 bins (3 families x 2 directions).
    Flags a bin if trades >= regime_min_trades and total profit < 0.

    Args:
        trade_log_is      : IS trades DataFrame with 'buy_time' and 'profit' columns
        timeframe         : strategy timeframe e.g. '4H', '1H', '6Hutc'
        data_folder_is    : path to IS data folder containing BTC parquets
        families          : family classification rules dict
        regime_min_trades : minimum trades per bin to trust result
        regime_lookback   : lookback bars for metric calculation
        family_source     : 'strategy' = BTC at strategy TF | 'macro' = BTC 1D
        hurst_window      : window for Hurst exponent
        er_window         : window for Efficiency Ratio
        atr_window        : window for ATR
        pe_window         : window for Permutation Entropy
        pe_order          : order for Permutation Entropy
        ma_period         : MA period for macro direction
        long_th           : multiplier threshold for uptrend
        short_th          : multiplier threshold for dwtrend
        strategy_direction: 'long' | 'short'
        force_direction_filter: if True, force filter by direction regardless of profit
        metrics_cache     : optional precomputed {timestamp: metrics} dict from build_metrics_cache

    Returns:
        bins_to_filter : set of bin keys to block e.g. {'trending_dwtrend', 'ranging_uptrend'}
    """
    btc_cache = {}
    btc_1d_df = load_btc_for_timeframe(data_folder_is, '1Dutc', btc_cache)
    btc_tf_df = load_btc_for_timeframe(data_folder_is, timeframe, btc_cache) \
                if family_source == 'strategy' else btc_1d_df

    directions = []
    families_  = []

# =============================================================================
#     # DEBUG no-lookahead check — remove after validation
#     first_trade = trade_log_is.iloc[0]
#     closed_1d = btc_1d_df[btc_1d_df['ts'] < first_trade['buy_time']]
#     closed_tf = btc_tf_df[btc_tf_df['ts'] < first_trade['buy_time']]
#     logger.info(f"DEBUG lookahead | buy_time={first_trade['buy_time']} | last 1D bar={closed_1d.iloc[-1]['ts']} | last {timeframe} bar={closed_tf.iloc[-1]['ts']}")
# 
# =============================================================================

    for _, trade in trade_log_is.iterrows():
        direction = get_btc_macro_direction(
            btc_1d_df  = btc_1d_df,
            trade_time = trade['buy_time'],
            ma_period  = ma_period,
            long_th    = long_th,
            short_th   = short_th,
        )

        if metrics_cache is not None:
            metrics = metrics_cache.get(pd.Timestamp(trade['buy_time']))
        else:
            metrics = calc_all_metrics_at_time(
                btc_df       = btc_tf_df,
                buy_time     = trade['buy_time'],
                lookback     = regime_lookback,
                hurst_window = hurst_window,
                er_window    = er_window,
                atr_window   = atr_window,
                pe_window    = pe_window,
                pe_order     = pe_order,
            )

        family = classify_trade_by_family(metrics, families) if metrics else 'unknown'
        directions.append(direction)
        families_.append(family)

    df = trade_log_is.copy()
    df['direction'] = directions
    df['family']    = families_

    df_valid = df[
        (df['family'] != 'unknown') &
        (df['direction'].isin(['uptrend', 'dwtrend']))
    ].copy()

    bins_to_filter = set()

    for family in ['trending', 'ranging', 'volatile']:
        for direction in ['uptrend', 'dwtrend']:
            subset = df_valid[(df_valid['family'] == family) & (df_valid['direction'] == direction)]
            n      = len(subset)
            profit = subset['profit'].sum() if n > 0 else 0.0
            if n >= regime_min_trades and profit < 0:
                bins_to_filter.add(f"{family}_{direction}")

    n_total    = len(trade_log_is)
    n_valid    = len(df_valid)
    n_filtered = df_valid[
        df_valid.apply(lambda r: f"{r['family']}_{r['direction']}" in bins_to_filter, axis=1)
    ].shape[0]
    pct_remain = round((n_valid - n_filtered) / n_valid * 100, 1) if n_valid > 0 else 0.0

    logger.info(
        f"STAGE 2  ── Regime IS Analysis     ── "
        f"total={n_total} | remaining={pct_remain}%"
    )

    if logger.isEnabledFor(logging.DEBUG):
        lines = []
        lines.append(f"\n  {'BIN':<30} {'CONF':>5} {'TRADES':>8} {'PROFIT':>12} {'WIN%':>8} {'DD%':>8} {'FILTER':>8}")
        lines.append("  " + "-" * 88)
        for fam in ['trending', 'ranging', 'volatile']:
            for dir_ in ['uptrend', 'dwtrend']:
                bin_key = f"{fam}_{dir_}"
                subset  = df_valid[(df_valid['family'] == fam) & (df_valid['direction'] == dir_)]
                n       = len(subset)
                profit  = subset['profit'].sum() if n > 0 else 0.0
                wr      = (subset['profit'] > 0).mean() * 100 if n > 0 else 0.0
                eq      = INITIAL_BALANCE + subset.sort_values('buy_time')['profit'].cumsum()
                dd      = calculate_max_dd_pct(eq) if n > 0 else 0.0
                conf    = "✓" if n >= regime_min_trades else "✗"
                flag    = "🚫 FILTER" if bin_key in bins_to_filter else ""
                lines.append(f"  {bin_key:<30} {conf:>5} {n:>8} {profit:>12.2f} {wr:>7.1f}% {dd:>7.2f}% {flag}")
        lines.append("  " + "-" * 88)
        logger.debug("\n".join(lines))

    if force_direction_filter:
        forced = 'dwtrend' if strategy_direction == 'long' else 'uptrend'
        for fam in ['trending', 'ranging', 'volatile']:
            bins_to_filter.add(f"{fam}_{forced}")

    return bins_to_filter

def _fmt_py_val(val):
    """Format a Python value for writing into a .py file."""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _load_py_module(path, module_name):
    """Load a .py file as a module and return it."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# =============================================================================
# HELPER — COMPARE BATCH VS PR_BATCH AND GENERATE CSV
# =============================================================================
def compare_and_generate_csv(strategies_batch_path, pr_batch_path, csv_path):
    """
    Compare strategies_batch.py (previous state) vs strategies_NN_batch.py (new state).
    Generate strategies_params.csv with change columns for diagnostics.
    """
    if not os.path.exists(strategies_batch_path):
        logger.warning(f"⚠️  strategies_batch.py not found — skipping CSV generation.")
        return
    if not os.path.exists(pr_batch_path):
        logger.warning(f"⚠️  strategies_NN_batch.py not found — skipping CSV generation.")
        return

    prev_map = {s["id"]: s for s in _load_py_module(strategies_batch_path, "strategies_batch").STRATEGIES}
    new_map  = {s["id"]: s for s in _load_py_module(pr_batch_path, "strategies_pr_batch").STRATEGIES}

    regime_bin_keys = (
        "regime_trending_uptrend", "regime_trending_dwtrend",
        "regime_ranging_uptrend",  "regime_ranging_dwtrend",
        "regime_volatile_uptrend", "regime_volatile_dwtrend",
    )

    rows = []
    for sid, new in new_map.items():
        prev = prev_map.get(sid, {})

        # Active change
        prev_active = prev.get("active", False)
        new_active  = new.get("active", False)
        change_active = (
            f"{'True' if prev_active else 'False'}→{'True' if new_active else 'False'}"
            if prev_active != new_active else "N/A"
        )

        # Param changes
        param_changes = []
        for k in PARAM_KEYS:
            prev_val = prev.get(k)
            new_val  = new.get(k)
            if prev_val is not None and new_val is not None and prev_val != new_val:
                param_changes.append(f"{k}: {prev_val}→{new_val}")
        change_params = " | ".join(param_changes) if param_changes else "N/A"

        # Regime bin changes
        regime_changes = []
        for bin_key in regime_bin_keys:
            prev_val = prev.get(bin_key)
            new_val  = new.get(bin_key)
            if prev_val is not None and new_val is not None and float(prev_val) != float(new_val):
                regime_changes.append(f"{bin_key}: {prev_val}→{new_val}")
        change_regime = " | ".join(regime_changes) if regime_changes else "N/A"

        row = {
            "id":                  sid,
            "name":                new["name"],
            "timeframe":           new["timeframe"],
            "active":              new_active,
            "direction":           new["direction"],
            "sell_after_ncandles": new.get("sell_after_ncandles", 0),
            "order_amount":        new.get("order_amount", 240),
            "lookback":            new.get("lookback"),
            "tolerance":           new.get("tolerance"),
            "ma_period":           new.get("ma_period"),
            "tp_pct":              new.get("tp_pct"),
            "sl_pct":              new.get("sl_pct"),
            "impulse":             new.get("impulse"),
            "flag":                new.get("flag"),
            "last_run":            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "validated":           new_active,
            "last_change_active":  change_active,
            "last_change_params":  change_params,
            "last_change_regime":  change_regime,
        }
        for bin_key in regime_bin_keys:
            row[bin_key] = new.get(bin_key, 1.0)

        rows.append(row)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.debug(f"✅ strategies_params.csv generated → {csv_path}")

def _render_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, btc_ts, btc_pct, title):
    """
    Core rendering function for equity curve comparison plots.
    Shared by plot_filter_comparison and plot_portfolio_comparison.
    """
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    if ts_r01 is not None and btc_ts is not None:
        btc_aligned = np.interp(
            pd.to_datetime(ts_r01).astype(np.int64) / 1e9,
            pd.to_datetime(btc_ts).astype(np.int64) / 1e9,
            btc_pct,
        )
        above = eq_r01 >= btc_aligned
        below = eq_r01 < btc_aligned
        ax.fill_between(ts_r01, eq_r01, 0, where=above, alpha=0.35, color="#00897B", interpolate=True)
        ax.fill_between(ts_r01, eq_r01, 0, where=below, alpha=0.35, color="#C62828", interpolate=True)

    lbl_base = (f"Baseline    NetGain={m_base['Net_Gain_pct']:>6.1f}%  "
                f"DD={m_base['Max_DD_pct']:>6.1f}%  R²={m_base['R_Squared']:.3f}")
    ax.plot(ts_base, eq_base, color="#2E86C1", linewidth=0.8, label=lbl_base)

    if ts_r01 is not None:
        lbl_r01 = (f"Regime 0+1  NetGain={m_r01['Net_Gain_pct']:>6.1f}%  "
                   f"DD={m_r01['Max_DD_pct']:>6.1f}%  R²={m_r01['R_Squared']:.3f}")
        ax.plot(ts_r01, eq_r01, color="#00897B", linewidth=1.4, label=lbl_r01)

    if btc_ts is not None:
        ax.plot(btc_ts, btc_pct, color="#FF8C00", linewidth=0.9,
                linestyle="--", alpha=0.6, label="_BTC")

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Net Gain (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.8, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()

    legend = ax.legend(
        loc="upper left",
        fontsize=10,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#AAAAAA",
        fancybox=False,
        borderpad=0.8,
        labelspacing=0.6,
        handlelength=2.5,
    )
    for text in legend.get_texts():
        text.set_fontfamily("monospace")

    plt.tight_layout()
    plt.show()


def _load_btc(data_folder, t_start, t_end):
    """Load and normalize BTC 1D data for a given time range."""
    btc_file = os.path.join(data_folder, "BTCUSDT_1Dutc.parquet")
    btc_df   = pd.read_parquet(btc_file)
    btc_df.columns = btc_df.columns.str.lower()
    btc_df["ts"] = pd.to_datetime(btc_df["timestamp"] if "timestamp" in btc_df.columns else btc_df.index)
    btc_df = btc_df.sort_values("ts").reset_index(drop=True)

    btc_sub = btc_df[(btc_df["ts"] >= t_start) & (btc_df["ts"] <= t_end)]
    if len(btc_sub) > 0:
        btc_ref = btc_sub["close"].iloc[0]
        return btc_sub["ts"].values, (btc_sub["close"].values / btc_ref - 1) * 100
    return None, None


def plot_filter_comparison(strategy_id, trade_log_baseline, trade_log_r01, data_folder, initial_balance):
    """
    Plot equity curves for a single strategy: baseline vs regime 0+1 vs BTC.
    """
    def _equity_pct(tl, t_start):
        tl  = tl.sort_values("sell_time").reset_index(drop=True)
        eq  = initial_balance + tl["profit"].cumsum().values
        pct = (eq - initial_balance) / initial_balance * 100
        m   = compute_metrics(tl, capital=initial_balance, name="")
        ts  = pd.to_datetime(tl["sell_time"]).values
        ts  = np.concatenate([[np.datetime64(t_start)], ts])
        pct = np.concatenate([[0.0], pct])
        return ts, pct, m

    t_start = pd.Timestamp(pd.to_datetime(trade_log_baseline["sell_time"]).min())
    t_end   = pd.Timestamp(pd.to_datetime(trade_log_baseline["sell_time"]).max())

    ts_base, eq_base, m_base = _equity_pct(trade_log_baseline, t_start)
    ts_r01,  eq_r01,  m_r01  = (
        _equity_pct(trade_log_r01, t_start)
        if trade_log_r01 is not None and len(trade_log_r01) > 0
        else (None, None, None)
    )
    btc_ts, btc_pct = _load_btc(data_folder, t_start, t_end)

    _render_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, btc_ts, btc_pct, strategy_id)


def plot_portfolio_comparison(trade_logs_baseline, trade_logs_regime01, data_folder, initial_balance, title="Portfolio"):
    """
    Plot combined portfolio equity curves: baseline vs regime 0+1 vs BTC.
    """
    if not trade_logs_baseline:
        return

    def _combined_equity_pct(trade_logs, capital_per_strategy):
        all_tl = pd.concat(
            [df for _, df in trade_logs], ignore_index=True
        ).sort_values("buy_time").reset_index(drop=True)
        total_capital = capital_per_strategy * len(trade_logs)
        eq  = total_capital + all_tl["profit"].cumsum().values
        pct = (eq - total_capital) / total_capital * 100
        ts  = pd.to_datetime(all_tl["buy_time"]).values
        m   = compute_metrics(all_tl, capital=total_capital, name="")
        t_start = pd.Timestamp(all_tl["buy_time"].min())
        ts  = np.concatenate([[np.datetime64(t_start)], ts])
        pct = np.concatenate([[0.0], pct])
        return ts, pct, m, t_start

    ts_base, eq_base, m_base, t_start_base = _combined_equity_pct(trade_logs_baseline, initial_balance)
    ts_r01, eq_r01, m_r01, t_start_r01 = (
        _combined_equity_pct(trade_logs_regime01, initial_balance)
        if trade_logs_regime01 else (None, None, None, None)
    )

    t_start = min(t_start_base, t_start_r01) if t_start_r01 else t_start_base
    t_end   = pd.Timestamp(pd.to_datetime(ts_base).max())
    btc_ts, btc_pct = _load_btc(data_folder, t_start, t_end)

    _render_comparison_plot(ts_base, eq_base, m_base, ts_r01, eq_r01, m_r01, btc_ts, btc_pct, title)
# =============================================================================
# HELPER — EXTRACT BEST PARAMS
# =============================================================================
def extract_best_params(df_summary, param_names, lists_for_grid, selection_percentile=None):
    """
    Extract optimal params from MC summary.
    Sorts by Net_Gain_pct_m (mean) or Net_Gain_pct_pN (percentile N) depending on selection_percentile.
    Preserves int/float types based on the original grid lists.
    """
    int_params  = {k for k, lst in zip(param_names, lists_for_grid) if all(isinstance(x, int) for x in lst)}
    sort_col = "Net_Gain_pct_m" if selection_percentile is None else "Net_Gain_pct_pN"
    best_row = df_summary.loc[df_summary[sort_col].idxmax()]
    best_params = {
        k: int(round(best_row[k])) if k in int_params else round(float(best_row[k]), 4)
        for k in param_names
    }
    logger.debug(f"Extracting optimal params (best {sort_col})...")
    logger.debug("Best params: " + " | ".join(f"{k}: {v}" for k, v in best_params.items()))
    return best_params


# =============================================================================
# HELPER — SELECT UNIVERSE
# =============================================================================
def select_universe(data_folder_is, data_folder_oos, timeframe, n_symbols, min_price, filter_symbols_fn, my_symbols=False, fix_symbols_mcis=False, n_symbols_mcis=20):
    """
    Select OOS universe (top N by volume) and match IS universe.
    If fix_symbols_mcis=True, IS universe is top n_symbols_mcis from IS by volume directly.
    Returns: symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos
    """
    raw_is  = sorted([f.split("_")[0] for f in os.listdir(data_folder_is)  if f.endswith(f"_{timeframe}.parquet")])
    raw_oos = sorted([f.split("_")[0] for f in os.listdir(data_folder_oos) if f.endswith(f"_{timeframe}.parquet")])

    ohlcv_oos, filtered_oos = filter_symbols_fn(raw_oos, min_vol_usdt=0, timeframe=timeframe, data_folder=data_folder_oos, min_price=min_price, vol_window=50, my_symbols=my_symbols)
    ohlcv_is,  filtered_is  = filter_symbols_fn(raw_is,  min_vol_usdt=0, timeframe=timeframe, data_folder=data_folder_is,  min_price=min_price, vol_window=50, my_symbols=my_symbols)

    def _vol_1d(sym, folder):
        path = os.path.join(folder, f"{sym}_1Dutc.parquet")
        if not os.path.exists(path):
            return 0.0
        df = pd.read_parquet(path, columns=["volume_quote"])
        return float(df["volume_quote"].tail(180).mean())

    vol_oos           = {sym: _vol_1d(sym, data_folder_oos) for sym in filtered_oos}
    oos_ranked        = sorted(filtered_oos, key=lambda s: vol_oos.get(s, 0), reverse=True)
    symbols_oos_final = oos_ranked[:n_symbols]

    if fix_symbols_mcis:
        vol_is           = {sym: _vol_1d(sym, data_folder_is) for sym in filtered_is}
        is_ranked        = sorted(filtered_is, key=lambda s: vol_is.get(s, 0), reverse=True)
        symbols_is_final = is_ranked[:n_symbols_mcis]
        logger.debug(f"FIX_SYMBOLS_MCIS_TRAINING=True — IS top {n_symbols_mcis} by volume: {symbols_is_final}")
    else:
        syms_is  = set(filtered_is)
        syms_oos = set(symbols_oos_final)
        in_both     = sorted(syms_is & syms_oos)
        only_in_oos = sorted(syms_oos - syms_is)

        vol_is               = {sym: _vol_1d(sym, data_folder_is) for sym in syms_is}
        is_candidates_by_vol = sorted(syms_is - syms_oos, key=lambda s: vol_is.get(s, 0), reverse=True)
        needed               = max(0, n_symbols - len(in_both))
        symbols_is_final     = sorted(in_both + is_candidates_by_vol[:needed])

        logger.debug(f"OOS pool ({len(filtered_oos):>3}): {len(filtered_oos)} candidates")
        logger.debug(f"IS  pool ({len(filtered_is):>3}): {len(filtered_is)} candidates")
        logger.debug(f"In both  ({len(in_both):>3}): {in_both}")
        logger.debug(f"Only in OOS ({len(only_in_oos):>3}): {only_in_oos}")

    logger.debug(f"OOS final universe ({len(symbols_oos_final):>3}): {sorted(symbols_oos_final)}")
    logger.debug(f"IS  final universe ({len(symbols_is_final):>3}): {symbols_is_final}")
    fix_str = "FIX=True" if fix_symbols_mcis else "FIX=False"
    logger.info(f"STAGE 0  ── Universe Selection     ── IS:{len(symbols_is_final)} symbols | OOS:{len(symbols_oos_final)} symbols | {fix_str}")

    if fix_symbols_mcis:
        if len(symbols_is_final) < n_symbols_mcis:
            logger.warning(f"⚠️  IS has only {len(symbols_is_final)} symbols — fewer than N_SYMBOLS_MCIS ({n_symbols_mcis}). Proceeding with available.")
    else:
        if len(symbols_is_final) < n_symbols:
            logger.warning(f"⚠️  IS has only {len(symbols_is_final)} symbols — fewer than N_SYMBOLS ({n_symbols}). Proceeding with available.")

    return symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos




# =============================================================================
# HELPER — SAVE DRIFT REFERENCE
# =============================================================================
def save_drift_reference(drift_results, output_path):
    """
    Write drift_montecarlo_batch.py with P5/P50 winrate reference values
    from Montecarlo OOS simulations.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = [
        '"""',
        'Montecarlo OOS reference values for drift detection.',
        'P5_WINRATE: Percentile 5 (floor - worst acceptable performance)',
        'P50_WINRATE: Percentile 50 (median - expected performance)',
        'These values come from Montecarlo simulations and represent the statistical',
        'boundaries for strategy health evaluation.',
        '"""',
        'DRIFT_REFERENCE = {',
    ]
    for entry in drift_results:
        lines += [
            f"    '{entry['strategy_id']}': {{",
            f"        'p5_winrate':  {entry['p5_winrate']},",
            f"        'p50_winrate': {entry['p50_winrate']},",
            f"    }},",
        ]
    lines.append("}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.debug(f"drift_montecarlo_batch.py updated → {output_path}")


# =============================================================================
# HELPER — SAVE STRATEGIES PR BATCH
# =============================================================================
def save_strategies_pr(strategies_batch_path, output_path, validation_results, best_params_map, strategy_ids_to_run=None, module_name="strategies_BT_batch"):
    """
    Generate strategies_PR_batch.py for production deployment.
    Reads strategies_BT_batch.py (never modified), applies dynamic fields from memory.

    strategies_batch_path : path to strategies_BT_batch.py (input, never modified)
    output_path           : path to write strategies_NN_batch.py
    validation_results    : list of dicts with strategy_id, verdict, bins_to_filter
    best_params_map       : dict {strategy_id: best_params dict}
    strategy_ids_to_run   : list of strategy IDs to include — None = all
    module_name           : module name to load from strategies_batch_path
    """
    if not os.path.exists(strategies_batch_path):
        logger.warning(f"⚠️  {os.path.basename(strategies_batch_path)} not found — skipping.")
        return

    strategies = _load_py_module(strategies_batch_path, module_name).STRATEGIES
    if strategy_ids_to_run is not None:
        strategies = [s for s in strategies if s["id"] in strategy_ids_to_run]
 
    val_map = {v["strategy_id"]: v for v in validation_results}
 
    all_bins = [
        "regime_trending_uptrend",
        "regime_trending_dwtrend",
        "regime_ranging_uptrend",
        "regime_ranging_dwtrend",
        "regime_volatile_uptrend",
        "regime_volatile_dwtrend",
    ]
 
    pr_lines = [
        '"""',
        'Trading Strategies Configuration',
        '',
        'Auto-generated by BOT_batch. Do not edit manually.',
        'Copy to BOT_trading/config/strategies_NN.py to deploy.',
        '"""',
        '',
        'STRATEGIES = [',
    ]
 
    for s in strategies:
        sid     = s["id"]
        v       = val_map.get(sid, {})
        bp      = best_params_map.get(sid, {})
        updated = dict(s)
 
        if v:
            updated["active"] = "VALIDATED" in v["verdict"]
            bins_to_filter    = v.get("bins_to_filter", set())
        else:
            bins_to_filter = set()
 
        if bp:
            for k, val in bp.items():
                updated[k.lower()] = val
 
        pr_lines.append("    {")
        pr_lines.append(f'        "id": "{sid}",')
        pr_lines.append(f'        "name": "{updated["name"]}",')
        pr_lines.append(f'        "timeframe": "{updated["timeframe"]}",')
        pr_lines.append(f'        "active": {updated.get("active", False)},')
        pr_lines.append(f'        "direction": "{updated["direction"]}",')
 
        for bin_key in all_bins:
            family, direction = bin_key.replace("regime_", "").rsplit("_", 1)
            blocked = f"{family}_{direction}" in bins_to_filter
            pr_lines.append(f'        "{bin_key}": {0 if blocked else 1},')
 
        pr_lines.append(f'        "sell_after_ncandles": {updated.get("sell_after_ncandles", 0)},')
        pr_lines.append(f'        "order_amount": {updated.get("order_amount_prod", 200)},')
 
        for k in SIGNAL_PARAM_KEYS:
            if k in updated:
                pr_lines.append(f'        "{k}": {_fmt_py_val(updated[k])},')
 
        for k in ("tp_pct", "sl_pct"):
            if k in updated:
                pr_lines.append(f'        "{k}": {_fmt_py_val(updated[k])},')
 
        pr_lines.append("    },")
 
    pr_lines.append("]")
 
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(pr_lines) + "\n")
 
    logger.info(f"\n{'─'*110}\n  ✅ strategies_OO/E1_batch.py generated\n{'─'*110}")
 


# =============================================================================
# HELPER — UPDATE STRATEGIES SYMBOLS CSV
# =============================================================================
def update_strategies_symbols(strategy_id, symbols_oos_final, timeframe=None, symbols_live_folder=None):
    symbols_changed = False

    if timeframe and symbols_live_folder:
        os.makedirs(symbols_live_folder, exist_ok=True)
        live_filename = f"symbols_live_{strategy_id}_{timeframe}.csv"
        live_path     = os.path.join(symbols_live_folder, live_filename)

        if os.path.exists(live_path):
            prev_symbols = pd.read_csv(live_path, header=None)[0].tolist()
            symbols_changed = prev_symbols != list(symbols_oos_final)
        else:
            symbols_changed = True

        pd.DataFrame(symbols_oos_final).to_csv(live_path, index=False, header=False)
        logger.debug(f"symbols_live saved → {live_path}")

    if symbols_changed:
        logger.debug(f"🔵 symbols_live — symbols updated for '{strategy_id}'")
    else:
        logger.debug(f"⚪ symbols_live — symbols unchanged for '{strategy_id}'")

    return {"symbols_changed": symbols_changed}

# =============================================================================
# PORTFOLIO ANALYSIS — EQUITY METRICS
# =============================================================================
def compute_metrics(trade_log, capital, name="Equity"):
    from sklearn.linear_model import LinearRegression as _LR

    tl      = trade_log.sort_values("sell_time").reset_index(drop=True)
    profits = tl["profit"].values

    win_rate = round((profits > 0).mean() * 100, 1)

    gains  = profits[profits > 0].sum()
    losses = -profits[profits < 0].sum()
    pf     = round(float(gains / losses), 3) if losses > 0 else np.inf

    tl["_date"]  = pd.to_datetime(tl["sell_time"]).dt.normalize()
    daily_profit = tl.groupby("_date")["profit"].sum()

    date_range   = pd.date_range(start=daily_profit.index.min(),
                                 end=daily_profit.index.max(),
                                 freq="1D")
    daily_profit = daily_profit.reindex(date_range, fill_value=0.0)
    eq           = capital + daily_profit.cumsum().values
    eq_series    = pd.Series(eq, index=date_range)

    cm         = np.maximum.accumulate(eq)
    max_dd     = ((eq - cm) / cm * 100).min()
    net_gain   = (eq[-1] - capital) / capital * 100
    profit_abs = round(float(eq[-1] - capital), 2)

    daily_returns = eq_series.pct_change().dropna()
    volatility    = daily_returns.std() * 100
    monthly       = eq_series.resample("ME").last().pct_change().dropna()
    consistency   = (monthly > 0).mean() * 100
    weekly = eq_series.resample("W").last().pct_change().dropna()
    weekly_pct = (weekly > 0).mean() * 100

    sharpe = (round(float(profits.mean() / profits.std() * np.sqrt(252)), 3)
              if profits.std() > 0 else np.nan)

    if "buy_time" in tl.columns and "sell_time" in tl.columns:
        duration_d = round(float(
            (pd.to_datetime(tl["sell_time"]) - pd.to_datetime(tl["buy_time"]))
            .dt.total_seconds().mean() / 86400
        ), 2)
    else:
        duration_d = np.nan

    X  = np.arange(len(eq)).reshape(-1, 1)
    y  = eq.reshape(-1, 1)
    r2 = round(_LR().fit(X, y).score(X, y), 3)

    return {
        "Curve":          name,
        "Net_Gain_pct":   round(float(net_gain), 2),
        "Max_DD_pct":     round(float(max_dd), 2),
        "Win_Rate":       win_rate,
        "R_Squared":      r2,
        "Profit_Factor":  pf,
        "Profit_abs":     profit_abs,
        "Sharpe":         sharpe,
        "Duration_d":     duration_d,
        "Weekly_pct": round(float(weekly_pct), 2),
        #"Monthly_pct":    round(float(consistency), 2),
    }


def print_metrics_table(metrics_list, title):
    df = pd.DataFrame(metrics_list)
    df['Curve'] = df['Curve'].astype(str)
    max_len = df['Curve'].str.len().max()
    df['Curve'] = df['Curve'].apply(lambda x: x.ljust(max_len))
    logger.debug(f"\n{title}\n{df.to_string(index=False)}")


# =============================================================================
# HELPER — PRINT STRATEGIES SUMMARY
# =============================================================================
def print_strategies_summary(validation_results):
    """Print validation summary table for all strategies."""
    if not validation_results:
        return
    lines = []
    lines.append(f"\n{'─'*110}")
    lines.append(f"  STRATEGIES SUMMARY")
    lines.append(f"{'─'*110}")
    lines.append(f"  {'Strategy':<27} {'Verdict':<14} {'Round':<16} {'NetGain%':>10} {'DD%':>8} {'WinRate%':>10} {'R2':>7} {'ProbNeg%':>10}")
    lines.append(f"  {'-'*110}")
    for v in validation_results:
        lines.append(
            f"  {v['strategy_id']:<27} {v['verdict']:<14} {v['round']:<16} "
            f"{v['net_gain_pct']:>9.2f}% {v['dd_pct']:>7.2f}% {v['win_ratio']:>9.1f}% "
            f"{v['r2']:>7.3f} {v['prob_neg_pct']:>9.2f}%"
        )
    lines.append(f" {'─'*110}")
    logger.info("\n".join(lines))


# =============================================================================
# HELPER — PRINT UPDATE STATUS (four separate tables, reads from CSV)
# =============================================================================
def print_update_status(csv_path, symbols_live_folder, validation_results):
    """
    Print update status tables reading from CSV:
      1. Active
      2. Market Regime
      3. Symbols
    """
    if not validation_results:
        return

    if not os.path.exists(csv_path):
        logger.warning(f"⚠️  strategies_params.csv not found — skipping update status tables.")
        return

    df = pd.read_csv(csv_path)
    strategy_ids = [v["strategy_id"] for v in validation_results]
    df = df[df["id"].isin(strategy_ids)].set_index("id")

    def _get(sid, col, default="—"):
        try:
            val = df.loc[sid, col]
            return str(val) if pd.notna(val) else default
        except KeyError:
            return default

    def _active_icon(sid):
        val = _get(sid, "active")
        if val == "True":  return "🟢 active"
        if val == "False": return "🔴 inactive"
        return "—"

    def _change_icon(val):
        if val in ("N/A", "—", ""):  return "⚪ no change"
        return f"🔵 {val}"

    # -------------------------------------------------------------------------
    # Table 1 — Active
    # -------------------------------------------------------------------------
    lines = []
    lines.append(f"\n{'─'*110}")
    lines.append(f"  ACTIVE")
    lines.append(f"{'─'*110}")
    lines.append(f"  {'Strategy':<27} {'Status':<16} {'Changes':<35}")
    lines.append(f"  {'-'*103}")
    for sid in strategy_ids:
        change = _get(sid, "last_change_active")
        lines.append(f"  {sid:<27} {_active_icon(sid):<16} {_change_icon(change):<35}")
    lines.append(f"  {'─'*103}")
    logger.info("\n".join(lines))

    # -------------------------------------------------------------------------
    # Table 2 — Params
    # -------------------------------------------------------------------------
    lines = []
    lines.append(f"\n{'─'*110}")
    lines.append(f"  PARAMS")
    lines.append(f"{'─'*110}")
    lines.append(f"  {'Strategy':<27} {'Changes':<50}")
    lines.append(f"  {'-'*103}")
    for sid in strategy_ids:
        change = _get(sid, "last_change_params")
        lines.append(f"  {sid:<27} {_change_icon(change):<50}")
    lines.append(f"  {'─'*103}")
    logger.info("\n".join(lines))

    # -------------------------------------------------------------------------
    # Table 3 — Market Regime
    # -------------------------------------------------------------------------
    lines = []
    lines.append(f"\n{'─'*110}")
    lines.append(f"  MARKET REGIME")
    lines.append(f"{'─'*110}")
    lines.append(f"  {'Strategy':<27} {'Trending':>10} {'Ranging':>10} {'Volatile':>10} {'Changes':<40}")
    lines.append(f"  {'-'*103}")
    for sid in strategy_ids:
        trending = _get(sid, "regime_trending")
        ranging  = _get(sid, "regime_ranging")
        volatile = _get(sid, "regime_volatile")
        change   = _get(sid, "last_change_regime")
        lines.append(
            f"  {sid:<27} {trending:>10} {ranging:>10} {volatile:>10} {_change_icon(change):<40}"
        )
    lines.append(f"  {'─'*110}")
    logger.info("\n".join(lines))

    # -------------------------------------------------------------------------
    # Table 4 — Symbols
    # -------------------------------------------------------------------------
    lines = []
    lines.append(f"\n{'─'*110}")
    lines.append(f"  SYMBOLS")
    lines.append(f"{'─'*110}")
    lines.append(f"  {'Strategy':<27} {'Status':<16}")
    lines.append(f"  {'-'*103}")
    for v in validation_results:
        sid  = v["strategy_id"]
        icon = "🔵 updated" if v.get("symbols_changed") else "⚪ no change"
        lines.append(f"  {sid:<27} {icon:<16}")
    lines.append(f"  {'─'*103}")
    logger.info("\n".join(lines))


# =============================================================================
# HELPER — PRINT PORTFOLIO METRICS TABLE
# =============================================================================
def print_portfolio_metrics_table(trade_logs, label, initial_balance):
    """Print individual + combined metrics table for a list of trade_logs."""
    named_logs = {sid: df for sid, df in trade_logs}
    metrics_list = []
    for sid, df in named_logs.items():
        metrics_list.append(compute_metrics(df, capital=initial_balance, name=sid))
    if len(named_logs) > 1:
        combined_tl      = pd.concat(list(named_logs.values()), ignore_index=True).sort_values("buy_time").reset_index(drop=True)
        combined_capital = initial_balance * len(named_logs)
        metrics_list.append(compute_metrics(combined_tl, capital=combined_capital, name="Combined"))
    print_metrics_table(metrics_list, f"📊 METRICS TABLE — {label}")


# =============================================================================
# HELPER — CALC R2 FROM EQUITY HISTORY
# =============================================================================
def calc_r2_from_equity_hist(equity_hist):
    """
    Compute R² of equity curve vs straight line from sim_balance_history dict.
    Returns np.nan if insufficient data.
    """
    if not equity_hist or len(equity_hist.get("balance", [])) < 2:
        return np.nan
    y = np.array(equity_hist["balance"]).reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    return round(LinearRegression().fit(X, y).score(X, y), 3)


# =============================================================================
# HELPER — PRINT ALL CURVES TABLE
# =============================================================================
def print_all_curves_table(trade_logs, label, initial_balance):
    """
    Print a metrics table for all curves plus long/short aggregate rows and a combined row.
    trade_logs: list of (strategy_id, trade_log_df)
    """
    named = {sid: df for sid, df in trade_logs}
    rows  = []
    for sid, df in named.items():
        rows.append(compute_metrics(df, capital=initial_balance, name=sid))

    # Aggregate longs and shorts
    long_logs  = [(sid, df) for sid, df in named.items() if "_long_"  in sid]
    short_logs = [(sid, df) for sid, df in named.items() if "_short_" in sid]

    if long_logs:
        long_tl  = pd.concat([df for _, df in long_logs], ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
        long_cap = initial_balance * len(long_logs)
        rows.append(compute_metrics(long_tl, capital=long_cap, name="── Longs"))

    if short_logs:
        short_tl  = pd.concat([df for _, df in short_logs], ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
        short_cap = initial_balance * len(short_logs)
        rows.append(compute_metrics(short_tl, capital=short_cap, name="── Shorts"))

    all_tl  = pd.concat(list(named.values()), ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
    all_cap = initial_balance * len(named)
    rows.append(compute_metrics(all_tl, capital=all_cap, name="── Combined"))

    df_out = pd.DataFrame(rows)

    # Compute Profit_pctT excluding aggregate rows
    strategy_rows = df_out[~df_out["Curve"].str.strip().str.startswith("──")]
    total_profit  = strategy_rows["Profit_abs"].sum()
    df_out["Profit_pctT"] = df_out["Profit_abs"].apply(
        lambda x: round(x / total_profit * 100, 1) if total_profit != 0 else np.nan
    )

    cols   = ["Curve", "Net_Gain_pct", "Max_DD_pct", "Win_Rate", "R_Squared", "Profit_Factor", "Profit_abs", "Profit_pctT", "Weekly_pct"]
    df_out = df_out[cols].copy()
    max_len = df_out["Curve"].str.len().max()
    df_out["Curve"]      = df_out["Curve"].apply(lambda x: x.ljust(max_len))
    df_out["Profit_abs"] = df_out["Profit_abs"].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    # Insert separator row before aggregate rows
    agg_mask = df_out["Curve"].str.strip().str.startswith("──")
    if agg_mask.any():
        agg_idx = agg_mask.idxmax()
        sep_row = pd.DataFrame(
            {col: ["─" * max(len(str(df_out[col].iloc[0])), 9)] for col in cols},
        )
        sep_row["Curve"] = "─" * max_len
        df_out = pd.concat([df_out.iloc[:agg_idx], sep_row, df_out.iloc[agg_idx:]], ignore_index=True)

    color = "\033[94m" if label == "Regime 0+1 — Validated only" else ""
    reset = "\033[0m" if color else ""
    lines = [f"\n{color}{'─'*110}\n📊 ALL CURVES COMBINED OOS1 — {label}\n{'─'*110}{reset}\n", df_out.to_string(index=False)]
    logger.info("\n".join(lines))


# =============================================================================
# HELPER — PRINT BEST COMBINATIONS
# =============================================================================
def find_best_r2_combination_ids(
    trade_logs_is: list,
    initial_balance: float,
    precomputed_metrics: dict = None,
) -> list:
    """
    Find the strategy combination with highest R² on IS trade logs.
    Returns list of strategy IDs (not trade logs).

    Args:
        trade_logs_is       : list of (strategy_id, trade_log_df) from IS period
        initial_balance     : capital per strategy
        precomputed_metrics : optional dict {strategy_id: metrics}

    Returns:
        list of strategy_ids forming the best R² combination
    """
    named   = {sid: df for sid, df in trade_logs_is}
    metrics = precomputed_metrics or {
        sid: compute_metrics(df, capital=initial_balance, name=sid)
        for sid, df in named.items()
    }

    best_r2    = -1.0
    best_combo = list(named.keys())

    for r in range(1, len(named) + 1):
        for combo in _combinations(named.keys(), r):
            if len(combo) == 1:
                r2 = metrics.get(combo[0], {}).get("R_Squared", -1.0)
            else:
                combo_tl = pd.concat(
                    [named[sid] for sid in combo], ignore_index=True
                ).sort_values("sell_time").reset_index(drop=True)
                r2 = compute_metrics(combo_tl, capital=initial_balance * len(combo), name="")["R_Squared"]

            if r2 > best_r2:
                best_r2    = r2
                best_combo = list(combo)

    logger.debug(f"Best R² combination on IS: {best_combo} — R²={best_r2:.3f}")
    return best_combo


def print_best_r2_robustness_table(
    combo_ids: list,
    trade_logs_per_period: list,
    initial_balance: float,
) -> None:
    """
    Evaluate a fixed strategy combination (selected on IS) across OOS periods.
    Prints robustness table with combined portfolio metrics per period.

    Args:
        combo_ids            : list of strategy IDs forming the fixed combination
        trade_logs_per_period: list of (period_label, trade_logs)
                               e.g. [("OOS1", validated_regime01), ("OOS2", ...), ...]
        initial_balance      : capital per strategy
    """
    def _weekly_returns(trade_logs, capital_per_strategy):
        if not trade_logs:
            return pd.Series(dtype=float)
        all_tl = pd.concat(
            [df for _, df in trade_logs], ignore_index=True
        ).sort_values("sell_time").reset_index(drop=True)
        total_capital = capital_per_strategy * len(trade_logs)
        all_tl["_date"] = pd.to_datetime(all_tl["sell_time"]).dt.normalize()
        daily = all_tl.groupby("_date")["profit"].sum()
        date_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
        daily = daily.reindex(date_range, fill_value=0.0)
        eq = total_capital + daily.cumsum()
        eq_series = pd.Series(eq.values, index=date_range)
        return eq_series.resample("W").last().pct_change().dropna() * 100

    def _neg_streak_stats(weekly):
        if len(weekly) == 0:
            return np.nan, np.nan
        streaks, current = [], 0
        for val in weekly:
            if val < 0:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)
        if not streaks:
            return 0.0, 0
        return round(float(np.mean(streaks)), 2), int(np.max(streaks))

    def _cvar(weekly, pct=10):
        if len(weekly) == 0:
            return np.nan
        threshold = np.percentile(weekly, pct)
        tail = weekly[weekly <= threshold]
        return round(float(tail.mean()), 2) if len(tail) > 0 else np.nan

    rows = []
    for label, trade_logs in trade_logs_per_period:
        filtered = [(sid, df) for sid, df in trade_logs if sid in combo_ids]
        if not filtered:
            continue

        all_tl        = pd.concat(
            [df for _, df in filtered], ignore_index=True
        ).sort_values("sell_time").reset_index(drop=True)
        total_capital = initial_balance * len(filtered)

        m              = compute_metrics(all_tl, capital=total_capital, name="")
        pf             = m["Profit_Factor"]
        weekly         = _weekly_returns(filtered, initial_balance)
        cvar10         = _cvar(weekly, pct=10)
        avg_neg, max_neg = _neg_streak_stats(weekly)

        rows.append({
            "Period":        label,
            "NetGain%":      m["Net_Gain_pct"],
            "MaxDD%":        m["Max_DD_pct"],
            "R2":            m["R_Squared"],
            "ProfitFactor":  pf if pf != float("inf") else np.nan,
            "CVaR10%":       cvar10,
            "AvgNegStreak":  avg_neg,
            "MaxNegStreak":  max_neg,
            "Weekly_pct": round(float((weekly > 0).mean() * 100), 2),
        })

    if not rows:
        logger.info("  No data for best R² robustness table.")
        return

    df = pd.DataFrame(rows)
    sep_row = {col: "─" * 8 for col in df.columns}
    sep_row["Period"] = "─" * 6
    mean_row = df.drop(columns="Period").mean().round(3).to_dict()
    mean_row["Period"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([sep_row, mean_row])], ignore_index=True)
    lines = [
        f"\n\033[94m{'─'*110}",
        f"  BEST R² COMBINATION — Robustness across OOS periods",
        f"{'─'*110}\033[0m",
        df.to_string(index=False),
        f"{'─'*110}",
    ]
    logger.info("\n".join(lines))

def _decorrelate(
    trade_logs_oos1: list,
    trade_logs_oos2: list,
    initial_balance: float,
    threshold: float,
    precomputed_metrics: dict,
    trade_logs_oos3: list,
    series_fn,
    label: str,
) -> list:
    def _num(sid):
        for part in sid.split("_"):
            if part.isdigit():
                return int(part)
        return 0

    metrics  = precomputed_metrics or {
        sid: compute_metrics(df, capital=initial_balance, name=sid)
        for sid, df in trade_logs_oos1
    }
    oos1_map = {sid: df for sid, df in trade_logs_oos1}
    oos2_map = {sid: df for sid, df in trade_logs_oos2}
    oos3_map = {sid: df for sid, df in (trade_logs_oos3 or [])}
    all_sids = [sid for sid, _ in trade_logs_oos1]

    series_combined = {}
    for sid in all_sids:
        parts = []
        if sid in oos1_map:
            parts.append(series_fn(oos1_map[sid], initial_balance))
        if sid in oos2_map:
            parts.append(series_fn(oos2_map[sid], initial_balance))
        if sid in oos3_map:
            parts.append(series_fn(oos3_map[sid], initial_balance))
        if parts:
            combined = pd.concat(parts).sort_index()
            series_combined[sid] = combined.groupby(level=0).mean()

    if len(series_combined) < 2:
        logger.info("  Not enough strategies for correlation analysis.")
        return trade_logs_oos1

    num_map = {sid: f"{_num(sid):02d}" for sid in series_combined}
    df_     = pd.DataFrame({num_map[sid]: s for sid, s in series_combined.items()}).fillna(0)
    corr_mx = df_.corr().round(2)
    logger.debug(f"\n{corr_mx.to_string()}")

    ranked    = sorted(all_sids, key=lambda s: metrics.get(s, {}).get("Net_Gain_pct", 0), reverse=True)
    selected  = []
    discarded = []
    lines     = [f"\n  {'Rank':<6} {'Strategy':<30} {'NetGain%':>10} {'Action':<20} {'Reason'}"]
    lines.append(f"  {'─'*85}")

    for sid in ranked:
        ng         = metrics.get(sid, {}).get("Net_Gain_pct", 0)
        num        = num_map.get(sid, sid)
        correlated = False
        reason     = ""
        for kept in selected:
            kept_num = num_map.get(kept, kept)
            val      = corr_mx.loc[num, kept_num] if num in corr_mx.index and kept_num in corr_mx.columns else 0.0
            if pd.notna(val) and val > threshold:
                correlated = True
                reason     = f"corr={val:.2f} with {kept}"
                discarded.append(sid)
                break
        if correlated:
            lines.append(f"  {ranked.index(sid)+1:<6} {sid:<30} {ng:>9.2f}%  {'❌ DISCARDED':<20} {reason}")
        else:
            selected.append(sid)
            lines.append(f"  {ranked.index(sid)+1:<6} {sid:<30} {ng:>9.2f}%  {'✅ SELECTED':<20}")

    lines.append(f"  {'─'*85}")
    logger.info("\n".join(lines))

    return [(sid, oos1_map[sid]) for sid in sorted(selected, key=_num) if sid in oos1_map]


def _dd_series(df: pd.DataFrame, capital: float) -> pd.Series:
    tl          = df.copy()
    tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
    daily       = tl.groupby("_date")["profit"].sum().groupby(level=0).sum()
    date_range  = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
    daily       = daily.reindex(date_range).ffill().fillna(0.0)
    equity      = capital + daily.cumsum()
    peak        = equity.cummax()
    return (equity - peak) / peak * 100


def _profit_series(df: pd.DataFrame, capital: float) -> pd.Series:
    tl          = df.copy()
    tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
    daily       = tl.groupby("_date")["profit"].sum().groupby(level=0).sum()
    date_range  = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
    return daily.reindex(date_range).fillna(0.0)


def decorrelate_by_dd(
    trade_logs_oos1: list,
    trade_logs_oos2: list,
    initial_balance: float,
    threshold: float = 0.7,
    precomputed_metrics: dict = None,
    trade_logs_oos3: list = None,
) -> list:
    """Greedy DD-correlation filter. Keeps best NetGain from each correlated pair."""
    return _decorrelate(
        trade_logs_oos1, trade_logs_oos2, initial_balance,
        threshold, precomputed_metrics, trade_logs_oos3,
        series_fn=_dd_series, label="DD",
    )


def decorrelate_by_profit(
    trade_logs_oos1: list,
    trade_logs_oos2: list,
    initial_balance: float,
    threshold: float = 0.7,
    precomputed_metrics: dict = None,
    trade_logs_oos3: list = None,
) -> list:
    """Greedy profit-correlation filter. Keeps best NetGain from each correlated pair."""
    return _decorrelate(
        trade_logs_oos1, trade_logs_oos2, initial_balance,
        threshold, precomputed_metrics, trade_logs_oos3,
        series_fn=_profit_series, label="Profit",
    )

def print_robustness_table(
    trade_logs_per_period: list[tuple[str, list[tuple[str, pd.DataFrame]]]],
    initial_balance: float,
) -> None:
    """
    Print a robustness table with one row per OOS period.
    Each row shows combined portfolio metrics for validated strategies.

    Args:
        trade_logs_per_period : list of (period_label, trade_logs)
                                e.g. [("OOS1", validated_regime01), ("OOS2", validated_oos2), ...]
        initial_balance       : capital per strategy
    """
    def _weekly_returns(trade_logs, capital_per_strategy):
        if not trade_logs:
            return pd.Series(dtype=float)
        all_tl = pd.concat(
            [df for _, df in trade_logs], ignore_index=True
        ).sort_values("sell_time").reset_index(drop=True)
        total_capital = capital_per_strategy * len(trade_logs)
        all_tl["_date"] = pd.to_datetime(all_tl["sell_time"]).dt.normalize()
        daily = all_tl.groupby("_date")["profit"].sum()
        date_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
        daily = daily.reindex(date_range, fill_value=0.0)
        eq = total_capital + daily.cumsum()
        eq_series = pd.Series(eq.values, index=date_range)
        weekly = eq_series.resample("W").last().pct_change().dropna() * 100
        return weekly

    def _neg_streak_stats(weekly):
        if len(weekly) == 0:
            return np.nan, np.nan
        streaks = []
        current = 0
        for val in weekly:
            if val < 0:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)
        if not streaks:
            return 0.0, 0
        return round(float(np.mean(streaks)), 2), int(np.max(streaks))

    def _cvar(weekly, pct=10):
        if len(weekly) == 0:
            return np.nan
        threshold = np.percentile(weekly, pct)
        tail = weekly[weekly <= threshold]
        return round(float(tail.mean()), 2) if len(tail) > 0 else np.nan

    rows = []
    for label, trade_logs in trade_logs_per_period:
        if not trade_logs:
            continue

        all_tl        = pd.concat(
            [df for _, df in trade_logs], ignore_index=True
        ).sort_values("sell_time").reset_index(drop=True)
        total_capital = initial_balance * len(trade_logs)

        m  = compute_metrics(all_tl, capital=total_capital, name="")
        pf = m["Profit_Factor"]

        weekly           = _weekly_returns(trade_logs, initial_balance)
        cvar10           = _cvar(weekly, pct=10)
        avg_neg, max_neg = _neg_streak_stats(weekly)

        rows.append({
            "Period":       label,
            "NetGain%":     m["Net_Gain_pct"],
            "MaxDD%":       m["Max_DD_pct"],
            "R2":           m["R_Squared"],
            "ProfitFactor": pf,
            "CVaR10%":      cvar10,
            "AvgNegStreak": avg_neg,
            "MaxNegStreak": max_neg,
            "Weekly_pct": round(float((weekly > 0).mean() * 100), 2),
        })

    if not rows:
        logger.info("  No data for robustness table.")
        return

    df = pd.DataFrame(rows)
    sep_row = {col: "─" * 8 for col in df.columns}
    sep_row["Period"] = "─" * 6
    mean_row = df.drop(columns="Period").mean().round(3).to_dict()
    mean_row["Period"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([sep_row, mean_row])], ignore_index=True)
    lines = [
        f"\n\033[94m{'─'*110}",
        f"  ROBUSTNESS TABLE — Validated Combined Portfolio",
        f"{'─'*110}\033[0m",
        df.to_string(index=False),
        f"{'─'*110}",
    ]
    logger.info("\n".join(lines))
