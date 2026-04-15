import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("BOT_batch.batch_utils")

# =============================================================================
# CSV COLUMN VALIDATION
# =============================================================================
EXPECTED_CSV_COLUMNS = [
    "id", "name", "timeframe", "active", "direction",
    "regime_trending", "regime_ranging", "regime_volatile",
    "direction_mode", "sell_after_ncandles", "order_amount",
    "lookback", "tolerance", "ma_period", "tp_pct", "sl_pct",
    "impulse", "ranges", "flag",
    "last_run", "bt_netgain_pct", "bt_r2", "prob_negative", "validated",
    "last_change_active", "last_change_params", "last_change_regime",
]

def validate_csv_columns(csv_path):
    """
    Validate that strategies_params.csv has exactly the expected columns.
    Raises ValueError and stops execution if columns do not match.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"strategies_params.csv not found at {csv_path}")

    df      = pd.read_csv(csv_path, nrows=0)
    actual  = list(df.columns)
    missing = [c for c in EXPECTED_CSV_COLUMNS if c not in actual]
    extra   = [c for c in actual if c not in EXPECTED_CSV_COLUMNS]

    if missing or extra:
        msg = "❌ strategies_params.csv column mismatch — aborting."
        if missing:
            msg += f"\n  Missing : {missing}"
        if extra:
            msg += f"\n  Extra   : {extra}"
        raise ValueError(msg)

    logger.info("✅ strategies_params.csv columns validated.")

def generate_csv_from_batch(strategies_batch_path, csv_path):
    """
    Generate strategies_params.csv from strategies_batch.py if it doesn't exist.
    strategies_batch.py is the source of truth — CSV is derived and diagnostic only.
    """
    if os.path.exists(csv_path):
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location("strategies_batch", strategies_batch_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    strategies = mod.STRATEGIES

    rows = []
    for s in strategies:
        rows.append({
            "id":                  s["strategy_id"],
            "name":                "_".join(s["strategy_id"].split("_")[1:]),
            "timeframe":           s["timeframe"],
            "active":              s.get("active", False),
            "direction":           s["side"],
            "regime_trending":     s.get("regime_trending", 1.0),
            "regime_ranging":      s.get("regime_ranging",  1.0),
            "regime_volatile":     s.get("regime_volatile", 1.0),
            "direction_mode":      s.get("direction_mode", "general"),
            "sell_after_ncandles": s.get("sell_after_ncandles", 0),
            "order_amount":        s.get("order_amount_prod", 200),
            "lookback":            s.get("lookback",   None),
            "tolerance":           s.get("tolerance",  None),
            "ma_period":           s.get("ma_period",  None),
            "tp_pct":              s.get("tp_pct",     None),
            "sl_pct":              s.get("sl_pct",     None),
            "impulse":             s.get("impulse",    None),
            "ranges":              s.get("ranges",     None),
            "flag":                s.get("flag",       None),
            "last_run":            None,
            "bt_netgain_pct":      None,
            "bt_r2":               None,
            "prob_negative":       None,
            "validated":           None,
            "last_change_active":  None,
            "last_change_params":  None,
            "last_change_regime":  None,
        })

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"✅ strategies_params.csv generated from strategies_batch.py → {csv_path}")



def plot_filter_comparison(strategy_id, trade_log_baseline, trade_log_r01, data_folder, initial_balance):
    """
    Single plot with 3 equity curves (baseline, regime 0+1) + BTC normalized.
    Blue = baseline, green = regime 0+1, orange dashed = BTC.
    """
    import matplotlib.dates as mdates

    def _equity_pct(tl, t_start):
        tl  = tl.sort_values("buy_time").reset_index(drop=True)
        eq  = initial_balance + tl["profit"].cumsum().values
        pct = (eq - initial_balance) / initial_balance * 100
        m   = compute_metrics(tl, capital=initial_balance, name="")
        ts  = pd.to_datetime(tl["buy_time"]).values
        # Prepend origin point at t_start with 0%
        ts  = np.concatenate([[np.datetime64(t_start)], ts])
        pct = np.concatenate([[0.0], pct])
        return ts, pct, m

    # Load BTC 1Dutc
    btc_file = os.path.join(data_folder, "BTCUSDT_1Dutc.parquet")
    btc_df   = pd.read_parquet(btc_file)
    btc_df.columns = btc_df.columns.str.lower()
    if "timestamp" in btc_df.columns:
        btc_df["ts"] = pd.to_datetime(btc_df["timestamp"])
    else:
        btc_df["ts"] = pd.to_datetime(btc_df.index)
    btc_df = btc_df.sort_values("ts").reset_index(drop=True)

    t_start = pd.Timestamp(pd.to_datetime(trade_log_baseline["buy_time"]).min())
    t_end   = pd.Timestamp(pd.to_datetime(trade_log_baseline["buy_time"]).max())

    ts_base, eq_base, m_base = _equity_pct(trade_log_baseline, t_start)
    ts_r01,  eq_r01,  m_r01  = _equity_pct(trade_log_r01, t_start) if trade_log_r01 is not None and len(trade_log_r01) > 0 else (None, None, None)
    btc_sub = btc_df[(btc_df["ts"] >= t_start) & (btc_df["ts"] <= t_end)]
    if len(btc_sub) > 0:
        btc_pct = (btc_sub["close"].values / btc_df["close"].iloc[0] - 1) * 100
        btc_ts  = btc_sub["ts"].values
    else:
        btc_pct, btc_ts = None, None

    fig, ax = plt.subplots(figsize=(14, 5))

    lbl_base = (f"Baseline    NetGain={m_base['Net_Gain_pct']:>6.1f}%  "
                f"DD={m_base['Max_DD_pct']:>6.1f}%  R²={m_base['R_Squared']:.3f}")
    ax.plot(ts_base, eq_base, color="steelblue", linewidth=1.2, label=lbl_base)

    if ts_r01 is not None:
        lbl_r01 = (f"Regime 0+1  NetGain={m_r01['Net_Gain_pct']:>6.1f}%  "
                   f"DD={m_r01['Max_DD_pct']:>6.1f}%  R²={m_r01['R_Squared']:.3f}")
        ax.plot(ts_r01, eq_r01, color="seagreen", linewidth=1.2, label=lbl_r01)

    if btc_ts is not None:
        ax.plot(btc_ts, btc_pct, color="darkorange", linewidth=0.8, linestyle="--", label="_BTC")

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title(strategy_id)
    ax.set_ylabel("Net Gain (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.grid(True, linestyle="--", alpha=0.4)
    legend = ax.legend(loc="upper left")
    for text in legend.get_texts():
        text.set_fontfamily("monospace")
    plt.tight_layout()
    plt.show()


# =============================================================================
# HELPER — PLOT PORTFOLIO COMPARISON (validated combined curves)
# =============================================================================
def plot_portfolio_comparison(trade_logs_baseline, trade_logs_regime01, data_folder, initial_balance):
    """
    Single plot with combined baseline + combined regime 0+1 + BTC.
    Used at the end of run_portfolio_analysis for validated strategies only.
    """
    import matplotlib.dates as mdates

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

    # Load BTC
    btc_file = os.path.join(data_folder, "BTCUSDT_1Dutc.parquet")
    btc_df   = pd.read_parquet(btc_file)
    btc_df.columns = btc_df.columns.str.lower()
    if "timestamp" in btc_df.columns:
        btc_df["ts"] = pd.to_datetime(btc_df["timestamp"])
    else:
        btc_df["ts"] = pd.to_datetime(btc_df.index)
    btc_df = btc_df.sort_values("ts").reset_index(drop=True)

    ts_base, eq_base, m_base, t_start_base = _combined_equity_pct(trade_logs_baseline, initial_balance)

    ts_r01, eq_r01, m_r01, t_start_r01 = (
        _combined_equity_pct(trade_logs_regime01, initial_balance)
        if trade_logs_regime01 else (None, None, None, None)
    )

    t_start = min(t_start_base, t_start_r01) if t_start_r01 else t_start_base
    t_end   = pd.Timestamp(pd.to_datetime(ts_base).max())
    btc_sub = btc_df[(btc_df["ts"] >= t_start) & (btc_df["ts"] <= t_end)]
    if len(btc_sub) > 0:
        btc_pct = (btc_sub["close"].values / btc_df["close"].iloc[0] - 1) * 100
        btc_ts  = btc_sub["ts"].values
    else:
        btc_pct, btc_ts = None, None

    fig, ax = plt.subplots(figsize=(14, 5))

    lbl_base = (f"Baseline    NetGain={m_base['Net_Gain_pct']:>6.1f}%  "
                f"DD={m_base['Max_DD_pct']:>6.1f}%  R²={m_base['R_Squared']:.3f}")
    ax.plot(ts_base, eq_base, color="steelblue", linewidth=1.2, label=lbl_base)

    if ts_r01 is not None:
        lbl_r01 = (f"Regime 0+1  NetGain={m_r01['Net_Gain_pct']:>6.1f}%  "
                   f"DD={m_r01['Max_DD_pct']:>6.1f}%  R²={m_r01['R_Squared']:.3f}")
        ax.plot(ts_r01, eq_r01, color="seagreen", linewidth=1.2, label=lbl_r01)

    if btc_ts is not None:
        ax.plot(btc_ts, btc_pct, color="darkorange", linewidth=0.8, linestyle="--", label="_BTC")

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Portfolio — Validated only")
    ax.set_ylabel("Net Gain (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.grid(True, linestyle="--", alpha=0.4)
    legend = ax.legend(loc="upper left")
    for text in legend.get_texts():
        text.set_fontfamily("monospace")
    plt.tight_layout()
    plt.show()


# =============================================================================
# HELPER — EXTRACT BEST PARAMS
# =============================================================================
def extract_best_params(df_summary, param_names, lists_for_grid):
    """
    Extract optimal params from MC summary (best Net_Gain_pct_m).
    Preserves int/float types based on the original grid lists.
    """
    int_params  = {k for k, lst in zip(param_names, lists_for_grid) if all(isinstance(x, int) for x in lst)}
    best_row    = df_summary.loc[df_summary["Net_Gain_pct_m"].idxmax()]
    best_params = {
        k: int(round(best_row[k])) if k in int_params else round(float(best_row[k]), 4)
        for k in param_names
    }
    logger.debug("Extracting optimal params (best Net_Gain_pct_m)...")
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
# HELPER — ENRICH TRADES WITH REGIME
# =============================================================================
def enrich_trades_with_regime(trade_log, ohlc_folder, timeframe, families, lookback_bars, ma_period, hurst_window, er_window, atr_window, pe_window, pe_order, load_btc_fn, calc_metrics_fn, classify_fn):
    """
    Enrich trade_log with regime family classification.
    Returns trade_log with 'family' column added.
    """
    btc_cache = {}
    btc_df    = load_btc_fn(ohlc_folder, timeframe, btc_cache)

    df = trade_log.copy()
    df["family"] = "unknown"

    for idx, trade in df.iterrows():
        metrics = calc_metrics_fn(
            btc_df, trade["buy_time"], lookback_bars,
            ma_period, hurst_window, er_window, atr_window, pe_window, pe_order
        )
        if metrics:
            df.at[idx, "family"] = classify_fn(metrics, families)

    return df


# =============================================================================
# HELPER — UPDATE STRATEGIES PARAMS CSV
# =============================================================================
def update_strategies_params(csv_path, strategy_id, best_params, param_keys, validated,
                             bt_netgain_pct, r2, prob_negative_oos, regime_stats=None):
    """
    Update strategies_params.csv with new params, validation result and regime flags.
    Production columns (params, regime) are only updated when validated=True.

    regime_stats: dict with keys 'trending', 'ranging', 'volatile', each containing
                  at least a 'profit' key. If None, regime columns are not updated.
    """
    def normalize(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    _no_change = {"params_changed": False, "param_changes": [], "regime_changes": [],
                  "active_prev": None, "active_new": None}

    if not os.path.exists(csv_path):
        logger.warning(f"⚠️  strategies_params.csv not found at {csv_path} — skipping update.")
        return _no_change

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df_params = pd.read_csv(csv_path)
    for col in ("bt_netgain_pct", "bt_r2", "prob_negative"):
        df_params[col] = pd.to_numeric(df_params[col], errors="coerce").astype("float64")

    mask = df_params["id"] == strategy_id
    if not mask.any():
        logger.warning(f"⚠️  Strategy id '{strategy_id}' not found in CSV — skipping update.")
        return _no_change

    idx      = df_params[mask].index[0]
    prev_row = df_params.loc[idx]

    # -------------------------------------------------------------------------
    # Detect param changes (computed always, applied only if validated)
    # -------------------------------------------------------------------------
    params_changed = False
    param_changes  = []
    logger.debug(f"ID : {strategy_id}")
    logger.debug(f"  {'Parameter':<20} {'Previous':>12} {'New':>12} {'Changed':>10}")
    logger.debug(f"  {'-'*56}")
    for k in param_keys:
        if k == "sell_after":
            continue
        prev_val = normalize(prev_row.get(k, None))
        new_val  = normalize(best_params.get(k.upper(), None))
        changed  = "⚠️  YES" if prev_val != new_val else "✅"
        if prev_val != new_val:
            params_changed = True
            param_changes.append(f"{k}: {prev_val}→{new_val}")
        logger.debug(f"  {k:<20} {str(prev_val):>12} {str(new_val):>12} {changed:>10}")

    logger.debug(f"  {'Metric':<20} {'Previous':>12} {'New':>12}")
    logger.debug(f"  {'-'*46}")
    for metric, new_val in [("bt_netgain_pct", round(bt_netgain_pct, 2)), ("bt_r2", round(r2, 3)), ("prob_negative", round(prob_negative_oos, 2))]:
        prev_val = prev_row.get(metric, None)
        logger.debug(f"  {metric:<20} {str(prev_val):>12} {str(new_val):>12}")

    # -------------------------------------------------------------------------
    # Detect regime changes (computed always, applied only if validated)
    # -------------------------------------------------------------------------
    regime_changes = []
    new_regime_flags = {}
    if regime_stats:
        for family in ("trending", "ranging", "volatile"):
            col       = f"regime_{family}"
            new_flag  = 1.0 if (regime_stats.get(family, {}).get("profit", 0) >= 0) else 0.0
            prev_flag = normalize(prev_row.get(col, None))
            new_regime_flags[col] = new_flag
            if prev_flag != new_flag:
                regime_changes.append(f"{col}: {prev_flag}→{new_flag}")

    # -------------------------------------------------------------------------
    # Detect active change
    # -------------------------------------------------------------------------
    prev_active = bool(prev_row.get("active", False)) if pd.notna(prev_row.get("active")) else False
    new_active  = bool(validated)

    # -------------------------------------------------------------------------
    # Always update diagnostic metrics
    # -------------------------------------------------------------------------
    df_params.at[idx, "validated"]      = validated
    df_params.at[idx, "bt_netgain_pct"] = round(bt_netgain_pct, 2)
    df_params.at[idx, "bt_r2"]          = round(float(r2), 2)
    df_params.at[idx, "prob_negative"]  = round(prob_negative_oos, 2)
    df_params.at[idx, "last_run"]       = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # Force string dtype to prevent NaN passthrough on empty cells
    for col in ("last_change_active", "last_change_params", "last_change_regime"):
        df_params[col] = df_params[col].astype(object)

    # Default change columns to N/A — overwritten below based on outcome
    df_params.at[idx, "last_change_active"] = "N/A"
    df_params.at[idx, "last_change_params"] = "N/A"
    df_params.at[idx, "last_change_regime"] = "N/A"

    # Active
    df_params.at[idx, "active"] = bool(validated)
    if prev_active and not validated:
        df_params.at[idx, "last_change_active"] = "True→False"
    elif not prev_active and validated:
        df_params.at[idx, "last_change_active"] = "False→True"

    # Params — always updated
    if params_changed:
        for k in param_keys:
            if k in df_params.columns:
                df_params.at[idx, k] = best_params.get(k.upper())
        df_params.at[idx, "last_change_params"] = " | ".join(param_changes)

    # Regime — always updated
    for col, new_flag in new_regime_flags.items():
        df_params.at[idx, col] = new_flag
    if regime_changes:
        df_params.at[idx, "last_change_regime"] = " | ".join(regime_changes)

    for col in ("last_change_active", "last_change_params", "last_change_regime"):
        df_params[col] = df_params[col].fillna("N/A")

    df_params.to_csv(csv_path, index=False)

    return {
        "params_changed": params_changed and validated,
        "param_changes":  param_changes if validated else [],
        "regime_changes": regime_changes if validated else [],
        "active_prev":    prev_active,
        "active_new":     new_active,
    }


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
# HELPER — SAVE STRATEGIES BATCH (update dynamic fields + write E1 output)
# =============================================================================
def save_strategies_e1(strategies_batch_path, output_path, validation_results, best_params_map):
    """
    Update strategies_batch.py with dynamic fields from memory and generate
    strategies_E1_batch.py for production deployment.

    strategies_batch_path : path to strategies_batch.py (source of truth)
    output_path           : path to write strategies_E1_batch.py
    validation_results    : list of dicts with strategy_id, verdict, regime_*
    best_params_map       : dict {strategy_id: best_params dict}
    """
    if not os.path.exists(strategies_batch_path):
        logger.warning(f"⚠️  strategies_batch.py not found — skipping.")
        return

    # Load current strategies_batch
    import importlib.util
    spec   = importlib.util.spec_from_file_location("strategies_batch", strategies_batch_path)
    mod    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    strategies = mod.STRATEGIES

    # Build lookup from validation results
    val_map = {v["strategy_id"]: v for v in validation_results}

    def _fmt(val):
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, str):
            return f'"{val}"'
        return str(val)

    # -------------------------------------------------------------------------
    # Update strategies_batch.py
    # -------------------------------------------------------------------------
    batch_lines = [
        '"""',
        'strategies_batch.py — Source of truth for BOT_batch.',
        '',
        'Static fields : strategy_id, signal, side, timeframe,',
        '                direction_mode, order_amount_prod, sell_after_ncandles.',
        '',
        'Dynamic fields (updated by batch): active, regime_trending,',
        '                                   regime_ranging, regime_volatile,',
        '                                   and all optimized params.',
        '"""',
        '',
        'STRATEGIES = [',
    ]

    for s in strategies:
        sid = s["strategy_id"]
        v   = val_map.get(sid, {})
        bp  = best_params_map.get(sid, {})

        updated = dict(s)
        if v:
            updated["active"]           = v["verdict"] == "🟢 VALIDATED"
            updated["regime_trending"]  = v.get("regime_trending", s.get("regime_trending", 1.0))
            updated["regime_ranging"]   = v.get("regime_ranging",  s.get("regime_ranging",  1.0))
            updated["regime_volatile"]  = v.get("regime_volatile", s.get("regime_volatile", 1.0))
        if bp:
            for k, val in bp.items():
                updated[k.lower()] = val

        batch_lines.append("    {")
        batch_lines.append(f'        # --- Identification ---')
        batch_lines.append(f'        "strategy_id": "{updated["strategy_id"]}",')
        batch_lines.append(f'        "signal": "{updated["signal"]}",')
        batch_lines.append(f'        "side": "{updated["side"]}",')
        batch_lines.append(f'        "timeframe": "{updated["timeframe"]}",')
        batch_lines.append(f'')
        batch_lines.append(f'        # --- Production config (static) ---')
        batch_lines.append(f'        "direction_mode": "{updated.get("direction_mode", "general")}",')
        batch_lines.append(f'        "order_amount_prod": {updated.get("order_amount_prod", 200)},')
        batch_lines.append(f'        "sell_after_ncandles": {updated.get("sell_after_ncandles", 0)},')
        batch_lines.append(f'')
        batch_lines.append(f'        # --- Updated by batch ---')
        batch_lines.append(f'        "active": {updated.get("active", False)},')
        batch_lines.append(f'        "regime_trending": {float(updated.get("regime_trending", 1.0))},')
        batch_lines.append(f'        "regime_ranging": {float(updated.get("regime_ranging", 1.0))},')
        batch_lines.append(f'        "regime_volatile": {float(updated.get("regime_volatile", 1.0))},')

        _PARAM_KEYS = {"lookback", "tolerance", "ma_period", "tp_pct", "sl_pct", "impulse", "flag", "ranges"}
        for k in _PARAM_KEYS:
            if k in updated:
                batch_lines.append(f'        "{k}": {_fmt(updated[k])},')
        batch_lines.append("    },")

    batch_lines.append("]")
    os.makedirs(os.path.dirname(os.path.abspath(strategies_batch_path)), exist_ok=True)
    with open(strategies_batch_path, "w") as f:
        f.write("\n".join(batch_lines) + "\n")
    logger.debug(f"strategies_batch.py updated → {strategies_batch_path}")

    # -------------------------------------------------------------------------
    # Write strategies_E1_batch.py (production format)
    # -------------------------------------------------------------------------
    e1_lines = [
        '"""',
        'Trading Strategies Configuration',
        '',
        'Auto-generated by BOT_batch. Do not edit manually.',
        'Copy to BOT_trading/config/strategies_E1.py to deploy.',
        '"""',
        '',
        'STRATEGIES = [',
    ]

    for s in strategies:
        sid = s["strategy_id"]
        v   = val_map.get(sid, {})
        bp  = best_params_map.get(sid, {})

        updated = dict(s)
        if v:
            updated["active"]          = v["verdict"] == "🟢 VALIDATED"
            updated["regime_trending"] = v.get("regime_trending", s.get("regime_trending", 1.0))
            updated["regime_ranging"]  = v.get("regime_ranging",  s.get("regime_ranging",  1.0))
            updated["regime_volatile"] = v.get("regime_volatile", s.get("regime_volatile", 1.0))
        if bp:
            for k, val in bp.items():
                updated[k.lower()] = val

        name = "_".join(sid.split("_")[1:])  # e.g. "reversal_long_4H"

        e1_lines.append("    {")
        e1_lines.append(f'        "id": "{sid}",')
        e1_lines.append(f'        "name": "{name}",')
        e1_lines.append(f'        "timeframe": "{updated["timeframe"]}",')
        e1_lines.append(f'        "active": {updated.get("active", False)},')
        e1_lines.append(f'        "direction": "{updated["side"]}",')
        e1_lines.append(f'        "regime_trending": {float(updated.get("regime_trending", 1.0))},')
        e1_lines.append(f'        "regime_ranging": {float(updated.get("regime_ranging", 1.0))},')
        e1_lines.append(f'        "regime_volatile": {float(updated.get("regime_volatile", 1.0))},')
        e1_lines.append(f'        "direction_mode": "{updated.get("direction_mode", "general")}",')
        e1_lines.append(f'        "sell_after_ncandles": {updated.get("sell_after_ncandles", 0)},')
        e1_lines.append(f'        "order_amount": {updated.get("order_amount_prod", 200)},')

        for k in _PARAM_KEYS:
            if k in updated:
                e1_lines.append(f'        "{k}": {_fmt(updated[k])},')
        e1_lines.append("    },")

    e1_lines.append("]")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(e1_lines) + "\n")
    logger.debug(f"strategies_E1_batch.py updated → {output_path}")


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
        logger.info(f"🔵 symbols_live — symbols updated for '{strategy_id}'")
    else:
        logger.debug(f"⚪ symbols_live — symbols unchanged for '{strategy_id}'")

    return {"symbols_changed": symbols_changed}


# =============================================================================
# HELPER — LOAD BTC 1D
# =============================================================================
def load_btc_1d(btc_file, ma_period=5):
    """
    Load BTC 1Dutc parquet and compute rolling MA.
    Returns DataFrame with 'ts' and 'ma{ma_period}' columns.
    """
    df = pd.read_parquet(btc_file)
    df.columns = df.columns.str.lower()
    df["ts"] = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df.index)
    df = df.sort_values("ts").reset_index(drop=True)
    df[f"ma{ma_period}"] = df["close"].rolling(window=ma_period).mean()
    return df


# =============================================================================
# HELPER — GET BTC DIRECTION AT TRADE TIME
# =============================================================================
def get_btc_direction(buy_time, btc_df, side, ma_period=5, long_th=1.00, short_th=1.00):
    """
    Classify a trade's BTC direction at buy_time (no lookahead).
    Returns 'uptrend', 'downtrend' or 'unknown'.
    """
    closed = btc_df[btc_df["ts"] < buy_time]
    if len(closed) < ma_period:
        return "unknown"
    last = closed.iloc[-1]
    if pd.isna(last[f"ma{ma_period}"]):
        return "unknown"
    if side == "long":
        return "uptrend" if last["close"] > last[f"ma{ma_period}"] * long_th else "downtrend"
    else:
        return "downtrend" if last["close"] < last[f"ma{ma_period}"] * short_th else "uptrend"


# =============================================================================
# PORTFOLIO ANALYSIS — EQUITY METRICS
# =============================================================================
RESAMPLE_FREQ = '1D'
BARS_PER_DAY  = 1


def resample_equity(df_indexed):
    common_index = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq=RESAMPLE_FREQ
    )
    df_r = df_indexed[['balance']].reindex(common_index)
    df_r['balance'] = df_r['balance'].ffill().bfill()
    df_r.index.name = 'timestamp'
    return df_r


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
        "Volatility_pct": round(float(volatility), 2),
        "Monthly_pct":    round(float(consistency), 2),
    }


def print_metrics_table(metrics_list, title, shorten_names=False, use_info=False):
    def _shorten(name):
        segments = name.strip().split("+")
        result = []
        for seg in segments:
            for part in seg.split("_"):
                if part.isdigit():
                    result.append(part)
                    break
        return "+".join(result) if result else name

    df = pd.DataFrame(metrics_list)
    df['Curve'] = df['Curve'].astype(str)
    if shorten_names:
        df['Curve'] = df['Curve'].apply(_shorten)
    max_len = df['Curve'].str.len().max()
    df['Curve'] = df['Curve'].apply(lambda x: x.ljust(max_len))
    msg = f"\n{title}\n{df.to_string(index=False)}"
    if use_info:
        logger.info(msg)
    else:
        logger.debug(msg)


# =============================================================================
# HELPER — PRINT STRATEGIES SUMMARY
# =============================================================================
def print_strategies_summary(validation_results):
    """Print validation summary table for all strategies."""
    if not validation_results:
        return
    lines = []
    lines.append(f"\n{'─'*105}")
    lines.append(f"  STRATEGIES SUMMARY")
    lines.append(f"{'─'*105}")
    lines.append(f"  {'Strategy':<25} {'Verdict':<14} {'Round':<16} {'NetGain%':>10} {'DD%':>8} {'WinRate%':>10} {'R2':>7} {'ProbNeg%':>10}")
    lines.append(f"  {'-'*103}")
    for v in validation_results:
        lines.append(
            f"  {v['strategy_id']:<25} {v['verdict']:<14} {v['round']:<16} "
            f"{v['net_gain_pct']:>9.2f}% {v['dd_pct']:>7.2f}% {v['win_ratio']:>9.1f}% "
            f"{v['r2']:>7.3f} {v['prob_neg_pct']:>9.2f}%"
        )
    lines.append(f"  {'─'*105}")
    logger.info("\n".join(lines))


# =============================================================================
# HELPER — PRINT UPDATE STATUS (four separate tables, reads from CSV)
# =============================================================================
def print_update_status(csv_path, symbols_live_folder, validation_results):
    """
    Print four update status tables reading from CSV as source of truth:
      1. Active
      2. Params
      3. Market Regime
      4. Symbols
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
        if val in ("N/A", "—", ""):    return "⚪ no change"
        if val == "REJECTED":                   return "🔴 REJECTED"
        return f"🔵 {val}"

    # -------------------------------------------------------------------------
    # Table 1 — Active
    # -------------------------------------------------------------------------
    lines = []
    lines.append(f"\n{'─'*105}")
    lines.append(f"  ACTIVE")
    lines.append(f"{'─'*105}")
    lines.append(f"  {'Strategy':<25} {'Status':<16} {'Changes':<35}")
    lines.append(f"  {'-'*103}")
    for sid in strategy_ids:
        change = _get(sid, "last_change_active")
        lines.append(f"  {sid:<25} {_active_icon(sid):<16} {_change_icon(change):<35}")
    lines.append(f"  {'─'*103}")
    logger.info("\n".join(lines))

    # -------------------------------------------------------------------------
    # Table 2 — Params
    # -------------------------------------------------------------------------
    lines = []
    lines.append(f"\n{'─'*105}")
    lines.append(f"  PARAMS")
    lines.append(f"{'─'*105}")
    lines.append(f"  {'Strategy':<25} {'Changes':<50}")
    lines.append(f"  {'-'*103}")
    for sid in strategy_ids:
        change = _get(sid, "last_change_params")
        lines.append(f"  {sid:<25} {_change_icon(change):<50}")
    lines.append(f"  {'─'*103}")
    logger.info("\n".join(lines))

    # -------------------------------------------------------------------------
    # Table 3 — Market Regime
    # -------------------------------------------------------------------------
    lines = []
    lines.append(f"\n{'─'*105}")
    lines.append(f"  MARKET REGIME")
    lines.append(f"{'─'*105}")
    lines.append(f"  {'Strategy':<25} {'Trending':>10} {'Ranging':>10} {'Volatile':>10} {'Changes':<40}")
    lines.append(f"  {'-'*103}")
    for sid in strategy_ids:
        trending = _get(sid, "regime_trending")
        ranging  = _get(sid, "regime_ranging")
        volatile = _get(sid, "regime_volatile")
        change   = _get(sid, "last_change_regime")
        lines.append(
            f"  {sid:<25} {trending:>10} {ranging:>10} {volatile:>10} {_change_icon(change):<40}"
        )
    lines.append(f"  {'─'*105}")
    logger.info("\n".join(lines))

    # -------------------------------------------------------------------------
    # Table 4 — Symbols
    # -------------------------------------------------------------------------
    lines = []
    lines.append(f"\n{'─'*105}")
    lines.append(f"  SYMBOLS")
    lines.append(f"{'─'*105}")
    lines.append(f"  {'Strategy':<25} {'Status':<16}")
    lines.append(f"  {'-'*103}")
    for v in validation_results:
        sid  = v["strategy_id"]
        icon = "🔵 updated" if v.get("symbols_changed") else "⚪ no change"
        lines.append(f"  {sid:<25} {icon:<16}")
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
    Print a metrics table for all curves plus a combined row.
    trade_logs: list of (strategy_id, trade_log_df)
    """
    named = {sid: df for sid, df in trade_logs}
    rows  = []
    for sid, df in named.items():
        rows.append(compute_metrics(df, capital=initial_balance, name=sid))

    all_tl  = pd.concat(list(named.values()), ignore_index=True).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
    all_cap = initial_balance * len(named)
    rows.append(compute_metrics(all_tl, capital=all_cap, name="── Combined"))

    df_out = pd.DataFrame(rows)

    # Compute Profit_pctT excluding the Combined row
    strategy_rows  = df_out[df_out["Curve"].str.strip() != "── Combined"]
    total_profit   = strategy_rows["Profit_abs"].sum()
    df_out["Profit_pctT"] = df_out["Profit_abs"].apply(
        lambda x: round(x / total_profit * 100, 1) if total_profit != 0 else np.nan
    )

    cols   = ["Curve", "Net_Gain_pct", "Max_DD_pct", "Win_Rate", "R_Squared", "Profit_Factor", "Profit_abs", "Profit_pctT", "Volatility_pct", "Monthly_pct"]
    df_out = df_out[cols].copy()
    max_len = df_out["Curve"].str.len().max()
    df_out["Curve"]      = df_out["Curve"].apply(lambda x: x.ljust(max_len))
    df_out["Profit_abs"] = df_out["Profit_abs"].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    lines = [f"\n📊 ALL CURVES COMBINED — {label}\n", df_out.to_string(index=False)]
    logger.info("\n".join(lines))


# =============================================================================
# HELPER — PRINT BEST COMBINATIONS
# =============================================================================
def print_best_combinations(trade_logs, label, initial_balance, precomputed_metrics=None):
    """
    Compute and print best strategy combinations by Net Gain, R² and Profit Factor.
    trade_logs         : list of (strategy_id, trade_log_df)
    precomputed_metrics: optional dict {strategy_id: metrics} to avoid recalculation
    """
    from itertools import combinations as _combinations

    def _num(sid):
        for part in sid.split("_"):
            if part.isdigit():
                return int(part)
        return 0

    named   = {sid: df for sid, df in trade_logs}
    metrics = precomputed_metrics or {
        sid: compute_metrics(df, capital=initial_balance, name=sid)
        for sid, df in named.items()
    }

    combo_results = []
    for r in range(1, len(named) + 1):
        for combo in _combinations(named.keys(), r):
            if len(combo) == 1:
                sid = combo[0]
                m   = metrics.get(sid)
                if m:
                    combo_results.append({**m, "Curve": str(_num(sid))})
            else:
                combo_tl = pd.concat(
                    [named[sid] for sid in combo], ignore_index=True
                ).sort_values(["buy_time", "symbol"]).reset_index(drop=True)
                capital  = initial_balance * len(combo)
                nums     = "+".join(str(_num(sid)) for sid in sorted(combo, key=_num))
                combo_results.append(compute_metrics(combo_tl, capital=capital, name=nums))

    combo_df = pd.DataFrame(combo_results)

    best_ng    = combo_df.loc[combo_df["Net_Gain_pct"].idxmax()]
    best_r2    = combo_df.loc[combo_df["R_Squared"].idxmax()]
    best_pf_df = combo_df[combo_df["Profit_Factor"] != float("inf")]
    best_pf    = best_pf_df.loc[best_pf_df["Profit_Factor"].idxmax()] if not best_pf_df.empty else best_ng

    rows = [
        ("💵 Net Gain",     best_ng),
        ("📈 R²",           best_r2),
        ("💰 ProfitFactor", best_pf),
    ]

    lines = []
    lines.append(f"\n{'─'*105}")
    lines.append(f"  BEST COMBINATIONS — {label}")
    lines.append(f"{'─'*105}")
    lines.append(f"  {'Metric':<16} {'Combo':<20} {'NetGain%':>10} {'DD%':>8} {'Win%':>7} {'R2':>7} {'ProfFactor':>12}")
    lines.append(f"  {'-'*103}")
    for lbl, row in rows:
        pf_str = f"{row['Profit_Factor']:>11.3f}" if row['Profit_Factor'] != float("inf") else f"{'∞':>12}"
        lines.append(
            f"  {lbl:<16} {str(row['Curve']):<20} {row['Net_Gain_pct']:>9.2f}% "
            f"{row['Max_DD_pct']:>7.2f}% {row['Win_Rate']:>6.1f}% {row['R_Squared']:>7.3f} {pf_str}"
        )
    lines.append(f"  {'─'*103}")
    logger.info("\n".join(lines))