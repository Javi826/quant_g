import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("BOT_trading.batch.batch_utils")

# =============================================================================
# HELPER — REPORT FILTERED TRADES
# =============================================================================
def report_filtered_trades(trade_log, initial_balance, data_folder, title="Filtered Trades"):
    """
    Recompute equity curve, key metrics and plot from a filtered trade_log.
    Mirrors the output format of report_backtesting.
    """
    df = trade_log.copy().sort_values("buy_time").reset_index(drop=True)
    df["duration_d"] = (pd.to_datetime(df["sell_time"]) - pd.to_datetime(df["buy_time"])).dt.total_seconds() / 86400

    # --- Equity curve ---
    df["equity"] = initial_balance + df["profit"].cumsum()
    balances     = df["equity"].values
    timestamps   = pd.to_datetime(df["buy_time"])

    # --- Metrics ---
    net_gain      = balances[-1] - initial_balance
    net_gain_pct  = net_gain / initial_balance * 100
    win_ratio     = (df["profit"] > 0).mean()
    num_signals   = len(df)
    duration_d    = df["duration_d"].mean()
    cummax        = np.maximum.accumulate(balances)
    dd_pct        = ((balances - cummax) / cummax * 100).min()
    X             = np.arange(len(balances)).reshape(-1, 1)
    y             = balances.reshape(-1, 1)
    r2            = round(LinearRegression().fit(X, y).score(X, y), 3)
    sharpe        = (df["profit"].mean() / df["profit"].std() * np.sqrt(252)) if df["profit"].std() > 0 else np.nan

    # --- Summary table ---
    _param_cols = [c for c in ["sell_after","lookback","tolerance","ma_period","tp_pct","sl_pct","impulse","range_str"] if c in trade_log.columns]
    df_summary = pd.DataFrame([{
        "Metric": "Net_Gain_pct",
        **{k.upper(): v for k, v in trade_log[_param_cols].iloc[0].items()},
        "Net_Gain_pct": round(net_gain_pct, 2), "Win_Ratio": round(win_ratio, 2),
        "R2": r2, "Sharpe": round(sharpe, 2), "DD_pct": round(dd_pct, 2),
        "Num_Signals": num_signals, "duration_d": round(duration_d, 2)
    }])
    logger.debug(df_summary.to_string(index=False))

    # --- Monthly stats ---
    df["month"] = pd.to_datetime(df["buy_time"]).dt.to_period("M")
    monthly      = df.groupby("month")["profit"].sum()
    winning_m    = (monthly > 0).sum()
    logger.debug(f"\n{'-'*60}")
    logger.debug("MONTHLY STATISTICS")
    logger.debug(f"{'-'*60}")
    logger.debug(f"Winning Months: {winning_m} / {len(monthly)} ({winning_m/len(monthly)*100:.2f}%)")

    # --- Plot ---
    net_gain_arr = (balances - initial_balance) / initial_balance * 100
    cummax_arr   = np.maximum.accumulate(balances)
    dd_arr       = (balances - cummax_arr) / cummax_arr * 100

    btc_file = os.path.join(data_folder, "BTCUSDT_4H.parquet")
    btc_df   = pd.read_parquet(btc_file)
    if "timestamp" not in btc_df.columns:
        btc_df = btc_df.reset_index().rename(columns={"index": "timestamp"})
    btc_df["timestamp"]    = pd.to_datetime(btc_df["timestamp"])
    btc_df["btc_net_gain"] = (btc_df["close"] / btc_df["close"].iloc[0] - 1) * 100
    btc_aligned = np.interp(
        timestamps.astype(np.int64) / 10**9,
        btc_df["timestamp"].astype(np.int64) / 10**9,
        btc_df["btc_net_gain"]
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    above = net_gain_arr >= btc_aligned
    ax1.fill_between(timestamps, net_gain_arr, 0, where=above,  alpha=0.2, color="green", interpolate=True)
    ax1.fill_between(timestamps, net_gain_arr, 0, where=~above, alpha=0.2, color="red",   interpolate=True)
    ax1.plot(timestamps, net_gain_arr, color="blue",       linewidth=1.2, label="Net Gain %")
    ax1.plot(btc_df["timestamp"], btc_df["btc_net_gain"], color="darkorange", linewidth=0.6, linestyle="--", label="BTC %")
    ax2 = ax1.twinx()
    ax2.plot(timestamps, dd_arr, color="lightcoral", linewidth=0.1, label="DD %")
    ax2.set_ylabel("Drawdown", color="red")
    textstr = (
        f"Net Gain STR : {net_gain_pct:.2f}%\n"
        f"Net Gain BTC : {btc_df['btc_net_gain'].iloc[-1]:.2f}%\n"
        f"Max DD       : {dd_pct:.2f}%\n"
        f"R²           : {r2:.3f}"
    )
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.suptitle(title)
    fig.autofmt_xdate()
    ax1.grid(True, linestyle="--", alpha=0.6)
    lines, labels   = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.show()
    return {
        "net_gain_pct": round(net_gain_pct, 2),
        "dd_pct":       round(dd_pct, 2),
        "win_ratio":    round(win_ratio * 100, 1),
        "r2":           r2,
    }


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
def select_universe(data_folder_is, data_folder_oos, timeframe, n_symbols, min_price, filter_symbols_fn, my_symbols=False):
    """
    Select OOS universe (top N by volume) and match IS universe.
    Returns: symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos
    """
    raw_is  = sorted([f.split("_")[0] for f in os.listdir(data_folder_is)  if f.endswith(f"_{timeframe}.parquet")])
    raw_oos = sorted([f.split("_")[0] for f in os.listdir(data_folder_oos) if f.endswith(f"_{timeframe}.parquet")])

    ohlcv_oos, filtered_oos = filter_symbols_fn(raw_oos, min_vol_usdt=0, timeframe=timeframe, data_folder=data_folder_oos, min_price=min_price, vol_window=50, my_symbols=my_symbols)
    ohlcv_is,  filtered_is  = filter_symbols_fn(raw_is,  min_vol_usdt=0, timeframe=timeframe, data_folder=data_folder_is,  min_price=min_price, vol_window=50, my_symbols=my_symbols)

    vol_oos           = {sym: ohlcv_oos[sym]["volume_quote"].tail(50).mean() for sym in filtered_oos}
    oos_ranked        = sorted(filtered_oos, key=lambda s: vol_oos.get(s, 0), reverse=True)
    symbols_oos_final = oos_ranked[:n_symbols]

    syms_is  = set(filtered_is)
    syms_oos = set(symbols_oos_final)
    in_both     = sorted(syms_is & syms_oos)
    only_in_oos = sorted(syms_oos - syms_is)

    vol_is               = {sym: ohlcv_is[sym]["volume_quote"].tail(50).mean() for sym in syms_is}
    is_candidates_by_vol = sorted(syms_is - syms_oos, key=lambda s: vol_is.get(s, 0), reverse=True)
    needed               = max(0, n_symbols - len(in_both))
    symbols_is_final     = sorted(in_both + is_candidates_by_vol[:needed])

    logger.debug(f"OOS pool ({len(filtered_oos):>3}): {len(filtered_oos)} candidates")
    logger.debug(f"IS  pool ({len(filtered_is):>3}): {len(filtered_is)} candidates")
    logger.debug(f"In both  ({len(in_both):>3}): {in_both}")
    logger.debug(f"Only in OOS ({len(only_in_oos):>3}): {only_in_oos}")
    logger.debug(f"OOS final universe ({len(symbols_oos_final):>3}): {sorted(symbols_oos_final)}")
    logger.debug(f"IS  final universe ({len(symbols_is_final):>3}): {symbols_is_final}")

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
def update_strategies_params(csv_path, strategy_id, best_params, param_keys, validated, bt_netgain_pct, r2, prob_negative_oos):
    def normalize(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    _no_change = {"params_changed": False, "active_prev": None, "active_new": None}

    if not os.path.exists(csv_path):
        logger.warning(f"⚠️  strategies_params.csv not found at {csv_path} — skipping update.")
        return _no_change

    df_params = pd.read_csv(csv_path)
    mask      = df_params["id"] == strategy_id

    if not mask.any():
        logger.warning(f"⚠️  Strategy id '{strategy_id}' not found in CSV — skipping update.")
        return _no_change

    idx      = df_params[mask].index[0]
    prev_row = df_params.loc[idx]

    # Detect param changes
    params_changed = False
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
        logger.debug(f"  {k:<20} {str(prev_val):>12} {str(new_val):>12} {changed:>10}")

    logger.debug(f"  {'Metric':<20} {'Previous':>12} {'New':>12}")
    logger.debug(f"  {'-'*46}")
    for metric, new_val in [("bt_netgain_pct", round(bt_netgain_pct, 2)), ("bt_r2", round(r2, 3)), ("prob_negative", round(prob_negative_oos, 2))]:
        prev_val = prev_row.get(metric, None)
        logger.debug(f"  {metric:<20} {str(prev_val):>12} {str(new_val):>12}")

    # Detect active change
    prev_active = bool(prev_row.get("active", False)) if pd.notna(prev_row.get("active")) else False
    new_active  = bool(validated)

    # Always update diagnostic metrics
    df_params.at[idx, "validated"]      = validated
    df_params.at[idx, "bt_netgain_pct"] = round(bt_netgain_pct, 2)
    df_params.at[idx, "bt_r2"]          = round(r2, 3)
    df_params.at[idx, "prob_negative"]  = round(prob_negative_oos, 2)
    df_params.at[idx, "last_run"]       = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    if validated:
        # Update production params only if validated
        for k in param_keys:
            if k in df_params.columns:
                df_params.at[idx, k] = best_params.get(k.upper())
        df_params.at[idx, "active"] = True
        if params_changed:
            logger.info(f"🔵 strategies_params.csv — params updated for '{strategy_id}'")
        if not prev_active:
            logger.info(f"🟠 strategies_params.csv — activated '{strategy_id}'")
    else:
        df_params.at[idx, "active"] = False
        if prev_active:
            logger.info(f"🔴 strategies_params.csv — deprecated '{strategy_id}'")

    df_params.to_csv(csv_path, index=False)

    return {
        "params_changed": params_changed and validated,
        "active_prev":    prev_active,
        "active_new":     new_active,
    }


# =============================================================================
# HELPER — UPDATE STRATEGIES SYMBOLS CSV
# =============================================================================
def update_strategies_symbols(csv_path, strategy_id, symbols_oos_final, timeframe=None, symbols_live_folder=None):
    symbols_str = str(symbols_oos_final)

    if not os.path.exists(csv_path):
        df_symbols = pd.DataFrame(columns=["id", "symbols"])
        symbols_changed = True
    else:
        df_symbols = pd.read_csv(csv_path)
        mask_sym = df_symbols["id"] == strategy_id
        prev_symbols_str = df_symbols.loc[mask_sym, "symbols"].values[0] if mask_sym.any() else None
        symbols_changed = prev_symbols_str != symbols_str

    mask_sym = df_symbols["id"] == strategy_id
    if mask_sym.any():
        df_symbols.loc[df_symbols["id"] == strategy_id, "symbols"] = symbols_str
    else:
        df_symbols = pd.concat([
            df_symbols,
            pd.DataFrame([{"id": strategy_id, "symbols": symbols_str}])
        ], ignore_index=True)

    df_symbols.to_csv(csv_path, index=False)

    # Generate symbols_live file
    if timeframe and symbols_live_folder:
        os.makedirs(symbols_live_folder, exist_ok=True)
        live_filename = f"symbols_live_{strategy_id}_{timeframe}.csv"
        live_path     = os.path.join(symbols_live_folder, live_filename)
        pd.DataFrame(symbols_oos_final).to_csv(live_path, index=False, header=False)
        logger.debug(f"symbols_live saved → {live_path}")

    if symbols_changed:
        logger.info(f"🔵 strategies_symbols.csv — symbols updated for '{strategy_id}'")
    else:
        logger.debug(f"⚪ strategies_symbols.csv — symbols unchanged for '{strategy_id}'")
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

    # Win rate — trade level
    win_rate = round((profits > 0).mean() * 100, 1)

    # Profit factor — trade level (standard definition)
    gains  = profits[profits > 0].sum()
    losses = -profits[profits < 0].sum()
    pf     = round(float(gains / losses), 3) if losses > 0 else np.inf

    # Daily equity curve
    tl["_date"]  = pd.to_datetime(tl["sell_time"]).dt.normalize()
    daily_profit = tl.groupby("_date")["profit"].sum()

    date_range   = pd.date_range(start=daily_profit.index.min(),
                                 end=daily_profit.index.max(),
                                 freq="1D")
    daily_profit = daily_profit.reindex(date_range, fill_value=0.0)
    eq           = capital + daily_profit.cumsum().values
    eq_series    = pd.Series(eq, index=date_range)

    # Drawdown & net gain
    cm       = np.maximum.accumulate(eq)
    max_dd   = ((eq - cm) / cm * 100).min()
    net_gain = (eq[-1] - capital) / capital * 100

    # Volatility & monthly consistency — daily returns
    daily_returns = eq_series.pct_change().dropna()
    volatility    = daily_returns.std() * 100
    monthly       = eq_series.resample("ME").last().pct_change().dropna()
    consistency   = (monthly > 0).mean() * 100

    # R²
    X  = np.arange(len(eq)).reshape(-1, 1)
    y  = eq.reshape(-1, 1)
    r2 = round(_LR().fit(X, y).score(X, y), 3)

    return {
        "Curve":          name,
        "Volatility_pct": round(float(volatility), 2),
        "Monthly_pct":    round(float(consistency), 2),
        "Net_Gain_pct":   round(float(net_gain), 2),
        "Max_DD_pct":     round(float(max_dd), 2),
        "Profit_Factor":  pf,
        "R_Squared":      r2,
        "Win_Rate":       win_rate,
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