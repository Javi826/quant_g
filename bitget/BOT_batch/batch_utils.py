import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
    df_summary = pd.DataFrame([{
        "Metric": "Net_Gain_pct",
        **{k.upper(): v for k, v in trade_log[["sell_after","lookback","tolerance","ma_period","tp_pct","sl_pct"]].iloc[0].items()},
        "Net_Gain_pct": round(net_gain_pct, 2), "Win_Ratio": round(win_ratio, 2),
        "R2": r2, "Sharpe": round(sharpe, 2), "DD_pct": round(dd_pct, 2),
        "Num_Signals": num_signals, "duration_d": round(duration_d, 2)
    }])
    print(df_summary.to_string(index=False))

    # --- Monthly stats ---
    df["month"] = pd.to_datetime(df["buy_time"]).dt.to_period("M")
    monthly      = df.groupby("month")["profit"].sum()
    winning_m    = (monthly > 0).sum()
    print(f"\n{'-'*60}")
    print("MONTHLY STATISTICS")
    print(f"{'-'*60}")
    print(f"Winning Months: {winning_m} / {len(monthly)} ({winning_m/len(monthly)*100:.2f}%)")

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
    print("\n   ▶ Extracting optimal params (best Net_Gain_pct_m)...")
    print("   Best params: " + " | ".join(f"{k}: {v}" for k, v in best_params.items()))
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

    print(f"\n  OOS pool         ({len(filtered_oos):>3}): {len(filtered_oos)} candidates")
    print(f"  IS  pool         ({len(filtered_is):>3}): {len(filtered_is)} candidates")
    print(f"\n  In both          ({len(in_both):>3}): {in_both}")
    print(f"  Only in OOS      ({len(only_in_oos):>3}): {only_in_oos}")
    print(f"\n  ▶ OOS final universe ({len(symbols_oos_final):>3}): {sorted(symbols_oos_final)}")
    print(f"  ▶ IS  final universe ({len(symbols_is_final):>3}): {symbols_is_final}")

    if len(symbols_is_final) < n_symbols:
        print(f"\n  ⚠️  IS has only {len(symbols_is_final)} symbols — fewer than N_SYMBOLS ({n_symbols}). Proceeding with available.")

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
def update_strategies_params(csv_path, strategy_id, best_params, param_keys, approved, bt_netgain_pct, r2, prob_negative_oos):
    def normalize(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    if not os.path.exists(csv_path):
        print(f"  ⚠️  strategies_params.csv not found at {csv_path} — skipping update.")
        return

    df_params = pd.read_csv(csv_path)
    mask      = df_params["id"] == strategy_id

    if not mask.any():
        print(f"  ⚠️  Strategy id '{strategy_id}' not found in CSV — skipping update.")
        return

    idx      = df_params[mask].index[0]
    prev_row = df_params.loc[idx]

    print(f"\n  ID : {strategy_id}")
    print(f"\n  {'Parameter':<20} {'Previous':>12} {'New':>12} {'Changed':>10}")
    print(f"  {'-'*56}")
    for k in param_keys:
        if k == "sell_after":
            continue
        prev_val = normalize(prev_row.get(k, None))
        new_val  = normalize(best_params.get(k.upper(), None))
        changed  = "⚠️  YES" if prev_val != new_val else "✅"
        print(f"  {k:<20} {str(prev_val):>12} {str(new_val):>12} {changed:>10}")

    print(f"\n  {'Metric':<20} {'Previous':>12} {'New':>12}")
    print(f"  {'-'*46}")
    for metric, new_val in [("bt_netgain_pct", round(bt_netgain_pct, 2)), ("bt_r2", round(r2, 3)), ("prob_negative", round(prob_negative_oos, 2))]:
        prev_val = prev_row.get(metric, None)
        print(f"  {metric:<20} {str(prev_val):>12} {str(new_val):>12}")

    df_params.at[idx, "approved"]       = approved
    df_params.at[idx, "bt_netgain_pct"] = round(bt_netgain_pct, 2)
    df_params.at[idx, "bt_r2"]          = round(r2, 3)
    df_params.at[idx, "prob_negative"]  = round(prob_negative_oos, 2)
    df_params.at[idx, "last_run"]       = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    if approved:
        for k in param_keys:
            if k in df_params.columns:
                df_params.at[idx, k] = best_params.get(k.upper())
        df_params.at[idx, "active"] = True
        print(f"\n  ✅ strategies_params.csv updated — params updated, active set to True for '{strategy_id}'")
    else:
        df_params.at[idx, "active"] = False
        print(f"\n  ❌ Strategy rejected — params NOT updated, active set to False for '{strategy_id}'")

    df_params.to_csv(csv_path, index=False)


# =============================================================================
# HELPER — UPDATE STRATEGIES SYMBOLS CSV
# =============================================================================
def update_strategies_symbols(csv_path, strategy_id, symbols_oos_final):
    symbols_str = str(symbols_oos_final)

    if not os.path.exists(csv_path):
        df_symbols = pd.DataFrame(columns=["id", "symbols"])
    else:
        df_symbols = pd.read_csv(csv_path)

    mask_sym = df_symbols["id"] == strategy_id
    if mask_sym.any():
        df_symbols.loc[df_symbols["id"] == strategy_id, "symbols"] = symbols_str
    else:
        df_symbols = pd.concat([
            df_symbols,
            pd.DataFrame([{"id": strategy_id, "symbols": symbols_str}])
        ], ignore_index=True)

    df_symbols.to_csv(csv_path, index=False)
    print(f"\n  ✅ strategies_symbols.csv updated — {len(symbols_oos_final)} symbols for '{strategy_id}'")
    
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