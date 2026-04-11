import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "development")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "BOT_trading")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from tools.ZX_st_tools import (
    extract_ohlcv_from_path,
    compile_MC_results,
    compile_grid_results,
    prepare_ohlcv_arrays,
    get_n_obs,
    save_all_trades_to_excel,
)
from tools.ZX_optimize_MCf_tf import generate_paths_for_all_symbols_functional
from utils.ZX_analysis import report_montecarlo, report_backtesting
from utils.ZX_utils import filter_symbols, final_prints
from signals.add_signals_parity import parity_long, parity_short

DTYPE      = np.float32
start_time = time.time()

# =============================================================================
# CONFIGURATION
# =============================================================================
STRATEGY_ID     = "03_parity_long_4H"
SIDE            = "long"           # "long" | "short"
N_JOBS          = -1
MY_SYMBOLS      = False

DATA_FOLDER_IS  = "../development/data/crypto_2022_IS"
DATA_FOLDER_OOS = "../development/data/crypto_2026_OOS"
TIMEFRAME_MINOR = "4H"
ORDER_AMOUNT    = 80
N_SYMBOLS       = 9               # top N OOS symbols by volume

# =============================================================================
# PARAMETER GRID
# =============================================================================
SELL_AFTER_LIST  = [0]
LOOKBACK_LIST    = [100, 150]
MA_PERIOD_LIST   = [25]
TOLERANCE_LIST   = [15, 30, 45]
TP_PCT_LIST      = [2, 3, 4]
SL_PCT_LIST      = [8, 9, 10]

SELL_AFTER_LIST  = [0]
LOOKBACK_LIST    = [150]
MA_PERIOD_LIST   = [50]
TOLERANCE_LIST   = [40]
TP_PCT_LIST      = [3]
SL_PCT_LIST      = [10]

param_names     = ["SELL_AFTER", "LOOKBACK", "TOLERANCE", "MA_PERIOD", "TP_PCT", "SL_PCT"]
lists_for_grid  = [globals()[f"{name}_LIST"] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

# =============================================================================
# MONTE CARLO SETTINGS
# =============================================================================
N_PATHS_IS           = 10
N_PATHS_OOS          = 200
FINAL_N_OBS_PER_PATH = get_n_obs(TIMEFRAME_MINOR)

# =============================================================================
# SIGNAL FUNCTION
# =============================================================================
signal_fn = parity_long if SIDE == "long" else parity_short

# =============================================================================
# SYMBOL DIAGNOSTICS & UNIVERSE SELECTION
# -----------------------------------------------------------------------------
# 1. Load all OOS symbols (no vol filter), rank by volume, take top N_SYMBOLS
# 2. Load all IS  symbols (no vol filter, enough bars only)
# 3. Build IS final: common symbols first, complete to N_SYMBOLS by IS volume rank
# 4. Print diagnostics and warn if IS universe is smaller than OOS
# =============================================================================
print("\n" + "="*60)
print(f"  Symbol Diagnostics & Universe Selection  |  {STRATEGY_ID}")
print("="*60)

raw_is  = sorted([f.split("_")[0] for f in os.listdir(DATA_FOLDER_IS)  if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")])
raw_oos = sorted([f.split("_")[0] for f in os.listdir(DATA_FOLDER_OOS) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")])

ohlcv_oos, filtered_oos = filter_symbols(raw_oos, min_vol_usdt=0, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER_OOS, min_price=MIN_PRICE, vol_window=50, my_symbols=MY_SYMBOLS)
ohlcv_is,  filtered_is  = filter_symbols(raw_is,  min_vol_usdt=0, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER_IS,  min_price=MIN_PRICE, vol_window=50, my_symbols=MY_SYMBOLS)

# Rank OOS by volume and select top N_SYMBOLS
vol_oos          = {sym: ohlcv_oos[sym]["volume_quote"].tail(50).mean() for sym in filtered_oos}
oos_ranked       = sorted(filtered_oos, key=lambda s: vol_oos.get(s, 0), reverse=True)
symbols_oos_final = oos_ranked[:N_SYMBOLS]

syms_is  = set(filtered_is)
syms_oos = set(symbols_oos_final)

in_both     = sorted(syms_is & syms_oos)
only_in_oos = sorted(syms_oos - syms_is)

# Build IS final universe: common first, then fill by IS volume rank
vol_is               = {sym: ohlcv_is[sym]["volume_quote"].tail(50).mean() for sym in syms_is}
is_candidates_by_vol = sorted(syms_is - syms_oos, key=lambda s: vol_is.get(s, 0), reverse=True)
needed               = max(0, N_SYMBOLS - len(in_both))
symbols_is_final     = sorted(in_both + is_candidates_by_vol[:needed])

print(f"\n  OOS pool         ({len(filtered_oos):>3}): {len(filtered_oos)} candidates")
print(f"  IS  pool         ({len(filtered_is):>3}): {len(filtered_is)} candidates")
print(f"\n  In both          ({len(in_both):>3}): {in_both}")
print(f"  Only in OOS      ({len(only_in_oos):>3}): {only_in_oos}")
print(f"\n  ▶ OOS final universe ({len(symbols_oos_final):>3}): {sorted(symbols_oos_final)}")
print(f"  ▶ IS  final universe ({len(symbols_is_final):>3}): {symbols_is_final}")

if len(symbols_is_final) < N_SYMBOLS:
    print(f"\n  ⚠️  IS has only {len(symbols_is_final)} symbols — fewer than N_SYMBOLS ({N_SYMBOLS}). Proceeding with available.")

# =============================================================================
# HELPER — REPORT FILTERED TRADES
# =============================================================================
def report_filtered_trades(trade_log, initial_balance, data_folder, title="Filtered Trades"):
    """
    Recompute equity curve, key metrics and plot from a filtered trade_log.
    Mirrors the output format of report_backtesting.
    """
    from sklearn.linear_model import LinearRegression

    df = trade_log.copy().sort_values("buy_time").reset_index(drop=True)
    df["duration_m"] = (pd.to_datetime(df["sell_time"]) - pd.to_datetime(df["buy_time"])).dt.total_seconds() / 60

    # --- Equity curve ---
    df["equity"] = initial_balance + df["profit"].cumsum()
    balances     = df["equity"].values
    timestamps   = pd.to_datetime(df["buy_time"])

    # --- Metrics ---
    net_gain      = balances[-1] - initial_balance
    net_gain_pct  = net_gain / initial_balance * 100
    win_ratio     = (df["profit"] > 0).mean()
    num_signals   = len(df)
    duration_m    = df["duration_m"].mean()
    cummax        = np.maximum.accumulate(balances)
    dd_pct        = ((balances - cummax) / cummax * 100).min()

    X  = np.arange(len(balances)).reshape(-1, 1)
    y  = balances.reshape(-1, 1)
    r2 = round(LinearRegression().fit(X, y).score(X, y), 3)

    sharpe = (df["profit"].mean() / df["profit"].std() * np.sqrt(252)) if df["profit"].std() > 0 else np.nan

    # --- Summary table ---
    df_summary = pd.DataFrame([{
        "Metric": "Net_Gain_pct", **{k.upper(): v for k, v in trade_log[["sell_after","lookback","tolerance","ma_period","tp_pct","sl_pct"]].iloc[0].items()},
        "Net_Gain_pct": round(net_gain_pct, 2), "Win_Ratio": round(win_ratio, 2),
        "R2": r2, "Sharpe": round(sharpe, 2), "DD_pct": round(dd_pct, 2),
        "Num_Signals": num_signals, "duration_m": round(duration_m, 2)
    }])
    print(df_summary.to_string(index=False))

    # --- Monthly stats ---
    df["month"] = pd.to_datetime(df["buy_time"]).dt.to_period("M")
    monthly     = df.groupby("month")["profit"].sum()
    winning_m   = (monthly > 0).sum()
    print(f"\n{'-'*60}")
    print("MONTHLY STATISTICS")
    print(f"{'-'*60}")
    print(f"Winning Months: {winning_m} / {len(monthly)} ({winning_m/len(monthly)*100:.2f}%)")

    # --- Plot ---
    import matplotlib.pyplot as plt
    net_gain_arr = (balances - initial_balance) / initial_balance * 100
    cummax_arr   = np.maximum.accumulate(balances)
    dd_arr       = (balances - cummax_arr) / cummax_arr * 100

    btc_file = os.path.join(data_folder, "BTCUSDT_4H.parquet")
    btc_df   = pd.read_parquet(btc_file)
    if "timestamp" not in btc_df.columns:
        btc_df = btc_df.reset_index().rename(columns={"index": "timestamp"})
    btc_df["timestamp"]     = pd.to_datetime(btc_df["timestamp"])
    btc_df["btc_net_gain"]  = (btc_df["close"] / btc_df["close"].iloc[0] - 1) * 100
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
# BLOCK 1 — MONTE CARLO IS
# =============================================================================
print("\n" + "="*60)
print(f"  BLOCK 1 — Monte Carlo IS  |  {STRATEGY_ID}")
print("="*60)

ohlcv_data_minor = {sym: ohlcv_is[sym] for sym in symbols_is_final}

paths_minor = generate_paths_for_all_symbols_functional(
    ohlcv_data_minor,
    n_paths=N_PATHS_IS,
    n_obs=FINAL_N_OBS_PER_PATH,
    raw_columns=[],
)


def _process_path(path_idx, paths_minor, param_dict_list):
    all_results = []
    for param_dict in param_dict_list:
        ohlcv_arrays = extract_ohlcv_from_path(paths_minor, path_idx, dtype=DTYPE)

        for sym in ohlcv_arrays:
            signals = signal_fn(
                ohlcv_arrays[sym],
                lookback=param_dict["LOOKBACK"],
                tolerance=param_dict["TOLERANCE"],
                ma_period=param_dict["MA_PERIOD"],
                live_trading=False,
            )
            ohlcv_arrays[sym]["signal"] = np.asarray(signals, dtype=DTYPE)

        result = run_grid_backtest(
            ohlcv_arrays,
            sell_after=param_dict["SELL_AFTER"],
            tp_pct=param_dict["TP_PCT"],
            sl_pct=param_dict["SL_PCT"],
            order_amount=ORDER_AMOUNT,
        )
        all_results.append(compile_MC_results(result, param_dict, path_idx, INITIAL_BALANCE, dtype=DTYPE))

    return all_results


final_prints(
    f"🎲 MC_{STRATEGY_ID} 🎲",
    DATA_FOLDER_IS,
    TIMEFRAME_MINOR,
    min_vol_usdt=0,
    order_amount=ORDER_AMOUNT,
    param_names=param_names,
    lists_for_grid=lists_for_grid,
)

with tqdm_joblib(tqdm(total=N_PATHS_IS, desc="🔄 Evaluating MC IS paths")):
    results_list = Parallel(n_jobs=N_JOBS)(
        delayed(_process_path)(i, paths_minor, param_dict_list)
        for i in range(N_PATHS_IS)
    )

all_results  = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

df_summary = report_montecarlo(
    df_portfolio=df_portfolio,
    param_names=param_names,
    initial_balance=INITIAL_BALANCE,
)

# -----------------------------------------------------------------------------
# Extracting optimal params (best Net_Gain_pct_m)
# -----------------------------------------------------------------------------
print("\n   ▶ Extracting optimal params (best Net_Gain_pct_m)...")

int_params = {k for k in param_names if all(isinstance(x, int) for x in globals()[f"{k}_LIST"])}

best_row    = df_summary.loc[df_summary["Net_Gain_pct_m"].idxmax()]
best_params = {
    k: int(round(best_row[k])) if k in int_params else round(float(best_row[k]), 4)
    for k in param_names
}

print("   Best params: " + " | ".join(f"{k}: {v}" for k, v in best_params.items()))

# =============================================================================
# BLOCK 2 — BACKTEST OOS
# =============================================================================
print("\n" + "="*60)
print(f"  BLOCK 2 — Backtest OOS  |  {STRATEGY_ID}")
print("="*60)

ohlcv_data_oos = {sym: ohlcv_oos[sym] for sym in symbols_oos_final}
ohlcv_arr_oos  = prepare_ohlcv_arrays(ohlcv_data_oos)

signal_params = {
    k.lower(): v
    for k, v in best_params.items()
    if k.lower() not in {"sell_after", "tp_pct", "sl_pct"}
}

ohlcv_arrays_oos = {}
for sym, arr in ohlcv_arr_oos.items():
    signals = signal_fn(arr, **signal_params, live_trading=False)
    ohlcv_arrays_oos[sym] = {**arr, "signal": signals}

oos_result = run_grid_backtest(
    ohlcv_arrays_oos,
    sell_after=best_params["SELL_AFTER"],
    tp_pct=best_params["TP_PCT"],
    sl_pct=best_params["SL_PCT"],
    order_amount=ORDER_AMOUNT,
)

best_comb = tuple(best_params[p] for p in param_names)
oos_df    = pd.DataFrame(compile_grid_results([(best_comb, oos_result)], param_names, INITIAL_BALANCE))

final_prints(
    f"🔭 OOS_{STRATEGY_ID}",
    DATA_FOLDER_OOS,
    TIMEFRAME_MINOR,
    0,
    ORDER_AMOUNT,
    param_names,
    lists_for_grid,
)

oos_bt_portfolio, _ = report_backtesting(
    df=oos_df,
    parameters=param_names,
    data_folder=DATA_FOLDER_OOS,
    initial_capital=INITIAL_BALANCE,
)

# =============================================================================
# BLOCK 3 — MONTE CARLO OOS
# =============================================================================
print("\n" + "="*60)
print(f"  BLOCK 3 — Monte Carlo OOS  |  {STRATEGY_ID}")
print("="*60)

n_obs_oos        = get_n_obs(TIMEFRAME_MINOR)
paths_oos        = generate_paths_for_all_symbols_functional(
    ohlcv_data_oos,
    n_paths=N_PATHS_OOS,
    n_obs=n_obs_oos,
    raw_columns=[],
)
best_params_list = [best_params]


def _process_path_oos(path_idx, paths, params_list):
    all_results = []
    for param_dict in params_list:
        ohlcv_arrays = extract_ohlcv_from_path(paths, path_idx, dtype=DTYPE)

        for sym in ohlcv_arrays:
            signals = signal_fn(
                ohlcv_arrays[sym],
                **{k.lower(): v for k, v in param_dict.items() if k.lower() not in {"sell_after", "tp_pct", "sl_pct"}},
                live_trading=False,
            )
            ohlcv_arrays[sym]["signal"] = np.asarray(signals, dtype=DTYPE)

        result = run_grid_backtest(
            ohlcv_arrays,
            sell_after=param_dict["SELL_AFTER"],
            tp_pct=param_dict["TP_PCT"],
            sl_pct=param_dict["SL_PCT"],
            order_amount=ORDER_AMOUNT,
        )
        all_results.append(compile_MC_results(result, param_dict, path_idx, INITIAL_BALANCE, dtype=DTYPE))

    return all_results


final_prints(
    f"🎲 MC_OOS_{STRATEGY_ID} 🎲",
    DATA_FOLDER_OOS,
    TIMEFRAME_MINOR,
    min_vol_usdt=0,
    order_amount=ORDER_AMOUNT,
    param_names=param_names,
    lists_for_grid=[[best_params[n]] for n in param_names],
)

with tqdm_joblib(tqdm(total=N_PATHS_OOS, desc="🔄 Evaluating MC OOS paths")):
    results_oos = Parallel(n_jobs=N_JOBS)(
        delayed(_process_path_oos)(i, paths_oos, best_params_list)
        for i in range(N_PATHS_OOS)
    )

all_results_oos  = [r for sublist in results_oos for r in sublist]
df_portfolio_oos = pd.DataFrame(all_results_oos)

report_montecarlo(
    df_portfolio=df_portfolio_oos,
    param_names=param_names,
    initial_balance=INITIAL_BALANCE,
)

# Extract best backtest row — used in Block 4 (regime) and Block 5 (validation)
best_bt_row = oos_df.loc[oos_df["Net_Gain"].idxmax()]

# =============================================================================
# BLOCK 4 — REGIME ANALYSIS
# =============================================================================
print("\n" + "="*60)
print(f"  BLOCK 4 — Regime Analysis  |  {STRATEGY_ID}")
print("="*60)

import sys as _sys
_sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "development", "market_regime")))
from regime1_performance_OOS import analyze_strategy, print_single_strategy_all_dimensions, FAMILIES
from regime1_performance_OOS import OHLC_FOLDER, MA_PERIOD, HURST_WINDOW, ER_WINDOW, ATR_WINDOW, PE_WINDOW, PE_ORDER, LOOKBACK_BARS
from regime_common import load_btc_for_timeframe, calc_all_metrics_at_time, classify_trade_by_family

# --- Save trades ---
save_all_trades_to_excel(
    [(best_comb, oos_result)],
    param_names,
    f"all_trades_{STRATEGY_ID}.xlsx",
    strategy_name=STRATEGY_ID,
    save=True,
)
TRADES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "development", "brief_trades", f"all_trades_{STRATEGY_ID}.xlsx"))
trade_log   = pd.read_excel(TRADES_PATH)
trade_log.columns = trade_log.columns.str.lower().str.strip()
trade_log["buy_time"] = pd.to_datetime(trade_log["buy_time"])
print(f"\n  ✅ Trades saved → {TRADES_PATH}  ({len(trade_log)} trades)")

# --- Run regime analysis (for print) ---
regime_result = analyze_strategy(TRADES_PATH, FAMILIES, INITIAL_BALANCE)
print_single_strategy_all_dimensions(regime_result)

# --- Enrich trade_log with family inline ---
btc_cache  = {}
btc_df     = load_btc_for_timeframe(OHLC_FOLDER, TIMEFRAME_MINOR, btc_cache)

trade_log["family"] = "unknown"
for idx, trade in trade_log.iterrows():
    metrics = calc_all_metrics_at_time(
        btc_df, trade["buy_time"], LOOKBACK_BARS,
        MA_PERIOD, HURST_WINDOW, ER_WINDOW, ATR_WINDOW, PE_WINDOW, PE_ORDER
    )
    if metrics:
        trade_log.at[idx, "family"] = classify_trade_by_family(metrics, FAMILIES)

# --- Filter: exclude regimes with negative profit ---
excluded_families = [
    fam for fam, stats in regime_result["family_stats"].items()
    if stats["profit"] < 0
]

if excluded_families:
    print(f"\n  ▶ Excluding regimes with negative profit: {excluded_families}")
    trade_log_filtered = trade_log[~trade_log["family"].isin(excluded_families)].reset_index(drop=True)
    print(f"  ▶ Trades after filter: {len(trade_log_filtered)} / {len(trade_log)}")

    report_filtered_trades(
        trade_log_filtered,
        initial_balance=INITIAL_BALANCE,
        data_folder=DATA_FOLDER_OOS,
        title=f"Filtered Trades — {STRATEGY_ID} (excl. {excluded_families})",
    )
else:
    print(f"\n  ✅ No regimes with negative profit — no filtering applied.")

# =============================================================================
# BLOCK 5 — VALIDATION
# =============================================================================
# Thresholds:
#   Backtest OOS : Net_Gain_pct > 20%  |  R2 > 0.7
#   MC OOS       : Probability of Negative Path < 31%
# =============================================================================
print("\n" + "="*60)
print(f"  BLOCK 5 — Validation  |  {STRATEGY_ID}")
print("="*60)

# --- Backtest OOS metrics ---
bt_netgain_pct = best_bt_row["Net_Gain"] / INITIAL_BALANCE * 100

equity_hist = best_bt_row.get("sim_balance_history", None)
if equity_hist and len(equity_hist.get("balance", [])) >= 2:
    from sklearn.linear_model import LinearRegression as _LR
    y  = np.array(equity_hist["balance"]).reshape(-1, 1)
    X  = np.arange(len(y)).reshape(-1, 1)
    r2 = round(_LR().fit(X, y).score(X, y), 3)
else:
    r2 = np.nan

# --- MC OOS metric ---
path_grouped_oos  = df_portfolio_oos.groupby("path_index")["Portfolio_Final_Balance"].mean().reset_index()
path_grouped_oos["Net_Gain_pct"] = (path_grouped_oos["Portfolio_Final_Balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
prob_negative_oos = (path_grouped_oos["Net_Gain_pct"] < 0).mean() * 100

# --- Thresholds ---
THRESHOLD_NETGAIN_PCT = 20.0
THRESHOLD_R2          = 0.7
THRESHOLD_PROB_NEG    = 31.0

ok_netgain  = bt_netgain_pct   > THRESHOLD_NETGAIN_PCT
ok_r2       = r2               > THRESHOLD_R2
ok_prob_neg = prob_negative_oos < THRESHOLD_PROB_NEG
approved    = ok_netgain and ok_r2 and ok_prob_neg

print(f"\n  Backtest OOS")
print(f"    Net_Gain_pct : {bt_netgain_pct:>7.2f}%   (threshold > {THRESHOLD_NETGAIN_PCT}%)   {'✅' if ok_netgain  else '❌'}")
print(f"    R2           : {r2:>7.3f}    (threshold > {THRESHOLD_R2})     {'✅' if ok_r2       else '❌'}")
print(f"\n  Monte Carlo OOS")
print(f"    Prob Negative: {prob_negative_oos:>7.2f}%   (threshold < {THRESHOLD_PROB_NEG}%)  {'✅' if ok_prob_neg else '❌'}")
print(f"\n  {'🟢 STRATEGY APPROVED' if approved else '🔴 STRATEGY REJECTED'}")
print("="*60)

# =============================================================================
# BLOCK 6 — UPDATE & COMPARE
# =============================================================================
print("\n" + "="*60)
print(f"  BLOCK 6 — Update & Compare  |  {STRATEGY_ID}")
print("="*60)

CSV_PARAMS  = os.path.join(os.path.dirname(__file__), "strategies_params.csv")
CSV_SYMBOLS = os.path.join(os.path.dirname(__file__), "strategies_symbols.csv")
PARAM_KEYS  = [p.lower() for p in param_names]

def normalize(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# -----------------------------------------------------------------------------
# strategies_params.csv
# -----------------------------------------------------------------------------
if not os.path.exists(CSV_PARAMS):
    print(f"  ⚠️  strategies_params.csv not found at {CSV_PARAMS} — skipping update.")
else:
    df_params = pd.read_csv(CSV_PARAMS)
    mask      = df_params["id"] == STRATEGY_ID

    if not mask.any():
        print(f"  ⚠️  Strategy id '{STRATEGY_ID}' not found in CSV — skipping update.")
    else:
        idx      = df_params[mask].index[0]
        prev_row = df_params.loc[idx]

        # --- Print comparison ---
        print(f"\n  ID : {STRATEGY_ID}")
        print(f"\n  {'Parameter':<20} {'Previous':>12} {'New':>12} {'Changed':>10}")
        print(f"  {'-'*56}")
        for k in PARAM_KEYS:
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

        # --- Always update metrics and approval status ---
        df_params.at[idx, "approved"]       = approved
        df_params.at[idx, "bt_netgain_pct"] = round(bt_netgain_pct, 2)
        df_params.at[idx, "bt_r2"]          = round(r2, 3)
        df_params.at[idx, "prob_negative"]  = round(prob_negative_oos, 2)
        df_params.at[idx, "last_run"]       = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

        # --- Update params only if approved ---
        if approved:
            for k in PARAM_KEYS:
                if k in df_params.columns:
                    df_params.at[idx, k] = best_params.get(k.upper())
            df_params.at[idx, "active"] = True
            print(f"\n  ✅ strategies_params.csv updated — params updated, active set to True for '{STRATEGY_ID}'")
        else:
            df_params.at[idx, "active"] = False
            print(f"\n  ❌ Strategy rejected — params NOT updated, active set to False for '{STRATEGY_ID}'")

        df_params.to_csv(CSV_PARAMS, index=False)

# -----------------------------------------------------------------------------
# strategies_symbols.csv
# -----------------------------------------------------------------------------
symbols_str = str(symbols_oos_final)

if not os.path.exists(CSV_SYMBOLS):
    df_symbols = pd.DataFrame(columns=["id", "symbols"])
else:
    df_symbols = pd.read_csv(CSV_SYMBOLS)

mask_sym = df_symbols["id"] == STRATEGY_ID
if mask_sym.any():
    df_symbols.loc[df_symbols["id"] == STRATEGY_ID, "symbols"] = symbols_str
else:
    df_symbols = pd.concat([
        df_symbols,
        pd.DataFrame([{"id": STRATEGY_ID, "symbols": symbols_str}])
    ], ignore_index=True)

df_symbols.to_csv(CSV_SYMBOLS, index=False)
print(f"\n  ✅ strategies_symbols.csv updated — {len(symbols_oos_final)} symbols for '{STRATEGY_ID}'")

# =============================================================================
# ELAPSED TIME
# =============================================================================
elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")