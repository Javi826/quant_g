import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "market_regime")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "market_regime")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "zdevelop", "analysis")))

import time
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from utils.st_tools import (
    extract_ohlcv_from_path,
    compile_MC_results,
    compile_grid_results,
    prepare_ohlcv_arrays,
    get_n_obs,
    save_all_trades_to_excel,
)
from tools.optimize_MCf_tf import generate_paths_for_all_symbols_functional
from utils.analysis import report_montecarlo, report_backtesting
from utils.utils import filter_symbols, final_prints
from regime_performance import analyze_strategy, print_single_strategy_all_dimensions
from regime_common import load_btc_for_timeframe, calc_all_metrics_at_time, classify_trade_by_family
from regime_performance import OHLC_FOLDER, MA_PERIOD, LOOKBACK_BARS
from shared_config import REGIME_FAMILIES as FAMILIES, REGIME_HURST_WINDOW as HURST_WINDOW, REGIME_ER_WINDOW as ER_WINDOW
from shared_config import REGIME_ATR_WINDOW as ATR_WINDOW, REGIME_PE_WINDOW as PE_WINDOW, REGIME_PE_ORDER as PE_ORDER
from batch_utils import (
    report_filtered_trades,
    extract_best_params,
    select_universe,
    enrich_trades_with_regime,
    update_strategies_params,
    update_strategies_symbols,
    load_btc_1d,
    get_btc_direction,
)
from Z_compose_equities_01 import (
    compute_metrics,
    build_combined_equity,
    resample_equity,
    print_metrics_table,
    plot_netgain_dd,
)

# =============================================================================
# SIGNAL REGISTRY
# =============================================================================
from signals.add_signals_parity   import parity_long, parity_short
from signals.add_signals_reversal import reversal_long, reversal_short
from signals.add_signals_flag     import flag_long, flag_short
from signals.add_signals_orderblocks import orderblocks_long, orderblocks_short
from signals.add_signals_ranging  import ranging_long, ranging_short

SIGNAL_REGISTRY = {
    "parity_long":       {"fn": parity_long,       "params": ["lookback", "tolerance", "ma_period"]},
    "parity_short":      {"fn": parity_short,      "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_long":     {"fn": reversal_long,     "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_short":    {"fn": reversal_short,    "params": ["lookback", "tolerance", "ma_period"]},
    "flag_long":         {"fn": flag_long,         "params": ["lookback", "impulse", "flag", "ma_period"]},
    "flag_short":        {"fn": flag_short,        "params": ["lookback", "impulse", "flag", "ma_period"]},
    "orderblocks_long":  {"fn": orderblocks_long,  "params": ["lookback", "tolerance", "impulse"]},
    "orderblocks_short": {"fn": orderblocks_short, "params": ["lookback", "tolerance", "impulse"]},
    "ranging_long":      {"fn": ranging_long,      "params": ["lookback", "tolerance", "ma_period", "range_str"]},
    "ranging_short":     {"fn": ranging_short,     "params": ["lookback", "tolerance", "ma_period", "range_str"]},
}

DTYPE = np.float32

# =============================================================================
# GLOBAL CONFIGURATION — shared across all strategies
# =============================================================================
DATA_FOLDER_IS  = "data/crypto_2022_IS"
DATA_FOLDER_OOS = "data/crypto_2026_OOS"
N_JOBS          = -1
MY_SYMBOLS      = False

N_PATHS_IS  = 10
N_PATHS_OOS = 200

# Validation thresholds — first round
THRESHOLD_NETGAIN_PCT = 20.0
THRESHOLD_R2          = 0.7
THRESHOLD_PROB_NEG    = 31.0

# Validation thresholds — second round (regime filtered)
THRESHOLD_WR_IMPROVEMENT = 5.0
THRESHOLD_R2_FILTERED    = 0.8
THRESHOLD_PROB_NEG_MAX   = 50.0

# Regime 0 settings
R0_MA_PERIOD = 5
R0_LONG_TH   = 1.00
R0_SHORT_TH  = 1.00

# Paths
CSV_PARAMS  = os.path.join(os.path.dirname(__file__), "strategies_params.csv")
CSV_SYMBOLS = os.path.join(os.path.dirname(__file__), "strategies_symbols.csv")

# Global equity curve accumulators (populated by each run_batch call)
_equity_curves_baseline : list = []   # (strategy_id, resampled_df)
_equity_curves_regime01 : list = []   # (strategy_id, resampled_df)


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def run_batch(strategy_config: dict) -> None:
    """
    Run the full batch pipeline for a single strategy.

    strategy_config keys:
        strategy_id   : str   e.g. "03_parity_long_4H"
        signal        : str   key in SIGNAL_REGISTRY
        side          : str   "long" | "short"
        timeframe     : str   e.g. "4H"
        n_symbols     : int   top N OOS symbols by volume
        order_amount  : int
        param_grid    : dict  {PARAM_NAME: [values], ...}
    """
    start_time = time.time()

    # -------------------------------------------------------------------------
    # Unpack config
    # -------------------------------------------------------------------------
    STRATEGY_ID   = strategy_config["strategy_id"]
    SIDE          = strategy_config["side"]
    TIMEFRAME     = strategy_config["timeframe"]
    N_SYMBOLS     = strategy_config["n_symbols"]
    ORDER_AMOUNT  = strategy_config["order_amount"]
    param_grid    = strategy_config["param_grid"]

    registry      = SIGNAL_REGISTRY[strategy_config["signal"]]
    signal_fn     = registry["fn"]
    signal_params_keys = registry["params"]  # e.g. ["lookback", "tolerance", "ma_period"]

    param_names     = list(param_grid.keys())
    lists_for_grid  = [param_grid[k] for k in param_names]
    param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

    FINAL_N_OBS_PER_PATH = get_n_obs(TIMEFRAME)
    TRADES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "brief_trades", f"all_trades_{STRATEGY_ID}.xlsx"))

    # -------------------------------------------------------------------------
    # Symbol Diagnostics & Universe Selection
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  Symbol Diagnostics & Universe Selection  |  {STRATEGY_ID}")
    print("="*60)

    symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos = select_universe(
        data_folder_is=DATA_FOLDER_IS,
        data_folder_oos=DATA_FOLDER_OOS,
        timeframe=TIMEFRAME,
        n_symbols=N_SYMBOLS,
        min_price=MIN_PRICE,
        filter_symbols_fn=filter_symbols,
        my_symbols=MY_SYMBOLS,
    )

    # -------------------------------------------------------------------------
    # BLOCK 1 — MONTE CARLO IS
    # -------------------------------------------------------------------------
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
                sig_kwargs = {k: param_dict[k.upper()] for k in signal_params_keys if k.upper() in param_dict}
                signals = signal_fn(ohlcv_arrays[sym], **sig_kwargs, live_trading=False)
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

    final_prints(f"🎲 MC_{STRATEGY_ID} 🎲", DATA_FOLDER_IS, TIMEFRAME, min_vol_usdt=0,
                 order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=lists_for_grid)

    with tqdm_joblib(tqdm(total=N_PATHS_IS, desc="🔄 Evaluating MC IS paths")):
        results_list = Parallel(n_jobs=N_JOBS)(
            delayed(_process_path)(i, paths_minor, param_dict_list)
            for i in range(N_PATHS_IS)
        )

    all_results  = [r for sublist in results_list for r in sublist]
    df_portfolio = pd.DataFrame(all_results)
    df_summary   = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)
    best_params  = extract_best_params(df_summary, param_names, lists_for_grid)

    # -------------------------------------------------------------------------
    # BLOCK 2 — BACKTEST OOS
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  BLOCK 2 — Backtest OOS  |  {STRATEGY_ID}")
    print("="*60)

    ohlcv_data_oos = {sym: ohlcv_oos[sym] for sym in symbols_oos_final}
    ohlcv_arr_oos  = prepare_ohlcv_arrays(ohlcv_data_oos)

    bt_signal_params = {k: best_params[k.upper()] for k in signal_params_keys if k.upper() in best_params}

    ohlcv_arrays_oos = {}
    for sym, arr in ohlcv_arr_oos.items():
        signals = signal_fn(arr, **bt_signal_params, live_trading=False)
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

    final_prints(f"🔭 OOS_{STRATEGY_ID}", DATA_FOLDER_OOS, TIMEFRAME, 0, ORDER_AMOUNT, param_names, lists_for_grid)
    oos_bt_portfolio, _ = report_backtesting(df=oos_df, parameters=param_names,
                                             data_folder=DATA_FOLDER_OOS, initial_capital=INITIAL_BALANCE)

    # -------------------------------------------------------------------------
    # BLOCK 3 — MONTE CARLO OOS
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  BLOCK 3 — Monte Carlo OOS  |  {STRATEGY_ID}")
    print("="*60)

    n_obs_oos    = get_n_obs(TIMEFRAME)
    paths_oos    = generate_paths_for_all_symbols_functional(
        ohlcv_data_oos, n_paths=N_PATHS_OOS, n_obs=n_obs_oos, raw_columns=[])
    best_params_list = [best_params]

    def _process_path_oos(path_idx, paths, params_list):
        all_results = []
        for param_dict in params_list:
            ohlcv_arrays = extract_ohlcv_from_path(paths, path_idx, dtype=DTYPE)
            for sym in ohlcv_arrays:
                sig_kwargs = {k: param_dict[k.upper()] for k in signal_params_keys if k.upper() in param_dict}
                signals = signal_fn(ohlcv_arrays[sym], **sig_kwargs, live_trading=False)
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

    final_prints(f"🎲 MC_OOS_{STRATEGY_ID} 🎲", DATA_FOLDER_OOS, TIMEFRAME, min_vol_usdt=0,
                 order_amount=ORDER_AMOUNT, param_names=param_names,
                 lists_for_grid=[[best_params[n]] for n in param_names])

    with tqdm_joblib(tqdm(total=N_PATHS_OOS, desc="🔄 Evaluating MC OOS paths")):
        results_oos = Parallel(n_jobs=N_JOBS)(
            delayed(_process_path_oos)(i, paths_oos, best_params_list)
            for i in range(N_PATHS_OOS)
        )

    all_results_oos  = [r for sublist in results_oos for r in sublist]
    df_portfolio_oos = pd.DataFrame(all_results_oos)
    report_montecarlo(df_portfolio=df_portfolio_oos, param_names=param_names, initial_balance=INITIAL_BALANCE)

    best_bt_row = oos_df.loc[oos_df["Net_Gain"].idxmax()]

    # -------------------------------------------------------------------------
    # BLOCK 4 — REGIME ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  BLOCK 4 — Regime Analysis  |  {STRATEGY_ID}")
    print("="*60)

    save_all_trades_to_excel(
        [(best_comb, oos_result)], param_names,
        f"all_trades_{STRATEGY_ID}.xlsx",
        strategy_name=STRATEGY_ID, save=True,
        output_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
    )
    trade_log = pd.read_excel(TRADES_PATH)
    trade_log.columns = trade_log.columns.str.lower().str.strip()
    trade_log["buy_time"] = pd.to_datetime(trade_log["buy_time"])
    print(f"\n  ✅ Trades saved → {TRADES_PATH}  ({len(trade_log)} trades)")

    regime_result = analyze_strategy(TRADES_PATH, FAMILIES, INITIAL_BALANCE)
    print_single_strategy_all_dimensions(regime_result)

    trade_log = enrich_trades_with_regime(
        trade_log=trade_log, ohlc_folder=OHLC_FOLDER, timeframe=TIMEFRAME,
        families=FAMILIES, lookback_bars=LOOKBACK_BARS, ma_period=MA_PERIOD,
        hurst_window=HURST_WINDOW, er_window=ER_WINDOW, atr_window=ATR_WINDOW,
        pe_window=PE_WINDOW, pe_order=PE_ORDER,
        load_btc_fn=load_btc_for_timeframe, calc_metrics_fn=calc_all_metrics_at_time,
        classify_fn=classify_trade_by_family,
    )

    excluded_families = [
        fam for fam, stats in regime_result["family_stats"].items()
        if stats["profit"] < 0
    ]

    if excluded_families:
        print(f"\n  ▶ Excluding regimes with negative profit: {excluded_families}")
        trade_log_filtered = trade_log[~trade_log["family"].isin(excluded_families)].reset_index(drop=True)
        print(f"  ▶ Trades after filter: {len(trade_log_filtered)} / {len(trade_log)}")
        report_filtered_trades(trade_log_filtered, initial_balance=INITIAL_BALANCE,
                               data_folder=DATA_FOLDER_OOS,
                               title=f"Filtered Trades — {STRATEGY_ID} (excl. {excluded_families})")
    else:
        print(f"\n  ✅ No regimes with negative profit — no filtering applied.")

    # -------------------------------------------------------------------------
    # BLOCK 5 — REGIME 0 — BTC DIRECTION FILTER
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  BLOCK 5 — Regime 0 (BTC Direction Filter)  |  {STRATEGY_ID}")
    print("="*60)

    R0_BTC_FILE = os.path.join(DATA_FOLDER_OOS, "BTCUSDT_1Dutc.parquet")

    if not os.path.exists(R0_BTC_FILE):
        print(f"\n  ⚠️  BTC 1Dutc file not found — skipping Regime 0 analysis.")
    else:
        r0_btc_df = load_btc_1d(R0_BTC_FILE, ma_period=R0_MA_PERIOD)

        def _get_btc_direction(buy_time):
            return get_btc_direction(buy_time, r0_btc_df, SIDE, R0_MA_PERIOD, R0_LONG_TH, R0_SHORT_TH)

        r0_trade_log = trade_log.copy()
        r0_trade_log["r0_direction"] = r0_trade_log["buy_time"].apply(_get_btc_direction)

        r0_counts = r0_trade_log["r0_direction"].value_counts()
        print(f"\n  BTC direction distribution (MA{R0_MA_PERIOD}):")
        for direction, count in r0_counts.items():
            print(f"    {direction:<12}: {count} trades")

        keep_direction = "uptrend" if SIDE == "long" else "downtrend"
        r0_filtered    = r0_trade_log[r0_trade_log["r0_direction"] == keep_direction].reset_index(drop=True)
        print(f"\n  ▶ Keeping '{keep_direction}' trades: {len(r0_filtered)} / {len(r0_trade_log)}")

        if len(r0_filtered) > 0:
            report_filtered_trades(r0_filtered, initial_balance=INITIAL_BALANCE,
                                   data_folder=DATA_FOLDER_OOS,
                                   title=f"Regime 0 Filtered — {STRATEGY_ID} ({keep_direction} only, MA{R0_MA_PERIOD})")

    # -------------------------------------------------------------------------
    # BLOCK 6 — REGIME 0 + 1 COMBINED FILTER
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  BLOCK 6 — Regime 0 + 1 Combined Filter  |  {STRATEGY_ID}")
    print("="*60)

    if not os.path.exists(R0_BTC_FILE):
        print(f"\n  ⚠️  BTC 1Dutc file not found — skipping combined analysis.")
    else:
        r01_base = trade_log_filtered if excluded_families and "trade_log_filtered" in vars() else trade_log
        print(f"\n  Base trades (after Regime 1): {len(r01_base)}")

        r01_trade_log = r01_base.copy()
        r01_trade_log["r0_direction"] = r01_trade_log["buy_time"].apply(_get_btc_direction)

        r01_filtered = r01_trade_log[r01_trade_log["r0_direction"] == keep_direction].reset_index(drop=True)
        print(f"  ▶ After Regime 0 filter ({keep_direction}): {len(r01_filtered)} / {len(r01_trade_log)}")

        if len(r01_filtered) > 0:
            report_filtered_trades(r01_filtered, initial_balance=INITIAL_BALANCE,
                                   data_folder=DATA_FOLDER_OOS,
                                   title=f"Regime 0+1 Combined — {STRATEGY_ID} (excl. {excluded_families}, {keep_direction} only)")
            r01_trades_path = os.path.join(os.path.dirname(__file__), "brief_trades", f"all_trades_{STRATEGY_ID}_regime01.xlsx")
            r01_filtered.to_excel(r01_trades_path, index=False)
            print(f"\n  ✅ Regime 0+1 trades saved → {r01_trades_path}  ({len(r01_filtered)} trades)")

    # -------------------------------------------------------------------------
    # SUMMARY TABLE — Baseline vs Regime 1 vs Regime 0 vs Regime 0+1
    # -------------------------------------------------------------------------
    def _calc_metrics(df, initial_balance):
        if df is None or len(df) == 0:
            return {"trades": 0, "net_gain_pct": np.nan, "win_rate": np.nan, "dd_pct": np.nan, "r2": np.nan}
        df = df.sort_values("buy_time").reset_index(drop=True)
        equity   = initial_balance + df["profit"].cumsum().values
        cummax   = np.maximum.accumulate(equity)
        dd_pct   = ((equity - cummax) / cummax * 100).min()
        X        = np.arange(len(equity)).reshape(-1, 1)
        y        = equity.reshape(-1, 1)
        r2       = round(LinearRegression().fit(X, y).score(X, y), 3)
        return {
            "trades":       len(df),
            "net_gain_pct": round((equity[-1] - initial_balance) / initial_balance * 100, 2),
            "win_rate":     round((df["profit"] > 0).mean() * 100, 1),
            "dd_pct":       round(dd_pct, 2),
            "r2":           r2,
        }

    r1_df   = trade_log_filtered if excluded_families and "trade_log_filtered" in vars() else None
    r0_df   = r0_filtered        if "r0_filtered"  in vars() and len(r0_filtered)  > 0 else None
    r01_df  = r01_filtered       if "r01_filtered" in vars() and len(r01_filtered) > 0 else None

    rows = [
        ("Baseline",    _calc_metrics(trade_log, INITIAL_BALANCE)),
        ("Regime 1",    _calc_metrics(r1_df,     INITIAL_BALANCE)),
        ("Regime 0",    _calc_metrics(r0_df,     INITIAL_BALANCE)),
        ("Regime 0+1",  _calc_metrics(r01_df,    INITIAL_BALANCE)),
    ]

    print(f"\n{'─'*75}")
    print(f"  FILTER COMPARISON SUMMARY — {STRATEGY_ID}")
    print(f"{'─'*75}")
    print(f"  {'Scenario':<14} {'Trades':>8} {'Net Gain%':>10} {'Win Rate%':>10} {'DD%':>8} {'R2':>7}")
    print(f"  {'-'*71}")
    for name, m in rows:
        if m["trades"] == 0:
            print(f"  {name:<14} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>7}")
        else:
            print(f"  {name:<14} {m['trades']:>8} {m['net_gain_pct']:>9.2f}% {m['win_rate']:>9.1f}% {m['dd_pct']:>7.2f}% {m['r2']:>7.3f}")
    print(f"  {'─'*71}")

    # -------------------------------------------------------------------------
    # BLOCK 7 — VALIDATION
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  BLOCK 7 — Validation  |  {STRATEGY_ID}")
    print("="*60)

    bt_netgain_pct = best_bt_row["Net_Gain"] / INITIAL_BALANCE * 100
    equity_hist    = best_bt_row.get("sim_balance_history", None)
    if equity_hist and len(equity_hist.get("balance", [])) >= 2:
        y_bt = np.array(equity_hist["balance"]).reshape(-1, 1)
        X_bt = np.arange(len(y_bt)).reshape(-1, 1)
        r2   = round(LinearRegression().fit(X_bt, y_bt).score(X_bt, y_bt), 3)
    else:
        r2 = np.nan

    path_grouped_oos = df_portfolio_oos.groupby("path_index")["Portfolio_Final_Balance"].mean().reset_index()
    path_grouped_oos["Net_Gain_pct"] = (path_grouped_oos["Portfolio_Final_Balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    prob_negative_oos = (path_grouped_oos["Net_Gain_pct"] < 0).mean() * 100

    ok_netgain  = bt_netgain_pct    > THRESHOLD_NETGAIN_PCT
    ok_r2       = r2                > THRESHOLD_R2
    ok_prob_neg = prob_negative_oos < THRESHOLD_PROB_NEG
    approved    = ok_netgain and ok_r2 and ok_prob_neg

    print(f"\n  Backtest OOS")
    print(f"    Net_Gain_pct : {bt_netgain_pct:>7.2f}%   (threshold > {THRESHOLD_NETGAIN_PCT}%)   {'✅' if ok_netgain  else '❌'}")
    print(f"    R2           : {r2:>7.3f}    (threshold > {THRESHOLD_R2})     {'✅' if ok_r2       else '❌'}")
    print(f"\n  Monte Carlo OOS")
    print(f"    Prob Negative: {prob_negative_oos:>7.2f}%   (threshold < {THRESHOLD_PROB_NEG}%)  {'✅' if ok_prob_neg else '❌'}")
    print(f"\n  {'🟢 STRATEGY APPROVED' if approved else '🔴 STRATEGY REJECTED — checking regime filter...'}")

    approved_regime = False
    if not approved and excluded_families and "trade_log_filtered" in vars():
        df_filt     = trade_log_filtered.copy().sort_values("buy_time").reset_index(drop=True)
        equity_filt = INITIAL_BALANCE + df_filt["profit"].cumsum().values
        wr_filtered = (df_filt["profit"] > 0).mean() * 100
        X2          = np.arange(len(equity_filt)).reshape(-1, 1)
        y2          = equity_filt.reshape(-1, 1)
        r2_filtered = round(LinearRegression().fit(X2, y2).score(X2, y2), 3)

        wr_original    = best_bt_row.get("Win_Ratio", 0) * 100
        wr_improvement = wr_filtered - wr_original

        ok_wr_improvement = wr_improvement    > THRESHOLD_WR_IMPROVEMENT
        ok_r2_filtered    = r2_filtered       > THRESHOLD_R2_FILTERED
        ok_prob_neg_max   = prob_negative_oos < THRESHOLD_PROB_NEG_MAX
        approved_regime   = ok_wr_improvement and ok_r2_filtered and ok_prob_neg_max

        print(f"\n  Second Round — Regime Filtered")
        print(f"    WR improvement : {wr_improvement:>7.2f}pp  (threshold > {THRESHOLD_WR_IMPROVEMENT}pp)   {'✅' if ok_wr_improvement else '❌'}  ({wr_original:.2f}% → {wr_filtered:.2f}%)")
        print(f"    R2 filtered    : {r2_filtered:>7.3f}    (threshold > {THRESHOLD_R2_FILTERED})      {'✅' if ok_r2_filtered    else '❌'}")
        print(f"    Prob Negative  : {prob_negative_oos:>7.2f}%   (threshold < {THRESHOLD_PROB_NEG_MAX}%)   {'✅' if ok_prob_neg_max   else '❌'}")
        print(f"\n  {'🟢 STRATEGY APPROVED (regime filtered)' if approved_regime else '🔴 STRATEGY REJECTED (regime filtered)'}")

    approved = approved or approved_regime
    print("="*60)

    # -------------------------------------------------------------------------
    # BLOCK 7 — EQUITY CURVES
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  BLOCK 7 — Equity Curves  |  {STRATEGY_ID}")
    print("="*60)

    import Z_compose_equities_01 as _ce
    _ce.DATA_FOLDER = os.path.abspath(DATA_FOLDER_OOS)

    def _trades_to_equity(df, initial_balance):
        """Build daily resampled equity curve from trade_log."""
        df = df.sort_values("buy_time").reset_index(drop=True)
        df["equity"] = initial_balance + df["profit"].cumsum()
        df_eq = df[["buy_time", "equity"]].rename(columns={"buy_time": "timestamp", "equity": "balance"})
        df_eq = df_eq.drop_duplicates(subset="timestamp", keep="last")
        df_eq = df_eq.set_index("timestamp")
        return resample_equity(df_eq)

    # Baseline equity
    eq_baseline = _trades_to_equity(trade_log, INITIAL_BALANCE)
    _equity_curves_baseline.append((STRATEGY_ID, eq_baseline))

    plot_netgain_dd(eq_baseline.reset_index(), capital=INITIAL_BALANCE,
                    title=f"Baseline — {STRATEGY_ID}")
    metrics_baseline = compute_metrics(eq_baseline.reset_index(), capital=INITIAL_BALANCE, name=STRATEGY_ID)
    print_metrics_table([metrics_baseline], f"  Metrics — {STRATEGY_ID} (Baseline)")

    # Regime 0+1 equity
    if "r01_filtered" in vars() and r01_filtered is not None and len(r01_filtered) > 0:
        eq_regime01 = _trades_to_equity(r01_filtered, INITIAL_BALANCE)
        _equity_curves_regime01.append((STRATEGY_ID, eq_regime01))

        plot_netgain_dd(eq_regime01.reset_index(), capital=INITIAL_BALANCE,
                        title=f"Regime 0+1 — {STRATEGY_ID}")
        metrics_regime01 = compute_metrics(eq_regime01.reset_index(), capital=INITIAL_BALANCE, name=f"{STRATEGY_ID}_r01")
        print_metrics_table([metrics_regime01], f"  Metrics — {STRATEGY_ID} (Regime 0+1)")

    # -------------------------------------------------------------------------
    # BLOCK 8 — UPDATE & COMPARE
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  BLOCK 8 — Update & Compare  |  {STRATEGY_ID}")
    print("="*60)

    PARAM_KEYS = [p.lower() for p in param_names]

    update_strategies_params(
        csv_path=CSV_PARAMS, strategy_id=STRATEGY_ID, best_params=best_params,
        param_keys=PARAM_KEYS, approved=approved, bt_netgain_pct=bt_netgain_pct,
        r2=r2, prob_negative_oos=prob_negative_oos,
    )
    update_strategies_symbols(
        csv_path=CSV_SYMBOLS, strategy_id=STRATEGY_ID, symbols_oos_final=symbols_oos_final,
    )

    elapsed = int(time.time() - start_time)
    print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")


# =============================================================================
# PORTFOLIO ANALYSIS — call after all run_batch() calls
# =============================================================================
def run_portfolio_analysis():
    """
    Compute combined portfolio metrics and best combinations
    for baseline and regime 0+1 equity curves.
    Call this after all run_batch() calls in the orchestrator.
    """
    from itertools import combinations

    for label, curves in [("Baseline", _equity_curves_baseline), ("Regime 0+1", _equity_curves_regime01)]:
        if not curves:
            continue

        print("\n" + "="*70)
        print(f"  PORTFOLIO ANALYSIS — {label}")
        print("="*70)

        named_dfs    = {sid: df for sid, df in curves}
        dfs          = list(named_dfs.values())
        metrics_list = [compute_metrics(df.reset_index(), capital=INITIAL_BALANCE, name=sid)
                        for sid, df in named_dfs.items()]

        # Combined portfolio
        if len(dfs) > 1:
            combined_df  = build_combined_equity(dfs)
            combined_cap = INITIAL_BALANCE * len(dfs)
            plot_netgain_dd(combined_df, capital=combined_cap, title=f"Combined Portfolio — {label}")
            metrics_list.append(compute_metrics(combined_df, capital=combined_cap, name="Combined"))

        print_metrics_table(metrics_list, f"📊 METRICS TABLE — {label}")

        # Best combinations
        combo_results = []
        for r in range(1, len(named_dfs) + 1):
            for combo in combinations(named_dfs.keys(), r):
                combo_dfs = [named_dfs[sid] for sid in combo]
                combined  = build_combined_equity(combo_dfs)
                capital   = INITIAL_BALANCE * len(combo_dfs)
                combo_results.append(
                    compute_metrics(combined, capital=capital, name="+".join(combo))
                )

        import pandas as _pd
        combo_df = _pd.DataFrame(combo_results)

        for metric, ascending, title in [
            ("Net_Gain_pct",  False, "📈 TOP 5 BY NET GAIN"),
            ("R_Squared",     False, "📐 TOP 5 BY R²"),
            ("Max_DD_pct",    False, "📉 TOP 5 BY LOWEST DD"),
        ]:
            top5 = combo_df.sort_values(metric, ascending=ascending).head(5)
            print_metrics_table(top5.to_dict("records"), f"\n{title} — {label}", shorten_names=True)

            # Plot best combination
            best_combo = top5.iloc[0]["Curve"].strip().split("+")
            best_dfs   = [named_dfs[sid] for sid in best_combo if sid in named_dfs]
            if best_dfs:
                best_combined = build_combined_equity(best_dfs)
                best_cap      = INITIAL_BALANCE * len(best_dfs)
                plot_netgain_dd(best_combined, capital=best_cap,
                                title=f"{title} — {label}: {'+'.join(best_combo)}")