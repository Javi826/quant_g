import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "market_regime")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "market_regime")))

import logging
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
    compute_metrics,
    print_metrics_table,
)

logger = logging.getLogger("BOT_trading.batch.main_batch")

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

# Global trade_log accumulators (populated by each run_batch call)
_trade_logs_baseline : list = []   # (strategy_id, trade_log_df)
_trade_logs_regime1  : list = []   # (strategy_id, trade_log_df)
_trade_logs_regime0  : list = []   # (strategy_id, trade_log_df)
_trade_logs_regime01 : list = []   # (strategy_id, trade_log_df)
_oos_metrics         : list = []   # {strategy_id, net_gain_pct, dd_pct, win_ratio, r2}
_oos_bt_regime1      : dict = {}   # strategy_id -> metrics dict from report_filtered_trades
_oos_bt_regime0      : dict = {}   # strategy_id -> metrics dict from report_filtered_trades
_oos_bt_regime01     : dict = {}   # strategy_id -> metrics dict from report_filtered_trades


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
    logger.debug(f"{'='*60}")
    logger.debug(f"  Symbol Diagnostics & Universe Selection  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

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
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 1 — Monte Carlo IS  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    logger.info(f"STAGE 1 [{STRATEGY_ID}] ── Monte Carlo IS        ── {N_PATHS_IS} paths | {len(param_dict_list)} combos")
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

# =============================================================================
#     final_prints(f"🎲 MC_{STRATEGY_ID} 🎲", DATA_FOLDER_IS, TIMEFRAME, min_vol_usdt=0,
#                  order_amount=ORDER_AMOUNT, param_names=param_names, lists_for_grid=lists_for_grid)
# 
# =============================================================================
    with tqdm_joblib(tqdm(total=N_PATHS_IS, desc="🔄 Evaluating MC IS paths", disable=logger.level > logging.DEBUG)):
        results_list = Parallel(n_jobs=N_JOBS)(
            delayed(_process_path)(i, paths_minor, param_dict_list)
            for i in range(N_PATHS_IS)
        )

    all_results  = [r for sublist in results_list for r in sublist]
    df_portfolio = pd.DataFrame(all_results)
    df_summary   = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)
    best_params  = extract_best_params(df_summary, param_names, lists_for_grid)

    params_str = " | ".join(f"{k}={v}" for k, v in best_params.items() if k not in ("SELL_AFTER",))
    logger.info(f"STAGE 2 [{STRATEGY_ID}] ── Backtest OOS           ── {params_str}")
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

# =============================================================================
#     final_prints(f"🔭 OOS_{STRATEGY_ID}", DATA_FOLDER_OOS, TIMEFRAME, 0, ORDER_AMOUNT, param_names, lists_for_grid)
#     oos_bt_portfolio, _ = report_backtesting(df=oos_df, parameters=param_names,
#                                              data_folder=DATA_FOLDER_OOS, initial_capital=INITIAL_BALANCE)
# =============================================================================

    best_bt_row = oos_df.loc[oos_df["Net_Gain"].idxmax()]

    _r2_oos = np.nan
    eq_hist = best_bt_row.get("sim_balance_history", None)
    if eq_hist and len(eq_hist.get("balance", [])) >= 2:
        y   = np.array(eq_hist["balance"]).reshape(-1, 1)
        X   = np.arange(len(y)).reshape(-1, 1)
        from sklearn.linear_model import LinearRegression as _LR
        _r2_oos = round(_LR().fit(X, y).score(X, y), 3)

    _oos_metrics.append({
        "strategy_id":   STRATEGY_ID,
        "net_gain_pct":  round(float(best_bt_row["Net_Gain"]) / INITIAL_BALANCE * 100, 2),
        "dd_pct":        round(-abs(float(best_bt_row["DD_pct"])), 2),
        "win_ratio":     round(float(best_bt_row["Win_Ratio"]) * 100, 1),
        "r2":            _r2_oos,
    })

    logger.info(f"STAGE 3 [{STRATEGY_ID}] ── Monte Carlo OOS        ── {N_PATHS_OOS} paths")
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

# =============================================================================
#     final_prints(f"🎲 MC_OOS_{STRATEGY_ID} 🎲", DATA_FOLDER_OOS, TIMEFRAME, min_vol_usdt=0,
#                  order_amount=ORDER_AMOUNT, param_names=param_names,
#                  lists_for_grid=[[best_params[n]] for n in param_names])
# 
# =============================================================================
    with tqdm_joblib(tqdm(total=N_PATHS_IS, desc="🔄 Evaluating MC IS paths", disable=logger.level > logging.DEBUG)):
        results_oos = Parallel(n_jobs=N_JOBS)(
            delayed(_process_path_oos)(i, paths_oos, best_params_list)
            for i in range(N_PATHS_OOS)
        )

    all_results_oos  = [r for sublist in results_oos for r in sublist]
    df_portfolio_oos = pd.DataFrame(all_results_oos)
    report_montecarlo(df_portfolio=df_portfolio_oos, param_names=param_names, initial_balance=INITIAL_BALANCE)

    # -------------------------------------------------------------------------
    # BLOCK 4 — REGIME ANALYSIS
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 4 — Regime Analysis  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    save_all_trades_to_excel(
        [(best_comb, oos_result)], param_names,
        f"all_trades_{STRATEGY_ID}.xlsx",
        strategy_name=STRATEGY_ID, save=True,
        output_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
    )
    trade_log = pd.read_excel(TRADES_PATH)
    trade_log.columns = trade_log.columns.str.lower().str.strip()
    trade_log["buy_time"] = pd.to_datetime(trade_log["buy_time"])
    logger.debug(f"Trades saved → {TRADES_PATH}  ({len(trade_log)} trades)")

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

    excl_str = str(excluded_families) if excluded_families else "none"
    logger.info(f"STAGE 4 [{STRATEGY_ID}] ── Regime Analysis        ── {len(trade_log)} trades | excl: {excl_str}")

    if excluded_families:
        logger.debug(f"Excluding regimes with negative profit: {excluded_families}")
        trade_log_filtered = trade_log[~trade_log["family"].isin(excluded_families)].reset_index(drop=True)
        logger.debug(f"Trades after filter: {len(trade_log_filtered)} / {len(trade_log)}")
        _r1_bt = report_filtered_trades(trade_log_filtered, initial_balance=INITIAL_BALANCE,
                                        data_folder=DATA_FOLDER_OOS,
                                        title=f"Filtered Trades — {STRATEGY_ID} (excl. {excluded_families})")
        if _r1_bt:
            _oos_bt_regime1[STRATEGY_ID] = _r1_bt
    else:
        logger.debug("No regimes with negative profit — no filtering applied.")

    # -------------------------------------------------------------------------
    # BLOCK 5 — REGIME 0 — BTC DIRECTION FILTER
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 5 — Regime 0 (BTC Direction Filter)  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    R0_BTC_FILE = os.path.join(DATA_FOLDER_OOS, "BTCUSDT_1Dutc.parquet")

    if not os.path.exists(R0_BTC_FILE):
        logger.warning("BTC 1Dutc file not found — skipping Regime 0 analysis.")
    else:
        r0_btc_df = load_btc_1d(R0_BTC_FILE, ma_period=R0_MA_PERIOD)

        def _get_btc_direction(buy_time):
            return get_btc_direction(buy_time, r0_btc_df, SIDE, R0_MA_PERIOD, R0_LONG_TH, R0_SHORT_TH)

        r0_trade_log = trade_log.copy()
        r0_trade_log["r0_direction"] = r0_trade_log["buy_time"].apply(_get_btc_direction)

        r0_counts = r0_trade_log["r0_direction"].value_counts()
        logger.debug(f"BTC direction distribution (MA{R0_MA_PERIOD}):")
        for direction, count in r0_counts.items():
            logger.debug(f"  {direction:<12}: {count} trades")

        keep_direction = "uptrend" if SIDE == "long" else "downtrend"
        r0_filtered    = r0_trade_log[r0_trade_log["r0_direction"] == keep_direction].reset_index(drop=True)
        logger.debug(f"Keeping '{keep_direction}' trades: {len(r0_filtered)} / {len(r0_trade_log)}")
        logger.info(f"STAGE 5 [{STRATEGY_ID}] ── Regime 0 (BTC filter)  ── {len(r0_filtered)} / {len(r0_trade_log)} kept ({keep_direction})")

        if len(r0_filtered) > 0:
            _r0_bt = report_filtered_trades(r0_filtered, initial_balance=INITIAL_BALANCE,
                                            data_folder=DATA_FOLDER_OOS,
                                            title=f"Regime 0 Filtered — {STRATEGY_ID} ({keep_direction} only, MA{R0_MA_PERIOD})")
            if _r0_bt:
                _oos_bt_regime0[STRATEGY_ID] = _r0_bt

    # -------------------------------------------------------------------------
    # BLOCK 6 — REGIME 0 + 1 COMBINED FILTER
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 6 — Regime 0 + 1 Combined Filter  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    if not os.path.exists(R0_BTC_FILE):
        logger.warning("BTC 1Dutc file not found — skipping combined analysis.")
    else:
        r01_base = trade_log_filtered if excluded_families and "trade_log_filtered" in vars() else trade_log
        logger.debug(f"Base trades (after Regime 1): {len(r01_base)}")

        r01_trade_log = r01_base.copy()
        r01_trade_log["r0_direction"] = r01_trade_log["buy_time"].apply(_get_btc_direction)

        r01_filtered = r01_trade_log[r01_trade_log["r0_direction"] == keep_direction].reset_index(drop=True)
        logger.debug(f"After Regime 0 filter ({keep_direction}): {len(r01_filtered)} / {len(r01_trade_log)}")
        logger.info(f"STAGE 6 [{STRATEGY_ID}] ── Regime 0+1 Combined    ── {len(r01_filtered)} / {len(r01_trade_log)} kept")

        if len(r01_filtered) > 0:
            _r01_bt = report_filtered_trades(r01_filtered, initial_balance=INITIAL_BALANCE,
                                             data_folder=DATA_FOLDER_OOS,
                                             title=f"Regime 0+1 Combined — {STRATEGY_ID} (excl. {excluded_families}, {keep_direction} only)")
            if _r01_bt:
                _oos_bt_regime01[STRATEGY_ID] = _r01_bt
            r01_trades_path = os.path.join(os.path.dirname(__file__), "brief_trades", f"all_trades_{STRATEGY_ID}_regime01.xlsx")
            r01_filtered.to_excel(r01_trades_path, index=False)
            logger.debug(f"Regime 0+1 trades saved → {r01_trades_path}  ({len(r01_filtered)} trades)")

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

    logger.debug(f"\n{'─'*75}")
    logger.debug(f"  FILTER COMPARISON SUMMARY — {STRATEGY_ID}")
    logger.debug(f"{'─'*75}")
    logger.debug(f"  {'Scenario':<14} {'Trades':>8} {'Net Gain%':>10} {'Win Rate%':>10} {'DD%':>8} {'R2':>7}")
    logger.debug(f"  {'-'*71}")
    for name, m in rows:
        if m["trades"] == 0:
            logger.debug(f"  {name:<14} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>7}")
        else:
            logger.debug(f"  {name:<14} {m['trades']:>8} {m['net_gain_pct']:>9.2f}% {m['win_rate']:>9.1f}% {m['dd_pct']:>7.2f}% {m['r2']:>7.3f}")
    logger.debug(f"  {'─'*71}")

    # -------------------------------------------------------------------------
    # BLOCK 7 — VALIDATION
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 7 — Validation  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

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

    verdict = "🟢 APPROVED" if approved else "🔴 REJECTED"
    logger.info(f"STAGE 7 [{STRATEGY_ID}] ── Validation             ── {verdict}  NetGain={bt_netgain_pct:.2f}% R2={r2:.3f} ProbNeg={prob_negative_oos:.1f}%")

    logger.debug(f"  Backtest OOS")
    logger.debug(f"    Net_Gain_pct : {bt_netgain_pct:>7.2f}%   (threshold > {THRESHOLD_NETGAIN_PCT}%)   {'✅' if ok_netgain  else '❌'}")
    logger.debug(f"    R2           : {r2:>7.3f}    (threshold > {THRESHOLD_R2})     {'✅' if ok_r2       else '❌'}")
    logger.debug(f"  Monte Carlo OOS")
    logger.debug(f"    Prob Negative: {prob_negative_oos:>7.2f}%   (threshold < {THRESHOLD_PROB_NEG}%)  {'✅' if ok_prob_neg else '❌'}")
    logger.debug(f"{'🟢 STRATEGY APPROVED' if approved else '🔴 STRATEGY REJECTED — checking regime filter...'} | {STRATEGY_ID}")

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

        logger.debug(f"  Second Round — Regime Filtered")
        logger.debug(f"    WR improvement : {wr_improvement:>7.2f}pp  (threshold > {THRESHOLD_WR_IMPROVEMENT}pp)   {'✅' if ok_wr_improvement else '❌'}  ({wr_original:.2f}% → {wr_filtered:.2f}%)")
        logger.debug(f"    R2 filtered    : {r2_filtered:>7.3f}    (threshold > {THRESHOLD_R2_FILTERED})      {'✅' if ok_r2_filtered    else '❌'}")
        logger.debug(f"    Prob Negative  : {prob_negative_oos:>7.2f}%   (threshold < {THRESHOLD_PROB_NEG_MAX}%)   {'✅' if ok_prob_neg_max   else '❌'}")
        logger.debug(f"{'🟢 STRATEGY APPROVED (regime filtered)' if approved_regime else '🔴 STRATEGY REJECTED (regime filtered)'} | {STRATEGY_ID}")

    approved = approved or approved_regime

    # -------------------------------------------------------------------------
    # BLOCK 7 — EQUITY CURVES
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 7 — Equity Curves  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    # Baseline
    _trade_logs_baseline.append((STRATEGY_ID, trade_log.copy()))

    metrics_baseline = compute_metrics(trade_log, capital=INITIAL_BALANCE, name=STRATEGY_ID)
    print_metrics_table([metrics_baseline], f"  Metrics — {STRATEGY_ID} (Baseline)")

    # Regime 1
    if excluded_families and "trade_log_filtered" in vars() and trade_log_filtered is not None and len(trade_log_filtered) > 0:
        _trade_logs_regime1.append((STRATEGY_ID, trade_log_filtered.copy()))

    # Regime 0
    if "r0_filtered" in vars() and r0_filtered is not None and len(r0_filtered) > 0:
        _trade_logs_regime0.append((STRATEGY_ID, r0_filtered.copy()))

    # Regime 0+1
    if "r01_filtered" in vars() and r01_filtered is not None and len(r01_filtered) > 0:
        _trade_logs_regime01.append((STRATEGY_ID, r01_filtered.copy()))

        metrics_regime01 = compute_metrics(r01_filtered, capital=INITIAL_BALANCE, name=f"{STRATEGY_ID}_r01")
        print_metrics_table([metrics_regime01], f"  Metrics — {STRATEGY_ID} (Regime 0+1)")

    # -------------------------------------------------------------------------
    # BLOCK 8 — UPDATE & COMPARE
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 8 — Update & Compare  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    active_str = "active=True" if approved else "active=False"
    logger.info(f"STAGE 8 [{STRATEGY_ID}] ── Update & Compare       ── params saved | {active_str}")

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
    logger.info(f"🏁 {STRATEGY_ID} — Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")

    # -------------------------------------------------------------------------
    # FINAL COMPARISON SUMMARY
    # -------------------------------------------------------------------------
    r1_tl  = trade_log_filtered if excluded_families and "trade_log_filtered" in vars() else None
    r0_tl  = r0_filtered        if "r0_filtered"  in vars() and len(r0_filtered)  > 0 else None
    r01_tl = r01_filtered       if "r01_filtered" in vars() and len(r01_filtered) > 0 else None

    def _quick(tl, cap):
        if tl is None or len(tl) == 0:
            return np.nan, np.nan, np.nan
        profits = tl.sort_values("buy_time")["profit"].values
        eq      = cap + np.cumsum(profits)
        cm      = np.maximum.accumulate(eq)
        dd      = ((eq - cm) / cm * 100).min()
        ng      = (eq[-1] - cap) / cap * 100
        wr      = (profits > 0).mean() * 100
        return round(ng, 2), round(dd, 2), round(wr, 1)

    bt_dd = float(best_bt_row.get("DD_pct", np.nan))

    rows = [
        ("OOS Backtest",  bt_netgain_pct,          bt_dd,                (best_bt_row.get("Win_Ratio", np.nan) * 100)),
        ("Regime 1",      *_quick(r1_tl,  INITIAL_BALANCE)),
        ("Regime 0",      *_quick(r0_tl,  INITIAL_BALANCE)),
        ("Regime 0+1",    *_quick(r01_tl, INITIAL_BALANCE)),
    ]

    logger.debug(f"\n{'─'*65}")
    logger.debug(f"  FINAL COMPARISON — {STRATEGY_ID}")
    logger.debug(f"{'─'*65}")
    logger.debug(f"  {'Scenario':<16} {'Net Gain%':>10} {'DD%':>8} {'Win Rate%':>10}")
    logger.debug(f"  {'-'*61}")
    for name, ng, dd, wr in rows:
        ng_str = f"{ng:>9.2f}%" if not np.isnan(ng) else f"{'N/A':>10}"
        dd_str = f"{dd:>7.2f}%" if not np.isnan(dd) else f"{'N/A':>8}"
        wr_str = f"{wr:>9.1f}%" if not np.isnan(wr) else f"{'N/A':>10}"
        logger.debug(f"  {name:<16} {ng_str} {dd_str} {wr_str}")
    logger.debug(f"  {'─'*61}")


# =============================================================================
# PORTFOLIO ANALYSIS — call after all run_batch() calls
# =============================================================================
def run_portfolio_analysis():
    """
    Compute combined portfolio metrics and best combinations
    for baseline and regime 0+1 trade_logs.
    Call this after all run_batch() calls in the orchestrator.
    """
    from itertools import combinations
    import pandas as _pd

    for label, trade_logs in [("Baseline", _trade_logs_baseline), ("Regime 0+1", _trade_logs_regime01)]:
        if not trade_logs:
            continue

        logger.debug(f"\n{'='*70}")
        logger.debug(f"  PORTFOLIO ANALYSIS — {label}")
        logger.debug(f"{'='*70}")

        named_logs = {sid: df for sid, df in trade_logs}

        # Individual metrics
        metrics_list = []
        for sid, df in named_logs.items():
            metrics_list.append(compute_metrics(df, capital=INITIAL_BALANCE, name=sid))

        # Combined portfolio
        if len(named_logs) > 1:
            combined_tl      = _pd.concat(list(named_logs.values()), ignore_index=True).sort_values("buy_time").reset_index(drop=True)
            combined_capital = INITIAL_BALANCE * len(named_logs)
            metrics_list.append(compute_metrics(combined_tl, capital=combined_capital, name="Combined"))

        print_metrics_table(metrics_list, f"📊 METRICS TABLE — {label}")

        # Best combinations
        combo_results = []
        for r in range(1, len(named_logs) + 1):
            for combo in combinations(named_logs.keys(), r):
                combo_tl = _pd.concat([named_logs[sid] for sid in combo], ignore_index=True).sort_values("buy_time").reset_index(drop=True)
                capital  = INITIAL_BALANCE * len(combo)
                combo_results.append(compute_metrics(combo_tl, capital=capital, name="+".join(combo)))

        combo_df = _pd.DataFrame(combo_results)

        for metric, ascending, title in [
            ("Net_Gain_pct",  False, "📈 TOP 5 BY NET GAIN"),
            ("R_Squared",     False, "📐 TOP 5 BY R²"),
            ("Max_DD_pct",    False, "📉 TOP 5 BY LOWEST DD"),
        ]:
            top5 = combo_df.sort_values(metric, ascending=ascending).head(5)
            print_metrics_table(top5.to_dict("records"), f"\n{title} — {label}", shorten_names=True)

    # OOS BACKTEST vs BASELINE COMPARISON
    if _oos_metrics and _trade_logs_baseline:
        b_metrics      = {sid: compute_metrics(df, capital=INITIAL_BALANCE, name=sid)
                          for sid, df in _trade_logs_baseline}
        r01_metrics    = {sid: compute_metrics(df, capital=INITIAL_BALANCE, name=sid)
                          for sid, df in _trade_logs_regime01}
        r1_metrics_map = {sid: compute_metrics(df, capital=INITIAL_BALANCE, name=sid)
                          for sid, df in _trade_logs_regime1}
        r0_metrics_map = {sid: compute_metrics(df, capital=INITIAL_BALANCE, name=sid)
                          for sid, df in _trade_logs_regime0}

        lines = []
        lines.append(f"\n{'─'*87}")
        lines.append(f"  OOS BACKTEST vs BASELINE COMPARISON")
        lines.append(f"{'─'*87}")
        lines.append(f"  {'Strategy':<25} {'Source':<14} {'Net Gain%':>10} {'DD%':>8} {'Win Rate%':>10} {'R2':>7}")
        lines.append(f"  {'-'*83}")

        for oos in _oos_metrics:
            sid    = oos["strategy_id"]
            b      = b_metrics.get(sid)
            r1     = r1_metrics_map.get(sid)
            r0     = r0_metrics_map.get(sid)
            r01    = r01_metrics.get(sid)
            r1_bt  = _oos_bt_regime1.get(sid)
            r0_bt  = _oos_bt_regime0.get(sid)
            r01_bt = _oos_bt_regime01.get(sid)
            lines.append(f"  {sid:<25} {'OOS BT':<14} {oos['net_gain_pct']:>9.2f}% {oos['dd_pct']:>7.2f}% {oos['win_ratio']:>9.1f}% {oos['r2']:>7.3f}")
            if b:
                lines.append(f"  {'':<25} {'Baseline':<14} {b['Net_Gain_pct']:>9.2f}% {b['Max_DD_pct']:>7.2f}% {b['Win_Rate']:>9.1f}% {b['R_Squared']:>7.3f}")
            if r1_bt:
                lines.append(f"  {'':<25} {'R1 (OOS BT)':<14} {r1_bt['net_gain_pct']:>9.2f}% {r1_bt['dd_pct']:>7.2f}% {r1_bt['win_ratio']:>9.1f}% {r1_bt['r2']:>7.3f}")
            if r1:
                lines.append(f"  {'':<25} {'R1 (metrics)':<14} {r1['Net_Gain_pct']:>9.2f}% {r1['Max_DD_pct']:>7.2f}% {r1['Win_Rate']:>9.1f}% {r1['R_Squared']:>7.3f}")
            if r0_bt:
                lines.append(f"  {'':<25} {'R0 (OOS BT)':<14} {r0_bt['net_gain_pct']:>9.2f}% {r0_bt['dd_pct']:>7.2f}% {r0_bt['win_ratio']:>9.1f}% {r0_bt['r2']:>7.3f}")
            if r0:
                lines.append(f"  {'':<25} {'R0 (metrics)':<14} {r0['Net_Gain_pct']:>9.2f}% {r0['Max_DD_pct']:>7.2f}% {r0['Win_Rate']:>9.1f}% {r0['R_Squared']:>7.3f}")
            if r01_bt:
                lines.append(f"  {'':<25} {'R0+1 (OOS BT)':<14} {r01_bt['net_gain_pct']:>9.2f}% {r01_bt['dd_pct']:>7.2f}% {r01_bt['win_ratio']:>9.1f}% {r01_bt['r2']:>7.3f}")
            if r01:
                lines.append(f"  {'':<25} {'R0+1 (metrics)':<14} {r01['Net_Gain_pct']:>9.2f}% {r01['Max_DD_pct']:>7.2f}% {r01['Win_Rate']:>9.1f}% {r01['R_Squared']:>7.3f}")
            lines.append(f"  {'-'*83}")
        lines.append(f"  {'─'*83}")

        logger.info("\n".join(lines))