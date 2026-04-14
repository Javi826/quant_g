import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "market_regime")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "market_regime")))

import logging
import contextlib
import time
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL  = logging.INFO   # Change to logging.DEBUG for full verbosity
SHOW_PLOTS = False           # Set to True to enable matplotlib plots

logging.basicConfig(level=LOG_LEVEL, format="%(message)s", force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

if not SHOW_PLOTS:
    import matplotlib
    matplotlib.use("Agg")

from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from utils.st_tools import extract_ohlcv_from_path, compile_MC_results
from utils.st_tools import compile_grid_results, prepare_ohlcv_arrays
from utils.st_tools import get_n_obs, save_all_trades_to_csv
from tools.optimize_MCf_tf import generate_paths_for_all_symbols_functional
from utils.analysis import report_montecarlo, report_backtesting
from utils.utils import filter_symbols, final_prints
from regime_performance import analyze_strategy, print_single_strategy_all_dimensions
from regime_common import load_btc_for_timeframe, calc_all_metrics_at_time, classify_trade_by_family
from regime_performance import MA_PERIOD, LOOKBACK_BARS
from shared_config import REGIME_FAMILIES as FAMILIES, REGIME_HURST_WINDOW as HURST_WINDOW, REGIME_ER_WINDOW as ER_WINDOW
from shared_config import REGIME_ATR_WINDOW as ATR_WINDOW, REGIME_PE_WINDOW as PE_WINDOW, REGIME_PE_ORDER as PE_ORDER
from batch_utils import report_filtered_trades, extract_best_params, select_universe
from batch_utils import enrich_trades_with_regime, update_strategies_params, update_strategies_symbols, load_btc_1d
from batch_utils import get_btc_direction, compute_metrics, print_metrics_table, calc_r2_from_equity_hist
from batch_utils import print_all_curves_table, print_best_combinations
from batch_utils import print_strategies_summary, print_update_status, print_portfolio_metrics_table
from batch_utils import validate_csv_columns

from signals.add_signals_parity      import parity_long, parity_short
from signals.add_signals_reversal    import reversal_long, reversal_short
from signals.add_signals_flag        import flag_long, flag_short
from signals.add_signals_orderblocks import orderblocks_long, orderblocks_short
from signals.add_signals_ranging     import ranging_long, ranging_short

SIGNAL_REGISTRY = {
    "parity_long":       {"fn": parity_long,       "params": ["lookback", "tolerance", "ma_period"]},
    "parity_short":      {"fn": parity_short,      "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_long":     {"fn": reversal_long,     "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_short":    {"fn": reversal_short,    "params": ["lookback", "tolerance", "ma_period"]},
    "flag_long":         {"fn": flag_long,         "params": ["lookback", "impulse", "flag", "ma_period"]},
    "flag_short":        {"fn": flag_short,        "params": ["lookback", "impulse", "flag", "ma_period"]},
    "orderblocks_long":  {"fn": orderblocks_long,  "params": ["lookback", "tolerance", "impulse"]},
    "orderblocks_short": {"fn": orderblocks_short, "params": ["lookback", "tolerance", "impulse"]},
    "ranging_long":      {"fn": ranging_long,      "params": ["lookback", "tolerance", "ma_period", "ranges"]},
    "ranging_short":     {"fn": ranging_short,     "params": ["lookback", "tolerance", "ma_period", "ranges"]},
}

DTYPE = np.float32

logger = logging.getLogger("BOT_batch.main_batch")

# =============================================================================
# GLOBAL CONFIGURATION — defaults, overridden by run_batch at runtime
# =============================================================================
DATA_FOLDER_IS  = "data/crypto_2022_IS"
DATA_FOLDER_OOS = "data/crypto_2026_OOS"
N_JOBS          = -1
MY_SYMBOLS      = False
SHOW_PROGRESS   = False

N_PATHS_IS  = 100
N_PATHS_OOS = 2000

# Validation thresholds — Round 1
R1_NETGAIN_ROUND1    = 20.0
R1_RSQUARED_ROUND1   = 0.8
R1_PROBNEG_ROUND1    = 15.0

# Validation thresholds — Round 2 path A (regime filtered)
R2A_NETGAIN_ROUND2   = 20.0
R2A_RSQUARED_ROUND2  = 0.90
R2A_PROBNEG_ROUND1   = 100.0

# Validation thresholds — Round 2 path B (high netgain OOS)
R2B_NETGAIN_ROUND1   = 80.0
R2B_PROBNEG_ROUND1   = 20.0

R0_MA_PERIOD = 5
R0_LONG_TH   = 1.00
R0_SHORT_TH  = 1.00

# IS symbol selection
FIX_SYMBOLS_MCIS_TRAINING = True   # If True, use top N_SYMBOLS_MCIS from IS by volume directly
N_SYMBOLS_MCIS            = 6      # Number of IS symbols when FIX_SYMBOLS_MCIS_TRAINING=True

# Strategy selection — set to None or [] to run all
SELECTED_STRATEGIES = [
    "02_reversal_long_4H",
    "03_parity_long_4H",
    "04_reversal_short_4H",
    "06_reversal_long_1H",
    "07_reversal_short_1H",
    "08_reversal_long_6Hutc",
    "09_reversal_short_6Hutc",
    "10_parity_long_1H",
    "11_parity_short_1H",
    "12_parity_long_6Hutc",
    "13_orderblocks_short_4H",
    "16_ranging_short_6Hutc",
    "17_flag_long_4H",
    "19_flag_short_4H",
    "20_flag_short_1H",
]

# Portfolio analysis flags
RUN_PORTFOLIO_ANALYSIS  = True   # Set to False to skip all portfolio analysis
RUN_BEST_COMBINATIONS   = False # Set to False to skip best combinations (expensive)
UPDATE_CSV              = True   # Set to False to skip CSV updates (tables will show last run data)

STRATEGIES_PARAMS_FOLDER = os.path.join(os.path.dirname(__file__), "strategies_params")
CSV_PARAMS          = os.path.join(STRATEGIES_PARAMS_FOLDER, "strategies_params.csv")
SYMBOLS_LIVE_FOLDER = os.path.join(os.path.dirname(__file__), "symbols_live")

# Global trade_log accumulators (populated by each run_batch call)
_trade_logs_baseline  : list = []   # (strategy_id, trade_log_df)
_trade_logs_regime01  : list = []   # (strategy_id, trade_log_df)
_oos_metrics          : list = []   # {strategy_id, net_gain_pct, dd_pct, win_ratio, r2}
_validation_results   : list = []   # {strategy_id, verdict, round, net_gain_pct, dd_pct, win_ratio, r2, prob_neg_pct}


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
    TRADES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "brief_trades", f"all_trades_{STRATEGY_ID}.csv"))

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
        fix_symbols_mcis=FIX_SYMBOLS_MCIS_TRAINING,
        n_symbols_mcis=N_SYMBOLS_MCIS,
    )

    # -------------------------------------------------------------------------
    # BLOCK 1 — MONTE CARLO IS
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 1 — Monte Carlo IS  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    logger.info(f"STAGE 1  ── Monte Carlo IS         ── {N_PATHS_IS} paths | {len(param_dict_list)} combos")
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

    with (tqdm_joblib(tqdm(total=N_PATHS_IS, desc="🔄 Evaluating MC IS paths")) if SHOW_PROGRESS else contextlib.nullcontext()):
        results_list = Parallel(n_jobs=N_JOBS)(
            delayed(_process_path)(i, paths_minor, param_dict_list)
            for i in range(N_PATHS_IS)
        )

    all_results  = [r for sublist in results_list for r in sublist]
    df_portfolio = pd.DataFrame(all_results)
    df_summary   = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE)
    best_params  = extract_best_params(df_summary, param_names, lists_for_grid)

    params_str = " | ".join(f"{k}={v}" for k, v in best_params.items() if k not in ("SELL_AFTER",))
    logger.info(f"STAGE 2  ── Backtest OOS           ── {params_str}")

    # -------------------------------------------------------------------------
    # BLOCK 2 — BACKTEST OOS
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 2 — Backtest OOS  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

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

    oos_bt_portfolio, _ = report_backtesting(df=oos_df, parameters=param_names,
                                             data_folder=DATA_FOLDER_OOS, initial_capital=INITIAL_BALANCE)

    best_bt_row = oos_df.loc[oos_df["Net_Gain"].idxmax()]

    _r2_oos = np.nan
    eq_hist = best_bt_row.get("sim_balance_history", None)
    _r2_oos = calc_r2_from_equity_hist(eq_hist)

    _oos_metrics.append({
        "strategy_id":   STRATEGY_ID,
        "net_gain_pct":  round(float(best_bt_row["Net_Gain"]) / INITIAL_BALANCE * 100, 2),
        "dd_pct":        round(-abs(float(best_bt_row["DD_pct"])), 2),
        "win_ratio":     round(float(best_bt_row["Win_Ratio"]) * 100, 1),
        "r2":            _r2_oos,
    })

    logger.info(f"STAGE 3  ── Monte Carlo OOS        ── {N_PATHS_OOS} paths")

    # -------------------------------------------------------------------------
    # BLOCK 3 — MONTE CARLO OOS
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 3 — Monte Carlo OOS  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    n_obs_oos = get_n_obs(TIMEFRAME)
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

    with (tqdm_joblib(tqdm(total=N_PATHS_OOS, desc="🔄 Evaluating MC OOS paths")) if SHOW_PROGRESS else contextlib.nullcontext()):
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

    save_all_trades_to_csv(
        [(best_comb, oos_result)], param_names,
        f"all_trades_{STRATEGY_ID}.csv",
        strategy_name=STRATEGY_ID, save=True,
        output_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
    )
    trade_log = pd.read_csv(TRADES_PATH)
    trade_log.columns = trade_log.columns.str.lower().str.strip()
    trade_log["buy_time"] = pd.to_datetime(trade_log["buy_time"])
    logger.debug(f"Trades saved → {TRADES_PATH}  ({len(trade_log)} trades)")

    regime_result = analyze_strategy(TRADES_PATH, FAMILIES, INITIAL_BALANCE, ohlc_folder=DATA_FOLDER_OOS)
    print_single_strategy_all_dimensions(regime_result)

    trade_log = enrich_trades_with_regime(
        trade_log=trade_log, ohlc_folder=DATA_FOLDER_OOS, timeframe=TIMEFRAME,
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
    logger.info(f"STAGE 4  ── Regime 1 Analysis      ── {len(trade_log)} trades | excl: {excl_str}")

    if excluded_families:
        logger.debug(f"Excluding regimes with negative profit: {excluded_families}")
        trade_log_filtered = trade_log[~trade_log["family"].isin(excluded_families)].reset_index(drop=True)
        logger.debug(f"Trades after filter: {len(trade_log_filtered)} / {len(trade_log)}")
        report_filtered_trades(trade_log_filtered, initial_balance=INITIAL_BALANCE,
                               data_folder=DATA_FOLDER_OOS,
                               title=f"Filtered Trades — {STRATEGY_ID} (excl. {excluded_families})")
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
        logger.warning("⚠️  BTC 1Dutc file not found — skipping Regime 0 analysis.")
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
        logger.info(f"STAGE 5  ── Regime 0 (BTC filter)  ── {len(r0_filtered)} / {len(r0_trade_log)} kept ({keep_direction})")

        if len(r0_filtered) > 0:
            report_filtered_trades(r0_filtered, initial_balance=INITIAL_BALANCE,
                                   data_folder=DATA_FOLDER_OOS,
                                   title=f"Regime 0 Filtered — {STRATEGY_ID} ({keep_direction} only, MA{R0_MA_PERIOD})")

    # -------------------------------------------------------------------------
    # BLOCK 6 — REGIME 0 + 1 COMBINED FILTER
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 6 — Regime 0 + 1 Combined Filter  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    if not os.path.exists(R0_BTC_FILE):
        logger.warning("⚠️  BTC 1Dutc file not found — skipping combined analysis.")
    else:
        r01_base = trade_log_filtered if excluded_families and "trade_log_filtered" in vars() else trade_log
        logger.debug(f"Base trades (after Regime 1): {len(r01_base)}")

        r01_trade_log = r01_base.copy()
        r01_trade_log["r0_direction"] = r01_trade_log["buy_time"].apply(_get_btc_direction)

        r01_filtered = r01_trade_log[r01_trade_log["r0_direction"] == keep_direction].reset_index(drop=True)
        logger.debug(f"After Regime 0 filter ({keep_direction}): {len(r01_filtered)} / {len(r01_trade_log)}")
        logger.info(f"STAGE 6  ── Regime 0+1 Combined    ── {len(r01_filtered)} / {len(r01_trade_log)} kept")

        if len(r01_filtered) > 0:
            report_filtered_trades(r01_filtered, initial_balance=INITIAL_BALANCE,
                                   data_folder=DATA_FOLDER_OOS,
                                   title=f"Regime 0+1 Combined — {STRATEGY_ID} (excl. {excluded_families}, {keep_direction} only)")
            r01_trades_path = os.path.join(os.path.dirname(__file__), "brief_trades", f"all_trades_{STRATEGY_ID}_regime01.csv")
            r01_filtered.to_csv(r01_trades_path, index=False)
            logger.debug(f"Regime 0+1 trades saved → {r01_trades_path}  ({len(r01_filtered)} trades)")

    # -------------------------------------------------------------------------
    # SUMMARY TABLE — Baseline vs Regime 1 vs Regime 0 vs Regime 0+1
    # -------------------------------------------------------------------------
    def _m(df):
        if df is None or len(df) == 0:
            return {"trades": 0, "net_gain_pct": np.nan, "win_rate": np.nan, "dd_pct": np.nan, "r2": np.nan}
        m = compute_metrics(df, capital=INITIAL_BALANCE, name="")
        return {"trades": len(df), "net_gain_pct": m["Net_Gain_pct"], "win_rate": m["Win_Rate"], "dd_pct": m["Max_DD_pct"], "r2": m["R_Squared"]}

    r1_df   = trade_log_filtered if excluded_families and "trade_log_filtered" in vars() else None
    r0_df   = r0_filtered        if "r0_filtered"  in vars() and len(r0_filtered)  > 0 else None
    r01_df  = r01_filtered       if "r01_filtered" in vars() and len(r01_filtered) > 0 else None

    rows = [
        ("Baseline",    _m(trade_log)),
        ("Regime 1",    _m(r1_df)),
        ("Regime 0",    _m(r0_df)),
        ("Regime 0+1",  _m(r01_df)),
    ]

    logger.debug(f"\n{'─'*105}")
    logger.debug(f"  FILTER COMPARISON SUMMARY — {STRATEGY_ID}")
    logger.debug(f"{'─'*105}")
    logger.debug(f"  {'Scenario':<14} {'Trades':>8} {'Net Gain%':>10} {'Win Rate%':>10} {'DD%':>8} {'R2':>7}")
    logger.debug(f"  {'-'*103}")
    for name, m in rows:
        if m["trades"] == 0:
            logger.debug(f"  {name:<14} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>7}")
        else:
            logger.debug(f"  {name:<14} {m['trades']:>8} {m['net_gain_pct']:>9.2f}% {m['win_rate']:>9.1f}% {m['dd_pct']:>7.2f}% {m['r2']:>7.3f}")
    logger.debug(f"  {'─'*105}")

    # -------------------------------------------------------------------------
    # BLOCK 7 — VALIDATION
    # -------------------------------------------------------------------------
    logger.debug(f"{'='*60}")
    logger.debug(f"  BLOCK 7 — Validation  |  {STRATEGY_ID}")
    logger.debug(f"{'='*60}")

    bt_netgain_pct = best_bt_row["Net_Gain"] / INITIAL_BALANCE * 100
    equity_hist    = best_bt_row.get("sim_balance_history", None)
    r2             = calc_r2_from_equity_hist(equity_hist)

    path_grouped_oos = df_portfolio_oos.groupby("path_index")["Portfolio_Final_Balance"].mean().reset_index()
    path_grouped_oos["Net_Gain_pct"] = (path_grouped_oos["Portfolio_Final_Balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    prob_negative_oos = (path_grouped_oos["Net_Gain_pct"] < 0).mean() * 100

    ok_netgain  = bt_netgain_pct    > R1_NETGAIN_ROUND1
    ok_r2       = r2                > R1_RSQUARED_ROUND1
    ok_prob_neg = prob_negative_oos < R1_PROBNEG_ROUND1
    approved    = ok_netgain and ok_r2 and ok_prob_neg

    verdict = "🟢 VALIDATED" if approved else "🔴 REJECTED"
    _v1 = ("REJECTED" if not approved else "VALIDATED").ljust(13)
    logger.info(f"STAGE 7  ── Validation             ── {'🔴' if not approved else '🟢'} {_v1} NetGain={bt_netgain_pct:.2f}% R2={r2:.2f} ProbNeg={prob_negative_oos:.1f}%")

    logger.debug(f"  Backtest OOS")
    logger.debug(f"    Net_Gain_pct : {bt_netgain_pct:>7.2f}%   (threshold > {R1_NETGAIN_ROUND1}%)   {'✅' if ok_netgain  else '❌'}")
    logger.debug(f"    R2           : {r2:>7.3f}    (threshold > {R1_RSQUARED_ROUND1})     {'✅' if ok_r2       else '❌'}")
    logger.debug(f"  Monte Carlo OOS")
    logger.debug(f"    Prob Negative: {prob_negative_oos:>7.2f}%   (threshold < {R1_PROBNEG_ROUND1}%)  {'✅' if ok_prob_neg else '❌'}")
    logger.debug(f"{'🟢 STRATEGY APPROVED' if approved else '🔴 STRATEGY REJECTED — checking regime filter...'} | {STRATEGY_ID}")

    approved_regime = False
    round_path      = ""

    approved = approved or approved_regime

    # -------------------------------------------------------------------------
    # SECOND ROUND VALIDATION — Regime 0+1
    # -------------------------------------------------------------------------
    if not approved and "r01_filtered" in vars() and r01_filtered is not None and len(r01_filtered) > 0:
        df_filt     = r01_filtered.copy().sort_values("buy_time").reset_index(drop=True)
        equity_filt = INITIAL_BALANCE + df_filt["profit"].cumsum().values
        r2_filtered = calc_r2_from_equity_hist({"balance": equity_filt.tolist()})

        netgain_filtered    = compute_metrics(r01_filtered, capital=INITIAL_BALANCE, name="")["Net_Gain_pct"]

        ok_netgain_filtered = netgain_filtered  > R2A_NETGAIN_ROUND2
        ok_r2_filtered      = r2_filtered       > R2A_RSQUARED_ROUND2
        ok_prob_neg_max     = prob_negative_oos < R2A_PROBNEG_ROUND1
        approved_path_a     = ok_netgain_filtered and ok_r2_filtered and ok_prob_neg_max

        ok_netgain_high    = bt_netgain_pct     > R2B_NETGAIN_ROUND1
        ok_prob_neg_strict = prob_negative_oos  < R2B_PROBNEG_ROUND1
        approved_path_b    = ok_netgain_high and ok_prob_neg_strict

        approved_regime = approved_path_a or approved_path_b
        round_path      = "A" if approved_path_a else ("B" if approved_path_b else "")

        logger.debug(f"  Second Round — Path A (regime filtered)")
        logger.debug(f"    Net Gain       : {netgain_filtered:>7.2f}%   (threshold > {R2A_NETGAIN_ROUND2}%)    {'✅' if ok_netgain_filtered else '❌'}")
        logger.debug(f"    R2 filtered    : {r2_filtered:>7.3f}    (threshold > {R2A_RSQUARED_ROUND2})      {'✅' if ok_r2_filtered  else '❌'}")
        logger.debug(f"    Prob Negative  : {prob_negative_oos:>7.2f}%   (threshold < {R2A_PROBNEG_ROUND1}%)   {'✅' if ok_prob_neg_max else '❌'}")
        logger.debug(f"  Second Round — Path B (high netgain)")
        logger.debug(f"    Net Gain OOS   : {bt_netgain_pct:>7.2f}%   (threshold > {R2B_NETGAIN_ROUND1}%)    {'✅' if ok_netgain_high    else '❌'}")
        logger.debug(f"    Prob Negative  : {prob_negative_oos:>7.2f}%   (threshold < {R2B_PROBNEG_ROUND1}%)   {'✅' if ok_prob_neg_strict else '❌'}")
        path_str       = f" ({round_path})" if approved_regime else ""
        verdict_r2     = f"{'🟢 VALIDATED' if approved_regime else '🔴 REJECTED'}{path_str}"
        _v2 = (f"VALIDATED ({round_path})" if approved_regime else "REJECTED").ljust(13)
        logger.info(f"STAGE 7  ── Validation (Round 2)   ── {'🟢' if approved_regime else '🔴'} {_v2} NetGain={netgain_filtered:.2f}% R2={r2_filtered:.2f} ProbNeg={prob_negative_oos:.1f}%")

        approved = approved or approved_regime

    _round = "—"
    if approved and not approved_regime:
        _round = "Round 1"
    elif approved and approved_regime:
        _round = f"Round 2 ({round_path})"

    # Build regime flags for validation_results (1.0 / 0.0 / None if no regime_result)
    _regime_trending = 1.0 if regime_result["family_stats"].get("trending", {}).get("profit", 0) >= 0 else 0.0
    _regime_ranging  = 1.0 if regime_result["family_stats"].get("ranging",  {}).get("profit", 0) >= 0 else 0.0
    _regime_volatile = 1.0 if regime_result["family_stats"].get("volatile", {}).get("profit", 0) >= 0 else 0.0

    _validation_results.append({
        "strategy_id":      STRATEGY_ID,
        "verdict":          "🟢 VALIDATED" if approved else "🔴 REJECTED",
        "round":            _round,
        "net_gain_pct":     round(bt_netgain_pct, 2),
        "dd_pct":           round(-abs(float(best_bt_row.get("DD_pct", np.nan))), 2),
        "win_ratio":        round(float(best_bt_row.get("Win_Ratio", np.nan)) * 100, 1),
        "r2":               r2,
        "prob_neg_pct":     round(prob_negative_oos, 2),
        "params_changed":   False,
        "active_prev":      None,
        "active_new":       None,
        "symbols_changed":  False,
        "regime_trending":  _regime_trending,
        "regime_ranging":   _regime_ranging,
        "regime_volatile":  _regime_volatile,
        "regime_changes":   [],
    })

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

    PARAM_KEYS = [p.lower() for p in param_names]

    _params_result  = update_strategies_params(
        csv_path=CSV_PARAMS, strategy_id=STRATEGY_ID, best_params=best_params,
        param_keys=PARAM_KEYS, validated=approved, bt_netgain_pct=bt_netgain_pct,
        r2=r2, prob_negative_oos=prob_negative_oos,
        regime_stats=regime_result["family_stats"],
    ) if UPDATE_CSV else None

    _symbols_result = update_strategies_symbols(
        strategy_id=STRATEGY_ID, symbols_oos_final=symbols_oos_final,
        timeframe=TIMEFRAME, symbols_live_folder=SYMBOLS_LIVE_FOLDER,
    ) if UPDATE_CSV else None

    _changes = []
    if _params_result:
        if _params_result.get("active_prev") != _params_result.get("active_new"):
            _changes.append("active")
        if _params_result.get("params_changed"):
            _changes.append("params")
        if _params_result.get("regime_changes"):
            _changes.append("regime")
    if _symbols_result and _symbols_result.get("symbols_changed"):
        _changes.append("symbols")
    _changes_str = " | ".join(_changes) if _changes else "no changes"
    _icon = "🔵" if _changes else "⚪"
    logger.info(f"STAGE 8  ── Update & Compare       ── {_icon} {_changes_str}")

    elapsed = int(time.time() - start_time)
    logger.info(f"DONE     ──  🏁 {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")

    # Backfill update info into last validation result
    if _validation_results:
        _validation_results[-1].update({
            "params_changed":  _params_result.get("params_changed", False)  if _params_result else False,
            "param_changes":   _params_result.get("param_changes", [])      if _params_result else [],
            "regime_changes":  _params_result.get("regime_changes", [])     if _params_result else [],
            "active_prev":     _params_result.get("active_prev", None)      if _params_result else None,
            "active_new":      _params_result.get("active_new", None)       if _params_result else None,
            "symbols_changed": _symbols_result.get("symbols_changed", False) if _symbols_result else False,
        })

    # -------------------------------------------------------------------------
    # FINAL COMPARISON SUMMARY
    # -------------------------------------------------------------------------
    r1_tl  = trade_log_filtered if excluded_families and "trade_log_filtered" in vars() else None
    r0_tl  = r0_filtered        if "r0_filtered"  in vars() and len(r0_filtered)  > 0 else None
    r01_tl = r01_filtered       if "r01_filtered" in vars() and len(r01_filtered) > 0 else None

    def _q(tl):
        if tl is None or len(tl) == 0:
            return np.nan, np.nan, np.nan
        m = compute_metrics(tl, capital=INITIAL_BALANCE, name="")
        return m["Net_Gain_pct"], m["Max_DD_pct"], m["Win_Rate"]

    bt_dd = float(best_bt_row.get("DD_pct", np.nan))

    rows = [
        ("OOS Backtest",  bt_netgain_pct,    bt_dd,                (best_bt_row.get("Win_Ratio", np.nan) * 100)),
        ("Regime 1",      *_q(r1_tl)),
        ("Regime 0",      *_q(r0_tl)),
        ("Regime 0+1",    *_q(r01_tl)),
    ]

    logger.debug(f"\n{'─'*105}")
    logger.debug(f"  FINAL COMPARISON — {STRATEGY_ID}")
    logger.debug(f"{'─'*105}")
    logger.debug(f"  {'Scenario':<16} {'Net Gain%':>10} {'DD%':>8} {'Win Rate%':>10}")
    logger.debug(f"  {'-'*103}")
    for name, ng, dd, wr in rows:
        ng_str = f"{ng:>9.2f}%" if not np.isnan(ng) else f"{'N/A':>10}"
        dd_str = f"{dd:>7.2f}%" if not np.isnan(dd) else f"{'N/A':>8}"
        wr_str = f"{wr:>9.1f}%" if not np.isnan(wr) else f"{'N/A':>10}"
        logger.debug(f"  {name:<16} {ng_str} {dd_str} {wr_str}")
    logger.debug(f"  {'─'*105}")


# =============================================================================
# PORTFOLIO ANALYSIS — call after all run_batch() calls
# =============================================================================
def run_portfolio_analysis():
    """
    Compute combined portfolio metrics and best combinations
    for baseline and regime 0+1 trade_logs.
    Call this after all run_batch() calls in the orchestrator.
    """
    if not RUN_PORTFOLIO_ANALYSIS:
        print_strategies_summary(_validation_results)
        print_update_status(CSV_PARAMS, SYMBOLS_LIVE_FOLDER, _validation_results)
        return

    for label, trade_logs in [("Baseline", _trade_logs_baseline), ("Regime 0+1", _trade_logs_regime01)]:
        if not trade_logs:
            continue
        logger.debug(f"\n{'='*70}")
        logger.debug(f"  PORTFOLIO ANALYSIS — {label}")
        logger.debug(f"{'='*70}")
        print_portfolio_metrics_table(trade_logs, label, INITIAL_BALANCE)

    r01_metrics = {sid: compute_metrics(df, capital=INITIAL_BALANCE, name=sid)
                   for sid, df in _trade_logs_regime01}

    # =========================================================================
    # STRATEGIES SUMMARY
    # =========================================================================
    logger.info(f"\n{'─'*105}")
    logger.info(f"  STRATEGIES SUMMARY")
    logger.info(f"{'─'*105}")
    print_strategies_summary(_validation_results)

    # =========================================================================
    # UPDATE STATUS TABLES
    # =========================================================================
    logger.info(f"\n{'─'*105}")
    logger.info(f"  UPDATE STATUS")
    logger.info(f"{'─'*105}")
    print_update_status(CSV_PARAMS, SYMBOLS_LIVE_FOLDER, _validation_results)

    # =========================================================================
    # ALL CURVES COMBINED — Baseline vs Regime 0+1
    # =========================================================================
    if _trade_logs_baseline:
        logger.info(f"\n{'─'*105}")
        logger.info(f"  PORTFOLIO ANALYSIS")
        logger.info(f"{'─'*105}")
        print_all_curves_table(_trade_logs_baseline, "Baseline", INITIAL_BALANCE)

        if _trade_logs_regime01:
            print_all_curves_table(_trade_logs_regime01, "Regime 0+1", INITIAL_BALANCE)

    # =========================================================================
    # ALL CURVES COMBINED — Validated only
    # =========================================================================
    validated_ids = {v["strategy_id"] for v in _validation_results if v["verdict"] == "🟢 VALIDATED"}

    validated_baseline = [(sid, df) for sid, df in _trade_logs_baseline if sid in validated_ids]
    validated_regime01 = [(sid, df) for sid, df in _trade_logs_regime01 if sid in validated_ids]

    if validated_baseline:
        logger.info(f"\n{'─'*105}")
        logger.info(f"  PORTFOLIO ANALYSIS — VALIDATED ONLY")
        logger.info(f"{'─'*105}")
        print_all_curves_table(validated_baseline, "Baseline — Validated only", INITIAL_BALANCE)
    if validated_regime01:
        print_all_curves_table(validated_regime01, "Regime 0+1 — Validated only", INITIAL_BALANCE)

    # =========================================================================
    # BEST COMBINATIONS
    # =========================================================================
    if RUN_BEST_COMBINATIONS:
        logger.info(f"\n{'─'*105}")
        logger.info(f"  BEST COMBINATIONS")
        logger.info(f"{'─'*105}")
        if validated_baseline:
            print_best_combinations(validated_baseline, "Baseline — Validated only", INITIAL_BALANCE)
        if validated_regime01:
            print_best_combinations(validated_regime01, "Regime 0+1 — Validated only", INITIAL_BALANCE, precomputed_metrics=r01_metrics)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    from strategies_config import STRATEGIES

    start  = time.time()
    logger = logging.getLogger("BOT_batch.main_batch")

    if UPDATE_CSV:
        validate_csv_columns(CSV_PARAMS)

    logger.info(f"\n{'='*105}")
    logger.info(f"  BATCH START")
    logger.info(f"{'='*105}")
    logger.info(f"  CSV update       : {'✅ enabled' if UPDATE_CSV else '⚪ disabled'}")
    logger.info(f"  Data IS          : {DATA_FOLDER_IS}")
    logger.info(f"  Data OOS         : {DATA_FOLDER_OOS}")
    logger.info(f"  Round 1          : NetGain>{R1_NETGAIN_ROUND1}%  R2>{R1_RSQUARED_ROUND1}  ProbNeg<{R1_PROBNEG_ROUND1}%")
    logger.info(f"  Round 2 (A)      : NetGain>{R2A_NETGAIN_ROUND2}%  R2>{R2A_RSQUARED_ROUND2}  ProbNeg<{R2A_PROBNEG_ROUND1}%")
    logger.info(f"  Round 2 (B)      : NetGain>{R2B_NETGAIN_ROUND1}%  ProbNeg<{R2B_PROBNEG_ROUND1}%")
    logger.info(f"  Regime 0         : MA{R0_MA_PERIOD}  long_th={R0_LONG_TH}  short_th={R0_SHORT_TH}")
    logger.info(f"{'='*105}\n")

    strategies_to_run = (
        [s for s in STRATEGIES if s["strategy_id"] in SELECTED_STRATEGIES]
        if SELECTED_STRATEGIES else STRATEGIES
    )
    for strategy in strategies_to_run:
        logger.info(f"\n{'='*105}")
        logger.info(f"  Running: {strategy['strategy_id']}")
        logger.info(f"{'='*105}")
        run_batch(strategy)

    run_portfolio_analysis()

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")