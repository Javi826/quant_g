#BOT_batch/main_batch_E1.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "market_regime")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch_regime")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_trading_batch_regime")))
import time
import logging
import matplotlib
import numpy as np
from itertools import product
from importlib import import_module
from shared_batchs.pipeline.wfo import run_wfo_is, run_wfo_mc_is, ANCHORED, METRIC_MODE, PARAM_SELECTION_MODE
# LOGGING CONFIGURATION
#------------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger("BOT_batch.main_batch")
logging.getLogger("BOT_batch.runs.run_best_portfolio").setLevel(logging.INFO)

# COMPUTE CONFIGURATION
#------------------------------------------------------------------------------
DTYPE         = np.float32
N_JOBS        = -1
SHOW_PROGRESS = False
SHOW_PLOTS    = True

if not SHOW_PLOTS:
    matplotlib.use("Agg")
from shared_batchs.pipeline.universe import filter_symbols, select_universe
from shared_batchs.backtesters.ZX_compute_BT import MIN_PRICE, INITIAL_BALANCE
from shared_batchs.pipeline.oos_period import run_oos_period
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.reporting import print_portfolio_metrics_table, print_wfo_summary, print_all_curves_table
from shared_batchs.utils.plotting import plot_portfolio_comparison, plot_filter_comparison
from shared_batchs.regime import regime_module
from shared_batchs.regime.regime_module import load_config_from_bins
from shared_batchs.runs.run_correlation import decorrelate_by_profit
from shared_batchs.runs.run_best_wfo_portfolio import find_best_portfolio_combination_wfo
from shared_batch_regime.regime_core import REGIME_TIMEFRAME
from shared_batchs.runs.run_deploy import run_deploy_train

regime_module._indicator_cache = {}

# Global accumulators
_strategy_trades_wfo_train     : list = []
_strategy_trades_wfo_test      : list = []
_strategy_trades_oos_baseline  : list = []
_strategy_trades_oos_regime    : list = []
_validation_results            : list = []
_wfo_results                   : list = []
_drift_results                 : list = []
_best_params_results           : dict = {}
_deploy_map                    : dict = {}

# =============================================================================
# RUN CONFIGURATION
# =============================================================================

# BATCH
#------------------------------------------------------------------------------
STRATEGIES_SET_NAME  = "E1"
STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_01"
SELECTION_MODE       = "WFO"   # "WFO" or "WFO_MC"

WFO_NET_GAIN_TH = 10
WFO_DD_TH       = 50

# RUNS
#------------------------------------------------------------------------------
RUN_CORRELATION        = False
RUN_BEST_WFO_PORTFOLIO = True
RUN_DEPLOY             = False
RUN_OOS                = False

# REGIME
#------------------------------------------------------------------------------
REGIME_ENABLED    = False

# OUTPUTS
#------------------------------------------------------------------------------
SAVE_TRADES       = False
DEBUG_WFO_WINDOW  = None  # int to debug a specific WFO test window, None to disable

# STRATEGY SELECTION
#------------------------------------------------------------------------------
SELECTED_STRATEGIES = [
    # 15m
# =============================================================================
#     "01_reversal_long_15m",
#     "02_reversal_short_15m",
#     "11_parity_long_15m",
#     "12_parity_short_15m",
#     "21_flag_long_15m",
#     "22_flag_short_15m",
#     "31_orderblocks_long_15m",
#     "32_orderblocks_short_15m",
#     # 30m
#     "03_reversal_long_30m",
#     "04_reversal_short_30m",
#     "13_parity_long_30m",
#     "14_parity_short_30m",
#     "23_flag_long_30m",
#     "24_flag_short_30m",
#     "33_orderblocks_long_30m",
#     "34_orderblocks_short_30m",
# =============================================================================
    # 1H
    "05_reversal_long_1H",
    "06_reversal_short_1H",
    "15_parity_long_1H",
    "16_parity_short_1H",
    "25_flag_long_1H",
    "26_flag_short_1H",
    "35_orderblocks_long_1H",
    "36_orderblocks_short_1H",
    # 4H
    "07_reversal_long_4H",
    "08_reversal_short_4H",
    "17_parity_long_4H",
    "18_parity_short_4H",
    "27_flag_long_4H",
    "28_flag_short_4H",
    "37_orderblocks_long_4H",
    "38_orderblocks_short_4H",
    # 6H UTC
    "09_reversal_long_6Hutc",
    "10_reversal_short_6Hutc",
    "19_parity_long_6Hutc",
    "20_parity_short_6Hutc",
    "29_flag_long_6Hutc",
    "30_flag_short_6Hutc",
    "39_orderblocks_long_6Hutc",
    "40_orderblocks_short_6Hutc",
]

# =============================================================================
# PIPELINES
# =============================================================================

# Correlation analysis
#------------------------------------------------------------------------------
CORRELATION_DD_THRESHOLD = 0.75

# FILES
#------------------------------------------------------------------------------
STRATEGIES_BATCH           = import_module(f"strategies_files.strategies_BT_{STRATEGIES_SET_NAME}_batch").STRATEGIES
STRATEGIES_BT_BATCH_MODULE = f"strategies_BT_{STRATEGIES_SET_NAME}_batch"
STRATEGIES_BT_BATCH_PATH   = os.path.join(os.path.dirname(__file__), "strategies_files", f"strategies_BT_{STRATEGIES_SET_NAME}_batch.py")
STRATEGIES_LOOP            = import_module(f"strategies_files.{STRATEGIES_LOOP_NAME}").STRATEGIES_LOOP
STRATEGIES_PARAMS_FOLDER   = os.path.join(os.path.dirname(__file__), f"strategies_{STRATEGIES_SET_NAME}")
CSV_PARAMS                 = os.path.join(STRATEGIES_PARAMS_FOLDER, f"strategies_{STRATEGIES_SET_NAME}.csv")
STRATEGIES_PR_BATCH_PATH   = os.path.join(STRATEGIES_PARAMS_FOLDER, f"strategies_{STRATEGIES_SET_NAME}_batch.py")
SYMBOLS_LIVE_FOLDER        = os.path.join(STRATEGIES_PARAMS_FOLDER, "symbols_live")
DRIFT_BATCH_PATH           = os.path.join(STRATEGIES_PARAMS_FOLDER, f"drift_reference_{STRATEGIES_SET_NAME}_batch.py")
REGIME_BINS_PATH           = os.path.join(os.path.dirname(__file__), "strategies_files", f"regime_bins_{STRATEGIES_SET_NAME}.py")

# DATA
#------------------------------------------------------------------------------
from shared_batch_regime.config_paths import DATA_FOLDER_IS, DATA_FOLDER_OOS1

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_batch(strategy_config: dict) -> None:

    start_time = time.time()

    STRATEGY_ID       = strategy_config["id"]
    SIDE              = strategy_config["direction"]
    TIMEFRAME         = strategy_config["timeframe"]
    N_SYMBOLS         = strategy_config["N_SYMBOLS"]
    ORDER_AMOUNT      = strategy_config["ORDER_AMOUNT"]
    ORDER_AMOUNT_PROD = strategy_config.get("order_amount_prod", strategy_config["ORDER_AMOUNT"])
    param_grid        = strategy_config["param_grid"]

    signal_key         = "_".join(strategy_config["name"].split("_")[:-1])
    registry           = SIGNAL_REGISTRY[signal_key]
    signal_fn          = registry["fn"]
    signal_params_keys = registry["params"]
    param_names        = list(param_grid.keys())
    lists_for_grid     = [param_grid[k] for k in param_names]
    param_dict_list    = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

    brief_trades_folder = os.path.join(os.path.dirname(__file__), "brief_trades")

    # -------------------------------------------------------------------------
    # BLOCK 0 — Universe Selection
    # -------------------------------------------------------------------------
    symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos1 = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        data_folder_oos   = DATA_FOLDER_OOS1,
        timeframe         = TIMEFRAME,
        n_symbols         = N_SYMBOLS,
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )
    logger.debug(f"IS full pool: {len(ohlcv_is)} symbols: {sorted(ohlcv_is.keys())}")

    bins_to_filter = regime_module.load_regime_bins(REGIME_BINS_PATH, STRATEGY_ID) if REGIME_ENABLED else "neutral"
    if REGIME_ENABLED:
        logger.info(f"STAGE 1 ── Regime bins    ── bins: {[bins_to_filter] if isinstance(bins_to_filter, str) else bins_to_filter}")

    # -------------------------------------------------------------------------
    # BLOCK 1 — Parameter Selection + Train/Test Trades (WFO or WFO_MC)
    # -------------------------------------------------------------------------
    if SELECTION_MODE == "WFO":
        best_params, approved_wfo, wfo_net_gain, wfo_max_dd, wfo_train_trades, wfo_test_trades, wfo_df_results = run_wfo_is(
            ohlcv_data          = ohlcv_is,
            param_names         = param_names,
            lists_for_grid      = lists_for_grid,
            signal_fn           = signal_fn,
            signal_params_keys  = signal_params_keys,
            order_amount        = ORDER_AMOUNT,
            timeframe           = TIMEFRAME,
            net_gain_th         = WFO_NET_GAIN_TH,
            dd_th               = WFO_DD_TH,
            dtype               = DTYPE,
            n_jobs              = N_JOBS,
            show_progress       = SHOW_PROGRESS,
            n_symbols           = N_SYMBOLS,
            bins_to_filter      = bins_to_filter,
            regime_enabled      = REGIME_ENABLED,
        )

    elif SELECTION_MODE == "WFO_MC":
        best_params, approved_wfo, wfo_net_gain, wfo_max_dd, wfo_train_trades, wfo_test_trades, wfo_df_results = run_wfo_mc_is(
            ohlcv_data          = ohlcv_is,
            param_names         = param_names,
            lists_for_grid      = lists_for_grid,
            signal_fn           = signal_fn,
            signal_params_keys  = signal_params_keys,
            order_amount        = ORDER_AMOUNT,
            timeframe           = TIMEFRAME,
            net_gain_th         = WFO_NET_GAIN_TH,
            dd_th               = WFO_DD_TH,
            dtype               = DTYPE,
            n_jobs              = N_JOBS,
            show_progress       = SHOW_PROGRESS,
            n_symbols           = N_SYMBOLS,
            bins_to_filter      = bins_to_filter,
            regime_enabled      = REGIME_ENABLED,
        )

    else:
        raise ValueError(f"Unknown SELECTION_MODE: {SELECTION_MODE}")

    _wfo_results.append({
        "strategy_id": STRATEGY_ID,
        "verdict":     "🟢 PASS" if approved_wfo else "🔴 FAIL",
        "net_gain":    wfo_net_gain,
        "max_dd":      wfo_max_dd,
    })

    if wfo_train_trades is not None and not wfo_train_trades.empty:
        _strategy_trades_wfo_train.append((STRATEGY_ID, wfo_train_trades))
        if SAVE_TRADES:
            os.makedirs(brief_trades_folder, exist_ok=True)
            wfo_train_trades.to_csv(os.path.join(brief_trades_folder, f"trades_wfo_train_{STRATEGY_ID}.csv"), index=False)

    if wfo_test_trades is not None and not wfo_test_trades.empty:
        _strategy_trades_wfo_test.append((STRATEGY_ID, wfo_test_trades))
        if SAVE_TRADES:
            os.makedirs(brief_trades_folder, exist_ok=True)
            wfo_test_trades.to_csv(os.path.join(brief_trades_folder, f"trades_wfo_test_{STRATEGY_ID}.csv"), index=False)
        if SHOW_PLOTS:
            plot_filter_comparison(
                strategy_id        = f"{STRATEGY_ID}_wfo_test{'_regime' if REGIME_ENABLED else ''}",
                trades_df_baseline = wfo_test_trades,
                trades_df_r01      = None,
                data_folder        = DATA_FOLDER_IS,
                initial_balance    = INITIAL_BALANCE,
                regime_enabled     = REGIME_ENABLED,
            )

    # STAGE 1 metrics + log — moved here so it prints before STAGE 2 (OOS)
    _wfo_metrics = None
    if wfo_test_trades is not None and not wfo_test_trades.empty:
        _wfo_metrics = compute_metrics(wfo_test_trades, capital=INITIAL_BALANCE, name="")
        _verdict     = "🟢 PASS" if approved_wfo else "🔴 FAIL"
        _params_str  = " | ".join(f"{k}={v}" for k, v in best_params.items() if k not in ("SELL_AFTER",))
        logger.info(
            f"STAGE 1 ── WFO results    ── {_verdict} NetGain={_wfo_metrics['Net_Gain_pct']:.1f}% DD={_wfo_metrics['Max_DD_pct']:.1f}% "
            f"WinRate%={_wfo_metrics['Win_Rate']:.1f}% R2={_wfo_metrics['R_Squared']:.3f} PF={_wfo_metrics['Profit_Factor']:.2f} "
            f"Calmar={_wfo_metrics['Calmar']:.2f} Trades={len(wfo_test_trades)}"
        )
        logger.info(f"STAGE 1 ── WFO params     ── {_params_str}")

    bt_signal_params = {k: best_params[k.upper()] for k in signal_params_keys if k.upper() in best_params}
    
    from shared_batchs.pipeline.wfo_validation import validate_wfo_window
    # -------------------------------------------------------------------------
    # BLOCK 1b — Validate WFO test window alignment
    # -------------------------------------------------------------------------
    if DEBUG_WFO_WINDOW is not None and wfo_df_results is not None:
        validate_wfo_window(
            window_idx         = DEBUG_WFO_WINDOW,
            df_results         = wfo_df_results,
            wfo_test_trades    = wfo_test_trades,
            ohlcv_is           = ohlcv_is,
            signal_fn          = signal_fn,
            signal_params_keys = signal_params_keys,
            best_params        = best_params,
            param_names        = param_names,
            order_amount       = ORDER_AMOUNT,
            timeframe          = TIMEFRAME,
        )
    # -------------------------------------------------------------------------
    # BLOCK 2 — OOS (baseline + regime) — informational only
    # -------------------------------------------------------------------------

    if RUN_OOS and DEBUG_WFO_WINDOW is None:
            ohlcv_data_oos = {sym: ohlcv_oos1[sym] for sym in symbols_oos_final}
            run_oos_period(
                strategy_id           = STRATEGY_ID,
                label                 = "OOS",
                ohlcv_data            = ohlcv_data_oos,
                signal_fn             = signal_fn,
                signal_params         = bt_signal_params,
                best_params           = best_params,
                param_names           = param_names,
                order_amount          = ORDER_AMOUNT,
                timeframe             = TIMEFRAME,
                data_folder           = DATA_FOLDER_OOS1,
                bins_to_filter        = bins_to_filter,
                trades_baseline_accum = _strategy_trades_oos_baseline,
                trades_regime_accum   = _strategy_trades_oos_regime,
                save_trades           = SAVE_TRADES,
                brief_trades_folder   = brief_trades_folder,
            )

    # -------------------------------------------------------------------------
    # BLOCK 3 — Build validation record
    # -------------------------------------------------------------------------
    _val_record = {
        "strategy_id":     STRATEGY_ID,
        "verdict":         "🟢 VALIDATED" if approved_wfo else "🔴 REJECTED",
        "round":           "—",
        "net_gain_pct":    0.0,
        "dd_pct":          0.0,
        "win_ratio":       0.0,
        "r2":              0.0,
        "prob_neg_pct":    0.0,
        "symbols_changed": False,
        "bins_to_filter":  bins_to_filter,
    }

    if _wfo_metrics is not None:
        _val_record.update({
            "net_gain_pct": round(_wfo_metrics["Net_Gain_pct"], 1),
            "dd_pct":       round(_wfo_metrics["Max_DD_pct"],   1),
            "win_ratio":    round(_wfo_metrics["Win_Rate"],      1),
            "r2":           _wfo_metrics["R_Squared"],
            "tn_trades":    len(wfo_test_trades),
        })

    _validation_results.append(_val_record)

    p_target_winrate = round((wfo_test_trades["profit"] > 0).mean() * 100, 1) if wfo_test_trades is not None and not wfo_test_trades.empty else 0.0

    _drift_results.append({
        "strategy_id":      STRATEGY_ID,
        "p_target_winrate": p_target_winrate,
    })

    _best_params_results[STRATEGY_ID] = best_params

    # -------------------------------------------------------------------------
    # BLOCK 5 — Deploy train (last train_months window, today as anchor)
    # -------------------------------------------------------------------------
    symbols_changed = False
    if RUN_DEPLOY:
        
        symbols_changed = run_deploy_train(
            strategy_id           = STRATEGY_ID,
            ohlcv_is              = ohlcv_is,
            param_names           = param_names,
            lists_for_grid        = lists_for_grid,
            signal_fn             = signal_fn,
            signal_params_keys    = signal_params_keys,
            order_amount          = ORDER_AMOUNT,
            timeframe             = TIMEFRAME,
            n_symbols             = N_SYMBOLS,
            approved_wfo          = approved_wfo,
            dtype                 = DTYPE,
            n_jobs                = N_JOBS,
            symbols_live_folder   = SYMBOLS_LIVE_FOLDER,
            output_path           = STRATEGIES_PR_BATCH_PATH,
            strategies_batch_path = STRATEGIES_BT_BATCH_PATH,
            module_name           = STRATEGIES_BT_BATCH_MODULE,
            regime_bins_path      = REGIME_BINS_PATH,
            deploy_map            = _deploy_map,
            strategy_ids_to_run   = [s["id"] for s in strategies_to_run],
            regime_enabled        = REGIME_ENABLED,
        )
        _icon = "🔵" if symbols_changed else "⚪"
        logger.debug(f"STAGE 4  ── Update & Compare       ── {_icon} {'symbols' if symbols_changed else 'no changes'}")

    if _validation_results:
        _validation_results[-1].update({
            "symbols_changed": symbols_changed,
        })

    elapsed = int(time.time() - start_time)
    logger.info(f"DONE  🏁 ── {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")


# =============================================================================
# RUN SUMMARY
# =============================================================================

def run_summary():
    print_wfo_summary(_wfo_results, _validation_results)

    validated_ids        = {w["strategy_id"] for w in _wfo_results if "PASS" in w["verdict"]} if _wfo_results else {v["strategy_id"] for v in _validation_results if "VALIDATED" in v["verdict"]}
    validated_baseline   = [(sid, df) for sid, df in _strategy_trades_oos_baseline if sid in validated_ids]
    validated_oos_regime = [(sid, df) for sid, df in _strategy_trades_oos_regime   if sid in validated_ids]

    # -------------------------------------------------------------------------
    # OOS ANALYSIS
    # -------------------------------------------------------------------------
    if RUN_OOS:
        _regime_label = "REGIME" if REGIME_ENABLED else "BASELINE"

        for label, strategy_trades in [("OOS — BASELINE (best WFO params)", _strategy_trades_oos_baseline), (f"OOS — {_regime_label} (best WFO params)", _strategy_trades_oos_regime)]:
            if not strategy_trades:
                continue
            print_portfolio_metrics_table(strategy_trades, label, INITIAL_BALANCE)
        if _strategy_trades_oos_baseline:
            logger.info(f"\n{'='*115}\n  PORTFOLIO ANALYSIS\n{'='*115}")
            if logger.isEnabledFor(logging.DEBUG):
                print_all_curves_table(_strategy_trades_oos_baseline, "OOS — BASELINE (best WFO params)", INITIAL_BALANCE)
            if _strategy_trades_oos_regime:
                print_all_curves_table(_strategy_trades_oos_regime, f"OOS — {_regime_label} (best WFO params)", INITIAL_BALANCE)
        if validated_baseline:
            if logger.isEnabledFor(logging.DEBUG):
                print_all_curves_table(validated_baseline, "OOS — BASELINE (best WFO params) — Validated only", INITIAL_BALANCE)
        if validated_oos_regime:
            print_all_curves_table(validated_oos_regime, f"OOS — {_regime_label} (best WFO params) — Validated only", INITIAL_BALANCE)
        plot_portfolio_comparison(
            strategy_trades_baseline = validated_baseline,
            strategy_trades_regime01 = validated_oos_regime,
            data_folder              = DATA_FOLDER_OOS1,
            initial_balance          = INITIAL_BALANCE,
            title                    = "Portfolio OOS — Validated only",
        )

    # -------------------------------------------------------------------------
    # WFO TRADES
    # -------------------------------------------------------------------------
    if _strategy_trades_wfo_train and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\n{'='*115}\n  WFO TRAIN TRADES\n{'='*115}")
        print_all_curves_table(_strategy_trades_wfo_train, "WFO Train", INITIAL_BALANCE)

    if _strategy_trades_wfo_test:
        logger.info(f"\n{'='*115}\n  WFO TEST TRADES\n{'='*115}")
        _all_validated_wfo_test = [(sid, df) for sid, df in _strategy_trades_wfo_test if sid in validated_ids]
        print_all_curves_table(_strategy_trades_wfo_test, "IS — WFO Test windows concatenated (per-window best params) — All", INITIAL_BALANCE)
        if _all_validated_wfo_test:
            print_all_curves_table(_all_validated_wfo_test, "IS — WFO Test windows concatenated (per-window best params) — Validated only", INITIAL_BALANCE)

    # -------------------------------------------------------------------------
    # CORRELATION + BEST WFO PORTFOLIO
    # -------------------------------------------------------------------------
    validated_wfo_test = [(sid, df) for sid, df in _strategy_trades_wfo_test if sid in validated_ids]

    if RUN_CORRELATION:
        logger.info(f"\n{'─'*115}\n  CORRELATION ANALYSIS WFO Test — Profit (threshold={CORRELATION_DD_THRESHOLD})\n{'─'*115}")
        validated_wfo_test = decorrelate_by_profit(
            strategy_trades_wfo_test = validated_wfo_test,
            initial_balance          = INITIAL_BALANCE,
            threshold                = CORRELATION_DD_THRESHOLD,
        )
        
    if SHOW_PLOTS and validated_wfo_test:
        plot_portfolio_comparison(
            strategy_trades_baseline = validated_wfo_test,
            strategy_trades_regime01 = None,
            data_folder              = DATA_FOLDER_IS,
            initial_balance          = INITIAL_BALANCE,
            title                    = "Portfolio WFO Test — Validated only",
        )

    if RUN_BEST_WFO_PORTFOLIO:
        find_best_portfolio_combination_wfo(
            validated_wfo_trades = validated_wfo_test,
            initial_balance      = INITIAL_BALANCE,
            show_plots           = SHOW_PLOTS,
        )

# =============================================================================
# MAIN
# =============================================================================
from pathlib import Path

def _short_path(full_path: str, from_part: str = "expanding") -> str:
    parts = Path(full_path).parts
    idx   = next((i for i, p in enumerate(parts) if p == from_part), None)
    return str(Path(*parts[idx:])) if idx is not None else full_path

if __name__ == "__main__":
    logger = logging.getLogger("BOT_batch.main_batch")
    start  = time.time()

    _loop_map  = {s["id"]: s for s in STRATEGIES_LOOP}
    STRATEGIES = []
    for s in STRATEGIES_BATCH:
        loop = _loop_map.get(s["id"], {})
        if not loop:
            logger.warning(f"⚠️  {s['id']} not found in strategies_loop — skipping.")
            continue
        STRATEGIES.append({**s, **loop})

    strategies_to_run = (
        [s for s in STRATEGIES if s["id"] in SELECTED_STRATEGIES]
        if SELECTED_STRATEGIES else STRATEGIES
    )

    logger.info(f"\n{'='*115}")
    logger.info(f"  BATCH START")
    logger.info(f"{'='*115}")
    logger.info(f"  Strategies set : {STRATEGIES_SET_NAME}-{len(strategies_to_run)} strategies")
    logger.info(f"  Loop config    : {STRATEGIES_LOOP_NAME}")
    logger.info(f"  Selection mode : {SELECTION_MODE}  |  Anchored={'🟢' if ANCHORED else '⚪'}  Metric={METRIC_MODE}  Selection={PARAM_SELECTION_MODE}")
    regime_module.REGIME_ENABLED = REGIME_ENABLED
    load_config_from_bins(REGIME_BINS_PATH)
    logger.info(f"  Regime         : {'🟢 enabled' if REGIME_ENABLED else '⚪ disabled'}  CFG={regime_module.INDICATOR_CFG}  TF={REGIME_TIMEFRAME}")
    logger.info(f"  Runs           : {'🟢' if RUN_CORRELATION else '⚪'} Correlation  {'🟢' if RUN_BEST_WFO_PORTFOLIO else '⚪'} BestPortfolio  {'🟢' if RUN_DEPLOY else '⚪'} Deploy  {'🟢' if RUN_OOS else '⚪'} OOS")
    logger.info(f"  Data IS        : 🔵 {_short_path(DATA_FOLDER_IS)}")
    logger.info(f"  Data OOS       : 🔵 {_short_path(DATA_FOLDER_OOS1)}")
    logger.info(f"{'='*115}\n")

    for strategy in strategies_to_run:
        logger.info(f"\n{'─'*115}\n  Running: {strategy['id']}\n{'─'*115}")
        run_batch(strategy)

    run_summary()

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")