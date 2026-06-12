#BOT_batch/main_batch_00.py
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
import shared_batchs.pipeline.universe as universe_cfg
from shared_batchs.backtesters.ZX_compute_BT import MIN_PRICE, INITIAL_BALANCE
from shared_batchs.pipeline.is_period import run_backtest_is
from shared_batchs.pipeline.oos_period import run_oos_period
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.pipeline.montecarlo import run_montecarlo_is, run_montecarlo_oos
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.reporting import print_portfolio_metrics_table, print_strategies_summary, print_all_curves_table, print_robustness_table
from shared_batchs.utils.plotting import plot_portfolio_comparison
from shared_batchs.utils.io import save_drift_reference, save_strategies_pr, compare_and_generate_csv, update_strategies_symbols, print_update_status
from shared_batchs.regime import regime_module
from shared_batchs.regime.regime_module import load_config_from_bins
from shared_batchs.runs.run_correlation import decorrelate_by_profit
from shared_batchs.runs.run_best_portfolio import find_best_portfolio_combination
from shared_batch_regime.regime_core import REGIME_TIMEFRAME

regime_module._indicator_cache = {}
# Global accumulators
_strategy_trades_is_baseline   : list = []
_strategy_trades_is_regime     : list = []
_strategy_trades_oos1_baseline : list = []
_strategy_trades_oos1_regime   : list = []
_strategy_trades_oos2_baseline : list = []
_strategy_trades_oos2_regime   : list = []
_strategy_trades_oos3_baseline : list = []
_strategy_trades_oos3_regime   : list = []
_validation_results            : list = []
_drift_results                 : list = []
_best_params_results           : dict = {}

# =============================================================================
# RUN CONFIGURATION
# =============================================================================

# BATCH 
#------------------------------------------------------------------------------
STRATEGIES_SET_NAME  = "00"  
STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_01"
N_PATHS_IS           = 1

# ELITE -- MA3
#------------------------------------------------------------------------------

OOS_NETGAIN_TH       = 34
OOS_MAX_DD_TH        = 8
OOS_R2_TH            = 0.82

#PRODUCTION
#OOS_NETGAIN_TH       = 1
#OOS_MAX_DD_TH        = 80
#OOS_R2_TH            = 0.082
# RUNS
#------------------------------------------------------------------------------
RUN_SUMMARY        = True
RUN_CORRELATION    = True
RUN_BEST_PORTFOLIO = True

# REGIME
#------------------------------------------------------------------------------
REGIME_ENABLED    = True

# OUTPUTS
#------------------------------------------------------------------------------
UPDATE_OUTPUTS  = False
SAVE_TRADES     = False


# STRATEGY SELECTION
#------------------------------------------------------------------------------
SELECTED_STRATEGIES = [
    "01_reversal_long_15m",
    "02_reversal_short_15m",
    "03_reversal_long_30m",
    "04_reversal_short_30m",
    "05_reversal_long_1H",
    "06_reversal_short_1H",
    "07_reversal_long_4H",
    "08_reversal_short_4H",
    "09_reversal_long_6Hutc",
    "10_reversal_short_6Hutc",
    "11_parity_long_15m",
    "12_parity_short_15m",
    "13_parity_long_30m",
    "14_parity_short_30m",
    "15_parity_long_1H",
    "16_parity_short_1H",
    "17_parity_long_4H",
    "18_parity_short_4H",
    "19_parity_long_6Hutc",
    "20_parity_short_6Hutc",
    "21_flag_long_15m",
    "22_flag_short_15m",
    "23_flag_long_30m",
    "24_flag_short_30m",
    "25_flag_long_1H",
    "26_flag_short_1H",
    "27_flag_long_4H",
    "28_flag_short_4H",
    "29_flag_long_6Hutc",
    "30_flag_short_6Hutc",
    "31_orderblocks_long_15m",
    "32_orderblocks_short_15m",
    "33_orderblocks_long_30m",
    "34_orderblocks_short_30m",
    "35_orderblocks_long_1H",
    "36_orderblocks_short_1H",
    "37_orderblocks_long_4H",
    "38_orderblocks_short_4H",
    "39_orderblocks_long_6Hutc",
    "40_orderblocks_short_6Hutc",
]

SELECTED_STRATEGIES = [
    "05_reversal_long_1H",
    "20_parity_short_6Hutc",
    "22_flag_short_15m",
    "31_orderblocks_long_15m",
    "34_orderblocks_short_30m",
]

# =============================================================================
# MONTECARLOS IS + OOS
# =============================================================================

MC_SELECTION_PERCENTILE = None  
RUN_MC_OOS              = False
N_PATHS_OOS1            = 500

# =============================================================================
# PIPELINES
# =============================================================================

# OOS2
#------------------------------------------------------------------------------
R_NETGAIN_OOS2      = OOS_NETGAIN_TH
R_MAX_DD_OOS2       = OOS_MAX_DD_TH
R_R2_OOS2           = OOS_R2_TH
OOS2_RUN_ANALYSIS   = True
OOS2_FOR_VALIDATION = True

# OOS3
#------------------------------------------------------------------------------
R_NETGAIN_OOS3      = OOS_NETGAIN_TH
R_MAX_DD_OOS3       = OOS_MAX_DD_TH
R_R2_OOS3           = OOS_R2_TH
OOS3_RUN_ANALYSIS   = True
OOS3_FOR_VALIDATION = True

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
from shared_batch_regime.config_paths import DATA_FOLDER_IS, DATA_FOLDER_OOS1, DATA_FOLDER_OOS2, DATA_FOLDER_OOS3

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_batch(strategy_config: dict) -> None:

    start_time = time.time()

    STRATEGY_ID       = strategy_config["id"]
    SIDE              = strategy_config["direction"]
    TIMEFRAME         = strategy_config["timeframe"]
    N_SYMBOLS         = strategy_config["n_symbols"]
    ORDER_AMOUNT      = strategy_config["order_amount"]
    ORDER_AMOUNT_PROD = strategy_config.get("order_amount_prod", strategy_config["order_amount"])
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

    # -------------------------------------------------------------------------
    # BLOCK 1 — Monte Carlo IS
    # -------------------------------------------------------------------------
    ohlcv_data_minor = {sym: ohlcv_is[sym] for sym in symbols_is_final}
    best_params, _   = run_montecarlo_is(
        ohlcv_data           = ohlcv_data_minor,
        param_dict_list      = param_dict_list,
        param_names          = param_names,
        lists_for_grid       = lists_for_grid,
        signal_fn            = signal_fn,
        signal_params_keys   = signal_params_keys,
        order_amount         = ORDER_AMOUNT,
        n_paths              = N_PATHS_IS,
        timeframe            = TIMEFRAME,
        dtype                = DTYPE,
        n_jobs               = N_JOBS,
        show_progress        = SHOW_PROGRESS,
        selection_percentile = MC_SELECTION_PERCENTILE,
    )
    bt_signal_params = {k: best_params[k.upper()] for k in signal_params_keys if k.upper() in best_params}

    # -------------------------------------------------------------------------
    # BLOCK 2 — Backtest IS + Regime Analysis
    # -------------------------------------------------------------------------
    bins_to_filter, _ = run_backtest_is(
        strategy_id           = STRATEGY_ID,
        ohlcv_is              = ohlcv_is,
        symbols_oos_final     = symbols_oos_final,
        signal_fn             = signal_fn,
        signal_params         = bt_signal_params,
        best_params           = best_params,
        order_amount          = ORDER_AMOUNT,
        timeframe             = TIMEFRAME,
        data_folder_is        = DATA_FOLDER_IS,
        strategy_direction    = SIDE,
        save_trades           = SAVE_TRADES,
        trades_is_baseline    = _strategy_trades_is_baseline,
        trades_is_regime      = _strategy_trades_is_regime,
        brief_trades_folder   = brief_trades_folder,
        regime_bins_path          = REGIME_BINS_PATH,
        regime_enabled            = REGIME_ENABLED
    )

    # -------------------------------------------------------------------------
    # BLOCK 3 — Monte Carlo OOS1 (informational)
    # -------------------------------------------------------------------------
    ohlcv_data_oos1    = {sym: ohlcv_oos1[sym] for sym in symbols_oos_final}
    p_target_winrate_oos1 = 0.0
    prob_negative_oos1    = 0.0

    if RUN_MC_OOS:
        df_portfolio_oos1, _, _ = run_montecarlo_oos(
            ohlcv_data         = ohlcv_data_oos1,
            best_params        = best_params,
            param_names        = param_names,
            signal_fn          = signal_fn,
            signal_params_keys = signal_params_keys,
            order_amount       = ORDER_AMOUNT,
            n_paths            = N_PATHS_OOS1,
            timeframe          = TIMEFRAME,
            dtype              = DTYPE,
            n_jobs             = N_JOBS,
            show_progress      = SHOW_PROGRESS,
        )
        path_grouped               = df_portfolio_oos1.groupby("path_index")["Portfolio_Final_Balance"].mean().reset_index()
        path_grouped["Net_Gain_pct"] = (path_grouped["Portfolio_Final_Balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        prob_negative_oos1         = (path_grouped["Net_Gain_pct"] < 0).mean() * 100
        logger.info(f"STAGE M ── MC OOS1                ── ProbNeg={prob_negative_oos1:.1f}%")

    # -------------------------------------------------------------------------
    # BLOCK 4  — OOS1 (baseline + regime + validation)
    # -------------------------------------------------------------------------
    ohlcv_data_oos1 = {sym: ohlcv_oos1[sym] for sym in symbols_oos_final}
    approved, trades_df_oos1_baseline, trades_df_oos1_regime, _metrics_baseline_oos1, _metrics_regime_oos1 = run_oos_period(
    
        strategy_id            = STRATEGY_ID,
        label                  = "OOS1",
        stage_baseline         = "STAGE 3",
        stage_regime           = "STAGE 4",
        ohlcv_data             = ohlcv_data_oos1,
        signal_fn              = signal_fn,
        signal_params          = bt_signal_params,
        best_params            = best_params,
        param_names            = param_names,
        order_amount           = ORDER_AMOUNT,
        timeframe              = TIMEFRAME,
        data_folder            = DATA_FOLDER_OOS1,
        bins_to_filter         = bins_to_filter,
        netgain_th             = OOS_NETGAIN_TH,
        max_dd_th              = OOS_MAX_DD_TH,
        r2_th                  = OOS_R2_TH,
        for_validation         = True,
        approved               = True,
        validation_record      = {},
        trades_baseline_accum  = _strategy_trades_oos1_baseline,
        trades_regime_accum    = _strategy_trades_oos1_regime,
        save_trades            = SAVE_TRADES,
        show_plots             = SHOW_PLOTS,
        brief_trades_folder    = brief_trades_folder,
        run_report_backtesting = True,
    )
    


    # -------------------------------------------------------------------------
    # BLOCK 5 — Build validation record
    # -------------------------------------------------------------------------
    _val_record = {
        "strategy_id":     STRATEGY_ID,
        "verdict":         "🟢 VALIDATED" if approved else "🔴 REJECTED",
        "round":           "—",
        "net_gain_pct":    round(trades_df_oos1_regime["profit"].sum() / INITIAL_BALANCE * 100, 1) if len(trades_df_oos1_regime) > 0 else round(trades_df_oos1_baseline["profit"].sum() / INITIAL_BALANCE * 100, 1),
        "dd_pct":          0.0,
        "win_ratio":       0.0,
        "r2":              0.0,
        "prob_neg_pct":    round(prob_negative_oos1, 2),
        "symbols_changed": False,
        "bins_to_filter":  bins_to_filter,
    }
    
    _m = compute_metrics(trades_df_oos1_regime if len(trades_df_oos1_regime) > 0 else trades_df_oos1_baseline, capital=INITIAL_BALANCE, name="")
    _val_record.update({
        "net_gain_pct": round(_m["Net_Gain_pct"], 1),
        "dd_pct":       round(_m["Max_DD_pct"], 1),
        "win_ratio":    round(_m["Win_Rate"], 1),
        "r2":           _m["R_Squared"],
    })
    _validation_results.append(_val_record)


    # -------------------------------------------------------------------------
    # BLOCK 6 — OOS2
    # -------------------------------------------------------------------------
    if OOS2_RUN_ANALYSIS:
        if universe_cfg.OOS23_MATCH_SYMBOLS:
            logging.disable(logging.INFO)
            _, _oos2_syms, _, ohlcv_oos2_all = select_universe(
                data_folder_is=DATA_FOLDER_IS, data_folder_oos=DATA_FOLDER_OOS2,
                timeframe=TIMEFRAME, n_symbols=N_SYMBOLS, min_price=MIN_PRICE,
                filter_symbols_fn=filter_symbols,
            )
            logging.disable(logging.NOTSET)
            ohlcv_oos2_data = {sym: ohlcv_oos2_all[sym] for sym in _oos2_syms if sym in ohlcv_oos2_all}
        else:
            ohlcv_oos2_raw, _ = filter_symbols(
                symbols_oos_final, min_vol_usdt=0, timeframe=TIMEFRAME,
                data_folder=DATA_FOLDER_OOS2, min_price=MIN_PRICE, vol_window=50,
            )
            ohlcv_oos2_data = ohlcv_oos2_raw

        approved, trades_df_oos2_baseline, trades_df_oos2_regime, _, _ = run_oos_period(
            strategy_id            = STRATEGY_ID,
            label                  = "OOS2",
            stage_baseline         = "STAGE 5",
            stage_regime           = "STAGE 5",
            ohlcv_data             = ohlcv_oos2_data,
            signal_fn              = signal_fn,
            signal_params          = bt_signal_params,
            best_params            = best_params,
            param_names            = param_names,
            order_amount           = ORDER_AMOUNT,
            timeframe              = TIMEFRAME,
            data_folder            = DATA_FOLDER_OOS2,
            bins_to_filter         = bins_to_filter,
            netgain_th             = OOS_NETGAIN_TH,
            max_dd_th              = OOS_MAX_DD_TH,
            r2_th                  = OOS_R2_TH,
            for_validation         = OOS2_FOR_VALIDATION,
            approved               = approved,
            validation_record      = _validation_results[-1],
            trades_baseline_accum  = _strategy_trades_oos2_baseline,
            trades_regime_accum    = _strategy_trades_oos2_regime,
            save_trades            = SAVE_TRADES,
            show_plots             = SHOW_PLOTS,
            brief_trades_folder    = brief_trades_folder,
            run_report_backtesting = False,
            run_baseline           = SHOW_PLOTS or SAVE_TRADES,
        )
        _df_oos2 = trades_df_oos2_regime if len(trades_df_oos2_regime) > 0 else trades_df_oos2_baseline
        if len(_df_oos2) > 0:
            _m_oos2 = compute_metrics(_df_oos2, capital=INITIAL_BALANCE, name="")
            _validation_results[-1].update({
                "net_gain_pct_oos2": round(_m_oos2["Net_Gain_pct"], 1),
                "dd_pct_oos2":       round(_m_oos2["Max_DD_pct"], 1),
                "r2_oos2":           _m_oos2["R_Squared"],
            })

    # -------------------------------------------------------------------------
    # BLOCK 7 — OOS3
    # -------------------------------------------------------------------------
    if OOS3_RUN_ANALYSIS:
        if universe_cfg.OOS23_MATCH_SYMBOLS:
            logging.disable(logging.INFO)
            _, _oos3_syms, _, ohlcv_oos3_all = select_universe(
                data_folder_is=DATA_FOLDER_IS, data_folder_oos=DATA_FOLDER_OOS3,
                timeframe=TIMEFRAME, n_symbols=N_SYMBOLS, min_price=MIN_PRICE,
                filter_symbols_fn=filter_symbols,
            )
            logging.disable(logging.NOTSET)
            ohlcv_oos3_data = {sym: ohlcv_oos3_all[sym] for sym in _oos3_syms if sym in ohlcv_oos3_all}
        else:
            ohlcv_oos3_raw, _ = filter_symbols(
                symbols_oos_final, min_vol_usdt=0, timeframe=TIMEFRAME,
                data_folder=DATA_FOLDER_OOS3, min_price=MIN_PRICE, vol_window=50,
            )
            ohlcv_oos3_data = ohlcv_oos3_raw

        approved, trades_df_oos3_baseline, trades_df_oos3_regime, _, _ = run_oos_period(
            strategy_id            = STRATEGY_ID,
            label                  = "OOS3",
            stage_baseline         = "STAGE 6",
            stage_regime           = "STAGE 6",
            ohlcv_data             = ohlcv_oos3_data,
            signal_fn              = signal_fn,
            signal_params          = bt_signal_params,
            best_params            = best_params,
            param_names            = param_names,
            order_amount           = ORDER_AMOUNT,
            timeframe              = TIMEFRAME,
            data_folder            = DATA_FOLDER_OOS3,
            bins_to_filter         = bins_to_filter,
            netgain_th             = OOS_NETGAIN_TH,
            max_dd_th              = OOS_MAX_DD_TH,
            r2_th                  = OOS_R2_TH,
            for_validation         = OOS3_FOR_VALIDATION,
            approved               = approved,
            validation_record      = _validation_results[-1],
            trades_baseline_accum  = _strategy_trades_oos3_baseline,
            trades_regime_accum    = _strategy_trades_oos3_regime,
            save_trades            = SAVE_TRADES,
            show_plots             = SHOW_PLOTS,
            brief_trades_folder    = brief_trades_folder,
            run_report_backtesting = False,
            run_baseline           = SHOW_PLOTS or SAVE_TRADES,
        )
        _df_oos3 = trades_df_oos3_regime if len(trades_df_oos3_regime) > 0 else trades_df_oos3_baseline
        if len(_df_oos3) > 0:
            _m_oos3 = compute_metrics(_df_oos3, capital=INITIAL_BALANCE, name="")
            _validation_results[-1].update({
                "net_gain_pct_oos3": round(_m_oos3["Net_Gain_pct"], 1),
                "dd_pct_oos3":       round(_m_oos3["Max_DD_pct"], 1),
                "r2_oos3":           _m_oos3["R_Squared"],
            })


    # -------------------------------------------------------------------------
    # BLOCK 8 — Update & Compare
    # -------------------------------------------------------------------------
    _symbols_result = update_strategies_symbols(
        strategy_id          = STRATEGY_ID,
        symbols_oos_final    = symbols_oos_final,
        timeframe            = TIMEFRAME,
        symbols_live_folder  = SYMBOLS_LIVE_FOLDER,
    ) if UPDATE_OUTPUTS else None

    _changes     = ["symbols"] if _symbols_result and _symbols_result.get("symbols_changed") else []
    _changes_str = " | ".join(_changes) if _changes else "no changes"
    _icon        = "🔵" if _changes else "⚪"
    logger.debug(f"STAGE 8  ── Update & Compare       ── {_icon} {_changes_str}")

    df_regime = trades_df_oos1_regime if len(trades_df_oos1_regime) > 0 else trades_df_oos1_baseline
    p_target_winrate_oos1 = round((df_regime["profit"] > 0).mean() * 100, 1) if len(df_regime) > 0 else 0.0
    
    _drift_results.append({
        "strategy_id":        STRATEGY_ID,
        "p_target_winrate":   p_target_winrate_oos1,
        })
    
    _best_params_results[STRATEGY_ID] = best_params

    if _validation_results:
        _validation_results[-1].update({
            "symbols_changed": _symbols_result.get("symbols_changed", False) if _symbols_result else False,
        })

    elapsed = int(time.time() - start_time)
    logger.info(f"DONE  🏁 ── {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")
    
# =============================================================================
# RUN SUMMARY
# =============================================================================

def run_summary():
    """Compute combined portfolio metrics. Call after all run_batch() calls."""
    if not RUN_SUMMARY:
        print_strategies_summary(_validation_results)
        print_update_status(CSV_PARAMS, SYMBOLS_LIVE_FOLDER, _validation_results)
        return

    for label, strategy_trades in [("Baseline", _strategy_trades_oos1_baseline), ("Regime 0+1", _strategy_trades_oos1_regime)]:
        if not strategy_trades:
            continue
        print_portfolio_metrics_table(strategy_trades, label, INITIAL_BALANCE)

    r01_metrics = {sid: compute_metrics(df, capital=INITIAL_BALANCE, name=sid)
                   for sid, df in _strategy_trades_oos1_regime}

    print_strategies_summary(_validation_results)

    if _strategy_trades_oos1_baseline:
        logger.info(f"\n{'='*115}\n  PORTFOLIO ANALYSIS\n{'='*115}")
        if logger.isEnabledFor(logging.DEBUG):
            print_all_curves_table(_strategy_trades_oos1_baseline, "Baseline", INITIAL_BALANCE)
        if _strategy_trades_oos1_regime:
            print_all_curves_table(_strategy_trades_oos1_regime, "Regime 0+1", INITIAL_BALANCE)

    validated_ids         = {v["strategy_id"] for v in _validation_results if "VALIDATED" in v["verdict"]}
    validated_baseline    = [(sid, df) for sid, df in _strategy_trades_oos1_baseline if sid in validated_ids]
    validated_oos1_regime = [(sid, df) for sid, df in _strategy_trades_oos1_regime if sid in validated_ids]

    if validated_baseline:
        if logger.isEnabledFor(logging.DEBUG):
            print_all_curves_table(validated_baseline, "Baseline — Validated only", INITIAL_BALANCE)
    if validated_oos1_regime:
        print_all_curves_table(validated_oos1_regime, "Regime 0+1 — Validated only", INITIAL_BALANCE)

    validated_oos2_regime = [(sid, df) for sid, df in _strategy_trades_oos2_regime if sid in validated_ids]
    validated_oos3_regime = [(sid, df) for sid, df in _strategy_trades_oos3_regime if sid in validated_ids]
    print_robustness_table(
        strategy_trades_per_period=[
            ("OOS1", validated_oos1_regime),
            ("OOS2", validated_oos2_regime),
            ("OOS3", validated_oos3_regime),
        ],
        initial_balance=INITIAL_BALANCE,
    )

    if _strategy_trades_oos1_baseline:
        plot_portfolio_comparison(
            strategy_trades_baseline=_strategy_trades_oos1_baseline,
            strategy_trades_regime01=_strategy_trades_oos1_regime,
            data_folder=DATA_FOLDER_OOS1,
            initial_balance=INITIAL_BALANCE,
            title="Portfolio — All strategies",
        )

    if validated_baseline:
        plot_portfolio_comparison(
            strategy_trades_baseline=validated_baseline,
            strategy_trades_regime01=validated_oos1_regime,
            data_folder=DATA_FOLDER_OOS1,
            initial_balance=INITIAL_BALANCE,
            title="Portfolio — Validated only",
        )

    if validated_oos2_regime:
        plot_portfolio_comparison(
            strategy_trades_baseline=validated_oos2_regime,
            strategy_trades_regime01=validated_oos2_regime,
            data_folder=DATA_FOLDER_OOS2,
            initial_balance=INITIAL_BALANCE,
            title="Portfolio OOS2 — Validated only",
        )

    if validated_oos3_regime:
        plot_portfolio_comparison(
            strategy_trades_baseline=validated_oos3_regime,
            strategy_trades_regime01=validated_oos3_regime,
            data_folder=DATA_FOLDER_OOS3,
            initial_balance=INITIAL_BALANCE,
            title="Portfolio OOS3 — Validated only",
        )
        
# =============================================================================
# RUNS
# =============================================================================        
    # CORRELATION ANALYSIS 
    # -------------------------------------------------------------------------
    if RUN_CORRELATION:
        logger.info(f"\n{'─'*115}\n  CORRELATION ANALYSIS OOSs — Profit (threshold={CORRELATION_DD_THRESHOLD})\n{'─'*115}")
        survivors_profit = decorrelate_by_profit(
            strategy_trades_oos1     = validated_oos1_regime,
            #strategy_trades_oos1     = [],
            #strategy_trades_oos2     = validated_oos2_regime,
            #strategy_trades_oos3     = validated_oos3_regime,
            strategy_trades_oos2     = [],
            strategy_trades_oos3     = [],
            initial_balance          = INITIAL_BALANCE,
            threshold                = CORRELATION_DD_THRESHOLD,
            precomputed_metrics      = r01_metrics,
        )
        if survivors_profit:
            print_all_curves_table(survivors_profit, "Decorrelated by Profit — Validated only", INITIAL_BALANCE)
            plot_portfolio_comparison(
                strategy_trades_baseline = survivors_profit,
                strategy_trades_regime01 = survivors_profit,
                data_folder              = DATA_FOLDER_OOS1,
                initial_balance          = INITIAL_BALANCE,
                title                    = "Portfolio — Decorrelated Validated (Profit filter)",
            )
    
            survivors_ids = {sid for sid, _ in survivors_profit}
            print_robustness_table(
                strategy_trades_per_period=[
                    ("OOS1", [(sid, df) for sid, df in validated_oos1_regime if sid in survivors_ids]),
                    ("OOS2", [(sid, df) for sid, df in validated_oos2_regime if sid in survivors_ids]),
                    ("OOS3", [(sid, df) for sid, df in validated_oos3_regime if sid in survivors_ids]),
                ],
                initial_balance=INITIAL_BALANCE,
            )   
    # BEST PORTFOLIO 
    # -------------------------------------------------------------------------
    if RUN_BEST_PORTFOLIO:
        if RUN_CORRELATION and survivors_profit:
            survivor_ids          = {sid for sid, _ in survivors_profit}
            portfolio_trades_oos1 = [(sid, df) for sid, df in validated_oos1_regime if sid in survivor_ids]
            portfolio_trades_oos2 = [(sid, df) for sid, df in validated_oos2_regime if sid in survivor_ids]
            portfolio_trades_oos3 = [(sid, df) for sid, df in validated_oos3_regime if sid in survivor_ids]
        else:
            portfolio_trades_oos1 = validated_oos1_regime
            portfolio_trades_oos2 = validated_oos2_regime
            portfolio_trades_oos3 = validated_oos3_regime

        find_best_portfolio_combination(
            validated_trades_oos1 = portfolio_trades_oos1,
            validated_trades_oos2 = portfolio_trades_oos2,
            validated_trades_oos3 = portfolio_trades_oos3,
            initial_balance       = INITIAL_BALANCE,
            data_folder_oos1      = DATA_FOLDER_OOS1,
            data_folder_oos2      = DATA_FOLDER_OOS2,
            data_folder_oos3      = DATA_FOLDER_OOS3,
            show_plots            = SHOW_PLOTS,
            validation_results    = _validation_results,
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
    logger.info(f"  Strategies set : {STRATEGIES_SET_NAME}-{len(strategies_to_run)} stratagies")
    logger.info(f"  Loop config    : {STRATEGIES_LOOP_NAME}")
    logger.info(f"  Outputs update : {'🟢 enabled' if UPDATE_OUTPUTS else '⚪ disabled'}")
    regime_module.REGIME_ENABLED = REGIME_ENABLED
    load_config_from_bins(REGIME_BINS_PATH)
    logger.info(f"  Regime         : {'🟢 enabled' if REGIME_ENABLED else '⚪ disabled'}  CFG={regime_module.INDICATOR_CFG}  TF={REGIME_TIMEFRAME}")
    logger.info(f"  Data IS        : 🔵 {_short_path(DATA_FOLDER_IS)}")
    logger.info(f"  Data OOS1      : 🔵 {_short_path(DATA_FOLDER_OOS1)}")
    logger.info(f"  Data OOS2      : {'🔵' if OOS2_FOR_VALIDATION else '⚪'} {_short_path(DATA_FOLDER_OOS2)}")
    logger.info(f"  Data OOS3      : {'🔵' if OOS3_FOR_VALIDATION else '⚪'} {_short_path(DATA_FOLDER_OOS3)}")
    logger.info(f"  Validation     : NetGain>{OOS_NETGAIN_TH}%  MaxDD<{OOS_MAX_DD_TH}%  R2>{OOS_R2_TH}")
    logger.info(f"{'='*115}\n")
    
    for strategy in strategies_to_run:
        logger.info(f"\n{'─'*115}\n  Running: {strategy['id']}\n{'─'*115}")
        run_batch(strategy)

    if UPDATE_OUTPUTS:
        save_drift_reference(_drift_results, DRIFT_BATCH_PATH)
        save_strategies_pr(
            strategies_batch_path=STRATEGIES_BT_BATCH_PATH,
            module_name=STRATEGIES_BT_BATCH_MODULE,
            output_path=STRATEGIES_PR_BATCH_PATH,
            validation_results=_validation_results,
            best_params_map=_best_params_results,
            strategy_ids_to_run=[s["id"] for s in strategies_to_run],
        )
        compare_and_generate_csv(
            strategies_batch_path=STRATEGIES_BT_BATCH_PATH,
            pr_batch_path=STRATEGIES_PR_BATCH_PATH,
            csv_path=CSV_PARAMS,
        )
        print_update_status(CSV_PARAMS, SYMBOLS_LIVE_FOLDER, _validation_results)
    
    run_summary()

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")