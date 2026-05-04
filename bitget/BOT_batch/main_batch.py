#BOT_batch/main_batch.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "market_regime")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))

import matplotlib
SHOW_PLOTS = False
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import logging
import contextlib
import time
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from importlib import import_module

# LOGGING CONFIGURATION
#------------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(message)s", force=True)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

from utils.utils import filter_symbols
from utils.analysis import report_montecarlo, report_backtesting
from shared_market_regime.regime_common import load_btc_for_timeframe, filter_signals_by_regime
from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from tools.optimize_MCf_tf import generate_paths_for_all_symbols_functional
from utils.torque import compile_grid_results, prepare_ohlcv_arrays,get_n_obs
from utils.torque import extract_ohlcv_from_path, compile_MC_results
from utils_batch import SIGNAL_REGISTRY,extract_best_params, select_universe,print_best_r2_robustness_table,print_robustness_table
from utils_batch import update_strategies_symbols, analyze_regime_is,decorrelate_by_dd, decorrelate_by_profit,print_update_status
from utils_batch import compute_metrics, print_metrics_table,accumulate_strategy_trades,print_portfolio_metrics_table
from utils_batch import save_drift_reference, save_strategies_pr, compare_and_generate_csv,print_strategies_summary
from utils_batch import print_all_curves_table, find_best_r2_combination_ids, plot_filter_comparison, plot_portfolio_comparison
from shared_config import REGIME_ATR_WINDOW as ATR_WINDOW, REGIME_PE_WINDOW as PE_WINDOW, REGIME_PE_ORDER as PE_ORDER
from shared_config import REGIME_FAMILIES as FAMILIES, REGIME_HURST_WINDOW as HURST_WINDOW, REGIME_ER_WINDOW as ER_WINDOW
from shared_config import REGIME0_MA_PERIOD as R0_MA_PERIOD, REGIME0_LONG_TH as R0_LONG_TH, REGIME0_SHORT_TH as R0_SHORT_TH

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
# GLOBAL CONFIGURATION
# =============================================================================
DTYPE         = np.float32
logger        = logging.getLogger("BOT_batch.main_batch")
N_JOBS        = -1
MY_SYMBOLS    = False
SHOW_PROGRESS = False

# =============================================================================
# RUN CONFIGURATION
# =============================================================================

# RUN + MC 
#------------------------------------------------------------------------------
STRATEGIES_SET_NAME  = "00"  
STRATEGIES_LOOP_NAME = f"strategies_loop_{STRATEGIES_SET_NAME}_03"
N_PATHS_IS           = 1000

# REGULAR -- MA4
#------------------------------------------------------------------------------
OOS_NETGAIN_TH       = 30
OOS_MAX_DD_TH        = 10
OOS_R2_TH            = 0.85  

# ELITE -- MA4
#----------------------------------------------------------------------------
# =============================================================================
# OOS_NETGAIN_TH       = 40
# OOS_MAX_DD_TH        = 5
# OOS_R2_TH            = 0.85  
# =============================================================================

# BATCH
#------------------------------------------------------------------------------
RUN_PORTFOLIO_ANALYSIS   = True
RUN_CORRELATION_ANALYSIS = False
RUN_BEST_COMBINATIONS    = False
UPDATE_OUTPUTS           = False
SAVE_TRADES              = False

# STRATEGY SELECTION
#------------------------------------------------------------------------------
SELECTED_STRATEGIES = [
    "07_reversal_short_1H",
    "10_parity_long_1H",
    "11_parity_short_1H",
    "18_flag_long_1H",
    "20_flag_short_1H",
    "27_orderblocks_short_1H",
    # ------------------------------------------------------------------------
    "04_reversal_short_4H",
    "06_reversal_long_1H",
    "19_flag_short_4H",
    "22_parity_short_6Hutc",
    "28_orderblocks_long_1H",
    # -------------------------------------------------------------------------
    "02_reversal_long_4H",
    "03_parity_long_4H",
    "04_reversal_short_4H",
    "08_reversal_long_6Hutc",
    "09_reversal_short_6Hutc",
    "12_parity_long_6Hutc",
    "13_orderblocks_short_4H",
    "14_orderblocks_long_4H",
    "17_flag_long_4H",
    "18_flag_long_1H",
    "21_parity_short_4H",
    "26_orderblocks_long_4H",
    "28_orderblocks_long_1H",
    "24_flag_long_6Hutc",
    "25_flag_short_6Hutc",
]
# =============================================================================
# -----------------------------------------------------------------------------
# =============================================================================

# FILES
#------------------------------------------------------------------------------
STRATEGIES_BATCH           = import_module(f"strategies_files.files_{STRATEGIES_SET_NAME}.strategies_BT_{STRATEGIES_SET_NAME}_batch").STRATEGIES
STRATEGIES_BT_BATCH_MODULE = f"strategies_BT_{STRATEGIES_SET_NAME}_batch"
STRATEGIES_BT_BATCH_PATH   = os.path.join(os.path.dirname(__file__), "strategies_files", f"files_{STRATEGIES_SET_NAME}", f"strategies_BT_{STRATEGIES_SET_NAME}_batch.py")
STRATEGIES_LOOP            = import_module(f"strategies_files.files_{STRATEGIES_SET_NAME}.{STRATEGIES_LOOP_NAME}").STRATEGIES_LOOP
STRATEGIES_PARAMS_FOLDER   = os.path.join(os.path.dirname(__file__), f"strategies_{STRATEGIES_SET_NAME}")
CSV_PARAMS                 = os.path.join(STRATEGIES_PARAMS_FOLDER, f"strategies_{STRATEGIES_SET_NAME}.csv")
STRATEGIES_PR_BATCH_PATH   = os.path.join(STRATEGIES_PARAMS_FOLDER, f"strategies_{STRATEGIES_SET_NAME}_batch.py")
SYMBOLS_LIVE_FOLDER        = os.path.join(STRATEGIES_PARAMS_FOLDER, "symbols_live")
DRIFT_MONTECARLO_FOLDER    = os.path.join(STRATEGIES_PARAMS_FOLDER, "drift_montecarlo")
DRIFT_BATCH_PATH           = os.path.join(DRIFT_MONTECARLO_FOLDER, f"drift_montecarlo_{STRATEGIES_SET_NAME}_batch.py")

# DATA
#------------------------------------------------------------------------------
SPLIT_MODE       = "expanding"
SPLIT_BASE       = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER_IS   = os.path.join(SPLIT_BASE, "IS",  "crypto_2024-01_2025-04_IS")
DATA_FOLDER_OOS1 = os.path.join(SPLIT_BASE, "OOS", "crypto_2025-04_2026-04_OOS")
DATA_FOLDER_OOS2 = os.path.join(SPLIT_BASE, "OOS", "crypto_2022-01_2023-01_OOS")
DATA_FOLDER_OOS3 = os.path.join(SPLIT_BASE, "OOS", "crypto_2023-01_2024-01_OOS")
#DATA_FOLDER_OOS1 = os.path.join(SPLIT_BASE, "OOS", "crypto_2026-01_2026-05_OOS")

#MONTECARLO
#------------------------------------------------------------------------------
N_SYMBOLS_MCIS            = 6
N_PATHS_OOS1              = 1
FIX_SYMBOLS_MCIS_TRAINING = True
MC_SELECTION_PERCENTILE   = None  # None = mean | int = percentile e.g. 25, 50

# Regime analysis params
#------------------------------------------------------------------------------
FORCE_DIRECTION_FILTER = True
REGIME_MIN_TRADES      = 10
REGIME_LOOKBACK_BARS   = 180
REGIME_FAMILY_SOURCE   = 'strategy'  # 'strategy' | 'macro'

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

# OOS1 - Validation thresholds — Round 1
#------------------------------------------------------------------------------
R1_NETGAIN_ROUND1    = 40
R1_RSQUARED_ROUND1   = 0.95
R1_MAX_DD_ROUND1     = 5  
R1_PROBNEG_ROUND1    = 60

# OOS1 - Validation thresholds — Round 2
#------------------------------------------------------------------------------
R2_NETGAIN_ROUND2   = OOS_NETGAIN_TH
R2_MAX_DD_ROUND2    = OOS_MAX_DD_TH
R2_R2_ROUND2        = OOS_R2_TH

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

# OOS2/3 symbol selection
#------------------------------------------------------------------------------
OOS23_MATCH_SYMBOLS = True  # True = top N by volume in OOS2/3 period | False = same symbols as OOS1

# Correlation analysis
#------------------------------------------------------------------------------
CORRELATION_DD_THRESHOLD = 0.70  # max allowed DD correlation between validated strategies

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_batch(strategy_config: dict) -> None:
    """
    Run the full batch pipeline for a single strategy.

    strategy_config keys:
        id               : str   e.g. "03_parity_long_4H"
        name             : str
        direction        : str   "long" | "short"
        timeframe        : str   e.g. "4H"
        n_symbols        : int
        order_amount     : int
        order_amount_prod: int
        direction_mode   : str
        sell_after_ncandles: int
        param_grid       : dict  {PARAM_NAME: [values], ...}
    """
        

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

    FINAL_N_OBS_PER_PATH = get_n_obs(TIMEFRAME)
    

    # -------------------------------------------------------------------------
    # BLOCK 0 — Universe Selection
    # -------------------------------------------------------------------------
    symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos1 = select_universe(
        data_folder_is=DATA_FOLDER_IS,
        data_folder_oos=DATA_FOLDER_OOS1,
        timeframe=TIMEFRAME,
        n_symbols=N_SYMBOLS,
        min_price=MIN_PRICE,
        filter_symbols_fn=filter_symbols,
        my_symbols=MY_SYMBOLS,
        fix_symbols_mcis=FIX_SYMBOLS_MCIS_TRAINING,
        n_symbols_mcis=N_SYMBOLS_MCIS,
    )

    # -------------------------------------------------------------------------
    # BLOCK 1 — Monte Carlo IS
    # -------------------------------------------------------------------------
    logger.info(f"STAGE 1  ── MC IS                  ── {N_PATHS_IS} paths | {len(param_dict_list)} combos")

    ohlcv_data_minor = {sym: ohlcv_is[sym] for sym in symbols_is_final}
    paths_minor      = generate_paths_for_all_symbols_functional(
        ohlcv_data_minor, n_paths=N_PATHS_IS, n_obs=FINAL_N_OBS_PER_PATH, raw_columns=[],
    )

    def _process_path(path_idx, paths, params_list):
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

    with (tqdm_joblib(tqdm(total=N_PATHS_IS, desc="🔄 Evaluating MC IS paths")) if SHOW_PROGRESS else contextlib.nullcontext()):
        results_list = Parallel(n_jobs=N_JOBS)(
            delayed(_process_path)(i, paths_minor, param_dict_list)
            for i in range(N_PATHS_IS)
        )

    all_results  = [r for sublist in results_list for r in sublist]
    df_portfolio = pd.DataFrame(all_results)
    df_summary, _, _ = report_montecarlo(df_portfolio=df_portfolio, param_names=param_names, initial_balance=INITIAL_BALANCE, selection_percentile=MC_SELECTION_PERCENTILE)
    best_params = extract_best_params(df_summary, param_names, lists_for_grid, selection_percentile=MC_SELECTION_PERCENTILE)

    params_str = " | ".join(f"{k}={v}" for k, v in best_params.items() if k not in ("SELL_AFTER",))
    logger.info(f"STAGE 1  ── MC Best params         ── {params_str}")

    bt_signal_params = {k: best_params[k.upper()] for k in signal_params_keys if k.upper() in best_params}
    
    # -------------------------------------------------------------------------
    # BLOCK 1b — Build metrics cache for regime analysis
    # -------------------------------------------------------------------------
    from shared_market_regime.regime_common import build_metrics_cache
    
    btc_cache_is_regime = {}
    btc_tf_is = load_btc_for_timeframe(DATA_FOLDER_IS, TIMEFRAME, btc_cache_is_regime) \
                if REGIME_FAMILY_SOURCE == 'strategy' \
                else load_btc_for_timeframe(DATA_FOLDER_IS, '1Dutc', btc_cache_is_regime)
    
    metrics_cache_is = build_metrics_cache(
        btc_df       = btc_tf_is,
        lookback     = REGIME_LOOKBACK_BARS,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    )

    # -------------------------------------------------------------------------
    # BLOCK 2 — Backtest IS + Regime Analysis
    # -------------------------------------------------------------------------
    logger.info(f"STAGE 2  ── Backtest IS + Regime   ── symbols: {len(symbols_oos_final)}")

    ohlcv_data_is_regime = {sym: ohlcv_is[sym] for sym in symbols_oos_final if sym in ohlcv_is}
    ohlcv_arr_is_regime  = prepare_ohlcv_arrays(ohlcv_data_is_regime)


    ohlcv_arrays_is = {}
    for sym, arr in ohlcv_arr_is_regime.items():
        signals              = signal_fn(arr, **bt_signal_params, live_trading=False)
        ohlcv_arrays_is[sym] = {**arr, "signal": signals}

    is_result = run_grid_backtest(
        ohlcv_arrays_is,
        sell_after=best_params["SELL_AFTER"],
        tp_pct=best_params["TP_PCT"],
        sl_pct=best_params["SL_PCT"],
        order_amount=ORDER_AMOUNT,
    )

    trades_df_is = is_result["__PORTFOLIO__"]["trade_log"].copy()
    trades_df_is.columns = trades_df_is.columns.str.lower().str.strip()
    trades_df_is["buy_time"] = pd.to_datetime(trades_df_is["buy_time"])


    bins_to_filter = analyze_regime_is(
        trades_df_is     = trades_df_is,
        timeframe        = TIMEFRAME,
        data_folder_is   = DATA_FOLDER_IS,
        families         = FAMILIES,
        regime_min_trades= REGIME_MIN_TRADES,
        regime_lookback  = REGIME_LOOKBACK_BARS,
        family_source    = REGIME_FAMILY_SOURCE,
        hurst_window     = HURST_WINDOW,
        er_window        = ER_WINDOW,
        atr_window       = ATR_WINDOW,
        pe_window        = PE_WINDOW,
        pe_order         = PE_ORDER,
        ma_period        = R0_MA_PERIOD,
        long_th          = R0_LONG_TH,
        short_th         = R0_SHORT_TH,
        strategy_direction    = SIDE,
        force_direction_filter= FORCE_DIRECTION_FILTER,
        metrics_cache         = metrics_cache_is,
    )
    if SAVE_TRADES:
        accumulate_strategy_trades(
            _strategy_trades_is_baseline, STRATEGY_ID, trades_df_is,
            csv_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
            label="is_baseline",
        )
    
    # -------------------------------------------------------------------------
    # BLOCK 2b — Backtest IS with regime filter (for best R² combination)
    # -------------------------------------------------------------------------
    if RUN_BEST_COMBINATIONS:
        btc_cache_is_r = {}
        btc_1d_df_is   = load_btc_for_timeframe(DATA_FOLDER_IS, '1Dutc', btc_cache_is_r)
        btc_tf_df_is   = load_btc_for_timeframe(DATA_FOLDER_IS, TIMEFRAME, btc_cache_is_r) \
                         if REGIME_FAMILY_SOURCE == 'strategy' else btc_1d_df_is

        ohlcv_arrays_is_regime = {}
        for sym, arr in ohlcv_arr_is_regime.items():
            signals = signal_fn(arr, **bt_signal_params, live_trading=False)
            if bins_to_filter:
                signals = filter_signals_by_regime(
                    signals        = signals,
                    ts             = arr['ts'],
                    btc_1d_df      = btc_1d_df_is,
                    btc_tf_df      = btc_tf_df_is,
                    bins_to_filter = bins_to_filter,
                    ma_period      = R0_MA_PERIOD,
                    long_th        = R0_LONG_TH,
                    short_th       = R0_SHORT_TH,
                    families       = FAMILIES,
                    lookback_bars  = REGIME_LOOKBACK_BARS,
                    hurst_window   = HURST_WINDOW,
                    er_window      = ER_WINDOW,
                    atr_window     = ATR_WINDOW,
                    pe_window      = PE_WINDOW,
                    pe_order       = PE_ORDER,
                    metrics_cache  = metrics_cache_is,
                )
            ohlcv_arrays_is_regime[sym] = {k: v.copy() if hasattr(v, "copy") else v for k, v in arr.items()}
            ohlcv_arrays_is_regime[sym]["signal"] = signals

        is_result_regime = run_grid_backtest(
            ohlcv_arrays_is_regime,
            sell_after=best_params["SELL_AFTER"],
            tp_pct=best_params["TP_PCT"],
            sl_pct=best_params["SL_PCT"],
            order_amount=ORDER_AMOUNT,
        )

        trades_df_is_regime             = is_result_regime["__PORTFOLIO__"]["trade_log"].copy()
        trades_df_is_regime.columns     = trades_df_is_regime.columns.str.lower().str.strip()
        trades_df_is_regime["buy_time"] = pd.to_datetime(trades_df_is_regime["buy_time"])

        if len(trades_df_is_regime) > 0:
            if SAVE_TRADES:
                accumulate_strategy_trades(
                    _strategy_trades_is_regime, STRATEGY_ID, trades_df_is_regime,
                    csv_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
                    label="is_regime",
                )
            else:
                _strategy_trades_is_regime.append((STRATEGY_ID, trades_df_is_regime.copy()))

    # -------------------------------------------------------------------------
    # BLOCK 3 — Backtest OOS1 Baseline
    # -------------------------------------------------------------------------
    logger.info(f"STAGE 3  ── Backtest OOS1 Baseline ── {len(symbols_oos_final)} symbols")

    ohlcv_data_oos1 = {sym: ohlcv_oos1[sym] for sym in symbols_oos_final}
    ohlcv_arr_oos1  = prepare_ohlcv_arrays(ohlcv_data_oos1)

    ohlcv_arrays_oos1_baseline = {}
    for sym, arr in ohlcv_arr_oos1.items():
        signals                         = signal_fn(arr, **bt_signal_params, live_trading=False)
        ohlcv_arrays_oos1_baseline[sym] = {**arr, "signal": signals}

    oos1_result_baseline = run_grid_backtest(
        ohlcv_arrays_oos1_baseline,
        sell_after=best_params["SELL_AFTER"],
        tp_pct=best_params["TP_PCT"],
        sl_pct=best_params["SL_PCT"],
        order_amount=ORDER_AMOUNT,
    )

    best_comb = tuple(best_params[p] for p in param_names)
    oos1_df    = pd.DataFrame(compile_grid_results([(best_comb, oos1_result_baseline)], param_names, INITIAL_BALANCE))

    _, _ = report_backtesting(
        df=oos1_df, parameters=param_names,
        data_folder=DATA_FOLDER_OOS1, initial_capital=INITIAL_BALANCE,
        strategy_id=STRATEGY_ID,
    )

    best_bt_row           = oos1_df.loc[oos1_df["Net_Gain"].idxmax()]
    trades_df             = oos1_result_baseline["__PORTFOLIO__"]["trade_log"].copy()
    trades_df.columns     = trades_df.columns.str.lower().str.strip()
    trades_df["buy_time"] = pd.to_datetime(trades_df["buy_time"])

    if SAVE_TRADES:
        accumulate_strategy_trades(
            _strategy_trades_oos1_baseline, STRATEGY_ID, trades_df,
            csv_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
            label="oos1_baseline",
        )
    else:
        _strategy_trades_oos1_baseline.append((STRATEGY_ID, trades_df.copy()))

    metrics_baseline = compute_metrics(trades_df, capital=INITIAL_BALANCE, name=STRATEGY_ID)
    print_metrics_table([metrics_baseline], f"  Metrics — {STRATEGY_ID} (Baseline)")

    # -------------------------------------------------------------------------
    # BLOCK 4 — Backtest OOS1 Regime 0+1
    # -------------------------------------------------------------------------
    logger.info(f"STAGE 4  ── Backtest OOS1 Regime   ── bins: {bins_to_filter if bins_to_filter else 'none'}")

    btc_cache_oos1 = {}
    btc_1d_df_oos1 = load_btc_for_timeframe(DATA_FOLDER_OOS1, '1Dutc', btc_cache_oos1)
    btc_tf_df_oos1 = load_btc_for_timeframe(DATA_FOLDER_OOS1, TIMEFRAME, btc_cache_oos1) \
                    if REGIME_FAMILY_SOURCE == 'strategy' else btc_1d_df_oos1

    metrics_cache_oos1 = build_metrics_cache(
        btc_df         = btc_tf_df_oos1,
        lookback       = REGIME_LOOKBACK_BARS,
        hurst_window   = HURST_WINDOW,
        er_window      = ER_WINDOW,
        atr_window     = ATR_WINDOW,
        pe_window      = PE_WINDOW,
        pe_order       = PE_ORDER,
    )

    ohlcv_arrays_oos1_regime = {}
    for sym, arr in ohlcv_arr_oos1.items():
        signals = signal_fn(arr, **bt_signal_params, live_trading=False)
        if bins_to_filter:
            signals = filter_signals_by_regime(
                signals        = signals,
                ts             = arr['ts'],
                btc_1d_df      = btc_1d_df_oos1,
                btc_tf_df      = btc_tf_df_oos1,
                bins_to_filter = bins_to_filter,
                ma_period      = R0_MA_PERIOD,
                long_th        = R0_LONG_TH,
                short_th       = R0_SHORT_TH,
                families       = FAMILIES,
                lookback_bars  = REGIME_LOOKBACK_BARS,
                hurst_window   = HURST_WINDOW,
                er_window      = ER_WINDOW,
                atr_window     = ATR_WINDOW,
                pe_window      = PE_WINDOW,
                pe_order       = PE_ORDER,
                metrics_cache  = metrics_cache_oos1,
            )
        ohlcv_arrays_oos1_regime[sym] = {k: v.copy() if hasattr(v, "copy") else v for k, v in arr.items()}
        ohlcv_arrays_oos1_regime[sym]["signal"] = signals

    oos1_result_regime = run_grid_backtest(
        ohlcv_arrays_oos1_regime,
        sell_after=best_params["SELL_AFTER"],
        tp_pct=best_params["TP_PCT"],
        sl_pct=best_params["SL_PCT"],
        order_amount=ORDER_AMOUNT,
    )

    trades_df_regime             = oos1_result_regime["__PORTFOLIO__"]["trade_log"].copy()
    trades_df_regime.columns     = trades_df_regime.columns.str.lower().str.strip()
    trades_df_regime["buy_time"] = pd.to_datetime(trades_df_regime["buy_time"])

    n_baseline = len(trades_df)
    n_regime   = len(trades_df_regime)
    logger.debug(
        f"STAGE 4  ── Filter results         ── "
        f"baseline={n_baseline} | regime={n_regime} | diff={n_baseline - n_regime}"
    )

    if len(trades_df_regime) > 0:
        metrics_regime = compute_metrics(trades_df_regime, capital=INITIAL_BALANCE, name=f"{STRATEGY_ID}_regime01")
        print_metrics_table([metrics_regime], f"  Metrics — {STRATEGY_ID} (Regime 0+1)")
        if SAVE_TRADES:
            accumulate_strategy_trades(
                _strategy_trades_oos1_regime, STRATEGY_ID, trades_df_regime,
                csv_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
                label="oos1_regime",
            )
        else:
            _strategy_trades_oos1_regime.append((STRATEGY_ID, trades_df_regime.copy()))

    # -------------------------------------------------------------------------
    # BLOCK 5 — Monte Carlo OOS1
    # -------------------------------------------------------------------------
    logger.info(f"STAGE 5  ── Monte Carlo OOS1       ── {N_PATHS_OOS1} paths")

    n_obs_oos1        = get_n_obs(TIMEFRAME)
    paths_oos1        = generate_paths_for_all_symbols_functional(
        ohlcv_data_oos1, n_paths=N_PATHS_OOS1, n_obs=n_obs_oos1, raw_columns=[],
    )
    best_params_list = [best_params]

    with (tqdm_joblib(tqdm(total=N_PATHS_OOS1, desc="🔄 Evaluating MC OOS paths")) if SHOW_PROGRESS else contextlib.nullcontext()):
        results_oos1 = Parallel(n_jobs=N_JOBS)(
            delayed(_process_path)(i, paths_oos1, best_params_list)
            for i in range(N_PATHS_OOS1)
        )

    all_results_oos1  = [r for sublist in results_oos1 for r in sublist]
    df_portfolio_oos1 = pd.DataFrame(all_results_oos1)
    _, p5_winrate_oos1, p50_winrate_oos1 = report_montecarlo(
        df_portfolio=df_portfolio_oos1, param_names=param_names, initial_balance=INITIAL_BALANCE,
    )

    # -------------------------------------------------------------------------
    # BLOCK 6 — Validation
    # -------------------------------------------------------------------------
    # prob_negative from MC OOS baseline (always from baseline)
    path_grouped_oos1  = df_portfolio_oos1.groupby("path_index")["Portfolio_Final_Balance"].mean().reset_index()
    path_grouped_oos1["Net_Gain_pct"] = (path_grouped_oos1["Portfolio_Final_Balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    prob_negative_oos1 = (path_grouped_oos1["Net_Gain_pct"] < 0).mean() * 100

    # Round 1 — evaluated on regime-filtered trades
    if len(trades_df_regime) > 0:
        metrics_oos1 = compute_metrics(trades_df_regime, capital=INITIAL_BALANCE, name="")
        netgain_oos1 = metrics_oos1["Net_Gain_pct"]
        r2_oos1      = metrics_oos1["R_Squared"]
        dd_oos1      = metrics_oos1["Max_DD_pct"]
    else:
        metrics_oos1 = compute_metrics(trades_df, capital=INITIAL_BALANCE, name="")
        netgain_oos1 = best_bt_row["Net_Gain"] / INITIAL_BALANCE * 100
        r2_oos1      = metrics_oos1["R_Squared"]
        dd_oos1      = metrics_oos1["Max_DD_pct"]

    ok_oos1_netgain  = netgain_oos1        >= R1_NETGAIN_ROUND1
    ok_oos1_r2       = r2_oos1             >= R1_RSQUARED_ROUND1
    ok_oos1_prob_neg = prob_negative_oos1  <= R1_PROBNEG_ROUND1
    ok_oos1_max_dd   = abs(dd_oos1)        <= R1_MAX_DD_ROUND1
    approved    = ok_oos1_netgain and ok_oos1_r2 and ok_oos1_prob_neg and ok_oos1_max_dd

    _v1 = ("REJECTED" if not approved else "VALIDATED").ljust(13)
    logger.info(f"STAGE 6  ── Backtest 00S1 R1       ── {'🔴' if not approved else '⭐'} {_v1} NetGain={netgain_oos1:.2f}% DD={round(dd_oos1, 2)}% R2={r2_oos1:.2f} ProbNeg={prob_negative_oos1:.1f}%")

    approved_regime = False

    if not approved and len(trades_df_regime) > 0:
        metrics_oos1        = metrics_oos1  # already computed above
        netgain_oos1        = metrics_oos1["Net_Gain_pct"]
        dd_oos1             = metrics_oos1["Max_DD_pct"]
        r2_oos1             = metrics_oos1["R_Squared"]

        ok_oos1_r2_netgain   = netgain_oos1  >= R2_NETGAIN_ROUND2
        ok_oos1_r2_max_dd    = abs(dd_oos1)  <= R2_MAX_DD_ROUND2
        ok_oos1_r2_r2        = r2_oos1       >= R2_R2_ROUND2
        approved_regime = ok_oos1_r2_netgain and ok_oos1_r2_max_dd and ok_oos1_r2_r2

        _v2 = ("VALIDATED" if approved_regime else "REJECTED").ljust(13)
        logger.info(f"STAGE 6  ── Backtest 00S1 R2       ── {'🟢' if approved_regime else '🔴'} {_v2} NetGain={netgain_oos1:.2f}% DD={dd_oos1:.2f}% R2={r2_oos1:.2f}")
        approved = approved or approved_regime

    _round = "—"
    if approved and not approved_regime:
        _round = "Round 1"
    elif approved and approved_regime:
        _round = "Round 2"

    _verdict = "🔴 REJECTED"
    if approved and not approved_regime:
        _verdict = "⭐ VALIDATED"
    elif approved and approved_regime:
        _verdict = "🟢 VALIDATED"

    _validation_results.append({
        "strategy_id":     STRATEGY_ID,
        "verdict":         _verdict,
        "round":           _round,
        "net_gain_pct":    round(netgain_oos1, 2),
        "dd_pct":          round(dd_oos1, 2),
        "win_ratio":       round(metrics_oos1["Win_Rate"], 1),
        "r2":              r2_oos1,
        "prob_neg_pct":    round(prob_negative_oos1, 2),
        "symbols_changed": False,
        "bins_to_filter":  bins_to_filter,
    })

    # If validated in Round 2, overwrite display metrics (same as Round 1 here since both use regime)
    if approved_regime:
        _validation_results[-1].update({
            "net_gain_pct": round(metrics_oos1["Net_Gain_pct"], 2),
            "dd_pct":       round(metrics_oos1["Max_DD_pct"], 2),
            "win_ratio":    round(metrics_oos1["Win_Rate"], 1),
            "r2":           r2_oos1,
        })

    # -------------------------------------------------------------------------
    # BLOCK 6b — OOS2 Analysis (informational + optional validation filter)
    # -------------------------------------------------------------------------
    approved_oos2  = False
    metrics_oos2   = None
    trades_df_oos2 = pd.DataFrame()

    if OOS2_RUN_ANALYSIS:
        if OOS23_MATCH_SYMBOLS:
            logging.disable(logging.INFO)
            _, _oos2_syms, _, ohlcv_oos2_all = select_universe(
                data_folder_is=DATA_FOLDER_IS,
                data_folder_oos=DATA_FOLDER_OOS2,
                timeframe=TIMEFRAME,
                n_symbols=N_SYMBOLS,
                min_price=MIN_PRICE,
                filter_symbols_fn=filter_symbols,
                my_symbols=MY_SYMBOLS,
                fix_symbols_mcis=FIX_SYMBOLS_MCIS_TRAINING,
                n_symbols_mcis=N_SYMBOLS_MCIS,
            )
            logging.disable(logging.NOTSET)
            ohlcv_oos2_raw = {sym: ohlcv_oos2_all[sym] for sym in _oos2_syms if sym in ohlcv_oos2_all}
        else:
            ohlcv_oos2_raw, _ = filter_symbols(
                symbols_oos_final,
                min_vol_usdt=0, timeframe=TIMEFRAME,
                data_folder=DATA_FOLDER_OOS2,
                min_price=MIN_PRICE, vol_window=50,
                my_symbols=MY_SYMBOLS,
            )
        ohlcv_oos2 = prepare_ohlcv_arrays(ohlcv_oos2_raw)
        logger.debug(f"STAGE 6b ── OOS2 symbols           ── {sorted(ohlcv_oos2.keys())}")

        btc_cache_oos2 = {}
        btc_1d_df_oos2 = load_btc_for_timeframe(DATA_FOLDER_OOS2, '1Dutc', btc_cache_oos2)
        btc_tf_df_oos2 = load_btc_for_timeframe(DATA_FOLDER_OOS2, TIMEFRAME, btc_cache_oos2) \
                         if REGIME_FAMILY_SOURCE == 'strategy' else btc_1d_df_oos2

        metrics_cache_oos2 = build_metrics_cache(
            btc_df       = btc_tf_df_oos2,
            lookback     = REGIME_LOOKBACK_BARS,
            hurst_window = HURST_WINDOW,
            er_window    = ER_WINDOW,
            atr_window   = ATR_WINDOW,
            pe_window    = PE_WINDOW,
            pe_order     = PE_ORDER,
        )

        ohlcv_arrays_oos2_regime = {}
        for sym, arr in ohlcv_oos2.items():
            signals = signal_fn(arr, **bt_signal_params, live_trading=False)
            if bins_to_filter:
                signals = filter_signals_by_regime(
                    signals        = signals,
                    ts             = arr['ts'],
                    btc_1d_df      = btc_1d_df_oos2,
                    btc_tf_df      = btc_tf_df_oos2,
                    bins_to_filter = bins_to_filter,
                    ma_period      = R0_MA_PERIOD,
                    long_th        = R0_LONG_TH,
                    short_th       = R0_SHORT_TH,
                    families       = FAMILIES,
                    lookback_bars  = REGIME_LOOKBACK_BARS,
                    hurst_window   = HURST_WINDOW,
                    er_window      = ER_WINDOW,
                    atr_window     = ATR_WINDOW,
                    pe_window      = PE_WINDOW,
                    pe_order       = PE_ORDER,
                    metrics_cache  = metrics_cache_oos2,
                )
            ohlcv_arrays_oos2_regime[sym] = {**arr, "signal": signals}

        oos2_result_regime = run_grid_backtest(
            ohlcv_arrays_oos2_regime,
            sell_after=best_params["SELL_AFTER"],
            tp_pct=best_params["TP_PCT"],
            sl_pct=best_params["SL_PCT"],
            order_amount=ORDER_AMOUNT,
        )

        trades_df_oos2             = oos2_result_regime["__PORTFOLIO__"]["trade_log"].copy()
        trades_df_oos2.columns     = trades_df_oos2.columns.str.lower().str.strip()
        trades_df_oos2["buy_time"] = pd.to_datetime(trades_df_oos2["buy_time"])

        logger.debug(f"STAGE 6b ── OOS2 Backtest Regime   ── {len(trades_df_oos2)} trades | bins: {bins_to_filter if bins_to_filter else 'none'}")

        if len(trades_df_oos2) > 0:
            metrics_oos2 = compute_metrics(trades_df_oos2, capital=INITIAL_BALANCE, name=f"{STRATEGY_ID}_oos2")
            print_metrics_table([metrics_oos2], f"  Metrics — {STRATEGY_ID} (OOS2 Regime)")

            ok_oos2_netgain = metrics_oos2["Net_Gain_pct"]    >= R_NETGAIN_OOS2
            ok_oos2_dd      = abs(metrics_oos2["Max_DD_pct"]) <= R_MAX_DD_OOS2
            ok_oos2_r2      = metrics_oos2["R_Squared"]       >= R_R2_OOS2
            approved_oos2   = ok_oos2_netgain and ok_oos2_dd and ok_oos2_r2

            _v_oos2 = ("VALIDATED" if approved_oos2 else "REJECTED").ljust(13)
            logger.info(
                f"STAGE 6b ── Backtest OOS2          ── "
                f"{'🟢' if approved_oos2 else '🔴'} {_v_oos2} "
                f"NetGain={metrics_oos2['Net_Gain_pct']:.2f}% "
                f"DD={metrics_oos2['Max_DD_pct']:.2f}% "
                f"R2={metrics_oos2['R_Squared']:.2f}  "
                f"trades={len(trades_df_oos2)}"
            )
            plot_filter_comparison(
                strategy_id=f"{STRATEGY_ID}_oos2",
                trades_df_baseline=trades_df_oos2,
                trades_df_r01=trades_df_oos2,
                data_folder=DATA_FOLDER_OOS2,
                initial_balance=INITIAL_BALANCE,
            )
        else:
            logger.info(f"STAGE 6b ── OOS2 Results           ── no trades after regime filter")

        if OOS2_FOR_VALIDATION and approved:
            approved = approved and approved_oos2
            if not approved_oos2:
                logger.info(f"STAGE 6b ── OOS2 Validation        ── 🔴 OOS2 failed — overriding verdict to REJECTED")
                _validation_results[-1]["verdict"] = "🔴 REJECTED"
                _validation_results[-1]["round"]   = "—"

        if len(trades_df_oos2) > 0:
            if SAVE_TRADES:
                accumulate_strategy_trades(
                    _strategy_trades_oos2_regime, STRATEGY_ID, trades_df_oos2,
                    csv_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
                    label="oos2_regime",
                )
            else:
                _strategy_trades_oos2_regime.append((STRATEGY_ID, trades_df_oos2.copy()))
                
        if SAVE_TRADES:
            ohlcv_arrays_oos2_baseline = {}
            for sym, arr in ohlcv_oos2.items():
                signals = signal_fn(arr, **bt_signal_params, live_trading=False)
                ohlcv_arrays_oos2_baseline[sym] = {**arr, "signal": signals}
            oos2_result_baseline = run_grid_backtest(
                ohlcv_arrays_oos2_baseline,
                sell_after=best_params["SELL_AFTER"],
                tp_pct=best_params["TP_PCT"],
                sl_pct=best_params["SL_PCT"],
                order_amount=ORDER_AMOUNT,
            )
            trades_df_oos2_baseline             = oos2_result_baseline["__PORTFOLIO__"]["trade_log"].copy()
            trades_df_oos2_baseline.columns     = trades_df_oos2_baseline.columns.str.lower().str.strip()
            trades_df_oos2_baseline["buy_time"] = pd.to_datetime(trades_df_oos2_baseline["buy_time"])
            if len(trades_df_oos2_baseline) > 0:
                accumulate_strategy_trades(
                    _strategy_trades_oos2_baseline, STRATEGY_ID, trades_df_oos2_baseline,
                    csv_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
                    label="oos2_baseline",
                )

    # -------------------------------------------------------------------------
    # BLOCK 6c — OOS3 Analysis (informational + optional validation filter)
    # -------------------------------------------------------------------------
    approved_oos3  = False
    metrics_oos3   = None
    trades_df_oos3 = pd.DataFrame()

    if OOS3_RUN_ANALYSIS:
        if OOS23_MATCH_SYMBOLS:
            logging.disable(logging.INFO)
            _, _oos3_syms, _, ohlcv_oos3_all = select_universe(
                data_folder_is=DATA_FOLDER_IS,
                data_folder_oos=DATA_FOLDER_OOS3,
                timeframe=TIMEFRAME,
                n_symbols=N_SYMBOLS,
                min_price=MIN_PRICE,
                filter_symbols_fn=filter_symbols,
                my_symbols=MY_SYMBOLS,
                fix_symbols_mcis=FIX_SYMBOLS_MCIS_TRAINING,
                n_symbols_mcis=N_SYMBOLS_MCIS,
            )
            logging.disable(logging.NOTSET)
            ohlcv_oos3_raw = {sym: ohlcv_oos3_all[sym] for sym in _oos3_syms if sym in ohlcv_oos3_all}
        else:
            ohlcv_oos3_raw, _ = filter_symbols(
                symbols_oos_final,
                min_vol_usdt=0, timeframe=TIMEFRAME,
                data_folder=DATA_FOLDER_OOS3,
                min_price=MIN_PRICE, vol_window=50,
                my_symbols=MY_SYMBOLS,
            )
        ohlcv_oos3 = prepare_ohlcv_arrays(ohlcv_oos3_raw)
        logger.debug(f"STAGE 6c ── OOS3 symbols           ── {sorted(ohlcv_oos3.keys())}")

        btc_cache_oos3 = {}
        btc_1d_df_oos3 = load_btc_for_timeframe(DATA_FOLDER_OOS3, '1Dutc', btc_cache_oos3)
        btc_tf_df_oos3 = load_btc_for_timeframe(DATA_FOLDER_OOS3, TIMEFRAME, btc_cache_oos3) \
                         if REGIME_FAMILY_SOURCE == 'strategy' else btc_1d_df_oos3

        metrics_cache_oos3 = build_metrics_cache(
            btc_df       = btc_tf_df_oos3,
            lookback     = REGIME_LOOKBACK_BARS,
            hurst_window = HURST_WINDOW,
            er_window    = ER_WINDOW,
            atr_window   = ATR_WINDOW,
            pe_window    = PE_WINDOW,
            pe_order     = PE_ORDER,
        )

        ohlcv_arrays_oos3_regime = {}
        for sym, arr in ohlcv_oos3.items():
            signals = signal_fn(arr, **bt_signal_params, live_trading=False)
            if bins_to_filter:
                signals = filter_signals_by_regime(
                    signals        = signals,
                    ts             = arr['ts'],
                    btc_1d_df      = btc_1d_df_oos3,
                    btc_tf_df      = btc_tf_df_oos3,
                    bins_to_filter = bins_to_filter,
                    ma_period      = R0_MA_PERIOD,
                    long_th        = R0_LONG_TH,
                    short_th       = R0_SHORT_TH,
                    families       = FAMILIES,
                    lookback_bars  = REGIME_LOOKBACK_BARS,
                    hurst_window   = HURST_WINDOW,
                    er_window      = ER_WINDOW,
                    atr_window     = ATR_WINDOW,
                    pe_window      = PE_WINDOW,
                    pe_order       = PE_ORDER,
                    metrics_cache  = metrics_cache_oos3,
                )
            ohlcv_arrays_oos3_regime[sym] = {**arr, "signal": signals}

        oos3_result_regime = run_grid_backtest(
            ohlcv_arrays_oos3_regime,
            sell_after=best_params["SELL_AFTER"],
            tp_pct=best_params["TP_PCT"],
            sl_pct=best_params["SL_PCT"],
            order_amount=ORDER_AMOUNT,
        )

        trades_df_oos3 = oos3_result_regime["__PORTFOLIO__"]["trade_log"].copy()
        trades_df_oos3.columns = trades_df_oos3.columns.str.lower().str.strip()
        trades_df_oos3["buy_time"] = pd.to_datetime(trades_df_oos3["buy_time"])

        logger.debug(f"STAGE 6c ── OOS3 Backtest Regime   ── {len(trades_df_oos3)} trades | bins: {bins_to_filter if bins_to_filter else 'none'}")

        if len(trades_df_oos3) > 0:
            metrics_oos3 = compute_metrics(trades_df_oos3, capital=INITIAL_BALANCE, name=f"{STRATEGY_ID}_oos3")
            print_metrics_table([metrics_oos3], f"  Metrics — {STRATEGY_ID} (OOS3 Regime)")
            ok_oos3_netgain = metrics_oos3["Net_Gain_pct"]    >= R_NETGAIN_OOS3
            ok_oos3_dd      = abs(metrics_oos3["Max_DD_pct"]) <= R_MAX_DD_OOS3
            ok_oos3_r2      = metrics_oos3["R_Squared"]       >= R_R2_OOS3
            approved_oos3   = ok_oos3_netgain and ok_oos3_dd and ok_oos3_r2

            _v_oos3 = ("VALIDATED" if approved_oos3 else "REJECTED").ljust(13)
            logger.info(
                f"STAGE 6c ── Backtest OOS3          ── "
                f"{'🟢' if approved_oos3 else '🔴'} {_v_oos3} "
                f"NetGain={metrics_oos3['Net_Gain_pct']:.2f}% "
                f"DD={metrics_oos3['Max_DD_pct']:.2f}% "
                f"R2={metrics_oos3['R_Squared']:.2f}  "
                f"trades={len(trades_df_oos3)}"
            )
            plot_filter_comparison(
                strategy_id=f"{STRATEGY_ID}_oos3",
                trades_df_baseline=trades_df_oos3,
                trades_df_r01=trades_df_oos3,
                data_folder=DATA_FOLDER_OOS3,
                initial_balance=INITIAL_BALANCE,
            )
        else:
            logger.info(f"STAGE 6c ── OOS3 Results           ── no trades after regime filter")

        if OOS3_FOR_VALIDATION and approved:
            approved = approved and approved_oos3
            if not approved_oos3:
                logger.info(f"STAGE 6c ── OOS3 Validation        ── 🔴 OOS3 failed — overriding verdict to REJECTED")
                _validation_results[-1]["verdict"] = "🔴 REJECTED"
                _validation_results[-1]["round"]   = "—"

        if len(trades_df_oos3) > 0:
            if SAVE_TRADES:
                accumulate_strategy_trades(
                    _strategy_trades_oos3_regime, STRATEGY_ID, trades_df_oos3,
                    csv_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
                    label="oos3_regime",
                )
            else:
                _strategy_trades_oos3_regime.append((STRATEGY_ID, trades_df_oos3.copy()))

        if SAVE_TRADES:
            ohlcv_arrays_oos3_baseline = {}
            for sym, arr in ohlcv_oos3.items():
                signals = signal_fn(arr, **bt_signal_params, live_trading=False)
                ohlcv_arrays_oos3_baseline[sym] = {**arr, "signal": signals}
            oos3_result_baseline = run_grid_backtest(
                ohlcv_arrays_oos3_baseline,
                sell_after=best_params["SELL_AFTER"],
                tp_pct=best_params["TP_PCT"],
                sl_pct=best_params["SL_PCT"],
                order_amount=ORDER_AMOUNT,
            )
            trades_df_oos3_baseline = oos3_result_baseline["__PORTFOLIO__"]["trade_log"].copy()
            trades_df_oos3_baseline.columns = trades_df_oos3_baseline.columns.str.lower().str.strip()
            trades_df_oos3_baseline["buy_time"] = pd.to_datetime(trades_df_oos3_baseline["buy_time"])
            if len(trades_df_oos3_baseline) > 0:
                accumulate_strategy_trades(
                    _strategy_trades_oos3_baseline, STRATEGY_ID, trades_df_oos3_baseline,
                    csv_folder=os.path.join(os.path.dirname(__file__), "brief_trades"),
                    label="oos3_baseline",
                )

    # -------------------------------------------------------------------------
    # BLOCK 7 — Equity Curves + Plot
    # -------------------------------------------------------------------------
    plot_filter_comparison(
        strategy_id=STRATEGY_ID,
        trades_df_baseline=trades_df,
        trades_df_r01=trades_df_regime if len(trades_df_regime) > 0 else None,
        data_folder=DATA_FOLDER_OOS1,
        initial_balance=INITIAL_BALANCE,
    )

    # -------------------------------------------------------------------------
    # BLOCK 8 — Update & Compare
    # -------------------------------------------------------------------------
    _symbols_result = update_strategies_symbols(
        strategy_id=STRATEGY_ID, symbols_oos_final=symbols_oos_final,
        timeframe=TIMEFRAME, symbols_live_folder=SYMBOLS_LIVE_FOLDER,
    ) if UPDATE_OUTPUTS else None

    _changes     = ["symbols"] if _symbols_result and _symbols_result.get("symbols_changed") else []
    _changes_str = " | ".join(_changes) if _changes else "no changes"
    _icon        = "🔵" if _changes else "⚪"
    logger.debug(f"STAGE 8  ── Update & Compare       ── {_icon} {_changes_str}")

    _drift_results.append({
        "strategy_id":  STRATEGY_ID,
        "p5_winrate":   round(float(p5_winrate_oos1) * 100, 1),
        "p50_winrate":  round(float(p50_winrate_oos1) * 100, 1),
    })
    _best_params_results[STRATEGY_ID] = best_params

    if _validation_results:
        _validation_results[-1].update({
            "symbols_changed": _symbols_result.get("symbols_changed", False) if _symbols_result else False,
        })

    elapsed = int(time.time() - start_time)
    logger.info(f"DONE  🏁 ──  {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")


# =============================================================================
# PORTFOLIO ANALYSIS
# =============================================================================

def run_portfolio_analysis():
    """Compute combined portfolio metrics. Call after all run_batch() calls."""
    if not RUN_PORTFOLIO_ANALYSIS:
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
        logger.info(f"\n{'─'*115}\n  PORTFOLIO ANALYSIS\n{'─'*115}")
        if logger.isEnabledFor(logging.DEBUG):
            print_all_curves_table(_strategy_trades_oos1_baseline, "Baseline", INITIAL_BALANCE)
        if _strategy_trades_oos1_regime:
            print_all_curves_table(_strategy_trades_oos1_regime, "Regime 0+1", INITIAL_BALANCE)

    validated_ids      = {v["strategy_id"] for v in _validation_results if "VALIDATED" in v["verdict"]}
    validated_baseline = [(sid, df) for sid, df in _strategy_trades_oos1_baseline if sid in validated_ids]
    validated_oos1_regime = [(sid, df) for sid, df in _strategy_trades_oos1_regime if sid in validated_ids]

    if validated_baseline:
        logger.info(f"\n{'─'*115}\n  PORTFOLIO ANALYSIS — VALIDATED ONLY\n{'─'*115}")
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

    if RUN_BEST_COMBINATIONS and validated_oos1_regime:
        best_r2_combo_ids = find_best_r2_combination_ids(
            strategy_trades_is = [(sid, df) for sid, df in _strategy_trades_is_regime if sid in validated_ids],
            initial_balance = INITIAL_BALANCE,
        )
        is_combined = pd.concat(
            [df for sid, df in _strategy_trades_is_regime if sid in best_r2_combo_ids],
            ignore_index=True
        ).sort_values("sell_time").reset_index(drop=True)
        is_r2 = compute_metrics(is_combined, capital=INITIAL_BALANCE * len(best_r2_combo_ids), name="")["R_Squared"]

        strategies_str = "\n".join(f"    · {sid}" for sid in best_r2_combo_ids)
        logger.info(
            f"\n{'─'*115}\n  BEST R² COMBINATION (selected on IS) — R²={is_r2:.3f}\n{'─'*115}\n{strategies_str}\n{'─'*115}"
        )
        print_best_r2_robustness_table(
            combo_ids            = best_r2_combo_ids,
            strategy_trades_per_period= [
                ("OOS1", validated_oos1_regime),
                ("OOS2", validated_oos2_regime),
                ("OOS3", validated_oos3_regime),
            ],
            initial_balance      = INITIAL_BALANCE,
        )
    # -------------------------------------------------------------------------
    # CORRELATION ANALYSIS 
    # -------------------------------------------------------------------------
    if RUN_CORRELATION_ANALYSIS and validated_oos1_regime:
        logger.info(f"\n{'-'*115}\n  CORRELATION ANALYSIS OOS1 — DD (threshold={CORRELATION_DD_THRESHOLD})\n{'─'*115}")
        survivors = decorrelate_by_dd(
            strategy_trades_oos1     = validated_oos1_regime,
# =============================================================================
#             strategy_trades_oos2     = [],
#             strategy_trades_oos3     = [],
# =============================================================================
            strategy_trades_oos2     = validated_oos2_regime,
            strategy_trades_oos3     = validated_oos3_regime,
            initial_balance     = INITIAL_BALANCE,
            threshold           = CORRELATION_DD_THRESHOLD,
            precomputed_metrics = r01_metrics,
        )
        if survivors:
            logger.debug(f"  Survivors after decorrelation: {[sid for sid, _ in survivors]}")
            print_metrics_table(
                [compute_metrics(df, capital=INITIAL_BALANCE, name=sid) for sid, df in survivors],
                "  Survivor Strategies — OOS1 Regime Metrics",
            )
            print_all_curves_table(survivors, "Decorrelated — Validated only", INITIAL_BALANCE)
            plot_portfolio_comparison(
                strategy_trades_baseline=survivors,
                strategy_trades_regime01=survivors,
                data_folder=DATA_FOLDER_OOS1,
                initial_balance=INITIAL_BALANCE,
                title="Portfolio — Decorrelated Validated (DD filter)",
            )

        logger.info(f"\n{'-'*115}\n  CORRELATION ANALYSIS OOS1 — Profit (threshold={CORRELATION_DD_THRESHOLD})\n{'-'*115}")
        survivors_profit = decorrelate_by_profit(
            strategy_trades_oos1     = validated_oos1_regime,
# =============================================================================
#             strategy_trades_oos2     = [],
#             strategy_trades_oos3     = [],
# =============================================================================
            strategy_trades_oos2     = validated_oos2_regime,
            strategy_trades_oos3     = validated_oos3_regime,
            initial_balance     = INITIAL_BALANCE,
            threshold           = CORRELATION_DD_THRESHOLD,
            precomputed_metrics = r01_metrics,
        )
        if survivors_profit:
            print_all_curves_table(survivors_profit, "Decorrelated by Profit — Validated only", INITIAL_BALANCE)
            plot_portfolio_comparison(
                strategy_trades_baseline=survivors_profit,
                strategy_trades_regime01=survivors_profit,
                data_folder=DATA_FOLDER_OOS1,
                initial_balance=INITIAL_BALANCE,
                title="Portfolio — Decorrelated Validated (Profit filter)",
            )
            
        discarded_dd     = {sid for sid, _ in validated_oos1_regime if sid not in {s for s, _ in survivors}}
        discarded_profit = {sid for sid, _ in validated_oos1_regime if sid not in {s for s, _ in survivors_profit}}
        discarded_both   = discarded_dd & discarded_profit
        combined_survivors = [(sid, df) for sid, df in validated_oos1_regime if sid not in discarded_both]
        if combined_survivors:
            logger.info(f"\n{'─'*115}\n  CORRELATION ANALYSIS OOS1 — DD + Profit combined (threshold={CORRELATION_DD_THRESHOLD})\n{'─'*115}")
            print_all_curves_table(combined_survivors, "Decorrelated DD + Profit — Validated only", INITIAL_BALANCE)
            plot_portfolio_comparison(
                strategy_trades_baseline=combined_survivors,
                strategy_trades_regime01=combined_survivors,
                data_folder=DATA_FOLDER_OOS1,
                initial_balance=INITIAL_BALANCE,
                title="Portfolio — Decorrelated Validated (DD + Profit filter)",
            )

# =============================================================================
# MAIN
# =============================================================================
from pathlib import Path

def _short_path(full_path: str, from_part: str = "expanding") -> str:
    parts = Path(full_path).parts
    idx = next((i for i, p in enumerate(parts) if p == from_part), None)
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
    logger.info(f"  Strategies set     : {STRATEGIES_SET_NAME}-{len(strategies_to_run)} stratagies")
    logger.info(f"  Loop config        : {STRATEGIES_LOOP_NAME}")
    logger.info(f"  Outputs update     : {'🟢 enabled' if UPDATE_OUTPUTS else '⚪ disabled'}")
    logger.info(f"  Data IS            : 🟢 {_short_path(DATA_FOLDER_IS)}")
    logger.info(f"  Data OOS1          : 🟢 {_short_path(DATA_FOLDER_OOS1)}")
    logger.info(f"  Data OOS2          : {'🟢' if OOS2_FOR_VALIDATION else '⚪'} {_short_path(DATA_FOLDER_OOS2)}")
    logger.info(f"  Data OOS3          : {'🟢' if OOS3_FOR_VALIDATION else '⚪'} {_short_path(DATA_FOLDER_OOS3)}")
    logger.info(f"  Round 1 OOS1       : NetGain>{R1_NETGAIN_ROUND1}%  MaxDD<{R1_MAX_DD_ROUND1}%  R2>{R1_RSQUARED_ROUND1}  ProbNeg<{R1_PROBNEG_ROUND1}%")
    logger.info(f"  Round 2 + All OOSs : NetGain>{R2_NETGAIN_ROUND2}%  MaxDD<{R2_MAX_DD_ROUND2}%  R2>{R2_R2_ROUND2}")
    logger.info(f"  Regime             : MA{R0_MA_PERIOD}  long_th={R0_LONG_TH}  short_th={R0_SHORT_TH}  min_trades={REGIME_MIN_TRADES}  source={REGIME_FAMILY_SOURCE}")
    logger.info(f"{'='*115}\n")
    
    for strategy in strategies_to_run:
        logger.info(f"\n{'='*115}\n  Running: {strategy['id']}\n{'='*115}")
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
    
    run_portfolio_analysis()

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")