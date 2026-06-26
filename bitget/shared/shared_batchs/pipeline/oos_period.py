#shared_batchs/pipeline/oos_period.py
import logging
import pandas as pd
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE, run_grid_backtest
from shared_batchs.utils.analysis import report_backtesting
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.utils.backtest_compiler import compile_grid_results
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.reporting import print_metrics_table
from shared_batchs.utils.plotting import plot_filter_comparison
from shared_batchs.utils.io import accumulate_strategy_trades
from shared_batchs.regime.regime_module import run_oos_backtest_with_regime
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays, apply_night_consolidation_filter, NIGHT_CONSOLIDATION_FILTER_ENABLED

logger = logging.getLogger("BOT_batch.pipeline.oos_period")


# =============================================================================
# RUN OOS PERIOD — GENERIC
# =============================================================================

def run_oos_period(
    strategy_id: str,
    label: str,
    ohlcv_data: dict,
    signal_fn: callable,
    signal_params: dict,
    best_params: dict,
    param_names: list,
    order_amount: int,
    timeframe: str,
    data_folder: str,
    bins_to_filter: set,
    trades_baseline_accum: list,
    trades_regime_accum: list,
    save_trades: bool,
    brief_trades_folder: str,
    run_baseline: bool = True,
    run_report_backtesting: bool = False,
) -> tuple:

    ohlcv_arrays     = prepare_ohlcv_arrays(ohlcv_data)
    trades_baseline  = pd.DataFrame()
    metrics_baseline = None

    # Baseline
    if run_baseline:
        logger.info(f"STAGE 2 ── Backtest {label} Baseline ── bins: {bins_to_filter if bins_to_filter else 'none'}")

        ohlcv_baseline = {}
        for sym, arr in ohlcv_arrays.items():
            signals = signal_fn(arr, **signal_params, live_trading=False)
            if NIGHT_CONSOLIDATION_FILTER_ENABLED:
                signals = apply_night_consolidation_filter(arr["ts"], signals)
            ohlcv_baseline[sym] = {**arr, "signal": signals}

        result_baseline = run_grid_backtest(
            ohlcv_baseline,
            sell_after   = best_params["SELL_AFTER"],
            tp_pct       = best_params["TP_PCT"],
            sl_pct       = best_params["SL_PCT"],
            order_amount = order_amount,
        )

        if run_report_backtesting:
            best_comb = tuple(best_params[p] for p in param_names)
            oos_df    = pd.DataFrame(compile_grid_results([(best_comb, result_baseline)], param_names, INITIAL_BALANCE))
            _, _ = report_backtesting(
                df              = oos_df,
                parameters      = param_names,
                data_folder     = data_folder,
                initial_capital = INITIAL_BALANCE,
                strategy_id     = strategy_id,
            )

        trades_baseline             = result_baseline["__PORTFOLIO__"]["trade_log"].copy()
        trades_baseline.columns     = trades_baseline.columns.str.lower().str.strip()
        trades_baseline["buy_time"] = pd.to_datetime(trades_baseline["buy_time"])

        if save_trades:
            accumulate_strategy_trades(
                trades_baseline_accum, strategy_id, trades_baseline,
                csv_folder=brief_trades_folder, label=f"{label.lower()}_baseline",
            )
        else:
            trades_baseline_accum.append((strategy_id, trades_baseline.copy()))

        metrics_baseline = compute_metrics(trades_baseline, capital=INITIAL_BALANCE, name=strategy_id)

    # Regime
    _symbols_str = f"{len(ohlcv_data)} symbols — " if run_baseline else ""
    logger.debug(f"STAGE 3 ── Backtest {label} Regime   ── {_symbols_str}bins: {bins_to_filter if bins_to_filter else 'none'}")

    from shared_batchs.regime.regime_module import REGIME_ENABLED as _REGIME_ENABLED
    if _REGIME_ENABLED:
        trades_regime, metrics_regime = run_oos_backtest_with_regime(
            strategy_id     = f"{strategy_id}_{label.lower()}_regime",
            ohlcv_arrays    = ohlcv_arrays,
            signal_fn       = signal_fn,
            signal_params   = signal_params,
            best_params     = best_params,
            order_amount    = order_amount,
            bins_to_filter  = bins_to_filter,
            initial_balance = INITIAL_BALANCE,
        )
    else:
        trades_regime  = trades_baseline.copy() if not trades_baseline.empty else pd.DataFrame()
        metrics_regime = metrics_baseline

    _b_profit = "N/A" if trades_baseline.empty else f"{trades_baseline['profit'].sum():.1f}"
    logger.debug(f"  [DEBUG {label}] baseline profit={_b_profit} | regime profit={trades_regime['profit'].sum():.1f}")
    logger.debug(
        f"STAGE 3 ── Filter results         ── "
        f"baseline={len(trades_baseline)} | regime={len(trades_regime)} | diff={len(trades_baseline) - len(trades_regime)}"
    )

    if len(trades_regime) > 0:
        print_metrics_table([metrics_regime], f"  Metrics — {strategy_id} ({label} Regime)")
        if save_trades:
            accumulate_strategy_trades(
                trades_regime_accum, strategy_id, trades_regime,
                csv_folder=brief_trades_folder, label=f"{label.lower()}_regime",
            )
        else:
            trades_regime_accum.append((strategy_id, trades_regime.copy()))

    if run_baseline and len(trades_baseline) > 0:
        plot_filter_comparison(
            strategy_id        = f"{strategy_id}_{label.lower()}",
            trades_df_baseline = trades_baseline,
            trades_df_r01      = trades_regime if len(trades_regime) > 0 else None,
            data_folder        = data_folder,
            initial_balance    = INITIAL_BALANCE,
        )

    return trades_baseline, trades_regime, metrics_baseline, metrics_regime