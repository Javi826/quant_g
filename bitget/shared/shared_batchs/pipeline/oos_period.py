#shared_batchs/pipeline/oos_period.py
import logging
import os
import pandas as pd
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE, run_grid_backtest
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays, apply_night_consolidation_filter, NIGHT_CONSOLIDATION_FILTER_ENABLED
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.reporting import print_metrics_table
from shared_batchs.utils.plotting import plot_filter_comparison
from shared_batchs.regime import regime_module
from shared_batchs.regime.regime_module import INDICATOR_CFG, _get_indicator_cache
from shared_batch_regime.regime_core import apply_regime_filter

logger = logging.getLogger("BOT_batch.pipeline.oos_period")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _accumulate_strategy_trades(
    registry: list,
    strategy_id: str,
    trade_log: pd.DataFrame,
    csv_folder: str = None,
    label: str = "",
) -> None:
    """Accumulate trades into a registry list, optionally saving to CSV."""
    registry.append((strategy_id, trade_log.copy()))
    if csv_folder and label:
        os.makedirs(csv_folder, exist_ok=True)
        path               = os.path.join(csv_folder, f"trades_{label}_{strategy_id}.csv")
        df_out             = trade_log.copy()
        df_out["strategy"] = strategy_id
        df_out.to_csv(path, index=False)
        logger.debug(f"trades saved → {path}")


# =============================================================================
# RUN OOS BACKTEST WITH REGIME
# =============================================================================

def run_oos_backtest_with_regime(
    strategy_id:     str,
    ohlcv_arrays:    dict,
    signal_fn,
    signal_params:   dict,
    best_params:     dict,
    order_amount:    int,
    bins_to_filter:  str | list[str],
    initial_balance: float,
    data_folder:     str,
) -> tuple:
    _bins_to_filter = [bins_to_filter] if isinstance(bins_to_filter, str) else bins_to_filter

    ohlcv_arrays_regime: dict = {}
    #FILTER-NIGHT
    for sym, arr in ohlcv_arrays.items():
        signals = signal_fn(arr, **signal_params, live_trading=False)
        if NIGHT_CONSOLIDATION_FILTER_ENABLED:
            signals = apply_night_consolidation_filter(arr["ts"], signals)

        if regime_module.REGIME_ENABLED and _bins_to_filter and _bins_to_filter != ["neutral"]:
            sym_cache = _get_indicator_cache(sym, data_folder)
            signals   = apply_regime_filter(
                signals        = signals,
                arr            = arr,
                sym_cache      = sym_cache,
                cfg            = INDICATOR_CFG,
                bins_to_filter = _bins_to_filter,
            )

        ohlcv_arrays_regime[sym] = {**arr, "signal": signals}

    result = run_grid_backtest(
        ohlcv_arrays_regime,
        sell_after   = best_params["SELL_AFTER"],
        tp_pct       = best_params["TP_PCT"],
        sl_pct       = best_params["SL_PCT"],
        order_amount = order_amount,
    )
    trades_df             = result["__PORTFOLIO__"]["trade_log"].copy()
    trades_df.columns     = trades_df.columns.str.lower().str.strip()
    trades_df["buy_time"] = pd.to_datetime(trades_df["buy_time"])
    metrics = compute_metrics(trades_df, capital=initial_balance, name=strategy_id) if len(trades_df) > 0 else None
    return trades_df, metrics


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
) -> None:

    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_data)

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------
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

    trades_baseline             = result_baseline["__PORTFOLIO__"]["trade_log"].copy()
    trades_baseline.columns     = trades_baseline.columns.str.lower().str.strip()
    trades_baseline["buy_time"] = pd.to_datetime(trades_baseline["buy_time"])

    if save_trades:
        _accumulate_strategy_trades(
            trades_baseline_accum, strategy_id, trades_baseline,
            csv_folder=brief_trades_folder, label=f"{label.lower()}_baseline",
        )
    else:
        trades_baseline_accum.append((strategy_id, trades_baseline.copy()))

    # -------------------------------------------------------------------------
    # Regime
    # -------------------------------------------------------------------------
    logger.debug(f"STAGE 3 ── Backtest {label} Regime   ── {len(ohlcv_data)} symbols — bins: {bins_to_filter if bins_to_filter else 'none'}")

    if regime_module.REGIME_ENABLED:
        trades_regime, metrics_regime = run_oos_backtest_with_regime(
            strategy_id     = f"{strategy_id}_{label.lower()}_regime",
            ohlcv_arrays    = ohlcv_arrays,
            signal_fn       = signal_fn,
            signal_params   = signal_params,
            best_params     = best_params,
            order_amount    = order_amount,
            bins_to_filter  = bins_to_filter,
            initial_balance = INITIAL_BALANCE,
            data_folder     = data_folder,
        )
    else:
        trades_regime  = trades_baseline.copy() if not trades_baseline.empty else pd.DataFrame()
        metrics_regime = compute_metrics(trades_baseline, capital=INITIAL_BALANCE, name=strategy_id)

    _b_profit = "N/A" if trades_baseline.empty else f"{trades_baseline['profit'].sum():.1f}"
    logger.debug(f"  [DEBUG {label}] baseline profit={_b_profit} | regime profit={trades_regime['profit'].sum():.1f}")
    logger.debug(
        f"STAGE 3 ── Filter results         ── "
        f"baseline={len(trades_baseline)} | regime={len(trades_regime)} | diff={len(trades_baseline) - len(trades_regime)}"
    )

    if len(trades_regime) > 0:
        print_metrics_table([metrics_regime], f"  Metrics — {strategy_id} ({label} Regime)")
        if save_trades:
            _accumulate_strategy_trades(
                trades_regime_accum, strategy_id, trades_regime,
                csv_folder=brief_trades_folder, label=f"{label.lower()}_regime",
            )
        else:
            trades_regime_accum.append((strategy_id, trades_regime.copy()))

    if len(trades_baseline) > 0:
        plot_filter_comparison(
            strategy_id        = f"{strategy_id}_{label.lower()}",
            trades_df_baseline = trades_baseline,
            trades_df_r01      = trades_regime if len(trades_regime) > 0 else None,
            data_folder        = data_folder,
            initial_balance    = INITIAL_BALANCE,
        )