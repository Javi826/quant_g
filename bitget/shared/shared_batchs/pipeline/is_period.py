#shared_batchs/pipeline/is_period.py
import logging
import pandas as pd
from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.regime.regime_filter import analyze_regime_is
from shared_batchs.utils.io import accumulate_strategy_trades

logger = logging.getLogger("BOT_batch.pipeline.backtest")


# =============================================================================
# BACKTEST IS
# =============================================================================

def run_backtest_is(
    strategy_id: str,
    ohlcv_is: dict,
    symbols_oos_final: list,
    signal_fn: callable,
    signal_params: dict,
    best_params: dict,
    order_amount: int,
    timeframe: str,
    data_folder_is: str,
    strategy_direction: str,
    metrics_cache_is: dict,
    save_trades: bool,
    trades_is_baseline: list,
    trades_is_regime: list,
    brief_trades_folder: str,
) -> tuple:
    """
    Run IS backtest, regime analysis and optionally IS regime backtest.

    Returns:
        tuple: (bins_to_filter, trades_df_is)
    """

    ohlcv_data_is = {sym: ohlcv_is[sym] for sym in symbols_oos_final if sym in ohlcv_is}
    ohlcv_arr_is  = prepare_ohlcv_arrays(ohlcv_data_is)

    ohlcv_arrays_is = {}
    for sym, arr in ohlcv_arr_is.items():
        signals              = signal_fn(arr, **signal_params, live_trading=False)
        ohlcv_arrays_is[sym] = {**arr, "signal": signals}

    is_result             = run_grid_backtest(
        ohlcv_arrays_is,
        sell_after   = best_params["SELL_AFTER"],
        tp_pct       = best_params["TP_PCT"],
        sl_pct       = best_params["SL_PCT"],
        order_amount = order_amount,
    )
    trades_df_is             = is_result["__PORTFOLIO__"]["trade_log"].copy()
    trades_df_is.columns     = trades_df_is.columns.str.lower().str.strip()
    trades_df_is["buy_time"] = pd.to_datetime(trades_df_is["buy_time"])

    bins_to_filter, pct_remain = analyze_regime_is(
        trades_df_is       = trades_df_is,
        timeframe          = timeframe,
        data_folder_is     = data_folder_is,
        strategy_direction = strategy_direction,
        metrics_cache      = metrics_cache_is,
    )
    logger.info(f"STAGE 2 ── Backtest IS + Regime   ── symbols: {len(symbols_oos_final)} | total={len(trades_df_is)} | remaining={pct_remain}%")

    if save_trades:
        accumulate_strategy_trades(
            trades_is_baseline, strategy_id, trades_df_is,
            csv_folder=brief_trades_folder, label="is_baseline",
        )

    return bins_to_filter, trades_df_is