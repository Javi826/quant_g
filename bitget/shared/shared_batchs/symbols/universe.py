#shared_batch/pipeline/symbols/universe.py (shared — forex + crypto)
import os
import sys
import logging
import pandas as pd
from shared_config import VOLUME_COL
logger = logging.getLogger("BOT_batch.pipeline.universe")

# =============================================================================
# UNIVERSE DATA REQUIREMENTS
# =============================================================================
# Symbols must have data available from this date onward (covers WFO window 0 train_start).
MIN_START_DATE = "2022-01-02"


# =============================================================================
# BUILD UNIVERSE — single entry point: validate every symbol across every
# timeframe, then load and return the OHLCV data. Reports pass/fail via
# logging and stops the process immediately if any symbol fails.
# =============================================================================
def build_universe(data_folder_is: str, symbols_by_timeframe: dict, min_price: float | None = None) -> dict:
    cutoff_ts    = pd.Timestamp(MIN_START_DATE)
    failures     = []
    ohlcv_loaded = {timeframe: {} for timeframe in symbols_by_timeframe}

    for timeframe, symbols in symbols_by_timeframe.items():
        for sym in symbols:
            file_path = os.path.join(data_folder_is, f"{sym}_{timeframe}.parquet")

            if not os.path.exists(file_path):
                failures.append(f"{sym} ({timeframe}): file missing")
                continue

            df = pd.read_parquet(file_path)

            if df.empty:
                failures.append(f"{sym} ({timeframe}): no data")
                continue

            if df.index[0] > cutoff_ts:
                failures.append(f"{sym} ({timeframe}): starts at {df.index[0]}, requires {MIN_START_DATE}")
                continue

            if min_price is not None and df['close'].iloc[-1] <= min_price:
                failures.append(f"{sym} ({timeframe}): last close {df['close'].iloc[-1]} <= min_price {min_price}")
                continue

            ohlcv_loaded[timeframe][sym] = df

    if failures:
        logger.error(f"⛔ Universe validation FAILED — {len(failures)} symbol(s):")
        for failure in failures:
            logger.error(f"  {failure}")
        sys.exit(1)

    counts_str = " | ".join(f"{timeframe}: {len(symbols)} symbol(s)" for timeframe, symbols in ohlcv_loaded.items())
    logger.info(f"✅ Universe validation OK — {counts_str}")

    ohlcv_by_timeframe = {}
    for timeframe, ohlcv_data in ohlcv_loaded.items():
        ohlcv_cut = {}
        for sym, df in ohlcv_data.items():
            start_before = df.index[0]
            df_cut = df[df.index >= cutoff_ts]
            ohlcv_cut[sym] = df_cut
            start_after = df_cut.index[0] if len(df_cut) else None
            logger.debug(f"  {sym:<12} before: {str(start_before):<19}  after: {str(start_after):<19}")
        ohlcv_by_timeframe[timeframe] = ohlcv_cut
        logger.debug(f"IS pool ({len(ohlcv_cut):>3}) [{timeframe}]: {list(ohlcv_cut.keys())}")

    return ohlcv_by_timeframe

# =============================================================================
# TOP-N SYMBOL SELECTION BY VOLUME — optional helper, kept for volume-based flows
# =============================================================================
def select_top_n_by_volume(ohlcv_data: dict, n_symbols: int | None) -> dict:

    if n_symbols is None or n_symbols >= len(ohlcv_data):
        return ohlcv_data

    avg_volume_by_symbol = {
        sym: float(df[VOLUME_COL].mean())
        for sym, df in ohlcv_data.items()
    }

    top_symbols = sorted(avg_volume_by_symbol, key=avg_volume_by_symbol.get, reverse=True)[:n_symbols]

    logger.debug(f"Top {n_symbols} symbols by avg volume: {top_symbols}")

    return {sym: ohlcv_data[sym] for sym in top_symbols}