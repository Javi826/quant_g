#BOT_trading/market_regime/regime_classifier.py
"""
Core module for market regime classification.
Uses GE indicator config (config_trading_batch.py) mirroring batch logic.
"""
import logging
import numpy as np
from typing import Optional, Dict
from market_data.data_utils import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from config.settings import ACCOUNTS

logger = logging.getLogger('BOT_trading.market_regime.regime_classifier')

REGIME_MA_WINDOW        = None
REGIME_TIMEFRAME        = None
REGIME_REFERENCE_SYMBOL = None

def configure_regime(account_number: str) -> None:
    global REGIME_MA_WINDOW, REGIME_TIMEFRAME, REGIME_REFERENCE_SYMBOL
    config                  = ACCOUNTS.get(account_number, {})
    REGIME_MA_WINDOW        = config.get('regime_ma_window', 3)
    REGIME_TIMEFRAME        = config.get('regime_timeframe', '1Dutc')
    REGIME_REFERENCE_SYMBOL = config.get('regime_reference_symbol', 'BTCUSDT')

# =============================================================================
# PUBLIC API
# =============================================================================

def _classify_ma(close: float, ma: float) -> str:
    if close > ma:
        return 'uptrend'
    elif close < ma:
        return 'dwtrend'
    return 'neutral'


def _calc_ma(close_arr: np.ndarray, window: int) -> Optional[float]:
    if len(close_arr) < window:
        return None
    return float(np.mean(close_arr[-window:]))


def get_symbol_regime(
    symbol:    str,
    timeframe: str,
    arr:       Optional[dict] = None,
) -> str:
    try:
        ohlcv_data = fetch_ohlcv_data([symbol], REGIME_TIMEFRAME)
        df         = ohlcv_data.get(symbol)
        if df is None or df.empty:
            logger.warning(f"[REGIME] No data for {symbol} {REGIME_TIMEFRAME} — defaulting to neutral")
            return 'neutral'

        df_norm   = normalize_live_ohlcv(df)
        arr_daily = df_to_arrays_live(df_norm)
        close_arr = arr_daily['close']
        close     = float(arr['close'][-1]) if arr is not None else float(close_arr[-1])
        ma        = _calc_ma(close_arr, REGIME_MA_WINDOW)

        if ma is None:
            logger.warning(f"[REGIME] Not enough data for MA({REGIME_MA_WINDOW}) on {symbol} — defaulting to neutral")
            return 'neutral'

        regime = _classify_ma(close, ma)
        logger.debug(f"[REGIME] {symbol} → {regime.upper()} | close={close:.4f} MA({REGIME_MA_WINDOW})={ma:.4f}")
        return regime

    except Exception as e:
        logger.error(f"[REGIME] Error computing regime for {symbol}: {e}")
        return 'neutral'

# =============================================================================
# FRONTEND API
# =============================================================================

def _get_regime_for_symbol(sym: str, close_timeframe: str = None) -> Dict:
    # Fetch daily data for MA
    daily_data = fetch_ohlcv_data([sym], REGIME_TIMEFRAME)
    df_daily   = daily_data.get(sym)
    if df_daily is None or df_daily.empty:
        return {'family': 'neutral', 'metrics': {}}

    arr_daily = df_to_arrays_live(normalize_live_ohlcv(df_daily))
    ma        = _calc_ma(arr_daily['close'], REGIME_MA_WINDOW)
    if ma is None:
        return {'family': 'neutral', 'metrics': {}}

    # Fetch close from requested timeframe (or fall back to daily)
    tf = close_timeframe if close_timeframe and close_timeframe != REGIME_TIMEFRAME else REGIME_TIMEFRAME
    if tf != REGIME_TIMEFRAME:
        tf_data = fetch_ohlcv_data([sym], tf)
        df_tf   = tf_data.get(sym)
        if df_tf is None or df_tf.empty:
            tf = REGIME_TIMEFRAME
            close = float(arr_daily['close'][-1])
        else:
            arr_tf = df_to_arrays_live(normalize_live_ohlcv(df_tf))
            close  = float(arr_tf['close'][-1])
    else:
        close = float(arr_daily['close'][-1])

    family  = _classify_ma(close, ma)
    metrics = {'close': close, f'ma_{REGIME_MA_WINDOW}': ma}
    return {'family': family, 'metrics': metrics}


def get_regime_info_front(
    timeframe: str,
    symbol=None,
) -> Dict:
    # ── Single-symbol mode (header) ──────────────────────────────────────────
    if symbol is None:
        try:
            data = _get_regime_for_symbol(REGIME_REFERENCE_SYMBOL, timeframe)
            return {
                'success':   True,
                'timeframe': timeframe,
                'family':    data['family'],
                'metrics':   data['metrics'],
            }
        except Exception as e:
            logger.error(f"[REGIME_FRONT] Error for {REGIME_REFERENCE_SYMBOL}: {e}")
            return {
                'success':   False,
                'timeframe': timeframe,
                'family':    'neutral',
                'metrics':   {},
                'error':     str(e),
            }
    # ── Multi-symbol mode (symbols grid) ────────────────────────────────────
    results = {}
    for sym in symbol:
        try:
            results[sym] = _get_regime_for_symbol(sym, timeframe)
        except Exception as e:
            logger.error(f"[REGIME_FRONT] Error for {sym}: {e}")
            results[sym] = {'family': 'neutral', 'metrics': {}, 'error': str(e)}
    return {
        'success':   True,
        'timeframe': timeframe,
        'symbols':   results,
    }
