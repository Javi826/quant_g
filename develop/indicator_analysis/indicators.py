import numpy as np
from numba import njit

# =============================================================================
# CONFIG - all indicator parameters
# =============================================================================

RSI_PERIOD          = 14
ADX_PERIOD           = 14
SMA_PERIOD           = 20
MOMENTUM_PERIOD      = 10

MACD_FAST            = 12
MACD_SLOW            = 26
MACD_SIGNAL          = 9

BOLLINGER_WINDOW     = 20
BOLLINGER_STD        = 2.0

ATR_PERIOD            = 14

STOCH_WINDOW          = 14
STOCH_SMOOTH         = 3

CCI_WINDOW            = 20
WILLIAMS_WINDOW      = 14
ROC_WINDOW            = 10
CMF_WINDOW            = 20
HIST_VOL_WINDOW      = 20
EFF_RATIO_WINDOW     = 10

AROON_WINDOW          = 25
DONCHIAN_WINDOW      = 20
KELTNER_WINDOW       = 20
KELTNER_ATR_MULT     = 2.0
TRIX_PERIOD           = 15

ULTOSC_SHORT          = 7
ULTOSC_MID            = 14
ULTOSC_LONG           = 28

VWAP_WINDOW           = 20
FORCE_INDEX_PERIOD   = 13
MASS_INDEX_EMA       = 9
MASS_INDEX_WINDOW    = 25
DPO_PERIOD            = 20

KAMA_PERIOD           = 10
KAMA_FAST             = 2
KAMA_SLOW             = 30

VORTEX_WINDOW         = 14
COPPOCK_ROC1          = 14
COPPOCK_ROC2          = 11
COPPOCK_WMA           = 10
RVI_WINDOW            = 10
CMO_WINDOW            = 14
PSAR_STEP             = 0.02
PSAR_MAX              = 0.2

# =============================================================================
# THRESHOLDS - manual thresholds for binarizing each indicator (interaction analysis)
#
# NOTE: ATR_TH, OBV_TH, CHAIKIN_OSC_TH, and FORCE_INDEX_TH default to 0.0
# (sign-based split) because atr, obv, chaikin_oscillator, and force_index
# are NOT normalized by price/volume — their absolute scale differs across
# symbols, so a single fixed threshold isn't meaningful. Treat results for
# these four with caution, or use atr_pct instead of atr where possible.
# =============================================================================

RSI_TH                = 70.0
ADX_TH                = 25.0
SMA_DISTANCE_TH        = 0.0
MOMENTUM_TH            = 0.0
MACD_LINE_TH           = 0.0
MACD_SIGNAL_TH         = 0.0
MACD_HISTOGRAM_TH      = 0.0
ROC_TH                 = 0.0
EFF_RATIO_TH           = 0.3
AROON_UP_TH           = 70.0
AROON_DOWN_TH         = 70.0
AROON_OSC_TH          = 50.0
TRIX_TH                = 0.0
DPO_TH                 = 0.0
KAMA_DISTANCE_TH       = 0.0
COPPOCK_TH             = 0.0
CMO_TH                = 50.0
BOLLINGER_PCT_B_TH      = 0.8
BOLLINGER_BANDWIDTH_TH  = 0.04
ATR_TH                  = 0.0   # scale-dependent, see note above
ATR_PCT_TH              = 0.02
HIST_VOL_TH             = 0.02
DONCHIAN_PCT_B_TH       = 0.8
DONCHIAN_WIDTH_TH       = 0.05
KELTNER_PCT_B_TH        = 0.8
KELTNER_WIDTH_TH        = 0.03
PSAR_DISTANCE_TH        = 0.0
MASS_INDEX_TH          = 27.0
STOCH_K_TH             = 80.0
STOCH_D_TH             = 80.0
CCI_TH                = 100.0
WILLIAMS_R_TH         = -20.0
ULTOSC_TH              = 70.0
RVI_TH                  = 0.0
VORTEX_OSC_TH           = 0.0
OBV_TH                  = 0.0   # scale-dependent, see note above
CMF_TH                  = 0.05
VWAP_DISTANCE_TH        = 0.0
CHAIKIN_OSC_TH          = 0.0   # scale-dependent, see note above
FORCE_INDEX_TH          = 0.0   # scale-dependent, see note above


# =============================================================================
# CORE INDICATORS (replicated from condition_bank.py)
# =============================================================================

@njit(cache=True)
def _sma(close: np.ndarray, window: int) -> np.ndarray:
    n   = len(close)
    out = np.full(n, np.nan)
    if window > n:
        return out
    cumsum = 0.0
    for i in range(window):
        cumsum += close[i]
    out[window - 1] = cumsum / window
    for i in range(window, n):
        cumsum += close[i] - close[i - window]
        out[i] = cumsum / window
    return out


@njit(cache=True)
def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    n     = len(close)
    out   = np.full(n, np.nan)
    alpha = 1.0 / window
    emaup = 0.0
    emadn = 0.0

    for i in range(1, n):
        diff = close[i] - close[i - 1]
        up   = diff  if diff > 0 else 0.0
        dn   = -diff if diff < 0 else 0.0

        if i == 1:
            emaup = alpha * up
            emadn = alpha * dn
        else:
            emaup = alpha * up + (1 - alpha) * emaup
            emadn = alpha * dn + (1 - alpha) * emadn

        if i >= window - 1:
            if emadn == 0.0:
                out[i] = 100.0
            else:
                out[i] = 100.0 - (100.0 / (1.0 + emaup / emadn))

    return out


@njit(cache=True)
def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n            = len(close)
    adx_full     = np.zeros(n)

    k       = n - (window - 1)
    if k <= window + 1:
        return adx_full

    trs     = np.zeros(k)
    dip     = np.zeros(k)
    din     = np.zeros(k)
    diff_dm = np.zeros(n)
    pos     = np.zeros(n)
    neg     = np.zeros(n)

    for i in range(1, n):
        cs         = close[i - 1]
        pdm        = high[i] if high[i] > cs else cs
        pdn        = low[i]  if low[i]  < cs else cs
        diff_dm[i] = pdm - pdn

        diff_up   = high[i] - high[i - 1]
        diff_down = low[i - 1] - low[i]
        if diff_up > diff_down and diff_up > 0:
            pos[i] = diff_up
        if diff_down > diff_up and diff_down > 0:
            neg[i] = diff_down

    trs_s = 0.0
    dip_s = 0.0
    din_s = 0.0
    for i in range(1, window + 1):
        trs_s += diff_dm[i]
        dip_s += pos[i]
        din_s += neg[i]
    trs[0] = trs_s
    dip[0] = dip_s
    din[0] = din_s

    for i in range(1, k - 1):
        trs[i] = trs[i - 1] - trs[i - 1] / window + diff_dm[window + i]
        dip[i] = dip[i - 1] - dip[i - 1] / window + pos[window + i]
        din[i] = din[i - 1] - din[i - 1] / window + neg[window + i]

    dx = np.zeros(k)
    for i in range(k):
        if trs[i] != 0:
            di_p = 100.0 * dip[i] / trs[i]
            di_n = 100.0 * din[i] / trs[i]
        else:
            di_p = 0.0
            di_n = 0.0
        denom = di_p + di_n
        dx[i] = 100.0 * abs(di_p - di_n) / denom if denom != 0 else 0.0

    adx_smooth = np.zeros(k)
    dx_sum     = 0.0
    for i in range(window):
        dx_sum += dx[i]
    adx_smooth[window] = dx_sum / window

    for i in range(window + 1, k):
        adx_smooth[i] = (adx_smooth[i - 1] * (window - 1) + dx[i - 1]) / window

    prefix = window - 1
    for i in range(k):
        adx_full[prefix + i] = adx_smooth[i]

    return adx_full


@njit(cache=True)
def _momentum(close: np.ndarray, nbar: int) -> np.ndarray:
    n   = len(close)
    out = np.full(n, np.nan)
    out[nbar:] = close[nbar:] - close[:-nbar]
    return out


# =============================================================================
# ROLLING HELPERS
# =============================================================================

@njit(cache=True)
def _ema(values: np.ndarray, period: int) -> np.ndarray:
    n     = len(values)
    out   = np.full(n, np.nan)
    alpha = 2.0 / (period + 1.0)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.std(values[i - window + 1:i + 1])
    return out


@njit(cache=True)
def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.max(values[i - window + 1:i + 1])
    return out


@njit(cache=True)
def _rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.min(values[i - window + 1:i + 1])
    return out


@njit(cache=True)
def _rolling_mean_skipnan(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        total = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(values[j]):
                total += values[j]
                count += 1
        if count > 0:
            out[i] = total / count
    return out


@njit(cache=True)
def _rolling_mean_abs_dev(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_slice = values[i - window + 1:i + 1]
        mean_val     = np.mean(window_slice)
        out[i]       = np.mean(np.abs(window_slice - mean_val))
    return out


@njit(cache=True)
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n   = len(close)
    out = np.full(n, np.nan)
    out[0] = high[0] - low[0]
    for i in range(1, n):
        tr1    = high[i] - low[i]
        tr2    = abs(high[i] - close[i - 1])
        tr3    = abs(low[i] - close[i - 1])
        out[i] = max(tr1, tr2, tr3)
    return out


@njit(cache=True)
def _wma(values: np.ndarray, window: int) -> np.ndarray:
    n          = len(values)
    out        = np.full(n, np.nan)
    weight_sum = window * (window + 1) / 2.0
    for i in range(window - 1, n):
        total = 0.0
        valid = True
        for j in range(window):
            v = values[i - window + 1 + j]
            if np.isnan(v):
                valid = False
                break
            total += v * (j + 1)
        if valid:
            out[i] = total / weight_sum
    return out


# =============================================================================
# CORE INDICATOR WRAPPERS (arr -> np.ndarray convention)
# =============================================================================

def rsi_14(arr: dict) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    return _rsi(close, RSI_PERIOD)


def adx_14(arr: dict) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    return _adx(high, low, close, ADX_PERIOD)


def sma_distance_20(arr: dict) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    sma   = _sma(close, SMA_PERIOD)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (close - sma) / sma


def momentum_10(arr: dict) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    return _momentum(close, MOMENTUM_PERIOD)


# =============================================================================
# TREND / MOMENTUM INDICATORS
# =============================================================================

def macd_line(arr: dict) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    return _ema(close, MACD_FAST) - _ema(close, MACD_SLOW)


def macd_signal(arr: dict) -> np.ndarray:
    line = macd_line(arr)
    return _ema(line, MACD_SIGNAL)


def macd_histogram(arr: dict) -> np.ndarray:
    return macd_line(arr) - macd_signal(arr)


def roc(arr: dict, window: int = ROC_WINDOW) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    n     = len(close)
    out   = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window:] = (close[window:] - close[:-window]) / close[:-window] * 100.0
    return out


def efficiency_ratio(arr: dict, window: int = EFF_RATIO_WINDOW) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    n     = len(close)
    out   = np.full(n, np.nan)
    for i in range(window, n):
        net_change  = abs(close[i] - close[i - window])
        path_length = np.sum(np.abs(np.diff(close[i - window:i + 1])))
        out[i]      = net_change / path_length if path_length != 0 else 0.0
    return out


@njit(cache=True)
def _aroon_up(high: np.ndarray, window: int) -> np.ndarray:
    n   = len(high)
    out = np.full(n, np.nan)
    for i in range(window, n):
        idx_max        = np.argmax(high[i - window:i + 1])
        periods_since  = window - idx_max
        out[i]         = 100.0 * (window - periods_since) / window
    return out


@njit(cache=True)
def _aroon_down(low: np.ndarray, window: int) -> np.ndarray:
    n   = len(low)
    out = np.full(n, np.nan)
    for i in range(window, n):
        idx_min        = np.argmin(low[i - window:i + 1])
        periods_since  = window - idx_min
        out[i]         = 100.0 * (window - periods_since) / window
    return out


def aroon_up(arr: dict, window: int = AROON_WINDOW) -> np.ndarray:
    high = np.ascontiguousarray(arr["high"], dtype=np.float64)
    return _aroon_up(high, window)


def aroon_down(arr: dict, window: int = AROON_WINDOW) -> np.ndarray:
    low = np.ascontiguousarray(arr["low"], dtype=np.float64)
    return _aroon_down(low, window)


def aroon_oscillator(arr: dict, window: int = AROON_WINDOW) -> np.ndarray:
    return aroon_up(arr, window) - aroon_down(arr, window)


def trix(arr: dict, period: int = TRIX_PERIOD) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    ema3  = _ema(_ema(_ema(close, period), period), period)
    n     = len(close)
    out   = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100.0
    return out


def dpo(arr: dict, period: int = DPO_PERIOD) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    n     = len(close)
    shift = period // 2 + 1
    sma_val = _sma(close, period)
    out     = np.full(n, np.nan)
    out[shift:] = close[:n - shift] - sma_val[shift:]
    return out


@njit(cache=True)
def _kama(close: np.ndarray, period: int, fast: int, slow: int) -> np.ndarray:
    n        = len(close)
    out      = np.full(n, np.nan)
    fastest  = 2.0 / (fast + 1.0)
    slowest  = 2.0 / (slow + 1.0)

    out[period] = close[period]
    for i in range(period + 1, n):
        change     = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        er = change / volatility if volatility != 0 else 0.0
        sc = (er * (fastest - slowest) + slowest) ** 2
        out[i] = out[i - 1] + sc * (close[i] - out[i - 1])
    return out


def kama_distance(arr: dict, period: int = KAMA_PERIOD, fast: int = KAMA_FAST, slow: int = KAMA_SLOW) -> np.ndarray:
    close    = np.ascontiguousarray(arr["close"], dtype=np.float64)
    kama_val = _kama(close, period, fast, slow)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (close - kama_val) / kama_val


def coppock_curve(arr: dict, roc1: int = COPPOCK_ROC1, roc2: int = COPPOCK_ROC2, wma_window: int = COPPOCK_WMA) -> np.ndarray:
    summed = roc(arr, roc1) + roc(arr, roc2)
    return _wma(summed, wma_window)


@njit(cache=True)
def _cmo(close: np.ndarray, window: int) -> np.ndarray:
    n   = len(close)
    out = np.full(n, np.nan)
    for i in range(window, n):
        sum_up   = 0.0
        sum_down = 0.0
        for j in range(i - window + 1, i + 1):
            diff = close[j] - close[j - 1]
            if diff > 0:
                sum_up += diff
            else:
                sum_down += -diff
        denom  = sum_up + sum_down
        out[i] = 100.0 * (sum_up - sum_down) / denom if denom != 0 else 0.0
    return out


def cmo(arr: dict, window: int = CMO_WINDOW) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    return _cmo(close, window)


# =============================================================================
# VOLATILITY / CHANNEL INDICATORS
# =============================================================================

def bollinger_percent_b(arr: dict, window: int = BOLLINGER_WINDOW, n_std: float = BOLLINGER_STD) -> np.ndarray:
    close      = np.ascontiguousarray(arr["close"], dtype=np.float64)
    mid        = _sma(close, window)
    std        = _rolling_std(close, window)
    upper      = mid + n_std * std
    lower      = mid - n_std * std
    band_range = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        return (close - lower) / band_range


def bollinger_bandwidth(arr: dict, window: int = BOLLINGER_WINDOW, n_std: float = BOLLINGER_STD) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    mid   = _sma(close, window)
    std   = _rolling_std(close, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (2 * n_std * std) / mid


def atr(arr: dict, window: int = ATR_PERIOD) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    tr    = _true_range(high, low, close)
    return _ema(tr, window)


def atr_pct(arr: dict, window: int = ATR_PERIOD) -> np.ndarray:
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return atr(arr, window) / close


def historical_volatility(arr: dict, window: int = HIST_VOL_WINDOW) -> np.ndarray:
    close       = np.ascontiguousarray(arr["close"], dtype=np.float64)
    n           = len(close)
    log_returns = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns[1:] = np.log(close[1:] / close[:-1])
    return _rolling_std(log_returns, window)


def donchian_percent_b(arr: dict, window: int = DONCHIAN_WINDOW) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    highest_high = _rolling_max(high, window)
    lowest_low   = _rolling_min(low,  window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (close - lowest_low) / (highest_high - lowest_low)


def donchian_width(arr: dict, window: int = DONCHIAN_WINDOW) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    highest_high = _rolling_max(high, window)
    lowest_low   = _rolling_min(low,  window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (highest_high - lowest_low) / close


def keltner_percent_b(arr: dict, window: int = KELTNER_WINDOW, atr_mult: float = KELTNER_ATR_MULT) -> np.ndarray:
    close  = np.ascontiguousarray(arr["close"], dtype=np.float64)
    middle = _ema(close, window)
    atr_val = atr(arr, window)
    upper  = middle + atr_mult * atr_val
    lower  = middle - atr_mult * atr_val
    with np.errstate(divide="ignore", invalid="ignore"):
        return (close - lower) / (upper - lower)


def keltner_width(arr: dict, window: int = KELTNER_WINDOW, atr_mult: float = KELTNER_ATR_MULT) -> np.ndarray:
    close  = np.ascontiguousarray(arr["close"], dtype=np.float64)
    middle = _ema(close, window)
    atr_val = atr(arr, window)
    upper  = middle + atr_mult * atr_val
    lower  = middle - atr_mult * atr_val
    with np.errstate(divide="ignore", invalid="ignore"):
        return (upper - lower) / middle


@njit(cache=True)
def _parabolic_sar(high: np.ndarray, low: np.ndarray, step: float, max_step: float) -> np.ndarray:
    n        = len(high)
    sar      = np.full(n, np.nan)
    trend_up = True
    sar[0]   = low[0]
    ep       = high[0]
    af       = step

    for i in range(1, n):
        prev_sar = sar[i - 1]
        prev_low  = low[i - 2]  if i >= 2 else low[i - 1]
        prev_high = high[i - 2] if i >= 2 else high[i - 1]

        if trend_up:
            candidate = prev_sar + af * (ep - prev_sar)
            candidate = min(candidate, low[i - 1], prev_low)
            if low[i] < candidate:
                trend_up = False
                sar[i]   = ep
                ep       = low[i]
                af       = step
            else:
                sar[i] = candidate
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            candidate = prev_sar + af * (ep - prev_sar)
            candidate = max(candidate, high[i - 1], prev_high)
            if high[i] > candidate:
                trend_up = True
                sar[i]   = ep
                ep       = high[i]
                af       = step
            else:
                sar[i] = candidate
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)

    return sar


def parabolic_sar_distance(arr: dict, step: float = PSAR_STEP, max_step: float = PSAR_MAX) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    sar_val = _parabolic_sar(high, low, step, max_step)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (close - sar_val) / close


def mass_index(arr: dict, ema_period: int = MASS_INDEX_EMA, window: int = MASS_INDEX_WINDOW) -> np.ndarray:
    high     = np.ascontiguousarray(arr["high"], dtype=np.float64)
    low      = np.ascontiguousarray(arr["low"],  dtype=np.float64)
    hl_range = high - low
    ema1     = _ema(hl_range, ema_period)
    ema2     = _ema(ema1, ema_period)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ema1 / ema2
    return _sma(ratio, window) * window


# =============================================================================
# OSCILLATORS
# =============================================================================

def stochastic_k(arr: dict, window: int = STOCH_WINDOW) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    highest_high = _rolling_max(high, window)
    lowest_low   = _rolling_min(low,  window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (close - lowest_low) / (highest_high - lowest_low)


def stochastic_d(arr: dict, window: int = STOCH_WINDOW, smooth: int = STOCH_SMOOTH) -> np.ndarray:
    k = stochastic_k(arr, window)
    return _rolling_mean_skipnan(k, smooth)


def cci(arr: dict, window: int = CCI_WINDOW) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    typical_price = (high + low + close) / 3.0
    sma_tp        = _sma(typical_price, window)
    mean_dev      = _rolling_mean_abs_dev(typical_price, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (typical_price - sma_tp) / (0.015 * mean_dev)


def williams_r(arr: dict, window: int = WILLIAMS_WINDOW) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    highest_high = _rolling_max(high, window)
    lowest_low   = _rolling_min(low,  window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -100.0 * (highest_high - close) / (highest_high - lowest_low)


@njit(cache=True)
def _bp_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray):
    n  = len(close)
    bp = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    bp[0] = 0.0
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        prev_close     = close[i - 1]
        min_low_close  = low[i]  if low[i]  < prev_close else prev_close
        max_high_close = high[i] if high[i] > prev_close else prev_close
        bp[i] = close[i] - min_low_close
        tr[i] = max_high_close - min_low_close
    return bp, tr


def ultimate_oscillator(arr: dict, short: int = ULTOSC_SHORT, mid: int = ULTOSC_MID, long: int = ULTOSC_LONG) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    bp, tr = _bp_tr(high, low, close)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_short = _sma(bp, short) / _sma(tr, short)
        avg_mid   = _sma(bp, mid)   / _sma(tr, mid)
        avg_long  = _sma(bp, long)  / _sma(tr, long)
        return 100.0 * (4 * avg_short + 2 * avg_mid + avg_long) / 7.0


@njit(cache=True)
def _rvi_num_denom(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray):
    n   = len(close)
    num = np.full(n, np.nan)
    den = np.full(n, np.nan)
    for i in range(3, n):
        num[i] = ((close[i] - open_[i]) + 2 * (close[i - 1] - open_[i - 1]) +
                   2 * (close[i - 2] - open_[i - 2]) + (close[i - 3] - open_[i - 3])) / 6.0
        den[i] = ((high[i] - low[i]) + 2 * (high[i - 1] - low[i - 1]) +
                   2 * (high[i - 2] - low[i - 2]) + (high[i - 3] - low[i - 3])) / 6.0
    return num, den


def rvi(arr: dict, window: int = RVI_WINDOW) -> np.ndarray:
    open_ = np.ascontiguousarray(arr["open"],  dtype=np.float64)
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    num, den = _rvi_num_denom(open_, high, low, close)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _rolling_mean_skipnan(num, window) / _rolling_mean_skipnan(den, window)


@njit(cache=True)
def _vortex(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n        = len(close)
    vm_plus  = np.full(n, np.nan)
    vm_minus = np.full(n, np.nan)
    tr       = np.full(n, np.nan)
    vm_plus[0]  = 0.0
    vm_minus[0] = 0.0
    tr[0]       = high[0] - low[0]

    for i in range(1, n):
        vm_plus[i]  = abs(high[i] - low[i - 1])
        vm_minus[i] = abs(low[i] - high[i - 1])
        tr1    = high[i] - low[i]
        tr2    = abs(high[i] - close[i - 1])
        tr3    = abs(low[i] - close[i - 1])
        tr[i]  = max(tr1, max(tr2, tr3))

    sum_vm_plus  = _sma(vm_plus,  window) * window
    sum_vm_minus = _sma(vm_minus, window) * window
    sum_tr       = _sma(tr,       window) * window

    vi_plus  = sum_vm_plus  / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    return vi_plus - vi_minus


def vortex_oscillator(arr: dict, window: int = VORTEX_WINDOW) -> np.ndarray:
    high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
    low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
    close = np.ascontiguousarray(arr["close"], dtype=np.float64)
    return _vortex(high, low, close, window)


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

def obv(arr: dict) -> np.ndarray:
    close  = np.ascontiguousarray(arr["close"],        dtype=np.float64)
    volume = np.ascontiguousarray(arr["volume_quote"], dtype=np.float64)
    n      = len(close)
    out    = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out


def cmf(arr: dict, window: int = CMF_WINDOW) -> np.ndarray:
    high   = np.ascontiguousarray(arr["high"],         dtype=np.float64)
    low    = np.ascontiguousarray(arr["low"],          dtype=np.float64)
    close  = np.ascontiguousarray(arr["close"],        dtype=np.float64)
    volume = np.ascontiguousarray(arr["volume_quote"], dtype=np.float64)

    high_low_range = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        mf_multiplier = ((close - low) - (high - close)) / high_low_range
    mf_volume = mf_multiplier * volume

    sum_mf_volume = _sma(mf_volume, window) * window
    sum_volume    = _sma(volume,    window) * window

    with np.errstate(divide="ignore", invalid="ignore"):
        return sum_mf_volume / sum_volume


def vwap_distance(arr: dict, window: int = VWAP_WINDOW) -> np.ndarray:
    high   = np.ascontiguousarray(arr["high"],         dtype=np.float64)
    low    = np.ascontiguousarray(arr["low"],          dtype=np.float64)
    close  = np.ascontiguousarray(arr["close"],        dtype=np.float64)
    volume = np.ascontiguousarray(arr["volume_quote"], dtype=np.float64)

    typical_price = (high + low + close) / 3.0
    tp_vol        = typical_price * volume

    sum_tp_vol = _sma(tp_vol,  window) * window
    sum_vol    = _sma(volume,  window) * window

    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = sum_tp_vol / sum_vol
        return (close - vwap) / vwap


def chaikin_oscillator(arr: dict, fast: int = 3, slow: int = 10) -> np.ndarray:
    high   = np.ascontiguousarray(arr["high"],         dtype=np.float64)
    low    = np.ascontiguousarray(arr["low"],          dtype=np.float64)
    close  = np.ascontiguousarray(arr["close"],        dtype=np.float64)
    volume = np.ascontiguousarray(arr["volume_quote"], dtype=np.float64)

    high_low_range = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        mf_multiplier = ((close - low) - (high - close)) / high_low_range
    mf_volume = np.nan_to_num(mf_multiplier * volume, nan=0.0)
    adl       = np.cumsum(mf_volume)

    return _ema(adl, fast) - _ema(adl, slow)


def force_index(arr: dict, period: int = FORCE_INDEX_PERIOD) -> np.ndarray:
    close  = np.ascontiguousarray(arr["close"],        dtype=np.float64)
    volume = np.ascontiguousarray(arr["volume_quote"], dtype=np.float64)
    n      = len(close)
    fi     = np.zeros(n)
    fi[1:] = (close[1:] - close[:-1]) * volume[1:]
    return _ema(fi, period)


# =============================================================================
# CANDIDATE POOL (40 indicators)
# =============================================================================

CANDIDATE_INDICATORS = {
    "rsi_14":                rsi_14,
    "adx_14":                adx_14,
    "sma_distance_20":       sma_distance_20,
    "momentum_10":           momentum_10,
    "macd_line":             macd_line,
    "macd_signal":           macd_signal,
    "macd_histogram":        macd_histogram,
    "roc":                   roc,
    "efficiency_ratio":      efficiency_ratio,
    "aroon_up":              aroon_up,
    "aroon_down":            aroon_down,
    "aroon_oscillator":      aroon_oscillator,
    "trix":                  trix,
    "dpo":                   dpo,
    "kama_distance":         kama_distance,
    "coppock_curve":         coppock_curve,
    "cmo":                   cmo,
    "bollinger_percent_b":   bollinger_percent_b,
    "bollinger_bandwidth":   bollinger_bandwidth,
    "atr":                   atr,
    "atr_pct":                atr_pct,
    "historical_volatility": historical_volatility,
    "donchian_percent_b":    donchian_percent_b,
    "donchian_width":        donchian_width,
    "keltner_percent_b":     keltner_percent_b,
    "keltner_width":         keltner_width,
    "parabolic_sar_distance": parabolic_sar_distance,
    "mass_index":            mass_index,
    "stochastic_k":          stochastic_k,
    "stochastic_d":          stochastic_d,
    "cci":                   cci,
    "williams_r":            williams_r,
    "ultimate_oscillator":   ultimate_oscillator,
    "rvi":                   rvi,
    "vortex_oscillator":     vortex_oscillator,
    "obv":                   obv,
    "cmf":                   cmf,
    "vwap_distance":         vwap_distance,
    "chaikin_oscillator":    chaikin_oscillator,
    "force_index":           force_index,
}

INDICATOR_THRESHOLDS = {
    "rsi_14":                 RSI_TH,
    "adx_14":                 ADX_TH,
    "sma_distance_20":        SMA_DISTANCE_TH,
    "momentum_10":            MOMENTUM_TH,
    "macd_line":              MACD_LINE_TH,
    "macd_signal":            MACD_SIGNAL_TH,
    "macd_histogram":         MACD_HISTOGRAM_TH,
    "roc":                    ROC_TH,
    "efficiency_ratio":       EFF_RATIO_TH,
    "aroon_up":               AROON_UP_TH,
    "aroon_down":             AROON_DOWN_TH,
    "aroon_oscillator":       AROON_OSC_TH,
    "trix":                   TRIX_TH,
    "dpo":                    DPO_TH,
    "kama_distance":          KAMA_DISTANCE_TH,
    "coppock_curve":          COPPOCK_TH,
    "cmo":                    CMO_TH,
    "bollinger_percent_b":    BOLLINGER_PCT_B_TH,
    "bollinger_bandwidth":    BOLLINGER_BANDWIDTH_TH,
    "atr":                    ATR_TH,
    "atr_pct":                ATR_PCT_TH,
    "historical_volatility":  HIST_VOL_TH,
    "donchian_percent_b":     DONCHIAN_PCT_B_TH,
    "donchian_width":         DONCHIAN_WIDTH_TH,
    "keltner_percent_b":      KELTNER_PCT_B_TH,
    "keltner_width":          KELTNER_WIDTH_TH,
    "parabolic_sar_distance": PSAR_DISTANCE_TH,
    "mass_index":             MASS_INDEX_TH,
    "stochastic_k":           STOCH_K_TH,
    "stochastic_d":           STOCH_D_TH,
    "cci":                    CCI_TH,
    "williams_r":             WILLIAMS_R_TH,
    "ultimate_oscillator":    ULTOSC_TH,
    "rvi":                    RVI_TH,
    "vortex_oscillator":      VORTEX_OSC_TH,
    "obv":                    OBV_TH,
    "cmf":                    CMF_TH,
    "vwap_distance":          VWAP_DISTANCE_TH,
    "chaikin_oscillator":     CHAIKIN_OSC_TH,
    "force_index":            FORCE_INDEX_TH,
}