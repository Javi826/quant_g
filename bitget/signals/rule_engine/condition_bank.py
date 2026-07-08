#signals/rule_engine/condition_bank.py
import numpy as np
from numba import njit


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


class ConditionBank:

    DEFAULT_RSI_PERIOD     = 14
    DEFAULT_ADX_PERIOD     = 14
    DEFAULT_RSI_THRESHOLDS = [30, 50, 70]
    DEFAULT_ADX_THRESHOLDS = [20, 25]
    DEFAULT_MA_PERIODS     = [20, 50, 100]
    DEFAULT_MOMENTUM_N     = [5, 10]

    def __init__(
        self,
        arr: dict,
        rsi_period: int      = DEFAULT_RSI_PERIOD,
        adx_period: int      = DEFAULT_ADX_PERIOD,
        rsi_thresholds: list = None,
        adx_thresholds: list = None,
        ma_periods: list     = None,
        momentum_n: list     = None,
    ):
        rsi_thresholds = rsi_thresholds if rsi_thresholds is not None else self.DEFAULT_RSI_THRESHOLDS
        adx_thresholds = adx_thresholds if adx_thresholds is not None else self.DEFAULT_ADX_THRESHOLDS
        ma_periods     = ma_periods     if ma_periods     is not None else self.DEFAULT_MA_PERIODS
        momentum_n     = momentum_n     if momentum_n     is not None else self.DEFAULT_MOMENTUM_N

        self.open  = np.ascontiguousarray(arr["open"],  dtype=np.float64)
        self.high  = np.ascontiguousarray(arr["high"],  dtype=np.float64)
        self.low   = np.ascontiguousarray(arr["low"],   dtype=np.float64)
        self.close = np.ascontiguousarray(arr["close"], dtype=np.float64)
        self.n     = len(self.close)

        self.rsi_period     = rsi_period
        self.adx_period     = adx_period
        self.rsi_thresholds = rsi_thresholds
        self.adx_thresholds = adx_thresholds
        self.ma_periods     = ma_periods
        self.momentum_n     = momentum_n

        self._rsi_cache = _rsi(self.close, self.rsi_period)
        self._adx_cache = _adx(self.high, self.low, self.close, self.adx_period)
        self._ma_cache  = {period: _sma(self.close, period) for period in self.ma_periods}

    def build_condition_specs(self) -> list:
        specs = []

        for th in self.rsi_thresholds:
            specs.append({"type": "rsi", "op": ">", "value": th})
            specs.append({"type": "rsi", "op": "<", "value": th})

        for th in self.adx_thresholds:
            specs.append({"type": "adx", "op": ">", "value": th})

        for period in self.ma_periods:
            specs.append({"type": "ma", "op": ">", "value": period})
            specs.append({"type": "ma", "op": "<", "value": period})

        for nbar in self.momentum_n:
            specs.append({"type": "momentum", "op": ">", "value": nbar})
            specs.append({"type": "momentum", "op": "<", "value": nbar})

        return specs

    def evaluate(self, spec: dict) -> np.ndarray:
        ctype = spec["type"]
        op    = spec["op"]
        value = spec["value"]

        if ctype == "rsi":
            series = self._rsi_cache
            return series > value if op == ">" else series < value

        if ctype == "adx":
            series = self._adx_cache
            return series > value

        if ctype == "ma":
            series = self._ma_cache[value]
            return self.close > series if op == ">" else self.close < series

        if ctype == "momentum":
            nbar   = value
            result = np.zeros(self.n, dtype=bool)
            if op == ">":
                result[nbar:] = self.close[nbar:] > self.close[:-nbar]
            else:
                result[nbar:] = self.close[nbar:] < self.close[:-nbar]
            return result

        raise ValueError(f"Unknown condition type: {ctype}")

    def describe(self, spec: dict) -> str:
        ctype = spec["type"]
        op    = spec["op"]
        value = spec["value"]

        if ctype == "rsi":
            return f"RSI{self.rsi_period}{op}{value}"
        if ctype == "adx":
            return f"ADX{self.adx_period}{op}{value}"
        if ctype == "ma":
            return f"CLOSE{op}MA{value}"
        if ctype == "momentum":
            return f"CLOSE{op}CLOSE[-{value}]"

        raise ValueError(f"Unknown condition type: {ctype}")
