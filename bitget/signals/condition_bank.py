import numpy as np
from numba import njit

# =============================================================================
# CONFIG
# =============================================================================

RSI_PERIODS                = [7,21]
RSI_THRESHOLDS             = [30,70]

ADX_PERIODS                = [7,21]
ADX_THRESHOLDS             = [20,30]

MA_PERIODS                 = [50,100]
MOMENTUM_PERIODS           = [5,10,20]
HISTVOL_BASE_PERIODS       = [30]
HISTVOL_REGIME_SMA_PERIODS = [20,50]

ATR_BASE_PERIODS           = [14]
ATR_REGIME_SMA_PERIODS     = [10,50]

OBV_REGIME_SMA_PERIODS     = [10,50]

# =============================================================================
# RSI_PERIODS                = [7,14,21]      # los 3 aparecen
# RSI_THRESHOLDS             = [30,50,70]     # los 3 aparecen
# 
# ADX_PERIODS                = [7,14]         # solo 7 y 14 aparecen (21 no aparece en ninguna)
# ADX_THRESHOLDS             = [20,25,30]     # los 3 aparecen
# 
# MA_PERIODS                 = [20,50,100]    # los 3 aparecen (20 solo 1 vez, pero aparece)
# 
# MOMENTUM_PERIODS           = [5,10,20]      # los 3 aparecen
# 
# HISTVOL_BASE_PERIODS       = [10,30]        # ambos aparecen
# HISTVOL_REGIME_SMA_PERIODS = [20,50]        # ambos aparecen
# 
# ATR_BASE_PERIODS           = [14]           # solo aparece 14
# ATR_REGIME_SMA_PERIODS     = [10,50]        # ambos aparecen
# 
# OBV_REGIME_SMA_PERIODS     = [10,20,50]     # los 3 aparecen
# =============================================================================

# =============================================================================
# CORE INDICATORS
# =============================================================================
#OLD

@njit(cache=False)
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

@njit(cache=False)
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

@njit(cache=False)
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

@njit(cache=False)
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


@njit(cache=False)
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


@njit(cache=False)
def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    tr  = _true_range(high, low, close)
    n   = len(tr)
    out = np.full(n, np.nan)

    if window > n:
        return out

    seed = 0.0
    for i in range(window):
        seed += tr[i]
    out[window - 1] = seed / window

    for i in range(window, n):
        out[i] = (out[i - 1] * (window - 1) + tr[i]) / window

    return out

@njit(cache=False)
def _historical_volatility(close: np.ndarray, window: int) -> np.ndarray:
    n           = len(close)
    log_returns = np.full(n, np.nan)
    for i in range(1, n):
        log_returns[i] = np.log(close[i] / close[i - 1])

    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = np.std(log_returns[i - window + 1:i + 1])
    return out

@njit(cache=False)
def _ema(values: np.ndarray, period: int) -> np.ndarray:
    n     = len(values)
    out   = np.full(n, np.nan)
    alpha = 2.0 / (period + 1.0)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


@njit(cache=False)
def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.max(values[i - window + 1:i + 1])
    return out


@njit(cache=False)
def _rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    n   = len(values)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.min(values[i - window + 1:i + 1])
    return out


@njit(cache=False)
def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n   = len(close)
    out = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]
    return out

# =============================================================================
# INDICATOR REGISTRY
# =============================================================================

def _build_specs_threshold(entry):
    specs = []
    for period in entry["periods"]:
        for th in entry["thresholds"]:
            for op in entry["ops"]:
                specs.append({"type": entry["type"], "period": period, "op": op, "value": th})
    return specs


def _build_specs_own_value(entry):
    specs = []
    for value in entry["periods"]:
        for op in entry["ops"]:
            specs.append({"type": entry["type"], "op": op, "value": value})
    return specs


def _build_specs_two_periods(entry):
    specs = []
    for period in entry["periods"]:
        for sma_period in entry["sma_periods"]:
            for op in entry["ops"]:
                specs.append({"type": entry["type"], "period": period, "sma_period": sma_period, "op": op})
    return specs


INDICATOR_REGISTRY = [
    {
        "type": "rsi",
        "periods": RSI_PERIODS,
        "thresholds": RSI_THRESHOLDS,
        "ops": [">", "<"],
        "build_cache": lambda o, h, l, c, v, entry: {p: _rsi(c, p) for p in entry["periods"]},
        "build_specs": _build_specs_threshold,
        "evaluate": lambda bank, spec: (bank._cache["rsi"][spec["period"]] > spec["value"] if spec["op"] == ">" else bank._cache["rsi"][spec["period"]] < spec["value"]),
        "describe": lambda spec: f"RSI{spec['period']}{spec['op']}{spec['value']}",
    },
    {
        "type": "adx",
        "periods": ADX_PERIODS,
        "thresholds": ADX_THRESHOLDS,
        "ops": [">"],
        "build_cache": lambda o, h, l, c, v, entry: {p: _adx(h, l, c, p) for p in entry["periods"]},
        "build_specs": _build_specs_threshold,
        "evaluate": lambda bank, spec: bank._cache["adx"][spec["period"]] > spec["value"],
        "describe": lambda spec: f"ADX{spec['period']}{spec['op']}{spec['value']}",
    },
    {
        "type": "ma",
        "periods": MA_PERIODS,
        "ops": [">", "<"],
        "build_cache": lambda o, h, l, c, v, entry: {p: _sma(c, p) for p in entry["periods"]},
        "build_specs": _build_specs_own_value,
        "evaluate": lambda bank, spec: (bank.close > bank._cache["ma"][spec["value"]] if spec["op"] == ">" else bank.close < bank._cache["ma"][spec["value"]]),
        "describe": lambda spec: f"CLOSE{spec['op']}MA{spec['value']}",
    },
# =============================================================================
#     {
#         "type": "momentum",
#         "periods": MOMENTUM_PERIODS,
#         "ops": [">", "<"],
#         "build_cache": None,
#         "build_specs": _build_specs_own_value,
#         "evaluate": lambda bank, spec: bank._momentum_mask(spec["op"], spec["value"]),
#         "describe": lambda spec: f"CLOSE{spec['op']}CLOSE[-{spec['value']}]",
#     },
# =============================================================================
    {
        "type": "atr_regime",
        "periods": ATR_BASE_PERIODS,
        "sma_periods": ATR_REGIME_SMA_PERIODS,
        "ops": [">", "<"],
        "build_cache": lambda o, h, l, c, v, entry: {(ap, sp): (_atr(h, l, c, ap), _rolling_mean_skipnan(_atr(h, l, c, ap), sp)) for ap in entry["periods"] for sp in entry["sma_periods"]},
        "build_specs": _build_specs_two_periods,
        "evaluate": lambda bank, spec: (bank._cache["atr_regime"][(spec["period"], spec["sma_period"])][0] > bank._cache["atr_regime"][(spec["period"], spec["sma_period"])][1] if spec["op"] == ">" else bank._cache["atr_regime"][(spec["period"], spec["sma_period"])][0] < bank._cache["atr_regime"][(spec["period"], spec["sma_period"])][1]),
        "describe": lambda spec: f"ATR{spec['period']}{spec['op']}SMA_ATR{spec['sma_period']}",
    },
    {
        "type": "histvol_regime",
        "periods": HISTVOL_BASE_PERIODS,
        "sma_periods": HISTVOL_REGIME_SMA_PERIODS,
        "ops": [">", "<"],
        "build_cache": lambda o, h, l, c, v, entry: {(hp, sp): (_historical_volatility(c, hp), _rolling_mean_skipnan(_historical_volatility(c, hp), sp)) for hp in entry["periods"] for sp in entry["sma_periods"]},
        "build_specs": _build_specs_two_periods,
        "evaluate": lambda bank, spec: (bank._cache["histvol_regime"][(spec["period"], spec["sma_period"])][0] > bank._cache["histvol_regime"][(spec["period"], spec["sma_period"])][1] if spec["op"] == ">" else bank._cache["histvol_regime"][(spec["period"], spec["sma_period"])][0] < bank._cache["histvol_regime"][(spec["period"], spec["sma_period"])][1]),
        "describe": lambda spec: f"HISTVOL{spec['period']}{spec['op']}SMA_HISTVOL{spec['sma_period']}",
    },
    {
        "type": "obv_regime",
        "periods": OBV_REGIME_SMA_PERIODS,
        "ops": [">", "<"],
        "build_cache": lambda o, h, l, c, v, entry: {p: _sma(_obv(c, v), p) for p in entry["periods"]},
        "build_specs": _build_specs_own_value,
        "evaluate": lambda bank, spec: (bank.obv_raw > bank._cache["obv_regime"][spec["value"]] if spec["op"] == ">" else bank.obv_raw < bank._cache["obv_regime"][spec["value"]]),
        "describe": lambda spec: f"OBV{spec['op']}SMA_OBV{spec['value']}",
    },
]


class ConditionBank:

    _REGISTRY_BY_TYPE = {entry["type"]: entry for entry in INDICATOR_REGISTRY}

    def __init__(self, arr: dict):
        self.open   = np.ascontiguousarray(arr["open"],         dtype=np.float64)
        self.high   = np.ascontiguousarray(arr["high"],         dtype=np.float64)
        self.low    = np.ascontiguousarray(arr["low"],          dtype=np.float64)
        self.close  = np.ascontiguousarray(arr["close"],        dtype=np.float64)
        self.volume = np.ascontiguousarray(arr["volume_quote"], dtype=np.float64)
        self.n      = len(self.close)

        self.obv_raw = _obv(self.close, self.volume)

        self._cache = {}
        for entry in INDICATOR_REGISTRY:
            if entry["build_cache"] is not None:
                self._cache[entry["type"]] = entry["build_cache"](
                    self.open, self.high, self.low, self.close, self.volume, entry
                )

    def _momentum_mask(self, op: str, nbar: int) -> np.ndarray:
        result = np.zeros(self.n, dtype=bool)
        if op == ">":
            result[nbar:] = self.close[nbar:] > self.close[:-nbar]
        else:
            result[nbar:] = self.close[nbar:] < self.close[:-nbar]
        return result

    def build_condition_specs(self) -> list:
        specs = []
        for entry in INDICATOR_REGISTRY:
            specs.extend(entry["build_specs"](entry))
        return specs

    def evaluate(self, spec: dict) -> np.ndarray:
        entry = self._REGISTRY_BY_TYPE.get(spec["type"])
        if entry is None:
            raise ValueError(f"Unknown condition type: {spec['type']}")
        return entry["evaluate"](self, spec)

    def describe(self, spec: dict) -> str:
        entry = self._REGISTRY_BY_TYPE.get(spec["type"])
        if entry is None:
            raise ValueError(f"Unknown condition type: {spec['type']}")
        return entry["describe"](spec)