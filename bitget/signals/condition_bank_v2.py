import numpy as np
from numba import njit

# =============================================================================
# CONFIG
# =============================================================================

ATR_BASE_PERIOD            = 14
ATR_REGIME_SMA_PERIODS     = [10, 20, 50]

HISTVOL_BASE_PERIOD        = 20
HISTVOL_REGIME_SMA_PERIODS = [10, 20, 50]

OBV_REGIME_SMA_PERIODS     = [10, 20, 50]

AROON_PERIODS              = [14]
AROON_THRESHOLDS           = [-50, 0, 50]


# =============================================================================
# CORE INDICATORS
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
def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    tr  = _true_range(high, low, close)
    n   = len(tr)
    out = np.full(n, np.nan)

    seed = 0.0
    for i in range(window):
        seed += tr[i]
    out[window - 1] = seed / window

    for i in range(window, n):
        out[i] = (out[i - 1] * (window - 1) + tr[i]) / window

    return out


@njit(cache=True)
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


@njit(cache=True)
def _historical_volatility(close: np.ndarray, window: int) -> np.ndarray:
    n           = len(close)
    log_returns = np.full(n, np.nan)
    for i in range(1, n):
        log_returns[i] = np.log(close[i] / close[i - 1])

    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = np.std(log_returns[i - window + 1:i + 1])
    return out


@njit(cache=True)
def _aroon_oscillator(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    n   = len(high)
    out = np.full(n, np.nan)
    for i in range(window, n):
        idx_max = np.argmax(high[i - window:i + 1])
        idx_min = np.argmin(low[i - window:i + 1])
        aroon_up   = 100.0 * idx_max / window
        aroon_down = 100.0 * idx_min / window
        out[i]     = aroon_up - aroon_down
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


INDICATOR_REGISTRY = [
    {
        "type":        "aroon_oscillator",
        "periods":     AROON_PERIODS,
        "thresholds":  AROON_THRESHOLDS,
        "ops":         [">", "<"],
        "build_cache": lambda o, h, l, c, v, periods: {p: _aroon_oscillator(h, l, p) for p in periods},
        "build_specs": _build_specs_threshold,
        "evaluate":    lambda bank, spec: (
            bank._cache["aroon_oscillator"][spec["period"]] > spec["value"]
            if spec["op"] == ">"
            else bank._cache["aroon_oscillator"][spec["period"]] < spec["value"]
        ),
        "describe":    lambda spec: f"AROON_OSC{spec['period']}{spec['op']}{spec['value']}",
    },
    {
        "type":        "atr_regime",
        "periods":     ATR_REGIME_SMA_PERIODS,
        "ops":         [">", "<"],
        "build_cache": lambda o, h, l, c, v, periods: {p: _rolling_mean_skipnan(_atr(h, l, c, ATR_BASE_PERIOD), p) for p in periods},
        "build_specs": _build_specs_own_value,
        "evaluate":    lambda bank, spec: (
            bank.atr_raw > bank._cache["atr_regime"][spec["value"]]
            if spec["op"] == ">"
            else bank.atr_raw < bank._cache["atr_regime"][spec["value"]]
        ),
        "describe":    lambda spec: f"ATR{spec['op']}SMA_ATR{spec['value']}",
    },
    {
        "type":        "obv_regime",
        "periods":     OBV_REGIME_SMA_PERIODS,
        "ops":         [">", "<"],
        "build_cache": lambda o, h, l, c, v, periods: {p: _sma(_obv(c, v), p) for p in periods},
        "build_specs": _build_specs_own_value,
        "evaluate":    lambda bank, spec: (
            bank.obv_raw > bank._cache["obv_regime"][spec["value"]]
            if spec["op"] == ">"
            else bank.obv_raw < bank._cache["obv_regime"][spec["value"]]
        ),
        "describe":    lambda spec: f"OBV{spec['op']}SMA_OBV{spec['value']}",
    },
    {
        "type":        "histvol_regime",
        "periods":     HISTVOL_REGIME_SMA_PERIODS,
        "ops":         [">", "<"],
        "build_cache": lambda o, h, l, c, v, periods: {p: _rolling_mean_skipnan(_historical_volatility(c, HISTVOL_BASE_PERIOD), p) for p in periods},
        "build_specs": _build_specs_own_value,
        "evaluate":    lambda bank, spec: (
            bank.histvol_raw > bank._cache["histvol_regime"][spec["value"]]
            if spec["op"] == ">"
            else bank.histvol_raw < bank._cache["histvol_regime"][spec["value"]]
        ),
        "describe":    lambda spec: f"HISTVOL{spec['op']}SMA_HISTVOL{spec['value']}",
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

        self.atr_raw     = _atr(self.high, self.low, self.close, ATR_BASE_PERIOD)
        self.obv_raw     = _obv(self.close, self.volume)
        self.histvol_raw = _historical_volatility(self.close, HISTVOL_BASE_PERIOD)

        self._cache = {}
        for entry in INDICATOR_REGISTRY:
            if entry["build_cache"] is not None:
                self._cache[entry["type"]] = entry["build_cache"](
                    self.open, self.high, self.low, self.close, self.volume, entry["periods"]
                )

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