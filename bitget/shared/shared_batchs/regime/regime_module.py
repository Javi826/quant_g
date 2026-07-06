#shared/shared_batchs/regime/regime_module.py
import os
import logging
from importlib.util import spec_from_file_location, module_from_spec
from shared_batch_regime.regime_core import load_ohlcv_raw
from shared_batch_regime.regime_core import precompute_indicators

logger = logging.getLogger("shared_batch.regime.regime_module")

# =============================================================================
# CONFIGURATION  (populated by load_config_from_bins)
# =============================================================================
REGIME_ENABLED  = None
INDICATOR_CFG:  dict = {}
# =============================================================================
# INDICATOR CACHE  (MA over daily close, keyed by symbol)
# =============================================================================
_indicator_cache: dict = {}

def load_config_from_bins(bins_path: str) -> None:
    """Load indicator config from a regime_bins file. Validates MA_TIMEFRAME against REGIME_TIMEFRAME."""
    global INDICATOR_CFG, _indicator_cache

    spec   = spec_from_file_location("regime_bins", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "INDICATOR_CFG"):
        raise ValueError("❌ regime_bins file must contain INDICATOR_CFG. Re-run regime_calibration.py to regenerate.")

    INDICATOR_CFG    = module.INDICATOR_CFG
    _indicator_cache = {}
    logger.debug(f"  [regime_module] config loaded — INDICATOR_CFG={INDICATOR_CFG}")

def _get_indicator_cache(symbol: str, data_folder: str) -> dict | None:
    if symbol not in _indicator_cache:
        df = load_ohlcv_raw(symbol, data_folder)
        if df.empty:
            return None
        _indicator_cache[symbol] = precompute_indicators(df, INDICATOR_CFG)
    return _indicator_cache[symbol]

# =============================================================================
# LOAD REGIME BINS
# =============================================================================

def load_regime_bins(bins_path: str, strategy_id: str) -> list[str]:

    if not os.path.exists(bins_path):
        logger.warning(f"regime_bins file not found: {bins_path} — defaulting to neutral.")
        return []
    spec   = spec_from_file_location("regime_bins", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    bins = getattr(module, "REGIME_BINS", {})

    return bins.get(strategy_id, [])