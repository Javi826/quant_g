# shared/shared_batchs/regime/weekday_module.py
"""
Weekday filter module — classifies signals by day of week (UTC).
Independent from regime_module and session_module. Same interface: load_weekday_bins / filter_signals_by_weekday.
"""
import os
import logging
import numpy as np
import pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

logger = logging.getLogger("shared_batch.regime.weekday_module")


# =============================================================================
# CONFIGURATION
# =============================================================================
WEEKDAY_ENABLED = True

WEEKDAY_BINS = {
    'monday':    0,
    'tuesday':   1,
    'wednesday': 2,
    'thursday':  3,
    'friday':    4,
    'saturday':  5,
    'sunday':    6,
}

ALL_WEEKDAY_BINS = list(WEEKDAY_BINS.keys())


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_signal_by_weekday(timestamp: pd.Timestamp) -> str:
    """Classify a signal timestamp into a weekday bin based on its UTC day of week."""
    return ALL_WEEKDAY_BINS[timestamp.weekday()]


# =============================================================================
# SIGNAL FILTERING
# =============================================================================

def filter_signals_by_weekday(
    signals:        np.ndarray,
    ts:             np.ndarray,
    bins_to_filter: set,
) -> np.ndarray:
    """
    Zero out signals that fall on weekday bins marked for filtering.
    Returns a filtered copy of the signals array.
    """
    if not bins_to_filter:
        return signals

    filtered    = signals.copy()
    signal_idxs = np.nonzero(signals)[0]

    for idx in signal_idxs:
        weekday = classify_signal_by_weekday(pd.Timestamp(ts[idx]))
        if weekday in bins_to_filter:
            filtered[idx] = 0

    return filtered


# =============================================================================
# LOAD WEEKDAY BINS
# =============================================================================

def load_weekday_bins(bins_path: str, strategy_id: str) -> set:
    """
    Load precomputed weekday bins for a strategy from a generated weekday_bins_{SET}.py file.
    Returns empty set if WEEKDAY_ENABLED is False, file not found, or strategy not present.
    """
    if not WEEKDAY_ENABLED:
        return set()

    if not os.path.exists(bins_path):
        logger.warning(f"⚠️  weekday_bins file not found: {bins_path} — using empty bins.")
        return set()

    spec   = spec_from_file_location("weekday_bins", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    bins   = getattr(module, "WEEKDAY_BINS_MAP", {})
    return set(bins.get(strategy_id, set()))