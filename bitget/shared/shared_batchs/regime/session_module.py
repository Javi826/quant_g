# shared/shared_batchs/regime/session_module.py
"""
Session filter module — classifies signals by trading session (UTC hours).
Independent from regime_module. Same interface: load_session_bins / filter_signals_by_session.
"""
import os
import logging
import numpy as np
import pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

logger = logging.getLogger("shared_batch.regime.session_module")


# =============================================================================
# CONFIGURATION
# =============================================================================
SESSION_ENABLED = True

SESSION_BINS = {
    'asian_early':    (0,  3),   # 00:00 - 03:00 UTC
    'asian_late':     (3,  6),   # 03:00 - 06:00 UTC
    'european_early': (6,  10),  # 06:00 - 10:00 UTC
    'european_late':  (10, 14),  # 10:00 - 14:00 UTC
    'american_early': (14, 18),  # 14:00 - 18:00 UTC
    'american_late':  (18, 24),  # 18:00 - 00:00 UTC
}

ALL_SESSION_BINS = list(SESSION_BINS.keys())


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_signal_by_session(timestamp: pd.Timestamp) -> str:
    """Classify a signal timestamp into a session bin based on its UTC hour."""
    hour = timestamp.hour
    for bin_name, (start, end) in SESSION_BINS.items():
        if start <= hour < end:
            return bin_name
    return 'american_late'


# =============================================================================
# SIGNAL FILTERING
# =============================================================================

def filter_signals_by_session(
    signals:        np.ndarray,
    ts:             np.ndarray,
    bins_to_filter: set,
) -> np.ndarray:
    """
    Zero out signals that fall within session bins marked for filtering.
    Returns a filtered copy of the signals array.
    """
    if not bins_to_filter:
        return signals

    filtered    = signals.copy()
    signal_idxs = np.nonzero(signals)[0]

    for idx in signal_idxs:
        session = classify_signal_by_session(pd.Timestamp(ts[idx]))
        if session in bins_to_filter:
            filtered[idx] = 0

    return filtered


# =============================================================================
# LOAD SESSION BINS
# =============================================================================

def load_session_bins(bins_path: str, strategy_id: str) -> set:
    """
    Load precomputed session bins for a strategy from a generated session_bins_{SET}.py file.
    Returns empty set if SESSION_ENABLED is False, file not found, or strategy not present.
    """
    if not SESSION_ENABLED:
        return set()

    if not os.path.exists(bins_path):
        logger.warning(f"⚠️  session_bins file not found: {bins_path} — using empty bins.")
        return set()

    spec   = spec_from_file_location("session_bins", bins_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    bins   = getattr(module, "SESSION_BINS_MAP", {})
    return set(bins.get(strategy_id, set()))