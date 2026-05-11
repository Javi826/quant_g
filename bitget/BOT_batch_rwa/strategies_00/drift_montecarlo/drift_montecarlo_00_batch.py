"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '34_reversal_short_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '38_parity_long_15m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
}
