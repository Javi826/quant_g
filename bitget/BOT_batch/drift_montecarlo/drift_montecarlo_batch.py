"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  79.0,
        'p50_winrate': 79.9,
    },
    '03_parity_long_4H': {
        'p5_winrate':  81.0,
        'p50_winrate': 82.5,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  73.0,
        'p50_winrate': 73.2,
    },
}
