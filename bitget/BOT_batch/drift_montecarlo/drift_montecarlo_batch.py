"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  54.3,
        'p50_winrate': 56.6,
    },
    '03_parity_long_4H': {
        'p5_winrate':  71.6,
        'p50_winrate': 78.0,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  70.6,
        'p50_winrate': 74.1,
    },
}
