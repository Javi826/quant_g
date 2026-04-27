"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '07_reversal_short_1H': {
        'p5_winrate':  62.1,
        'p50_winrate': 64.9,
    },
    '10_parity_long_1H': {
        'p5_winrate':  63.4,
        'p50_winrate': 66.8,
    },
}