"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  59.1,
        'p50_winrate': 59.1,
    },
    '03_parity_long_4H': {
        'p5_winrate':  85.1,
        'p50_winrate': 85.1,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  70.2,
        'p50_winrate': 70.2,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  86.2,
        'p50_winrate': 86.2,
    },
}
