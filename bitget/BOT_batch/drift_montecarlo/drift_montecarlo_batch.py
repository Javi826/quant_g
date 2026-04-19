"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  56.3,
        'p50_winrate': 67.0,
    },
    '03_parity_long_4H': {
        'p5_winrate':  71.2,
        'p50_winrate': 77.4,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  67.4,
        'p50_winrate': 77.4,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  77.9,
        'p50_winrate': 81.5,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  71.7,
        'p50_winrate': 74.2,
    },
}
