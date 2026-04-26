"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '06_reversal_long_1H': {
        'p5_winrate':  66.1,
        'p50_winrate': 66.7,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  61.8,
        'p50_winrate': 63.6,
    },
    '10_parity_long_1H': {
        'p5_winrate':  64.8,
        'p50_winrate': 67.8,
    },
    '20_flag_short_1H': {
        'p5_winrate':  61.9,
        'p50_winrate': 64.7,
    },
    '21_parity_short_4H': {
        'p5_winrate':  56.6,
        'p50_winrate': 59.0,
    },
    '23_flag_long_1H': {
        'p5_winrate':  65.1,
        'p50_winrate': 65.6,
    },
}
