"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '10_parity_long_1H': {
        'p5_winrate':  59.7,
        'p50_winrate': 59.7,
    },
    '20_flag_short_1H': {
        'p5_winrate':  68.2,
        'p50_winrate': 68.2,
    },
}
