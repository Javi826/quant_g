"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '10_parity_long_1H': {
        'p5_winrate':  32.0,
        'p50_winrate': 33.3,
    },
    '11_parity_short_1H': {
        'p5_winrate':  45.1,
        'p50_winrate': 45.4,
    },
    '23_flag_long_1H': {
        'p5_winrate':  40.8,
        'p50_winrate': 45.6,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  41.2,
        'p50_winrate': 41.7,
    },
}
