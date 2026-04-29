"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '07_reversal_short_1H': {
        'p5_winrate':  64.1,
        'p50_winrate': 66.7,
    },
    '10_parity_long_1H': {
        'p5_winrate':  65.0,
        'p50_winrate': 66.3,
    },
    '11_parity_short_1H': {
        'p5_winrate':  65.4,
        'p50_winrate': 66.4,
    },
    '23_flag_long_1H': {
        'p5_winrate':  58.4,
        'p50_winrate': 63.9,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  63.5,
        'p50_winrate': 66.0,
    },
}
