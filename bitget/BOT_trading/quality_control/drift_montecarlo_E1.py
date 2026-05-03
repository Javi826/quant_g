"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '07_reversal_short_1H': {
        'p5_winrate':  63.8,
        'p50_winrate': 63.8,
    },
    '10_parity_long_1H': {
        'p5_winrate':  67.8,
        'p50_winrate': 67.8,
    },
    '11_parity_short_1H': {
        'p5_winrate':  65.3,
        'p50_winrate': 65.3,
    },
    '18_flag_long_1H': {
        'p5_winrate':  66.1,
        'p50_winrate': 66.1,
    },
    '20_flag_short_1H': {
        'p5_winrate':  68.2,
        'p50_winrate': 68.2,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  63.2,
        'p50_winrate': 63.2,
    },
}
