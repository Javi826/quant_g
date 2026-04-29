"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '03_parity_long_4H': {
        'p5_winrate':  44.9,
        'p50_winrate': 44.9,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  55.9,
        'p50_winrate': 55.9,
    },
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
        'p5_winrate':  70.1,
        'p50_winrate': 70.1,
    },
    '19_flag_short_4H': {
        'p5_winrate':  62.7,
        'p50_winrate': 62.7,
    },
    '20_flag_short_1H': {
        'p5_winrate':  67.7,
        'p50_winrate': 67.7,
    },
    '21_parity_short_4H': {
        'p5_winrate':  47.6,
        'p50_winrate': 47.6,
    },
    '22_parity_short_6Hutc': {
        'p5_winrate':  34.8,
        'p50_winrate': 34.8,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  63.2,
        'p50_winrate': 63.2,
    },
}
