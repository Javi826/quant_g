"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '03_parity_long_4H': {
        'p5_winrate':  32.8,
        'p50_winrate': 43.4,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  49.0,
        'p50_winrate': 55.1,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  64.4,
        'p50_winrate': 69.6,
    },
    '10_parity_long_1H': {
        'p5_winrate':  61.2,
        'p50_winrate': 65.9,
    },
    '11_parity_short_1H': {
        'p5_winrate':  64.1,
        'p50_winrate': 67.9,
    },
    '18_flag_long_1H': {
        'p5_winrate':  58.2,
        'p50_winrate': 63.7,
    },
    '19_flag_short_4H': {
        'p5_winrate':  46.5,
        'p50_winrate': 57.9,
    },
    '20_flag_short_1H': {
        'p5_winrate':  63.5,
        'p50_winrate': 68.4,
    },
    '21_parity_short_4H': {
        'p5_winrate':  36.5,
        'p50_winrate': 47.6,
    },
    '22_parity_short_6Hutc': {
        'p5_winrate':  40.8,
        'p50_winrate': 49.8,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  63.2,
        'p50_winrate': 68.7,
    },
}
