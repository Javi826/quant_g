"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '03_parity_long_4H': {
        'p5_winrate':  51.3,
        'p50_winrate': 52.1,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  58.7,
        'p50_winrate': 61.9,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  63.0,
        'p50_winrate': 64.2,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  62.5,
        'p50_winrate': 67.6,
    },
    '10_parity_long_1H': {
        'p5_winrate':  65.0,
        'p50_winrate': 66.3,
    },
    '11_parity_short_1H': {
        'p5_winrate':  65.0,
        'p50_winrate': 66.4,
    },
    '19_flag_short_4H': {
        'p5_winrate':  55.1,
        'p50_winrate': 56.1,
    },
    '20_flag_short_1H': {
        'p5_winrate':  61.9,
        'p50_winrate': 67.3,
    },
    '21_parity_short_4H': {
        'p5_winrate':  55.1,
        'p50_winrate': 56.8,
    },
    '23_flag_long_1H': {
        'p5_winrate':  60.4,
        'p50_winrate': 64.2,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  63.8,
        'p50_winrate': 66.1,
    },
    '28_orderblocks_long_1H': {
        'p5_winrate':  49.6,
        'p50_winrate': 53.3,
    },
}
