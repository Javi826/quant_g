"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  50.7,
        'p50_winrate': 51.0,
    },
    '03_parity_long_4H': {
        'p5_winrate':  45.2,
        'p50_winrate': 47.1,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  48.5,
        'p50_winrate': 48.9,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  52.4,
        'p50_winrate': 54.1,
    },
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
    '13_orderblocks_short_4H': {
        'p5_winrate':  41.3,
        'p50_winrate': 42.8,
    },
    '17_flag_long_4H': {
        'p5_winrate':  44.7,
        'p50_winrate': 49.6,
    },
    '19_flag_short_4H': {
        'p5_winrate':  57.6,
        'p50_winrate': 60.1,
    },
    '20_flag_short_1H': {
        'p5_winrate':  67.9,
        'p50_winrate': 69.2,
    },
    '21_parity_short_4H': {
        'p5_winrate':  47.6,
        'p50_winrate': 47.6,
    },
    '23_flag_long_1H': {
        'p5_winrate':  58.4,
        'p50_winrate': 63.9,
    },
    '26_orderblocks_long_4H': {
        'p5_winrate':  40.8,
        'p50_winrate': 43.6,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  63.5,
        'p50_winrate': 66.0,
    },
    '28_orderblocks_long_1H': {
        'p5_winrate':  51.2,
        'p50_winrate': 54.4,
    },
}
