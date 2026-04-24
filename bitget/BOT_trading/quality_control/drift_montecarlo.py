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
        'p50_winrate': 50.7,
    },
    '03_parity_long_4H': {
        'p5_winrate':  44.9,
        'p50_winrate': 44.9,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  48.4,
        'p50_winrate': 48.4,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  55.9,
        'p50_winrate': 55.9,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  70.7,
        'p50_winrate': 70.7,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  57.0,
        'p50_winrate': 57.0,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  43.6,
        'p50_winrate': 43.6,
    },
    '10_parity_long_1H': {
        'p5_winrate':  75.0,
        'p50_winrate': 75.0,
    },
    '11_parity_short_1H': {
        'p5_winrate':  65.3,
        'p50_winrate': 65.3,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  45.6,
        'p50_winrate': 45.6,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  41.1,
        'p50_winrate': 41.1,
    },
    '17_flag_long_4H': {
        'p5_winrate':  51.9,
        'p50_winrate': 51.9,
    },
    '19_flag_short_4H': {
        'p5_winrate':  68.6,
        'p50_winrate': 68.6,
    },
    '20_flag_short_1H': {
        'p5_winrate':  65.4,
        'p50_winrate': 65.4,
    },
}