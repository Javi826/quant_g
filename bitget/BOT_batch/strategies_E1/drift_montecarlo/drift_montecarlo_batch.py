"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  26.2,
        'p50_winrate': 35.5,
    },
    '03_parity_long_4H': {
        'p5_winrate':  26.3,
        'p50_winrate': 32.8,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  29.9,
        'p50_winrate': 38.4,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  29.4,
        'p50_winrate': 35.8,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  42.7,
        'p50_winrate': 48.9,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  25.7,
        'p50_winrate': 34.0,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  31.0,
        'p50_winrate': 39.5,
    },
    '10_parity_long_1H': {
        'p5_winrate':  25.0,
        'p50_winrate': 33.3,
    },
    '11_parity_short_1H': {
        'p5_winrate':  41.7,
        'p50_winrate': 47.8,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  25.1,
        'p50_winrate': 31.9,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  28.8,
        'p50_winrate': 37.1,
    },
    '17_flag_long_4H': {
        'p5_winrate':  32.2,
        'p50_winrate': 43.0,
    },
    '19_flag_short_4H': {
        'p5_winrate':  30.7,
        'p50_winrate': 39.6,
    },
    '20_flag_short_1H': {
        'p5_winrate':  41.8,
        'p50_winrate': 47.6,
    },
}
