"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  35.5,
        'p50_winrate': 46.7,
    },
    '03_parity_long_4H': {
        'p5_winrate':  42.8,
        'p50_winrate': 51.1,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  39.1,
        'p50_winrate': 48.6,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  48.2,
        'p50_winrate': 55.2,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  70.5,
        'p50_winrate': 76.0,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  35.1,
        'p50_winrate': 45.8,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  38.2,
        'p50_winrate': 50.0,
    },
    '10_parity_long_1H': {
        'p5_winrate':  66.0,
        'p50_winrate': 72.0,
    },
    '11_parity_short_1H': {
        'p5_winrate':  64.3,
        'p50_winrate': 69.2,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  34.1,
        'p50_winrate': 42.5,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  37.5,
        'p50_winrate': 47.8,
    },
    '17_flag_long_4H': {
        'p5_winrate':  47.1,
        'p50_winrate': 56.6,
    },
    '19_flag_short_4H': {
        'p5_winrate':  53.7,
        'p50_winrate': 64.3,
    },
    '20_flag_short_1H': {
        'p5_winrate':  63.3,
        'p50_winrate': 68.3,
    },
}
