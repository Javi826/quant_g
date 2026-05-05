"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  55.3,
        'p50_winrate': 55.3,
    },
    '03_parity_long_4H': {
        'p5_winrate':  47.3,
        'p50_winrate': 47.3,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  49.0,
        'p50_winrate': 49.0,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  55.9,
        'p50_winrate': 55.9,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  63.8,
        'p50_winrate': 63.8,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  57.0,
        'p50_winrate': 57.0,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  44.3,
        'p50_winrate': 44.3,
    },
    '10_parity_long_1H': {
        'p5_winrate':  59.7,
        'p50_winrate': 59.7,
    },
    '11_parity_short_1H': {
        'p5_winrate':  64.9,
        'p50_winrate': 64.9,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  48.7,
        'p50_winrate': 48.7,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  40.6,
        'p50_winrate': 40.6,
    },
    '17_flag_long_4H': {
        'p5_winrate':  66.0,
        'p50_winrate': 66.0,
    },
    '18_flag_long_1H': {
        'p5_winrate':  65.7,
        'p50_winrate': 65.7,
    },
    '19_flag_short_4H': {
        'p5_winrate':  58.6,
        'p50_winrate': 58.6,
    },
    '20_flag_short_1H': {
        'p5_winrate':  68.2,
        'p50_winrate': 68.2,
    },
    '21_parity_short_4H': {
        'p5_winrate':  47.9,
        'p50_winrate': 47.9,
    },
    '22_parity_short_6Hutc': {
        'p5_winrate':  40.6,
        'p50_winrate': 40.6,
    },
    '24_flag_long_6Hutc': {
        'p5_winrate':  65.4,
        'p50_winrate': 65.4,
    },
    '25_flag_short_6Hutc': {
        'p5_winrate':  48.2,
        'p50_winrate': 48.2,
    },
    '26_orderblocks_long_4H': {
        'p5_winrate':  47.6,
        'p50_winrate': 47.6,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  63.2,
        'p50_winrate': 63.2,
    },
    '28_orderblocks_long_1H': {
        'p5_winrate':  58.1,
        'p50_winrate': 58.1,
    },
}
