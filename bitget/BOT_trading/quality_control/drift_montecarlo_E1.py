"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  46.9,
        'p50_winrate': 50.7,
    },
    '03_parity_long_4H': {
        'p5_winrate':  43.7,
        'p50_winrate': 48.5,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  53.8,
        'p50_winrate': 57.3,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  66.1,
        'p50_winrate': 66.7,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  61.8,
        'p50_winrate': 63.6,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  52.4,
        'p50_winrate': 55.1,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  48.0,
        'p50_winrate': 50.8,
    },
    '10_parity_long_1H': {
        'p5_winrate':  64.8,
        'p50_winrate': 67.8,
    },
    '11_parity_short_1H': {
        'p5_winrate':  62.2,
        'p50_winrate': 65.1,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  55.9,
        'p50_winrate': 61.8,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  51.9,
        'p50_winrate': 55.7,
    },
    '17_flag_long_4H': {
        'p5_winrate':  47.8,
        'p50_winrate': 48.4,
    },
    '19_flag_short_4H': {
        'p5_winrate':  51.2,
        'p50_winrate': 54.4,
    },
    '20_flag_short_1H': {
        'p5_winrate':  61.9,
        'p50_winrate': 64.7,
    },
}
