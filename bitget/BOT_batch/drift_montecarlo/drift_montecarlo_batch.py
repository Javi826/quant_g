"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  38.7,
        'p50_winrate': 46.1,
    },
    '03_parity_long_4H': {
        'p5_winrate':  35.9,
        'p50_winrate': 44.4,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  46.7,
        'p50_winrate': 53.7,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  38.6,
        'p50_winrate': 45.9,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  66.8,
        'p50_winrate': 72.5,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  36.8,
        'p50_winrate': 46.1,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  39.2,
        'p50_winrate': 47.9,
    },
    '10_parity_long_1H': {
        'p5_winrate':  65.7,
        'p50_winrate': 70.8,
    },
    '11_parity_short_1H': {
        'p5_winrate':  55.7,
        'p50_winrate': 65.4,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  20.0,
        'p50_winrate': 42.1,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  38.2,
        'p50_winrate': 46.7,
    },
    '17_flag_long_4H': {
        'p5_winrate':  46.0,
        'p50_winrate': 55.7,
    },
    '19_flag_short_4H': {
        'p5_winrate':  53.5,
        'p50_winrate': 62.7,
    },
    '20_flag_short_1H': {
        'p5_winrate':  59.9,
        'p50_winrate': 65.5,
    },
}
