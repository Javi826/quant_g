"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '03_parity_long_4H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '10_parity_long_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '11_parity_short_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '17_flag_long_4H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '18_flag_long_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '19_flag_short_4H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '20_flag_short_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '21_parity_short_4H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '22_parity_short_6Hutc': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '24_flag_long_6Hutc': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '25_flag_short_6Hutc': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '26_orderblocks_long_4H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '27_orderblocks_short_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '28_orderblocks_long_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
}
