"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  67.0,
        'p50_winrate': 74.5,
    },
    '03_parity_long_4H': {
        'p5_winrate':  67.6,
        'p50_winrate': 76.6,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  74.5,
        'p50_winrate': 81.5,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  73.8,
        'p50_winrate': 79.7,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  74.5,
        'p50_winrate': 79.4,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  55.5,
        'p50_winrate': 68.8,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  65.4,
        'p50_winrate': 74.9,
    },
    '10_parity_long_1H': {
        'p5_winrate':  69.6,
        'p50_winrate': 76.3,
    },
    '11_parity_short_1H': {
        'p5_winrate':  74.4,
        'p50_winrate': 79.6,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  64.9,
        'p50_winrate': 73.7,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  69.9,
        'p50_winrate': 79.0,
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate':  57.3,
        'p50_winrate': 66.0,
    },
    '17_flag_long_4H': {
        'p5_winrate':  61.6,
        'p50_winrate': 71.2,
    },
    '19_flag_short_4H': {
        'p5_winrate':  75.1,
        'p50_winrate': 83.1,
    },
    '20_flag_short_1H': {
        'p5_winrate':  76.0,
        'p50_winrate': 80.6,
    },
}
