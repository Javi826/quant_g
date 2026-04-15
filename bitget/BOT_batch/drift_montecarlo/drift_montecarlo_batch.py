"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  63.5,
        'p50_winrate': 73.2,
    },
    '03_parity_long_4H': {
        'p5_winrate':  61.9,
        'p50_winrate': 72.8,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  71.8,
        'p50_winrate': 80.6,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  71.7,
        'p50_winrate': 78.2,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  71.7,
        'p50_winrate': 76.6,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  56.7,
        'p50_winrate': 69.1,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  65.0,
        'p50_winrate': 74.4,
    },
    '10_parity_long_1H': {
        'p5_winrate':  69.0,
        'p50_winrate': 75.2,
    },
    '11_parity_short_1H': {
        'p5_winrate':  71.5,
        'p50_winrate': 76.4,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  59.0,
        'p50_winrate': 71.8,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  67.0,
        'p50_winrate': 77.7,
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate':  54.4,
        'p50_winrate': 66.2,
    },
    '17_flag_long_4H': {
        'p5_winrate':  57.8,
        'p50_winrate': 68.5,
    },
    '19_flag_short_4H': {
        'p5_winrate':  73.7,
        'p50_winrate': 82.1,
    },
    '20_flag_short_1H': {
        'p5_winrate':  76.3,
        'p50_winrate': 80.6,
    },
}
