"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  63.2,
        'p50_winrate': 71.7,
    },
    '03_parity_long_4H': {
        'p5_winrate':  64.6,
        'p50_winrate': 72.2,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  73.0,
        'p50_winrate': 79.6,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  65.5,
        'p50_winrate': 73.2,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  69.2,
        'p50_winrate': 77.6,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  46.5,
        'p50_winrate': 63.1,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  61.3,
        'p50_winrate': 76.6,
    },
    '10_parity_long_1H': {
        'p5_winrate':  66.5,
        'p50_winrate': 69.8,
    },
    '11_parity_short_1H': {
        'p5_winrate':  72.2,
        'p50_winrate': 76.8,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  60.4,
        'p50_winrate': 71.0,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  69.0,
        'p50_winrate': 77.8,
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate':  57.3,
        'p50_winrate': 69.3,
    },
    '17_flag_long_4H': {
        'p5_winrate':  61.1,
        'p50_winrate': 69.3,
    },
    '19_flag_short_4H': {
        'p5_winrate':  76.1,
        'p50_winrate': 83.8,
    },
    '20_flag_short_1H': {
        'p5_winrate':  76.7,
        'p50_winrate': 82.1,
    },
}
