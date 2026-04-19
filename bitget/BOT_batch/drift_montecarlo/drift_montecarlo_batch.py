"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  69.3,
        'p50_winrate': 78.4,
    },
    '03_parity_long_4H': {
        'p5_winrate':  69.5,
        'p50_winrate': 78.6,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  66.2,
        'p50_winrate': 74.3,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  64.2,
        'p50_winrate': 72.5,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  75.5,
        'p50_winrate': 81.0,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  72.4,
        'p50_winrate': 80.7,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  70.2,
        'p50_winrate': 79.5,
    },
    '10_parity_long_1H': {
        'p5_winrate':  76.0,
        'p50_winrate': 81.5,
    },
    '11_parity_short_1H': {
        'p5_winrate':  73.6,
        'p50_winrate': 79.1,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  63.5,
        'p50_winrate': 74.5,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  65.4,
        'p50_winrate': 74.9,
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate':  64.3,
        'p50_winrate': 71.5,
    },
    '17_flag_long_4H': {
        'p5_winrate':  70.2,
        'p50_winrate': 80.0,
    },
    '19_flag_short_4H': {
        'p5_winrate':  65.8,
        'p50_winrate': 75.6,
    },
    '20_flag_short_1H': {
        'p5_winrate':  77.4,
        'p50_winrate': 81.7,
    },
}
