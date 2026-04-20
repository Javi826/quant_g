"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  37.7,
        'p50_winrate': 45.1,
    },
    '03_parity_long_4H': {
        'p5_winrate':  38.1,
        'p50_winrate': 45.8,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  46.7,
        'p50_winrate': 53.6,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  39.2,
        'p50_winrate': 46.3,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  65.4,
        'p50_winrate': 72.8,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  38.0,
        'p50_winrate': 47.3,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  39.4,
        'p50_winrate': 48.2,
    },
    '10_parity_long_1H': {
        'p5_winrate':  65.7,
        'p50_winrate': 70.7,
    },
    '11_parity_short_1H': {
        'p5_winrate':  56.7,
        'p50_winrate': 63.2,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  29.4,
        'p50_winrate': 45.8,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  37.0,
        'p50_winrate': 46.5,
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate':  37.1,
        'p50_winrate': 43.7,
    },
    '17_flag_long_4H': {
        'p5_winrate':  46.3,
        'p50_winrate': 54.7,
    },
    '19_flag_short_4H': {
        'p5_winrate':  55.5,
        'p50_winrate': 62.7,
    },
    '20_flag_short_1H': {
        'p5_winrate':  59.9,
        'p50_winrate': 65.0,
    },
}
