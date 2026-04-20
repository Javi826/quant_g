"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  40.9,
        'p50_winrate': 48.4,
    },
    '03_parity_long_4H': {
        'p5_winrate':  37.7,
        'p50_winrate': 44.4,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  48.7,
        'p50_winrate': 50.0,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  44.1,
        'p50_winrate': 51.6,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  53.9,
        'p50_winrate': 58.9,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  47.5,
        'p50_winrate': 53.8,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  44.1,
        'p50_winrate': 49.4,
    },
    '10_parity_long_1H': {
        'p5_winrate':  51.7,
        'p50_winrate': 55.4,
    },
    '11_parity_short_1H': {
        'p5_winrate':  59.6,
        'p50_winrate': 67.5,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  37.4,
        'p50_winrate': 50.0,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  48.1,
        'p50_winrate': 51.4,
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate':  43.3,
        'p50_winrate': 46.0,
    },
    '17_flag_long_4H': {
        'p5_winrate':  41.6,
        'p50_winrate': 50.2,
    },
    '19_flag_short_4H': {
        'p5_winrate':  39.0,
        'p50_winrate': 46.7,
    },
    '20_flag_short_1H': {
        'p5_winrate':  66.3,
        'p50_winrate': 67.9,
    },
}
