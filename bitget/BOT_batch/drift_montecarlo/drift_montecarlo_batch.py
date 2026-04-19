"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  38.0,
        'p50_winrate': 45.7,
    },
    '03_parity_long_4H': {
        'p5_winrate':  35.1,
        'p50_winrate': 43.9,
    },
    '04_reversal_short_4H': {
        'p5_winrate':  38.6,
        'p50_winrate': 46.9,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  38.8,
        'p50_winrate': 45.7,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  60.7,
        'p50_winrate': 66.2,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  36.9,
        'p50_winrate': 45.9,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  39.2,
        'p50_winrate': 47.8,
    },
    '10_parity_long_1H': {
        'p5_winrate':  54.9,
        'p50_winrate': 60.5,
    },
    '11_parity_short_1H': {
        'p5_winrate':  53.5,
        'p50_winrate': 64.0,
    },
    '12_parity_long_6Hutc': {
        'p5_winrate':  35.4,
        'p50_winrate': 43.2,
    },
    '13_orderblocks_short_4H': {
        'p5_winrate':  38.1,
        'p50_winrate': 46.6,
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate':  39.3,
        'p50_winrate': 44.2,
    },
    '17_flag_long_4H': {
        'p5_winrate':  37.3,
        'p50_winrate': 48.2,
    },
    '19_flag_short_4H': {
        'p5_winrate':  52.3,
        'p50_winrate': 61.6,
    },
    '20_flag_short_1H': {
        'p5_winrate':  61.0,
        'p50_winrate': 66.1,
    },
}
