"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '02_reversal_long_4H': {
        'p5_winrate':  50.7,
        'p50_winrate': 51.0,
    },
    '03_parity_long_4H': {
        'p5_winrate':  45.2,
        'p50_winrate': 47.1,
    },
    '06_reversal_long_1H': {
        'p5_winrate':  52.4,
        'p50_winrate': 54.1,
    },
    '07_reversal_short_1H': {
        'p5_winrate':  64.1,
        'p50_winrate': 66.7,
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate':  46.5,
        'p50_winrate': 51.5,
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate':  36.0,
        'p50_winrate': 39.6,
    },
    '10_parity_long_1H': {
        'p5_winrate':  65.0,
        'p50_winrate': 66.3,
    },
    '11_parity_short_1H': {
        'p5_winrate':  65.0,
        'p50_winrate': 66.4,
    },
    '17_flag_long_4H': {
        'p5_winrate':  46.6,
        'p50_winrate': 49.1,
    },
    '19_flag_short_4H': {
        'p5_winrate':  57.6,
        'p50_winrate': 60.1,
    },
    '20_flag_short_1H': {
        'p5_winrate':  65.7,
        'p50_winrate': 68.8,
    },
    '21_parity_short_4H': {
        'p5_winrate':  47.6,
        'p50_winrate': 47.6,
    },
    '23_flag_long_1H': {
        'p5_winrate':  58.4,
        'p50_winrate': 63.9,
    },
}
