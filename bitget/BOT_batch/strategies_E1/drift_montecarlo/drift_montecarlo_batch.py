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
        'p5_winrate':  71.0,
        'p50_winrate': 73.9,
    },
    '10_parity_long_1H': {
        'p5_winrate':  72.0,
        'p50_winrate': 73.4,
    },
    '11_parity_short_1H': {
        'p5_winrate':  65.4,
        'p50_winrate': 66.4,
    },
    '20_flag_short_1H': {
        'p5_winrate':  65.7,
        'p50_winrate': 68.8,
    },
}
