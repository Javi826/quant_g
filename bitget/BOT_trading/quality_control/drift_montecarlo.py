"""
Montecarlo OOS reference values for drift detection.

P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)

These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""

DRIFT_REFERENCE = {
    # Format: 'strategy_id': {'p5_winrate': X, 'p50_winrate': Y}
    
    '01_double_top_long_4H': {
        'p5_winrate': 61.0,
        'p50_winrate': 71.0
    },
    '02_reversal_long_4H': {
        'p5_winrate': 69.0,
        'p50_winrate': 78.0
    },
    '03_parity_long_4H': {
        'p5_winrate': 72.0,
        'p50_winrate': 80.0
    },
    '04_reversal_short_4H': {
        'p5_winrate': 74.0,
        'p50_winrate': 82.0
    },
    '05_parity_short_4H': {
        'p5_winrate': 74.0,
        'p50_winrate': 82.0
    },
    '06_reversal_long_1H': {
        'p5_winrate': 78.0,
        'p50_winrate': 84.0
    },
    '07_reversal_short_1H': {
        'p5_winrate': 75.0,
        'p50_winrate': 81.0
    },
    '08_reversal_long_6Hutc': {
        'p5_winrate': 64.0,
        'p50_winrate': 75.0
    },
    '09_reversal_short_6Hutc': {
        'p5_winrate': 64.0,
        'p50_winrate': 75.0
    },
    '10_parity_long_1H': {
        'p5_winrate': 77.0,
        'p50_winrate': 82.0
    },
    '11_parity_short_1H': {
        'p5_winrate': 78.0,
        'p50_winrate': 84.0
    },
    '12_parity_long_6Hutc': {
        'p5_winrate': 67.0,
        'p50_winrate': 77.0
    },
    '13_orderblocks_short_4H': {
        'p5_winrate': 70.0,
        'p50_winrate': 80.0
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate': 57.0,
        'p50_winrate': 67.0
    },
    '17_flag_long_4H': {
        'p5_winrate': 65.0,
        'p50_winrate': 75.0
    },
    '18_flag_long_1H': {
        'p5_winrate': 71.0,
        'p50_winrate': 78.0
    },
    '19_flag_short_4H': {
        'p5_winrate': 75.0,
        'p50_winrate': 80.0
    },
    '20_flag_short_1H': {
        'p5_winrate': 78.0,
        'p50_winrate': 83.0
    }
}