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
        'p5_winrate':  36.5,
        'p50_winrate': 44.8,
    },
    '16_ranging_short_6Hutc': {
        'p5_winrate':  28.5,
        'p50_winrate': 44.3,
    },
}
