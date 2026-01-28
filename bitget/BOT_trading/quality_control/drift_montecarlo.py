"""
Montecarlo OOS reference values for drift detection.

P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)

These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""

DRIFT_REFERENCE = {
    '01_example_strategy': {
        'p5_winrate': 52.0,
        'p50_winrate': 60.0
    },
    '02_another_strategy': {
        'p5_winrate': 48.0,
        'p50_winrate': 58.0
    },
    '03_third_strategy': {
        'p5_winrate': 55.0,
        'p50_winrate': 63.0
    }
    # TODO: Add all 18 strategies with real Montecarlo OOS values
}