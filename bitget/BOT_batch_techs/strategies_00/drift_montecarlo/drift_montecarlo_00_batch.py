"""
Montecarlo OOS reference values for drift detection.
P5_WINRATE: Percentile 5 (floor - worst acceptable performance)
P50_WINRATE: Percentile 50 (median - expected performance)
These values come from Montecarlo simulations and represent the statistical
boundaries for strategy health evaluation.
"""
DRIFT_REFERENCE = {
    '30_reversal_long_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '31_reversal_long_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '32_reversal_long_15m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '33_reversal_short_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '34_reversal_short_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '36_parity_long_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '37_parity_long_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '38_parity_long_15m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '39_parity_short_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '40_parity_short_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '41_parity_short_15m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '42_flag_long_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '43_flag_long_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '44_flag_long_15m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '45_flag_short_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '46_flag_short_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '47_flag_short_15m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '48_orderblocks_long_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '49_orderblocks_long_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '50_orderblocks_long_15m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '51_orderblocks_short_1H': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '52_orderblocks_short_30m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
    '53_orderblocks_short_15m': {
        'p5_winrate':  0.0,
        'p50_winrate': 0.0,
    },
}
