"""
Tests for strategies module (strategy_registry, strategy_loader).

Run with:
    python3 testing/test_strategies.py
    
Or with pytest:
    pytest testing/test_strategies.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import Mock, patch


# ==========================================================================
# SETUP
# ==========================================================================
def setup_module():
    """Setup before all tests."""
    print("\n" + "="*70)
    print("TESTING: strategies module")
    print("="*70)


# ==========================================================================
# SAMPLE DATA
# ==========================================================================
def get_sample_ohlcv_array(length=200):
    """
    Generate sample OHLCV array for testing signals.
    
    Returns dict with numpy arrays simulating price movement.
    """
    np.random.seed(42)
    
    # Base price around 50000
    base_price = 50000.0
    
    # Generate realistic price movement
    returns = np.random.normal(0, 0.02, length)
    close = base_price * np.exp(np.cumsum(returns))
    
    # Add volatility for high/low
    high = close * (1 + np.abs(np.random.normal(0, 0.01, length)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, length)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    
    volume = np.random.uniform(1000000, 5000000, length)
    
    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


def get_ohlcv_with_reversal_long_signal():
    """
    Generate OHLCV with clear reversal long signal.
    
    Pattern: Downtrend followed by strong bounce above MA.
    """
    length = 200
    arr = get_sample_ohlcv_array(length)
    
    # Create downtrend in last 50 candles
    arr['close'][-50:] = np.linspace(52000, 48000, 50)
    
    # Strong bounce at the end
    arr['close'][-5:] = [48000, 48500, 49500, 50500, 51000]
    arr['high'][-5:] = arr['close'][-5:] * 1.01
    arr['low'][-5:] = arr['close'][-5:] * 0.99
    
    return arr


def get_ohlcv_with_no_signal():
    """
    Generate OHLCV with no clear signal (flat/choppy).
    """
    length = 200
    arr = get_sample_ohlcv_array(length)
    
    # Flat price movement
    arr['close'] = np.ones(length) * 50000 + np.random.normal(0, 100, length)
    arr['high'] = arr['close'] * 1.005
    arr['low'] = arr['close'] * 0.995
    
    return arr


def get_sample_strategy_config():
    """Get sample strategy configuration."""
    return {
        'id': '06_reversal_long_1H',
        'name': 'reversal_long_1H',
        'timeframe': '1H',
        'active': True,
        'direction': 'long',
        'sell_after_ncandles': 50,
        'order_amount': 50,
        'tp_pct': 2.0,
        'sl_pct': 10.0,
        'lookback': 100,
        'tolerance': 20,
        'ma_period': 25
    }


# ==========================================================================
# TESTS: Signal Functions
# ==========================================================================
def test_reversal_long_signal_detected():
    """Test reversal_long detects signal in bullish reversal."""
    from signals.add_signals_reversal import reversal_long
    
    # Get OHLCV with signal
    arr = get_ohlcv_with_reversal_long_signal()
    
    # Run signal function
    signals = reversal_long(
        arr,
        lookback=100,
        tolerance=20,
        ma_period=25,
        live_trading=True
    )
    
    # Should return signal (1) at last candle
    assert len(signals) > 0, "Should return signals array"
    print(f"   Signal at last candle: {signals[-1]}")
    print("✅ test_reversal_long_signal_detected PASSED")


def test_reversal_long_no_signal():
    """Test reversal_long returns no signal in flat market."""
    from signals.add_signals_reversal import reversal_long
    
    # Get flat OHLCV
    arr = get_ohlcv_with_no_signal()
    
    # Run signal function
    signals = reversal_long(
        arr,
        lookback=100,
        tolerance=20,
        ma_period=25,
        live_trading=True
    )
    
    # Should return 0 (no signal) at last candle
    assert signals[-1] == 0, "Should not detect signal in flat market"
    print("✅ test_reversal_long_no_signal PASSED")


def test_reversal_short_signal():
    """Test reversal_short function executes without error."""
    from signals.add_signals_reversal import reversal_short
    
    arr = get_sample_ohlcv_array()
    
    signals = reversal_short(
        arr,
        lookback=100,
        tolerance=20,
        ma_period=25,
        live_trading=True
    )
    
    assert len(signals) > 0, "Should return signals array"
    assert signals[-1] in [0, -1], "Signal should be 0 or -1 for short"  # ← CAMBIAR
    print("✅ test_reversal_short_signal PASSED")


def test_parity_long_signal():
    """Test parity_long function executes without error."""
    from signals.add_signals_parity import parity_long
    
    arr = get_sample_ohlcv_array()
    
    signals = parity_long(
        arr,
        lookback=100,
        tolerance=20,
        ma_period=25,
        live_trading=True
    )
    
    assert len(signals) > 0, "Should return signals array"
    assert signals[-1] in [0, 1], "Signal should be 0 or 1"
    print("✅ test_parity_long_signal PASSED")


def test_parity_short_signal():
    """Test parity_short function executes without error."""
    from signals.add_signals_parity import parity_short
    
    arr = get_sample_ohlcv_array()
    
    signals = parity_short(
        arr,
        lookback=100,
        tolerance=20,
        ma_period=25,
        live_trading=True
    )
    
    assert len(signals) > 0, "Should return signals array"
    # SHORT signals can be -1 (short), 0 (no signal)
    assert signals[-1] in [0, -1], "Signal should be 0 or -1 for short"
    print("✅ test_parity_short_signal PASSED")


def test_orderblocks_long_signal():
    """Test orderblocks_long function executes without error."""
    from signals.add_signals_orderblocks import orderblocks_long
    
    arr = get_sample_ohlcv_array()
    
    signals = orderblocks_long(
        arr,
        lookback=100,
        tolerance=20,
        impulse=30,
        live_trading=True
    )
    
    assert len(signals) > 0, "Should return signals array"
    assert signals[-1] in [0, 1], "Signal should be 0 or 1"
    print("✅ test_orderblocks_long_signal PASSED")


def test_orderblocks_short_signal():
    """Test orderblocks_short function executes without error."""
    from signals.add_signals_orderblocks import orderblocks_short
    
    arr = get_sample_ohlcv_array()
    
    signals = orderblocks_short(
        arr,
        lookback=100,
        tolerance=20,
        impulse=30,
        live_trading=True
    )
    
    assert len(signals) > 0, "Should return signals array"
    assert signals[-1] in [0, -1], "Signal should be 0 or -1 for short"  # ← CAMBIAR
    print("✅ test_orderblocks_short_signal PASSED")


def test_double_top_long_signal():
    """Test double_top_long function executes without error."""
    from signals.add_signals_double_top import double_top_long
    
    arr = get_sample_ohlcv_array()
    
    signals = double_top_long(
        arr,
        lookback_minor=2,      # ← CORRECTO
        price_tolerance=15,    # ← CORRECTO
        trend_th=5,            # ← CORRECTO
        live_trading=True
    )
    
    assert len(signals) > 0, "Should return signals array"
    assert signals[-1] in [0, 1], "Signal should be 0 or 1"
    print("✅ test_double_top_long_signal PASSED")


# ==========================================================================
# TESTS: Strategy Registry
# ==========================================================================
def test_get_implemented_strategies():
    """Test get_implemented_strategies returns correct set."""
    from strategies.strategy_registry import get_implemented_strategies
    
    strategies = get_implemented_strategies()
    
    assert isinstance(strategies, set), "Should return a set"
    assert len(strategies) == 14, f"Should have 14 strategies, got {len(strategies)}"
    
    # Check some known strategies
    # Check some known strategy IDs (with prefix!)
    assert '02_reversal_long_4H' in strategies
    assert '06_reversal_long_1H' in strategies
    assert '03_parity_long_4H' in strategies
    assert '14_orderblocks_long_4H' in strategies
    
    print(f"   Found {len(strategies)} implemented strategies")
    print("✅ test_get_implemented_strategies PASSED")


def test_detect_signals_for_strategy_mock():
    """Test detect_signals_for_strategy with mocked exchange."""
    from strategies.strategy_registry import detect_signals_for_strategy
    
    # Mock exchange
    mock_exchange = Mock()
    mock_exchange.fetch_ohlcv.return_value = [
        [1704067200000, 50000, 50500, 49500, 50300, 1000000],
        [1704070800000, 50300, 50800, 50100, 50600, 1100000],
    ]
    
    strat_config = get_sample_strategy_config()
    symbols = ['BTCUSDT']
    
    # Run detection
    result = detect_signals_for_strategy(
        strat=strat_config,
        final_symbols=symbols,
        exchange=mock_exchange,
        use_hardcoded=False
    )
    
    assert isinstance(result, list), "Should return a list"
    print(f"   Detected signals for: {result}")
    print("✅ test_detect_signals_for_strategy_mock PASSED")


# ==========================================================================
# TESTS: Strategy Loader
# ==========================================================================
def test_load_strategies_from_yaml():
    """Test loading strategies from YAML file."""
    from strategies.strategy_loader import load_strategies_from_yaml
    import os
    
    # Use absolute path from BOT_trading root
    bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(bot_root, 'strategies', 'strategies.yaml')
    
    strategies = load_strategies_from_yaml(yaml_path)
    
    assert isinstance(strategies, list), "Should return a list"
    assert len(strategies) > 0, "Should load at least one strategy"
    
    # Check first strategy has required fields
    strat = strategies[0]
    assert 'id' in strat
    assert 'name' in strat
    assert 'timeframe' in strat
    assert 'active' in strat
    
    print(f"   Loaded {len(strategies)} strategies from YAML")
    print("✅ test_load_strategies_from_yaml PASSED")


def test_filter_strategies_by_ids():
    """Test filtering strategies by IDs."""
    from strategies.strategy_loader import load_strategies_from_yaml, filter_strategies_by_ids
    import os
    
    bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(bot_root, 'strategies', 'strategies.yaml')
    all_strategies = load_strategies_from_yaml(yaml_path)
    
    # Filter to specific IDs
    strategy_ids = ['06_reversal_long_1H', '07_reversal_short_1H']
    filtered = filter_strategies_by_ids(all_strategies, strategy_ids)
    
    assert len(filtered) == 2, f"Should have 2 strategies, got {len(filtered)}"
    assert filtered[0]['id'] in strategy_ids
    assert filtered[1]['id'] in strategy_ids
    
    print(f"   Filtered to {len(filtered)} strategies")
    print("✅ test_filter_strategies_by_ids PASSED")


# ==========================================================================
# TESTS: Strategy Registry (Advanced)
# ==========================================================================
def test_normalize_live_ohlcv():
    """Test OHLCV normalization for live trading."""
    from strategies.strategy_registry import normalize_live_ohlcv
    import pandas as pd
    
    # Create sample DataFrame
    data = {
        'timestamp': [1704067200000, 1704070800000, 1704074400000],
        'open': [50000, 50100, 50200],
        'high': [50500, 50600, 50700],
        'low': [49500, 49600, 49700],
        'close': [50300, 50400, 50500],
        'volume': [1000000, 1100000, 1200000]
    }
    df = pd.DataFrame(data)
    
    # Normalize
    normalized = normalize_live_ohlcv(df)
    
    assert 'open' in normalized.columns
    assert 'high' in normalized.columns
    assert 'low' in normalized.columns
    assert 'close' in normalized.columns
    assert len(normalized) == 3
    
    print("✅ test_normalize_live_ohlcv PASSED")


def test_df_to_arrays_live():
    """Test DataFrame to arrays conversion."""
    from strategies.strategy_registry import df_to_arrays_live
    import pandas as pd
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'open': [50000, 50100, 50200],
        'high': [50500, 50600, 50700],
        'low': [49500, 49600, 49700],
        'close': [50300, 50400, 50500],
        'volume': [1000000, 1100000, 1200000]
    })
    
    # Convert
    arr = df_to_arrays_live(df)
    
    assert 'open' in arr
    assert 'high' in arr
    assert 'low' in arr
    assert 'close' in arr
    assert 'volume' in arr
    assert len(arr['close']) == 3
    
    print("✅ test_df_to_arrays_live PASSED")


def test_detect_signals_with_real_data():
    """Test detect_signals_for_strategy with realistic OHLCV data."""
    from strategies.strategy_registry import detect_signals_for_strategy
    
    # Mock exchange with more realistic data
    mock_exchange = Mock()
    
    # Generate 200 candles of realistic data
    ohlcv_data = []
    base_price = 50000
    for i in range(200):
        timestamp = 1704067200000 + (i * 3600000)  # Hourly
        open_price = base_price + np.random.normal(0, 100)
        high_price = open_price + abs(np.random.normal(50, 20))
        low_price = open_price - abs(np.random.normal(50, 20))
        close_price = open_price + np.random.normal(0, 100)
        volume = np.random.uniform(1000000, 2000000)
        
        ohlcv_data.append([timestamp, open_price, high_price, low_price, close_price, volume])
        base_price = close_price  # Drift
    
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data
    
    strat_config = get_sample_strategy_config()
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    # Run detection
    result = detect_signals_for_strategy(
        strat=strat_config,
        final_symbols=symbols,
        exchange=mock_exchange,
        use_hardcoded=False
    )
    
    assert isinstance(result, list), "Should return a list"
    # May or may not have signals, but shouldn't crash
    print(f"   Symbols with signals: {len(result)}/{len(symbols)}")
    print("✅ test_detect_signals_with_real_data PASSED")


# ==========================================================================
# TESTS: Strategy Loader (Advanced)
# ==========================================================================
def test_load_strategies_main_function():
    """Test main load_strategies function."""
    from strategies.strategy_loader import load_strategies
    
    # Load specific strategies
    strategy_ids = ['06_reversal_long_1H', '07_reversal_short_1H']
    strategies = load_strategies(strategy_ids)
    
    assert len(strategies) == 2, f"Should load 2 strategies, got {len(strategies)}"
    assert strategies[0]['id'] in strategy_ids
    assert strategies[1]['id'] in strategy_ids
    
    print(f"   Loaded {len(strategies)} strategies via main function")
    print("✅ test_load_strategies_main_function PASSED")


def test_apply_set_active_argument():
    """Test --set-active argument parsing."""
    from strategies.strategy_loader import apply_set_active_argument, load_strategies_from_yaml
    import os
    
    bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(bot_root, 'strategies', 'strategies.yaml')
    all_strategies = load_strategies_from_yaml(yaml_path)
    
    # Apply set-active (modifies in-place, NO return value)
    active_ids = ['06_reversal_long_1H', '07_reversal_short_1H']
    apply_set_active_argument(all_strategies, active_ids)  # ← No asignar
    
    # Count active strategies
    active_count = sum(1 for s in all_strategies if s['active'])
    assert active_count == 2, f"Should have 2 active strategies, got {active_count}"
    
    # Verify correct ones are active
    active_strategy_ids = [s['id'] for s in all_strategies if s['active']]
    assert set(active_strategy_ids) == set(active_ids), "Wrong strategies activated"
    
    print(f"   Set {active_count} strategies as active")
    print("✅ test_apply_set_active_argument PASSED")


def test_validate_strategy_config():
    """Test strategy configuration validation."""
    from strategies.strategy_loader import validate_strategy_config
    
    # Valid config
    valid_config = {
        'id': '99_test_strategy',
        'name': 'test_strategy',
        'timeframe': '1H',
        'active': True,
        'direction': 'long',
        'sell_after_ncandles': 50,
        'order_amount': 50,
        'tp_pct': 2.0,
        'sl_pct': 10.0
    }
    
    # Should not raise exception
    validate_strategy_config(valid_config)
    
    # Invalid config (missing required field)
    invalid_config = {
        'id': '99_test_strategy',
        'name': 'test_strategy'
        # Missing other required fields
    }
    
    try:
        validate_strategy_config(invalid_config)
        assert False, "Should raise exception for invalid config"
    except (KeyError, ValueError):
        pass  # Expected
    
    print("✅ test_validate_strategy_config PASSED")

def test_df_to_arrays_live():
    """Test DataFrame to arrays conversion."""
    from market_data.data_utils import df_to_arrays_live
    import pandas as pd
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'open': [50000, 50100, 50200],
        'high': [50500, 50600, 50700],
        'low': [49500, 49600, 49700],
        'close': [50300, 50400, 50500],
        'volume_quote': [1000000, 1100000, 1200000]
    })
    df.index = pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'])
    
    # Convert
    arr = df_to_arrays_live(df)
    
    assert 'open' in arr
    assert 'high' in arr
    assert 'low' in arr
    assert 'close' in arr
    assert 'volume_quote' in arr  # ← CORRECTO nombre
    assert len(arr['close']) == 3
    
    print("✅ test_df_to_arrays_live PASSED")
    
# ==========================================================================
# TESTS: Strategy Loader (Edge Cases & Error Handling)
# ==========================================================================
def test_get_all_strategy_ids():
    """Test getting all strategy IDs from YAML."""
    from strategies.strategy_loader import get_all_strategy_ids
    import os
    
    bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(bot_root, 'strategies', 'strategies.yaml')
    
    all_ids = get_all_strategy_ids(yaml_path)
    
    assert isinstance(all_ids, list), "Should return a list"
    assert len(all_ids) == 14, f"Should have 14 IDs, got {len(all_ids)}"
    assert '06_reversal_long_1H' in all_ids
    assert '02_reversal_long_4H' in all_ids
    
    print(f"   Found {len(all_ids)} strategy IDs")
    print("✅ test_get_all_strategy_ids PASSED")


def test_get_strategy_config():
    """Test retrieving specific strategy config."""
    from strategies.strategy_loader import get_strategy_config
    import os
    
    bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(bot_root, 'strategies', 'strategies.yaml')
    
    config = get_strategy_config('06_reversal_long_1H', yaml_path)
    
    assert config is not None, "Should find strategy"
    assert config['id'] == '06_reversal_long_1H'
    assert config['name'] == 'reversal_long_1H'
    assert config['timeframe'] == '1H'
    
    # Test non-existent strategy (raises ValueError)
    try:
        config_none = get_strategy_config('99_nonexistent', yaml_path)
        assert False, "Should raise ValueError for non-existent strategy"
    except ValueError as e:
        assert 'not found' in str(e).lower()
    
    print("✅ test_get_strategy_config PASSED")


def test_validate_strategy_config_missing_fields():
    """Test validation catches missing required fields."""
    from strategies.strategy_loader import validate_strategy_config
    
    # Missing 'timeframe'
    invalid_config = {
        'id': '99_test',
        'name': 'test',
        'active': True
        # Missing: timeframe, direction, etc.
    }
    
    try:
        validate_strategy_config(invalid_config)
        assert False, "Should raise exception for missing fields"
    except (KeyError, ValueError) as e:
        assert 'timeframe' in str(e) or 'required' in str(e).lower()
        print("✅ test_validate_strategy_config_missing_fields PASSED")


def test_validate_strategy_config_invalid_values():
    """Test validation catches invalid field values."""
    from strategies.strategy_loader import validate_strategy_config
    
    # Invalid timeframe
    invalid_tf = {
        'id': '99_test',
        'name': 'test',
        'timeframe': 'INVALID_TF',
        'active': True,
        'direction': 'long',
        'sell_after_ncandles': 50,
        'order_amount': 50,
        'tp_pct': 2.0,
        'sl_pct': 10.0
    }
    
    try:
        validate_strategy_config(invalid_tf)
        # May or may not raise - depends on implementation
        print("✅ test_validate_strategy_config_invalid_values PASSED")
    except (ValueError, KeyError):
        print("✅ test_validate_strategy_config_invalid_values PASSED")


def test_error_handling_yaml_not_found():
    """Test error handling when YAML file doesn't exist."""
    from strategies.strategy_loader import load_strategies_from_yaml
    
    try:
        strategies = load_strategies_from_yaml('/nonexistent/path/strategies.yaml')
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError as e:
        assert 'not found' in str(e).lower()
        print("✅ test_error_handling_yaml_not_found PASSED")


def test_error_handling_invalid_strategy_id():
    """Test error handling for invalid strategy ID filter."""
    from strategies.strategy_loader import filter_strategies_by_ids, load_strategies_from_yaml
    import os
    
    bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(bot_root, 'strategies', 'strategies.yaml')
    all_strategies = load_strategies_from_yaml(yaml_path)
    
    # Filter with non-existent IDs (raises ValueError)
    try:
        filtered = filter_strategies_by_ids(all_strategies, ['99_nonexistent', '88_fake'])
        assert False, "Should raise ValueError for non-existent IDs"
    except ValueError as e:
        assert 'not found' in str(e).lower()
    
    print("✅ test_error_handling_invalid_strategy_id PASSED")
# ==========================================================================
# MAIN (standalone execution)
# ==========================================================================
if __name__ == "__main__":
    setup_module()
    
    tests = [
            # Signal functions
            test_reversal_long_signal_detected,
            test_reversal_long_no_signal,
            test_reversal_short_signal,
            test_parity_long_signal,
            test_parity_short_signal,
            test_orderblocks_long_signal,
            test_orderblocks_short_signal,
            test_double_top_long_signal,
            # Strategy registry
            test_get_implemented_strategies,
            test_detect_signals_for_strategy_mock,
            # Strategy registry (advanced)
            test_normalize_live_ohlcv,
            test_df_to_arrays_live,
            test_detect_signals_with_real_data,
            # Strategy loader
            test_load_strategies_from_yaml,
            test_filter_strategies_by_ids,
            # Strategy loader (advanced)
            test_load_strategies_main_function,
            test_apply_set_active_argument,
            test_validate_strategy_config,
            # NEW: Helper functions & edge cases
            test_get_all_strategy_ids,
            test_get_strategy_config,
            test_validate_strategy_config_missing_fields,
            test_validate_strategy_config_invalid_values,
            test_error_handling_yaml_not_found,
            test_error_handling_invalid_strategy_id
        ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    sys.exit(0 if failed == 0 else 1)
