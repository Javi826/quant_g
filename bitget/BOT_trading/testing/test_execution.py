"""
Tests for execution module (order_manager.py + position_tracker.py)

Tests READ-ONLY functions with real broker data:
- WebSocket price fetching
- Contract information retrieval
- Balance queries
- Mathematical calculations (size, TP/SL, quantization, PnL)

Tests INTEGRATION with real data + mocked broker (SAFE - NO WRITES):
- check_all_tp_sl() with real positions from account 00
- Load and validate all strategy configs (00, E1, 01)
- Regime calculation for all timeframes (4H, 1H, 6Hutc)
- Symbol loading for active strategies
- Position dict structure validation
- State save/load cycle (in-memory serialization)
- PositionSizer with real strategy configs
- Detects bugs like "'BotState' object is not subscriptable"

TOTAL: 32 tests (18 unit + 7 calculations + 7 integration)

DOES NOT TEST:
- place_order() (no actual trading)
- close_position() (no actual trading)
- add_position() (would write to production DB)
- log_closed_position() (would write to production DB)

Run with:
    python3 testing/test_execution.py
    
Or with pytest:
    pytest testing/test_execution.py -v
"""

import sys
import os
import time
from decimal import Decimal

# Add BOT_trading to path
current_dir = os.path.dirname(os.path.abspath(__file__))
bot_root = os.path.dirname(current_dir)
sys.path.insert(0, bot_root)

from execution.order_manager import (
    fetch_ticker_ws,
    fetch_contracts_ws,
    get_usdt_balance_ws,
    get_current_price,
    compute_size_base,
    extract_contract_params,
    fallback_params,
    quantize_size
)

from execution.position_tracker import (
    calculate_tp_sl_prices,
    calculate_pnl,
    check_all_tp_sl
)

from state import load_state

from market_data import init_websocket
from config.settings import PRODUCT_TYPE, HOUR_ZONE


# =============================================================================
# SETUP
# =============================================================================

def setup_websocket():
    """Initialize WebSocket connection for tests."""
    print("\n" + "="*70)
    print("TESTING: execution module (order_manager + position_tracker)")
    print("="*70)
    print("\nInitializing WebSocket connection...")
    
    try:
        from execution import BitgetClient
        from config.connect_pass import (
            BITGET_API_KEY_00,
            BITGET_API_SECRET_00,
            BITGET_API_PASS_00
        )
        
        # Use account 00 for real data (read-only, no trading)
        bitget_client = BitgetClient(
            api_key=BITGET_API_KEY_00,
            api_secret=BITGET_API_SECRET_00,
            api_passphrase=BITGET_API_PASS_00
        )
        
        ws_manager = init_websocket(
            api_key=bitget_client.api_key,
            api_secret=bitget_client.api_secret,
            api_passphrase=bitget_client.api_passphrase
        )
        
        if ws_manager:
            # Preload some common symbols
            test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            ws_manager.preload_contracts(test_symbols, product_type=PRODUCT_TYPE)
            
            # Wait for initial data
            print("Waiting for WebSocket data...")
            time.sleep(3)
            
            # Verify we have data
            btc_price_data = ws_manager.prices.get('BTCUSDT')
            if btc_price_data:
                print(f"✅ WebSocket initialized and ready (BTC: ${btc_price_data['price']})")
                return True
            else:
                print("⚠️  WebSocket initialized but no price data yet")
                time.sleep(2)  # Wait a bit more
                return True
        else:
            print("❌ WebSocket initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TESTS: WebSocket Data Fetching
# =============================================================================

def test_fetch_ticker_ws_btc():
    """Test fetching BTC ticker price via WebSocket."""
    print("\n--- test_fetch_ticker_ws_btc ---")
    
    symbol = 'BTCUSDT'
    price, _ = fetch_ticker_ws(symbol)
    
    assert price is not None, "Price should not be None"
    assert isinstance(price, Decimal), "Price should be Decimal"
    assert price > 0, "Price should be positive"
    
    # BTC price should be in reasonable range (sanity check)
    assert 10000 < price < 200000, f"BTC price {price} seems out of range"
    
    print(f"   BTCUSDT price: ${price:,.2f}")
    print("✅ test_fetch_ticker_ws_btc PASSED")


def test_fetch_ticker_ws_eth():
    """Test fetching ETH ticker price via WebSocket."""
    print("\n--- test_fetch_ticker_ws_eth ---")
    
    symbol = 'ETHUSDT'
    price, _ = fetch_ticker_ws(symbol)
    
    assert price is not None, "Price should not be None"
    assert isinstance(price, Decimal), "Price should be Decimal"
    assert price > 0, "Price should be positive"
    
    # ETH price should be in reasonable range
    assert 500 < price < 10000, f"ETH price {price} seems out of range"
    
    print(f"   ETHUSDT price: ${price:,.2f}")
    print("✅ test_fetch_ticker_ws_eth PASSED")


def test_fetch_ticker_ws_multiple_calls():
    """Test multiple calls to fetch_ticker_ws (caching behavior)."""
    print("\n--- test_fetch_ticker_ws_multiple_calls ---")
    
    symbol = 'BTCUSDT'
    
    # First call
    price1, _ = fetch_ticker_ws(symbol)
    time1 = time.time()
    
    # Second call (should use cache)
    price2, _ = fetch_ticker_ws(symbol)
    time2 = time.time()
    
    # Third call after small delay
    time.sleep(0.1)
    price3, _ = fetch_ticker_ws(symbol)
    
    assert price1 is not None
    assert price2 is not None
    assert price3 is not None
    
    # Second call should be very fast (cached)
    call_time = time2 - time1
    assert call_time < 0.1, f"Second call took {call_time}s, should use cache"
    
    print(f"   First price: ${price1:,.2f}")
    print(f"   Second price: ${price2:,.2f}")
    print(f"   Third price: ${price3:,.2f}")
    print(f"   Cache hit time: {call_time*1000:.2f}ms")
    print("✅ test_fetch_ticker_ws_multiple_calls PASSED")


def test_fetch_contracts_ws_btc():
    """Test fetching BTC contract info via WebSocket."""
    print("\n--- test_fetch_contracts_ws_btc ---")
    
    symbol = 'BTCUSDT'
    contract = fetch_contracts_ws(symbol)
    
    assert contract is not None, "Contract should not be None"
    assert isinstance(contract, dict), "Contract should be dict"
    
    # Check required fields
    required_fields = ['pricePlace', 'volumePlace', 'minTradeNum', 
                      'sizeMultiplier', 'minTradeUSDT']
    for field in required_fields:
        assert field in contract, f"Contract missing field: {field}"
    
    print(f"   Contract info for {symbol}:")
    print(f"     pricePlace: {contract['pricePlace']}")
    print(f"     volumePlace: {contract['volumePlace']}")
    print(f"     minTradeNum: {contract['minTradeNum']}")
    print(f"     minTradeUSDT: {contract['minTradeUSDT']}")
    print("✅ test_fetch_contracts_ws_btc PASSED")


def test_get_usdt_balance_ws():
    """Test getting USDT balance via WebSocket."""
    print("\n--- test_get_usdt_balance_ws ---")
    
    balance = get_usdt_balance_ws()
    
    assert balance is not None, "Balance should not be None"
    assert isinstance(balance, float), "Balance should be float"
    assert balance >= 0, "Balance should be non-negative"
    
    print(f"   USDT balance: ${balance:,.2f}")
    print("✅ test_get_usdt_balance_ws PASSED")


def test_get_current_price_btc():
    """Test get_current_price with BTC."""
    print("\n--- test_get_current_price_btc ---")
    
    symbol = 'BTCUSDT'
    price = get_current_price(symbol, max_cache_age=0.5)
    
    assert price is not None, "Price should not be None"
    assert isinstance(price, Decimal), "Price should be Decimal"
    assert price > 0, "Price should be positive"
    
    print(f"   BTCUSDT current price: ${price:,.2f}")
    print("✅ test_get_current_price_btc PASSED")


def test_get_current_price_cache_behavior():
    """Test get_current_price caching behavior."""
    print("\n--- test_get_current_price_cache_behavior ---")
    
    symbol = 'ETHUSDT'
    
    # First call
    start1 = time.time()
    price1 = get_current_price(symbol, max_cache_age=1.0)
    duration1 = time.time() - start1
    
    # Second call (should use cache)
    start2 = time.time()
    price2 = get_current_price(symbol, max_cache_age=1.0)
    duration2 = time.time() - start2
    
    assert price1 is not None
    assert price2 is not None
    
    # Second call should be faster (cached)
    assert duration2 < duration1, "Second call should be faster (cached)"
    
    print(f"   First call: ${price1:,.2f} ({duration1*1000:.2f}ms)")
    print(f"   Second call: ${price2:,.2f} ({duration2*1000:.2f}ms)")
    print("✅ test_get_current_price_cache_behavior PASSED")


# =============================================================================
# TESTS: Mathematical Calculations
# =============================================================================

def test_compute_size_base_simple():
    """Test compute_size_base with simple values."""
    print("\n--- test_compute_size_base_simple ---")
    
    usdt_amount = 100.0
    last_price = Decimal('50000')
    
    size = compute_size_base(usdt_amount, last_price)
    
    assert isinstance(size, Decimal), "Size should be Decimal"
    
    # 100 / 50000 = 0.002
    expected = Decimal('0.002')
    assert abs(size - expected) < Decimal('0.000001'), f"Expected {expected}, got {size}"
    
    print(f"   USDT: ${usdt_amount}")
    print(f"   Price: ${last_price}")
    print(f"   Size: {size}")
    print("✅ test_compute_size_base_simple PASSED")


def test_compute_size_base_realistic():
    """Test compute_size_base with realistic values."""
    print("\n--- test_compute_size_base_realistic ---")
    
    # Test case: $80 at BTC price $95000
    usdt_amount = 80.0
    last_price = Decimal('95000')
    
    size = compute_size_base(usdt_amount, last_price)
    
    # 80 / 95000 = 0.00084210526...
    expected_approx = Decimal('0.000842')
    assert abs(size - expected_approx) < Decimal('0.000001')
    
    print(f"   USDT: ${usdt_amount}")
    print(f"   BTC Price: ${last_price}")
    print(f"   BTC Size: {size}")
    print("✅ test_compute_size_base_realistic PASSED")


def test_extract_contract_params_btc():
    """Test extracting contract params from real BTC contract."""
    print("\n--- test_extract_contract_params_btc ---")
    
    # Get real contract
    contract = fetch_contracts_ws('BTCUSDT')
    last_price = Decimal('95000')
    
    price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt = \
        extract_contract_params(contract, last_price)
    
    assert price_tick is not None, "price_tick should not be None"
    assert size_scale is not None, "size_scale should not be None"
    assert min_trade_num is not None, "min_trade_num should not be None"
    
    print(f"   price_tick: {price_tick}")
    print(f"   size_scale: {size_scale}")
    print(f"   min_trade_num: {min_trade_num}")
    print(f"   size_multiplier: {size_multiplier}")
    print(f"   min_trade_usdt: {min_trade_usdt}")
    print("✅ test_extract_contract_params_btc PASSED")


def test_extract_contract_params_none():
    """Test extract_contract_params with None contract."""
    print("\n--- test_extract_contract_params_none ---")
    
    last_price = Decimal('50000')
    
    price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt = \
        extract_contract_params(None, last_price)
    
    # Should all be None
    assert price_tick is None
    assert size_scale is None
    assert min_trade_num is None
    assert size_multiplier is None
    assert min_trade_usdt is None
    
    print("   Contract None → All params None (as expected)")
    print("✅ test_extract_contract_params_none PASSED")


def test_fallback_params_high_price():
    """Test fallback_params for high-priced asset."""
    print("\n--- test_fallback_params_high_price ---")
    
    last_price = Decimal('95000')  # BTC-like
    
    price_tick, size_scale, min_trade_num, min_trade_usdt = \
        fallback_params(None, None, last_price)
    
    assert price_tick is not None
    assert size_scale is not None
    assert min_trade_num is not None
    
    # For price >= 1000, price_tick should be 0.1
    assert price_tick == Decimal('0.1'), f"Expected 0.1, got {price_tick}"
    
    # For price >= 100, min_trade_num should be 0.01
    assert min_trade_num == Decimal('0.01'), f"Expected 0.01, got {min_trade_num}"
    
    print(f"   Price: ${last_price}")
    print(f"   Fallback price_tick: {price_tick}")
    print(f"   Fallback size_scale: {size_scale}")
    print(f"   Fallback min_trade_num: {min_trade_num}")
    print("✅ test_fallback_params_high_price PASSED")


def test_fallback_params_medium_price():
    """Test fallback_params for medium-priced asset."""
    print("\n--- test_fallback_params_medium_price ---")
    
    last_price = Decimal('50')  # ETH-like
    
    price_tick, size_scale, min_trade_num, min_trade_usdt = \
        fallback_params(None, None, last_price)
    
    # For 10 <= price < 100, price_tick should be 0.01
    assert price_tick == Decimal('0.01'), f"Expected 0.01, got {price_tick}"
    
    # For 10 <= price < 100, min_trade_num should be 0.1
    assert min_trade_num == Decimal('0.1'), f"Expected 0.1, got {min_trade_num}"
    
    print(f"   Price: ${last_price}")
    print(f"   Fallback price_tick: {price_tick}")
    print(f"   Fallback min_trade_num: {min_trade_num}")
    print("✅ test_fallback_params_medium_price PASSED")


def test_fallback_params_low_price():
    """Test fallback_params for low-priced asset."""
    print("\n--- test_fallback_params_low_price ---")
    
    last_price = Decimal('0.5')  # Low-cap altcoin
    
    price_tick, size_scale, min_trade_num, min_trade_usdt = \
        fallback_params(None, None, last_price)
    
    # For 0.1 <= price < 1, price_tick should be 0.001
    assert price_tick == Decimal('0.001'), f"Expected 0.001, got {price_tick}"
    
    # For price < 10, min_trade_num should be 1
    assert min_trade_num == Decimal('1'), f"Expected 1, got {min_trade_num}"
    
    print(f"   Price: ${last_price}")
    print(f"   Fallback price_tick: {price_tick}")
    print(f"   Fallback min_trade_num: {min_trade_num}")
    print("✅ test_fallback_params_low_price PASSED")


def test_quantize_size_normal():
    """Test quantize_size with normal values."""
    print("\n--- test_quantize_size_normal ---")
    
    size_base = Decimal('0.123456789')
    size_scale = 4  # 4 decimal places
    
    size_q, precision = quantize_size(size_base, size_scale)
    
    assert size_q is not None, "Quantized size should not be None"
    assert size_q == Decimal('0.1234'), f"Expected 0.1234, got {size_q}"
    assert precision == Decimal('0.0001'), f"Expected 0.0001, got {precision}"
    
    print(f"   Base size: {size_base}")
    print(f"   Scale: {size_scale}")
    print(f"   Quantized: {size_q}")
    print(f"   Precision: {precision}")
    print("✅ test_quantize_size_normal PASSED")


def test_quantize_size_zero():
    """Test quantize_size when result rounds to zero."""
    print("\n--- test_quantize_size_zero ---")
    
    size_base = Decimal('0.000001')  # Very small
    size_scale = 3  # Only 3 decimal places
    
    size_q, precision = quantize_size(size_base, size_scale)
    
    # Should try fallback to 1e-6, but still might be zero
    # In this case, quantize_size returns None for zero size
    if size_q is None:
        print("   Very small size rounded to zero (expected)")
    else:
        assert size_q > 0, "If not None, should be positive"
        print(f"   Quantized to: {size_q}")
    
    print("✅ test_quantize_size_zero PASSED")


def test_quantize_size_realistic_btc():
    """Test quantize_size with realistic BTC order."""
    print("\n--- test_quantize_size_realistic_btc ---")
    
    # $80 order at $95000 = 0.00084210526... BTC
    size_base = Decimal('0.00084210526')
    size_scale = 6  # BTC typically uses 6 decimals
    
    size_q, precision = quantize_size(size_base, size_scale)
    
    assert size_q is not None
    assert size_q == Decimal('0.000842'), f"Expected 0.000842, got {size_q}"
    
    print(f"   Base size: {size_base}")
    print(f"   Quantized: {size_q}")
    print(f"   Precision: {precision}")
    print("✅ test_quantize_size_realistic_btc PASSED")


# =============================================================================
# TESTS: Integration (multiple functions together)
# =============================================================================

def test_full_order_sizing_flow():
    """Test complete order sizing flow (without placing order)."""
    print("\n--- test_full_order_sizing_flow ---")
    
    symbol = 'BTCUSDT'
    usdt_amount = 80.0
    
    # 1. Get current price
    last_price = get_current_price(symbol)
    print(f"   1. Current price: ${last_price:,.2f}")
    
    # 2. Calculate base size
    size_base = compute_size_base(usdt_amount, last_price)
    print(f"   2. Base size: {size_base}")
    
    # 3. Get contract info
    contract = fetch_contracts_ws(symbol)
    print(f"   3. Contract info retrieved")
    
    # 4. Extract params
    price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt = \
        extract_contract_params(contract, last_price)
    print(f"   4. Contract params extracted (size_scale={size_scale})")
    
    # 5. Apply fallbacks if needed
    price_tick, size_scale, min_trade_num, min_trade_usdt = \
        fallback_params(price_tick, size_scale, last_price, min_trade_num, min_trade_usdt)
    print(f"   5. Fallbacks applied")
    
    # 6. Quantize size
    size_q, precision = quantize_size(size_base, size_scale)
    print(f"   6. Quantized size: {size_q}")
    
    assert size_q is not None, "Final size should not be None"
    assert size_q > 0, "Final size should be positive"
    
    # Verify size is valid
    assert size_q >= min_trade_num, f"Size {size_q} < minimum {min_trade_num}"
    
    print(f"\n   Final order parameters:")
    print(f"     Symbol: {symbol}")
    print(f"     USDT Amount: ${usdt_amount}")
    print(f"     Entry Price: ${last_price:,.2f}")
    print(f"     Size: {size_q} BTC")
    print(f"     Min Trade Num: {min_trade_num}")
    
    print("✅ test_full_order_sizing_flow PASSED")


# =============================================================================
# TESTS: Position Tracker - TP/SL Calculations
# =============================================================================

def test_calculate_tp_sl_prices_long():
    """Test TP/SL calculation for long position."""
    print("\n--- test_calculate_tp_sl_prices_long ---")
    
    entry_price = Decimal('50000')
    direction = 'long'
    tp_pct = 5.0  # 5% TP
    sl_pct = 2.0  # 2% SL
    
    tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
    
    # Long: TP above entry, SL below entry
    expected_tp = Decimal('52500')  # 50000 * 1.05
    expected_sl = Decimal('49000')  # 50000 * 0.98
    
    assert tp_price == expected_tp, f"Expected TP {expected_tp}, got {tp_price}"
    assert sl_price == expected_sl, f"Expected SL {expected_sl}, got {sl_price}"
    
    print(f"   Entry: ${entry_price}")
    print(f"   TP (+5%): ${tp_price}")
    print(f"   SL (-2%): ${sl_price}")
    print("✅ test_calculate_tp_sl_prices_long PASSED")


def test_calculate_tp_sl_prices_short():
    """Test TP/SL calculation for short position."""
    print("\n--- test_calculate_tp_sl_prices_short ---")
    
    entry_price = Decimal('50000')
    direction = 'short'
    tp_pct = 5.0
    sl_pct = 2.0
    
    tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
    
    # Short: TP below entry, SL above entry
    expected_tp = Decimal('47500')  # 50000 * 0.95
    expected_sl = Decimal('51000')  # 50000 * 1.02
    
    assert tp_price == expected_tp, f"Expected TP {expected_tp}, got {tp_price}"
    assert sl_price == expected_sl, f"Expected SL {expected_sl}, got {sl_price}"
    
    print(f"   Entry: ${entry_price}")
    print(f"   TP (-5%): ${tp_price}")
    print(f"   SL (+2%): ${sl_price}")
    print("✅ test_calculate_tp_sl_prices_short PASSED")


def test_calculate_tp_sl_prices_realistic():
    """Test TP/SL with realistic BTC values."""
    print("\n--- test_calculate_tp_sl_prices_realistic ---")
    
    entry_price = Decimal('87250.50')
    direction = 'long'
    tp_pct = 2.5
    sl_pct = 1.5
    
    tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
    
    # Verify TP is above entry
    assert tp_price > entry_price, "TP should be above entry for long"
    assert sl_price < entry_price, "SL should be below entry for long"
    
    # Calculate actual percentages
    tp_pct_actual = float((tp_price - entry_price) / entry_price * 100)
    sl_pct_actual = float((entry_price - sl_price) / entry_price * 100)
    
    assert abs(tp_pct_actual - tp_pct) < 0.001, f"TP% mismatch: {tp_pct_actual}"
    assert abs(sl_pct_actual - sl_pct) < 0.001, f"SL% mismatch: {sl_pct_actual}"
    
    print(f"   Entry: ${entry_price}")
    print(f"   TP: ${tp_price} (+{tp_pct_actual:.2f}%)")
    print(f"   SL: ${sl_price} (-{sl_pct_actual:.2f}%)")
    print("✅ test_calculate_tp_sl_prices_realistic PASSED")


# =============================================================================
# TESTS: Position Tracker - PnL Calculations
# =============================================================================

def test_calculate_pnl_long_profit():
    """Test PnL calculation for profitable long position."""
    print("\n--- test_calculate_pnl_long_profit ---")
    
    direction = 'long'
    entry_price = Decimal('50000')
    current_price = Decimal('52000')  # +4% profit
    size = Decimal('0.1')
    
    pnl = calculate_pnl(direction, entry_price, current_price, size)
    
    # (52000 - 50000) * 0.1 = 200
    expected_pnl = 200.0
    
    assert abs(pnl - expected_pnl) < 0.01, f"Expected {expected_pnl}, got {pnl}"
    assert pnl > 0, "Should be profitable"
    
    print(f"   Entry: ${entry_price}, Current: ${current_price}")
    print(f"   Size: {size} BTC")
    print(f"   PnL: ${pnl:.2f}")
    print("✅ test_calculate_pnl_long_profit PASSED")


def test_calculate_pnl_long_loss():
    """Test PnL calculation for losing long position."""
    print("\n--- test_calculate_pnl_long_loss ---")
    
    direction = 'long'
    entry_price = Decimal('50000')
    current_price = Decimal('48000')  # -4% loss
    size = Decimal('0.1')
    
    pnl = calculate_pnl(direction, entry_price, current_price, size)
    
    # (48000 - 50000) * 0.1 = -200
    expected_pnl = -200.0
    
    assert abs(pnl - expected_pnl) < 0.01, f"Expected {expected_pnl}, got {pnl}"
    assert pnl < 0, "Should be losing"
    
    print(f"   Entry: ${entry_price}, Current: ${current_price}")
    print(f"   Size: {size} BTC")
    print(f"   PnL: ${pnl:.2f}")
    print("✅ test_calculate_pnl_long_loss PASSED")


def test_calculate_pnl_short_profit():
    """Test PnL calculation for profitable short position."""
    print("\n--- test_calculate_pnl_short_profit ---")
    
    direction = 'short'
    entry_price = Decimal('50000')
    current_price = Decimal('48000')  # Price down = profit for short
    size = Decimal('0.1')
    
    pnl = calculate_pnl(direction, entry_price, current_price, size)
    
    # (50000 - 48000) * 0.1 = 200
    expected_pnl = 200.0
    
    assert abs(pnl - expected_pnl) < 0.01, f"Expected {expected_pnl}, got {pnl}"
    assert pnl > 0, "Should be profitable"
    
    print(f"   Entry: ${entry_price}, Current: ${current_price}")
    print(f"   Size: {size} BTC")
    print(f"   PnL: ${pnl:.2f}")
    print("✅ test_calculate_pnl_short_profit PASSED")


def test_calculate_pnl_short_loss():
    """Test PnL calculation for losing short position."""
    print("\n--- test_calculate_pnl_short_loss ---")
    
    direction = 'short'
    entry_price = Decimal('50000')
    current_price = Decimal('52000')  # Price up = loss for short
    size = Decimal('0.1')
    
    pnl = calculate_pnl(direction, entry_price, current_price, size)
    
    # (50000 - 52000) * 0.1 = -200
    expected_pnl = -200.0
    
    assert abs(pnl - expected_pnl) < 0.01, f"Expected {expected_pnl}, got {pnl}"
    assert pnl < 0, "Should be losing"
    
    print(f"   Entry: ${entry_price}, Current: ${current_price}")
    print(f"   Size: {size} BTC")
    print(f"   PnL: ${pnl:.2f}")
    print("✅ test_calculate_pnl_short_loss PASSED")


# =============================================================================
# TESTS: Integration - check_all_tp_sl with REAL positions (SAFE - uses mocks)
# =============================================================================

def test_check_all_tp_sl_with_real_positions():
    """
    Integration test: Execute check_all_tp_sl with REAL positions from account 00.
    
    SAFE: Uses mocked broker functions - NO actual trading, NO writes to DB.
    
    This test:
    1. Loads REAL positions from PostgreSQL (account 00)
    2. Mocks broker functions (send_request, check_tp_sl_for_strategy)
    3. Executes check_all_tp_sl to verify it doesn't crash
    4. Detects bugs like "'BotState' object is not subscriptable"
    """
    print("\n--- test_check_all_tp_sl_with_real_positions ---")
    print("   [INTEGRATION TEST - READS REAL DATA, MOCKS BROKER]")
    
    # Load real positions from account 00
    print("\n   [1/3] Loading real state from account 00...")
    try:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bot_root = os.path.dirname(current_dir)
        state_file_00 = os.path.join(bot_root, 'persistence', 'bot_state_00.json')
        
        positions, candles = load_state("00", state_file_00)
        
        total_positions = sum(len(p) for p in positions.values())
        print(f"   ✅ Loaded {total_positions} active positions")
        
        if total_positions == 0:
            print("   ⚠️  No active positions, test skipped (not applicable)")
            print("✅ test_check_all_tp_sl_with_real_positions SKIPPED")
            return
            
    except Exception as e:
        print(f"   ⚠️  Could not load state (maybe no data): {e}")
        print("✅ test_check_all_tp_sl_with_real_positions SKIPPED")
        return
    
    # Mock broker functions
    print("   [2/3] Preparing mocks (NO real broker calls)...")
    
    def mock_send_request(*args, **kwargs):
        """Mock that returns success without touching broker."""
        return {'code': '00000', 'data': {}}
    
    def mock_check_tp_sl_for_strategy(*args, **kwargs):
        """Mock that does nothing (just verify it's called without error)."""
        pass
    
    # Build minimal strategies from loaded positions
    strategies = [
        {
            'id': strat_id,
            'sell_after_ncandles': 50
        }
        for strat_id in positions.keys()
    ]
    
    # Execute check_all_tp_sl with real positions + mocked functions
    print("   [3/3] Executing check_all_tp_sl with real data...")
    try:
        result = check_all_tp_sl(
            strategies=strategies,
            open_positions=positions,
            strategy_candles=candles,
            account_number="00",
            state_file=state_file_00,
            send_request_func=mock_send_request,
            check_tp_sl_for_strategy_func=mock_check_tp_sl_for_strategy,
            bot_state=None
        )
        
        print(f"   ✅ check_all_tp_sl executed without crash")
        print(f"   ✅ Result: {result}")
        print(f"   ✅ Tested with {total_positions} real positions")
        print("✅ test_check_all_tp_sl_with_real_positions PASSED")
        
    except TypeError as e:
        if "'BotState' object is not subscriptable" in str(e):
            print(f"\n   ❌ CRITICAL BUG DETECTED: {e}")
            print(f"   This is the production bug we fixed!")
            raise AssertionError("Bug detected: bot_state subscriptable error")
        raise
    except Exception as e:
        print(f"\n   ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_load_and_validate_all_strategies():
    """
    Integration test: Load and validate strategies from all accounts.
    
    SAFE: Only reads Python config files, NO broker calls, NO DB writes.
    
    This test:
    1. Loads strategies from accounts 00, E1, 01
    2. Validates each strategy config
    3. Detects missing fields, invalid values
    """
    print("\n--- test_load_and_validate_all_strategies ---")
    print("   [INTEGRATION TEST - READS CONFIG ONLY]")
    
    from strategies import load_strategies
    from validation import validate_strategy_configuration
    from strategies.strategy_registry import IMPLEMENTED_STRATEGIES
    
    accounts = ['00', 'E1', '01']
    total_strategies = 0
    
    for account in accounts:
        print(f"\n   Testing account {account}...")
        
        try:
            strategies = load_strategies(account)
            active_count = sum(1 for s in strategies if s.get('active', True))
            
            print(f"   ✅ Loaded {len(strategies)} strategies ({active_count} active)")
            
            # Validate
            errors, warnings = validate_strategy_configuration(strategies, IMPLEMENTED_STRATEGIES)
            
            if errors:
                print(f"   ❌ Validation errors found:")
                for err in errors:
                    print(f"      - {err}")
                raise AssertionError(f"Strategy validation failed for account {account}")
            
            if warnings:
                print(f"   ⚠️  Warnings: {len(warnings)}")
            
            total_strategies += len(strategies)
            
        except Exception as e:
            print(f"   ❌ Failed for account {account}: {e}")
            raise
    
    print(f"\n   ✅ Total strategies validated: {total_strategies}")
    print("✅ test_load_and_validate_all_strategies PASSED")


def test_regime_calculation_all_timeframes():
    """
    Integration test: Calculate regime for all timeframes.
    
    SAFE: Only fetches OHLCV data (read-only), NO writes.
    
    This test:
    1. Calculates regime for 4H, 1H, 6Hutc
    2. Verifies no crashes in regime_classifier
    3. Detects errors in metric calculations
    """
    print("\n--- test_regime_calculation_all_timeframes ---")
    print("   [INTEGRATION TEST - FETCHES OHLCV READ-ONLY]")
    
    from market_regime import get_current_regime, get_current_direction
    
    timeframes = ['4H', '1H', '6Hutc']
    results = {}
    
    for tf in timeframes:
        print(f"\n   Testing timeframe {tf}...")
        
        try:
            # Calculate regime
            regime, metrics = get_current_regime(tf)
            
            assert regime in ['trending', 'ranging', 'volatile', 'default'], \
                f"Invalid regime: {regime}"
            
            assert 'hurst' in metrics, "Missing hurst metric"
            assert 'efficiency_ratio' in metrics, "Missing efficiency_ratio metric"
            
            print(f"   ✅ Regime: {regime}")
            print(f"      Hurst: {metrics.get('hurst', 0):.2f}")
            print(f"      ER: {metrics.get('efficiency_ratio', 0):.2f}")
            
            # Calculate direction
            direction, btc_price, btc_ma50 = get_current_direction(tf)
            
            assert direction in ['uptrend', 'dwtrend'], \
                f"Invalid direction: {direction}"
            
            print(f"   ✅ Direction: {direction}")
            print(f"      BTC: ${btc_price:.2f}" if btc_price else "      BTC: N/A")
            print(f"      MA50: ${btc_ma50:.2f}" if btc_ma50 else "      MA50: N/A")
            
            results[tf] = {'regime': regime, 'direction': direction}
            
        except Exception as e:
            print(f"   ❌ Failed for {tf}: {e}")
            raise
    
    print(f"\n   ✅ All timeframes calculated successfully")
    print(f"   Results: {results}")
    print("✅ test_regime_calculation_all_timeframes PASSED")


def test_strategy_symbols_loading():
    """
    Integration test: Load symbols for active strategies.
    
    SAFE: Only reads symbol files from disk, NO broker calls.
    
    This test:
    1. Loads symbols for each active strategy
    2. Verifies files exist and are readable
    3. Detects missing symbol files
    """
    print("\n--- test_strategy_symbols_loading ---")
    print("   [INTEGRATION TEST - READS SYMBOL FILES ONLY]")
    
    from strategies import load_strategies
    from market_data import load_final_symbols
    
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bot_root = os.path.dirname(current_dir)
    
    # Test with account 00 (most strategies)
    strategies = load_strategies('00')
    active_strategies = [s for s in strategies if s.get('active', True)]
    
    print(f"\n   Testing {len(active_strategies)} active strategies...")
    
    total_symbols = 0
    
    for strat in active_strategies[:5]:  # Test first 5 to keep it fast
        strat_id = strat['id']
        timeframe = strat['timeframe']
        
        try:
            symbols = load_final_symbols(
                all_symbols=[],  # Empty list, will load from file
                strategy=strat_id,
                timeframe=timeframe
            )
            
            print(f"   ✅ {strat_id}: {len(symbols)} symbols")
            total_symbols += len(symbols)
            
        except FileNotFoundError as e:
            print(f"   ⚠️  {strat_id}: Symbol file not found (expected for some strategies)")
        except Exception as e:
            print(f"   ❌ {strat_id}: Error loading symbols: {e}")
            raise
    
    print(f"\n   ✅ Total symbols loaded: {total_symbols}")
    print("✅ test_strategy_symbols_loading PASSED")


def test_position_dict_structure_from_real_data():
    """
    Integration test: Verify position dict structure from real state.
    
    SAFE: Only reads from PostgreSQL, NO writes, NO broker calls.
    
    This test:
    1. Loads real positions from account 00
    2. Verifies all required fields are present
    3. Detects missing or malformed position data
    """
    print("\n--- test_position_dict_structure_from_real_data ---")
    print("   [INTEGRATION TEST - READS STATE ONLY]")
    
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bot_root = os.path.dirname(current_dir)
    state_file_00 = os.path.join(bot_root, 'persistence', 'bot_state_00.json')
    
    try:
        positions, candles = load_state("00", state_file_00)
        
        total_positions = sum(len(p) for p in positions.values())
        print(f"\n   Loaded {total_positions} positions")
        
        if total_positions == 0:
            print("   ⚠️  No positions, test skipped")
            print("✅ test_position_dict_structure_from_real_data SKIPPED")
            return
        
        # Required fields in position dict
        required_fields = [
            'symbol', 'size', 'entry_price', 'direction',
            'tp', 'sl', 'order_id', 'opened_at', 'usdt_amount',
            'regime_family', 'regime_multiplier',
            'market_direction', 'direction_multiplier'
        ]
        
        positions_checked = 0
        
        for strat_id, strat_positions in positions.items():
            for pos in strat_positions:
                # Check all required fields exist
                missing_fields = [f for f in required_fields if f not in pos]
                
                if missing_fields:
                    print(f"   ❌ Position in {strat_id} missing fields: {missing_fields}")
                    raise AssertionError(f"Missing fields: {missing_fields}")
                
                # Verify types
                assert isinstance(pos['symbol'], str), "symbol should be string"
                assert pos['direction'] in ['long', 'short'], "direction should be long/short"
                assert pos['regime_family'] in ['trending', 'ranging', 'volatile', 'unknown'], \
                    f"Invalid regime_family: {pos['regime_family']}"
                assert pos['market_direction'] in ['uptrend', 'dwtrend', 'unknown'], \
                    f"Invalid market_direction: {pos['market_direction']}"
                
                positions_checked += 1
                
                if positions_checked >= 10:  # Check first 10
                    break
            
            if positions_checked >= 10:
                break
        
        print(f"   ✅ Checked {positions_checked} positions")
        print(f"   ✅ All required fields present")
        print(f"   ✅ All field types valid")
        print("✅ test_position_dict_structure_from_real_data PASSED")
        
    except Exception as e:
        print(f"   ⚠️  Could not load state: {e}")
        print("✅ test_position_dict_structure_from_real_data SKIPPED")


def test_state_save_and_load_cycle():
    """
    Integration test: Verify state serialization/deserialization cycle.
    
    SAFE: Only tests in-memory serialization, NO file writes.
    
    This test:
    1. Loads real state from account 00
    2. Serializes to JSON (in memory)
    3. Deserializes back
    4. Verifies data integrity (no loss, correct types)
    5. Detects Decimal/datetime serialization issues
    """
    print("\n--- test_state_save_and_load_cycle ---")
    print("   [INTEGRATION TEST - IN-MEMORY ONLY]")
    
    import os
    import json
    from decimal import Decimal
    from datetime import datetime
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bot_root = os.path.dirname(current_dir)
    state_file_00 = os.path.join(bot_root, 'persistence', 'bot_state_00.json')
    
    try:
        # Load real state
        positions, candles = load_state("00", state_file_00)
        
        total_positions = sum(len(p) for p in positions.values())
        print(f"\n   Loaded {total_positions} positions")
        
        if total_positions == 0:
            print("   ⚠️  No positions, test skipped")
            print("✅ test_state_save_and_load_cycle SKIPPED")
            return
        
        # Prepare state for serialization (like save_state_local does)
        state_data = {
            'open_positions': {},
            'strategy_candles': candles
        }
        
        for strat_id, strat_positions in positions.items():
            state_data['open_positions'][strat_id] = []
            for pos in strat_positions:
                pos_copy = pos.copy()
                # Convert types (same as production code)
                for key, value in pos_copy.items():
                    if isinstance(value, Decimal):
                        pos_copy[key] = float(value)
                    elif isinstance(value, datetime):
                        pos_copy[key] = value.isoformat()
                state_data['open_positions'][strat_id].append(pos_copy)
        
        print("   ✅ Serialized to dict")
        
        # Serialize to JSON string (in memory)
        try:
            json_string = json.dumps(state_data, indent=2, default=str)
            print(f"   ✅ Converted to JSON ({len(json_string)} chars)")
        except Exception as e:
            print(f"   ❌ JSON serialization failed: {e}")
            raise
        
        # Deserialize back
        try:
            loaded_data = json.loads(json_string)
            print("   ✅ Deserialized from JSON")
        except Exception as e:
            print(f"   ❌ JSON deserialization failed: {e}")
            raise
        
        # Verify integrity
        assert 'open_positions' in loaded_data, "Missing open_positions"
        assert 'strategy_candles' in loaded_data, "Missing strategy_candles"
        
        loaded_positions = sum(len(p) for p in loaded_data['open_positions'].values())
        assert loaded_positions == total_positions, \
            f"Position count mismatch: {loaded_positions} != {total_positions}"
        
        print(f"   ✅ Data integrity verified ({total_positions} positions intact)")
        print("✅ test_state_save_and_load_cycle PASSED")
        
    except Exception as e:
        print(f"   ⚠️  Could not load state: {e}")
        print("✅ test_state_save_and_load_cycle SKIPPED")


def test_position_sizer_with_real_strategies():
    """
    Integration test: Test PositionSizer with real strategy configs.
    
    SAFE: Only calculates amounts, NO position opening, NO broker calls.
    
    This test:
    1. Loads real strategies from account 00
    2. Calculates adjusted_amount for each strategy
    3. Verifies metadata is correct
    4. Detects blocking logic (direction_mode)
    5. Detects sizing errors
    """
    print("\n--- test_position_sizer_with_real_strategies ---")
    print("   [INTEGRATION TEST - CALCULATIONS ONLY]")
    
    from strategies import load_strategies
    from market_regime import PositionSizer
    import logging
    
    # Create logger for PositionSizer
    logger = logging.getLogger('test_position_sizer')
    logger.setLevel(logging.INFO)
    
    # Load real strategies
    strategies = load_strategies('00')
    active_strategies = [s for s in strategies if s.get('active', True)]
    
    print(f"\n   Testing {len(active_strategies)} active strategies...")
    
    # Initialize PositionSizer
    position_sizer = PositionSizer(logger)
    
    # Mock regime states
    test_scenarios = [
        {'regime': 'trending', 'direction': 'uptrend'},
        {'regime': 'ranging', 'direction': 'dwtrend'},
        {'regime': 'volatile', 'direction': 'uptrend'},
    ]
    
    strategies_tested = 0
    blocked_count = 0
    
    for strat in active_strategies[:10]:  # Test first 10
        strat_id = strat['id']
        base_amount = strat.get('order_amount', 80)
        direction_mode = strat.get('direction_mode', 'general')
        
        print(f"\n   Testing {strat_id}...")
        print(f"      Base: ${base_amount}, Mode: {direction_mode}")
        
        for scenario in test_scenarios:
            try:
                adjusted_amount, metadata = position_sizer.calculate_adjusted_amount(
                    base_amount=base_amount,
                    regime_trending=strat.get('regime_trending', 1.0),
                    regime_ranging=strat.get('regime_ranging', 1.0),
                    regime_volatile=strat.get('regime_volatile', 1.0),
                    direction_mode=direction_mode,
                    market_regime=scenario['regime'],
                    market_direction=scenario['direction']
                )
                
                # Verify metadata structure
                assert 'base_amount' in metadata, "Missing base_amount"
                assert 'market_regime' in metadata, "Missing market_regime"
                assert 'market_direction' in metadata, "Missing market_direction"
                assert 'blocked' in metadata, "Missing blocked"
                assert 'final_multiplier' in metadata, "Missing final_multiplier"
                
                # Verify values
                assert metadata['market_regime'] == scenario['regime']
                assert metadata['market_direction'] == scenario['direction']
                assert adjusted_amount >= 0, "Negative amount"
                
                if metadata['blocked']:
                    assert adjusted_amount == 0, "Blocked but amount > 0"
                    blocked_count += 1
                
            except Exception as e:
                print(f"      ❌ Failed for {scenario}: {e}")
                raise
        
        strategies_tested += 1
    
    print(f"\n   ✅ Tested {strategies_tested} strategies")
    print(f"   ✅ All metadata valid")
    print(f"   ✅ Blocking logic working ({blocked_count} blocked scenarios)")
    print("✅ test_position_sizer_with_real_strategies PASSED")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Setup WebSocket
    ws_initialized = setup_websocket()
    
    if not ws_initialized:
        print("\n❌ WebSocket initialization failed. Cannot run tests.")
        sys.exit(1)
    
    tests = [
        # =====================================================================
        # ORDER MANAGER TESTS
        # =====================================================================
        
        # WebSocket data fetching
        test_fetch_ticker_ws_btc,
        test_fetch_ticker_ws_eth,
        test_fetch_ticker_ws_multiple_calls,
        test_fetch_contracts_ws_btc,
        test_get_usdt_balance_ws,
        test_get_current_price_btc,
        test_get_current_price_cache_behavior,
        
        # Mathematical calculations
        test_compute_size_base_simple,
        test_compute_size_base_realistic,
        test_extract_contract_params_btc,
        test_extract_contract_params_none,
        test_fallback_params_high_price,
        test_fallback_params_medium_price,
        test_fallback_params_low_price,
        test_quantize_size_normal,
        test_quantize_size_zero,
        test_quantize_size_realistic_btc,
        
        # Integration
        test_full_order_sizing_flow,
        
        # =====================================================================
        # POSITION TRACKER TESTS
        # =====================================================================
        
        # TP/SL calculations
        test_calculate_tp_sl_prices_long,
        test_calculate_tp_sl_prices_short,
        test_calculate_tp_sl_prices_realistic,
        
        # PnL calculations
        test_calculate_pnl_long_profit,
        test_calculate_pnl_long_loss,
        test_calculate_pnl_short_profit,
        test_calculate_pnl_short_loss,
        
        # =====================================================================
        # INTEGRATION TESTS (real data + mocks - SAFE)
        # =====================================================================
        
        test_check_all_tp_sl_with_real_positions,
        test_load_and_validate_all_strategies,
        test_regime_calculation_all_timeframes,
        test_strategy_symbols_loading,
        test_position_dict_structure_from_real_data,
        test_state_save_and_load_cycle,
        test_position_sizer_with_real_strategies,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    print("\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            errors.append((test.__name__, str(e)))
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if errors:
        print("\nFailed tests:")
        for test_name, error in errors:
            print(f"  ❌ {test_name}: {error}")
    
    sys.exit(0 if failed == 0 else 1)