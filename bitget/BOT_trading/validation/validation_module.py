"""
Validation functions for bot configuration
"""
import re
import logging

logger = logging.getLogger('BOT_trading.validation.validation_module')

from config.settings import MIN_ORDER_AMOUNT, MAX_ORDER_AMOUNT, MIN_TP_PCT, MAX_TP_PCT
from config.settings import MIN_SL_PCT, MAX_SL_PCT, MIN_CANDLES, MAX_CANDLES
from config.settings import VALID_TIMEFRAMES
from config.settings import REGIME_FAMILIES, REGIME_GENERAL, REGIME_REFERENCE_SYMBOL
from config.settings import STRATEGY_TYPE_REQUIRED_PARAMS, COMMON_REQUIRED_PARAMS
from config.settings import ACCOUNTS, BASE_URL
from config.settings import POSTGRES_CONFIG
import psycopg2
from psycopg2 import OperationalError

# ==========================================================================
# POSTGRESQL VALIDATION
# ==========================================================================

def validate_postgresql_connection():
    """
    Validates PostgreSQL connection is available before bot starts.
    
    Raises:
        SystemExit: If PostgreSQL is not accessible
    
    Returns:
        bool: True if connection successful
    """
    try:
        logger.info("Validating PostgreSQL connection...")
        
        # Try to connect
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        conn.close()
        
        logger.info("✓ PostgreSQL connection validated successfully")
        return True
        
    except OperationalError as e:
        logger.error("✗ FATAL: PostgreSQL connection failed")
        logger.error(f"Error: {e}")
        logger.error("Ensure PostgreSQL is running: sudo systemctl status postgresql")
        raise SystemExit(1)
        
    except Exception as e:
        logger.error(f"✗ FATAL: Unexpected error validating PostgreSQL: {e}")
        raise SystemExit(1)
        
# ==========================================================================
# SETTINGS VALIDATION
# ==========================================================================

def validate_settings():
    """
    Validates system configuration and market regime settings.
    
    Returns:
        tuple: (errors, warnings) - lists of validation messages
    """
    
    errors = []
    warnings = []
    
    # ========================================================================
    # Val S1: Unique dashboard ports
    # ========================================================================
    validation_s1_errors = 0
    ports = [acc['dashboard_port'] for acc in ACCOUNTS.values()]
    if len(ports) != len(set(ports)):
        errors.append("Dashboard ports must be unique across accounts")
        validation_s1_errors += 1
    
    if validation_s1_errors == 0:
        logger.info("Val S1: Dashboard ports are unique")
    
    # ========================================================================
    # Val S2: Valid timeframes
    # ========================================================================
    validation_s2_errors = 0
    if not VALID_TIMEFRAMES:
        errors.append("VALID_TIMEFRAMES cannot be empty")
        validation_s2_errors += 1
    
    if validation_s2_errors == 0:
        logger.info("Val S2: VALID_TIMEFRAMES configured")
    
    # ========================================================================
    # Val S3: Order amount limits
    # ========================================================================
    validation_s3_errors = 0
    if MIN_ORDER_AMOUNT >= MAX_ORDER_AMOUNT:
        errors.append("MIN_ORDER_AMOUNT must be less than MAX_ORDER_AMOUNT")
        validation_s3_errors += 1
    
    if validation_s3_errors == 0:
        logger.info("Val S3: Order amount limits valid")
    
    # ========================================================================
    # Val S4: TP percentage limits
    # ========================================================================
    validation_s4_errors = 0
    if MIN_TP_PCT >= MAX_TP_PCT:
        errors.append("MIN_TP_PCT must be less than MAX_TP_PCT")
        validation_s4_errors += 1
    
    if validation_s4_errors == 0:
        logger.info("Val S4: TP percentage limits valid")
    
    # ========================================================================
    # Val S5: BASE_URL uses HTTPS
    # ========================================================================
    validation_s5_errors = 0
    if not BASE_URL.startswith("https://"):
        errors.append("BASE_URL must use HTTPS")
        validation_s5_errors += 1
    
    if validation_s5_errors == 0:
        logger.info("Val S5: BASE_URL uses HTTPS")
    
    
    # ========================================================================
    # Val S7: SL percentage limits
    # ========================================================================
    validation_s7_errors = 0
    if MIN_SL_PCT >= MAX_SL_PCT:
        errors.append("MIN_SL_PCT must be less than MAX_SL_PCT")
        validation_s7_errors += 1
    
    if validation_s7_errors == 0:
        logger.info("Val S7: SL percentage limits valid")
    
    # ========================================================================
    # Val S8: Candles limits
    # ========================================================================
    validation_s8_errors = 0
    if MIN_CANDLES >= MAX_CANDLES:
        errors.append("MIN_CANDLES must be less than MAX_CANDLES")
        validation_s8_errors += 1
    
    if validation_s8_errors == 0:
        logger.info("Val S8: Candles limits valid")
    
    # ========================================================================
    # Val S9: REGIME_FAMILIES required families
    # ========================================================================
    validation_s9_errors = 0
    required_families = {'volatile', 'ranging', 'trending'}
    configured_families = set(REGIME_FAMILIES.keys())
    
    missing_families = required_families - configured_families
    if missing_families:
        errors.append(
            f"REGIME_FAMILIES missing required families: {missing_families}"
        )
        validation_s9_errors += 1
    
    if validation_s9_errors == 0:
        logger.info("Val S9: REGIME_FAMILIES has all required families")
    
    # ========================================================================
    # Val S10: REGIME_FAMILIES structure validation
    # ========================================================================
    validation_s10_errors = 0
    
    # Validate structure of each family
    for family, rules in REGIME_FAMILIES.items():
        if not isinstance(rules, dict):
            errors.append(
                f"REGIME_FAMILIES['{family}'] must be a dict, got {type(rules)}"
            )
            validation_s10_errors += 1
            continue
        
        # Validate rule structure: {metric: (operator, threshold)}
        for metric, rule in rules.items():
            if not isinstance(rule, tuple) or len(rule) != 2:
                errors.append(
                    f"REGIME_FAMILIES['{family}']['{metric}'] must be tuple "
                    f"(operator, threshold), got {type(rule)}"
                )
                validation_s10_errors += 1
                continue
            
            operator, threshold = rule
            
            # Validate operator
            if operator not in ['>', '<', '>=', '<=', '==']:
                errors.append(
                    f"REGIME_FAMILIES['{family}']['{metric}'] has invalid operator "
                    f"'{operator}' (valid: >, <, >=, <=, ==)"
                )
                validation_s10_errors += 1
            
            # Validate threshold is numeric
            if not isinstance(threshold, (int, float)):
                errors.append(
                    f"REGIME_FAMILIES['{family}']['{metric}'] threshold must be "
                    f"numeric, got {type(threshold)}"
                )
                validation_s10_errors += 1
            elif threshold < 0:
                errors.append(
                    f"REGIME_FAMILIES['{family}']['{metric}'] threshold = {threshold} "
                    f"(must be >= 0)"
                )
                validation_s10_errors += 1
    
    if validation_s10_errors == 0:
        logger.info("Val S10: REGIME_FAMILIES structure valid")
    
    # ========================================================================
    # Val S11: REGIME_GENERAL required families
    # ========================================================================
    validation_s11_errors = 0
    required_families = {'volatile', 'ranging', 'trending'}
    configured_families = set(REGIME_GENERAL.keys())
    
    missing_families = required_families - configured_families
    if missing_families:
        warnings.append(
            f"REGIME_GENERAL missing families: {missing_families} "
            f"(will use fallback 1.0)"
        )
    
    if validation_s11_errors == 0:
        logger.info("Val S11: REGIME_GENERAL families checked")
    
    # ========================================================================
    # Val S12: REGIME_GENERAL multipliers validation
    # ========================================================================
    validation_s12_errors = 0
    
    # Validate multipliers
    for family, multiplier in REGIME_GENERAL.items():
        if not isinstance(multiplier, (int, float)):
            errors.append(
                f"REGIME_GENERAL['{family}'] must be numeric, "
                f"got {type(multiplier)}"
            )
            validation_s12_errors += 1
            continue
        
        # Warn about extreme values
        if multiplier < 0:
            errors.append(
                f"REGIME_GENERAL['{family}'] = {multiplier} (must be >= 0)"
            )
            validation_s12_errors += 1
        elif multiplier > 5.0:
            warnings.append(
                f"REGIME_GENERAL['{family}'] = {multiplier} "
                f"(>5.0 is very aggressive)"
            )
        elif multiplier == 0 and family != 'volatile':
            warnings.append(
                f"REGIME_GENERAL['{family}'] = 0 "
                f"(blocks trading in this regime)"
            )
    
    if validation_s12_errors == 0:
        logger.info("Val S12: REGIME_GENERAL multipliers valid")
    
    # ========================================================================
    # Val S13: REGIME thresholds coherence
    # ========================================================================
    validation_s13_errors = 0
    
    for family, rules in REGIME_FAMILIES.items():
        for metric, rule in rules.items():
            if not isinstance(rule, tuple) or len(rule) != 2:
                continue
            
            operator, threshold = rule
            
            # Validate Hurst exponent range (0-1)
            if 'hurst' in metric.lower():
                if threshold < 0 or threshold > 1:
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] threshold = {threshold} "
                        f"(Hurst exponent must be between 0 and 1)"
                    )
                    validation_s13_errors += 1
            
            # Validate Efficiency Ratio range (0-1)
            if 'efficiency' in metric.lower():
                if threshold < 0 or threshold > 1:
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] threshold = {threshold} "
                        f"(Efficiency Ratio must be between 0 and 1)"
                    )
                    validation_s13_errors += 1
            
            # Validate Permutation Entropy range (0-1)
            if 'entropy' in metric.lower():
                if threshold < 0 or threshold > 1:
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] threshold = {threshold} "
                        f"(Permutation Entropy must be between 0 and 1)"
                    )
                    validation_s13_errors += 1
            
            # Validate ATR percentage reasonable range
            if 'atr' in metric.lower() and 'pct' in metric.lower():
                if threshold < 0.1:
                    warnings.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] = {threshold} "
                        f"(<0.1% is very low, may trigger too often)"
                    )
                elif threshold > 15:
                    warnings.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] = {threshold} "
                        f"(>15% is very high, may never trigger)"
                    )
    
    if validation_s13_errors == 0:
        logger.info("Val S13: REGIME thresholds coherent")
    
    # ========================================================================
    # Val S14: REGIME reference symbol
    # ========================================================================
    validation_s14_errors = 0
    
    if not isinstance(REGIME_REFERENCE_SYMBOL, str):
        errors.append(
            f"REGIME_REFERENCE_SYMBOL must be string, got {type(REGIME_REFERENCE_SYMBOL)}"
        )
        validation_s14_errors += 1
    elif not REGIME_REFERENCE_SYMBOL:
        errors.append("REGIME_REFERENCE_SYMBOL is empty")
        validation_s14_errors += 1
    elif not REGIME_REFERENCE_SYMBOL.endswith('USDT'):
        warnings.append(
            f"REGIME_REFERENCE_SYMBOL = '{REGIME_REFERENCE_SYMBOL}' "
            f"doesn't end with 'USDT'. Ensure it's a valid futures symbol."
        )
    
    if validation_s14_errors == 0:
        logger.info(f"Val S14: REGIME_REFERENCE_SYMBOL = '{REGIME_REFERENCE_SYMBOL}'")
    
  
    
    # ========================================================================
    # Val S17: Account numbers format
    # ========================================================================
    validation_s17_errors = 0
    
    for account_num in ACCOUNTS.keys():
        # Check format: 2 chars alphanumeric
        if not re.match(r'^[A-Z0-9]{2}$', account_num):
            errors.append(
                f"Account number '{account_num}' has invalid format "
                f"(must be 2 alphanumeric characters, e.g., '00', 'E1')"
            )
            validation_s17_errors += 1
    
    if validation_s17_errors == 0:
        logger.info("Val S17: All account numbers have valid format")
    
    return errors, warnings


# ==========================================================================
# STRATEGY CONFIGURATION VALIDATION
# ==========================================================================

def validate_strategy_configuration(strategies, implemented_strategies):
    """
    Validates strategy configuration against all validation rules.
    
    Args:
        strategies: List of strategy dicts loaded from YAML
        implemented_strategies: Set of strategy IDs that have implementations
    
    Returns:
        tuple: (errors, warnings) - lists of validation messages
    """
    
    # Use IDs instead of names for validation
    declared_strategies = {s['id'] for s in strategies}
    missing_implementation = declared_strategies - implemented_strategies
    unused_implementation = implemented_strategies - declared_strategies
    
    errors = []
    warnings = []
    
    # ========================================================================
    # Val Y1: Strategy IDs have implementations
    # ========================================================================
    if missing_implementation:
        errors.append(f"Strategies WITHOUT implementation: {missing_implementation}")
    
    if unused_implementation:
        warnings.append(f"Implemented but NOT declared: {unused_implementation}")
    
    if not missing_implementation:
        logger.info("Val Y1: All strategy IDs implemented")
    
    # ========================================================================
    # Val Y2: Direction coherence with name
    # ========================================================================
    validation_y2_errors = 0
    for strat in strategies:
        name = strat.get('name', '')
        direction = strat.get('direction', '')
        strat_id = strat.get('id', 'UNKNOWN')

        name_indicates_long = '_long_' in name.lower()
        name_indicates_short = '_short_' in name.lower()
        
        if name_indicates_long and direction != 'long':
            errors.append(
                f"Strategy '{strat_id}' has name='{name}' (indicates LONG) "
                f"but direction='{direction}'"
            )
            validation_y2_errors += 1
        
        if name_indicates_short and direction != 'short':
            errors.append(
                f"Strategy '{strat_id}' has name='{name}' (indicates SHORT) "
                f"but direction='{direction}'"
            )
            validation_y2_errors += 1
        
        if direction not in ['long', 'short']:
            errors.append(
                f"Strategy '{strat_id}' has invalid direction='{direction}' "
                f"(must be 'long' or 'short')"
            )
            validation_y2_errors += 1
    
    if validation_y2_errors == 0:
        logger.info("Val Y2: All directions coherent with names")
    
    # ========================================================================
    # Val Y3: Order amount within valid range
    # ========================================================================
    validation_y3_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        order_amount = strat.get('order_amount', None)
        
        if order_amount is None:
            errors.append(
                f"Strategy '{strat_id}' is missing 'order_amount' parameter"
            )
            validation_y3_errors += 1
        elif not isinstance(order_amount, (int, float)):
            errors.append(
                f"Strategy '{strat_id}' has invalid order_amount='{order_amount}' "
                f"(must be a number)"
            )
            validation_y3_errors += 1
        elif order_amount < MIN_ORDER_AMOUNT:
            errors.append(
                f"Strategy '{strat_id}' has order_amount={order_amount} "
                f"(minimum is {MIN_ORDER_AMOUNT})"
            )
            validation_y3_errors += 1
        elif order_amount > MAX_ORDER_AMOUNT:
            errors.append(
                f"Strategy '{strat_id}' has order_amount={order_amount} "
                f"(maximum is {MAX_ORDER_AMOUNT})"
            )
            validation_y3_errors += 1
    
    if validation_y3_errors == 0:
        logger.info("Val Y3: All order amounts in range")
    
    # ========================================================================
    # Val Y4: Required parameters for each strategy type
    # ========================================================================
    validation_y4_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        strat_name = strat.get('name', '')
        
        # Check common parameters
        for param in COMMON_REQUIRED_PARAMS:
            if param not in strat:
                errors.append(
                    f"Strategy '{strat_id}' is missing required parameter: '{param}'"
                )
                validation_y4_errors += 1
        
        # Determine base strategy type
        base_type = None
        for tf in VALID_TIMEFRAMES:
            suffix = f'_{tf}'
            if strat_name.endswith(suffix):
                base_type = strat_name[:-len(suffix)]
                break
        
        # Check strategy-specific parameters
        if base_type and base_type in STRATEGY_TYPE_REQUIRED_PARAMS:
            required_params = STRATEGY_TYPE_REQUIRED_PARAMS[base_type]
            for param in required_params:
                if param not in strat:
                    errors.append(
                        f"Strategy '{strat_id}' (type: {base_type}) is missing "
                        f"required parameter: '{param}'"
                    )
                    validation_y4_errors += 1
        elif base_type:
            warnings.append(
                f"Strategy '{strat_id}' base type '{base_type}' has no parameter "
                f"requirements defined. Add it to STRATEGY_TYPE_REQUIRED_PARAMS."
            )
    
    if validation_y4_errors == 0:
        logger.info("Val Y4: All strategies have required parameters")
    
    # ========================================================================
    # Val Y5: TP/SL within valid ranges
    # ========================================================================
    validation_y5_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        tp_pct = strat.get('tp_pct', None)
        sl_pct = strat.get('sl_pct', None)
        
        if tp_pct and (tp_pct < MIN_TP_PCT or tp_pct > MAX_TP_PCT):
            errors.append(
                f"Strategy '{strat_id}' has tp_pct={tp_pct} "
                f"(valid range: {MIN_TP_PCT}-{MAX_TP_PCT}%)"
            )
            validation_y5_errors += 1
        
        if sl_pct and (sl_pct < MIN_SL_PCT or sl_pct > MAX_SL_PCT):
            errors.append(
                f"Strategy '{strat_id}' has sl_pct={sl_pct} "
                f"(valid range: {MIN_SL_PCT}-{MAX_SL_PCT}%)"
            )
            validation_y5_errors += 1
    
    if validation_y5_errors == 0:
        logger.info("Val Y5: All TP/SL percentages within valid ranges")
    
    # ========================================================================
    # Val Y6: Unique strategy IDs
    # ========================================================================
    validation_y6_errors = 0
    ids = [s.get('id') for s in strategies]
    duplicates = [id for id in set(ids) if ids.count(id) > 1]
    
    if duplicates:
        errors.append(f"Duplicate strategy IDs found: {duplicates}")
        validation_y6_errors += 1
    
    if validation_y6_errors == 0:
        logger.info("Val Y6: All strategy IDs are unique")
    
    # ========================================================================
    # Val Y7: Candles within reasonable range
    # ========================================================================
    validation_y7_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        candles = strat.get('sell_after_ncandles', None)
        
        if candles and (candles < MIN_CANDLES or candles > MAX_CANDLES):
            errors.append(
                f"Strategy '{strat_id}' has sell_after_ncandles={candles} "
                f"(typical range: {MIN_CANDLES}-{MAX_CANDLES})"
            )
            validation_y7_errors += 1
    
    if validation_y7_errors == 0:
        logger.info("Val Y7: All sell_after_ncandles within range")
    
    # ========================================================================
    # Y8: Unique strategy names
    # ========================================================================
    validation_y8_errors = 0
    names = [s.get('name') for s in strategies]
    duplicates = [name for name in set(names) if names.count(name) > 1]
    
    if duplicates:
        errors.append(f"Duplicate strategy names found: {duplicates}")
        validation_y8_errors += 1
    
    if validation_y8_errors == 0:
        logger.info("Val Y8: All strategy names are unique")
    
    # ========================================================================
    # Val Y9: Valid timeframes
    # ========================================================================
    validation_y9_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        timeframe = strat.get('timeframe', '')
        
        if timeframe not in VALID_TIMEFRAMES:
            errors.append(
                f"Strategy '{strat_id}' has invalid timeframe='{timeframe}' "
                f"(valid: {', '.join(VALID_TIMEFRAMES)})"
            )
            validation_y9_errors += 1
    
    if validation_y9_errors == 0:
        logger.info("Val Y9: All timeframes are valid")
    
    # ========================================================================
    # Val Y10: ID format with numeric prefix
    # ========================================================================
    validation_y10_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', '')
        
        if not re.match(r'^\d{2}_\w+', strat_id):
            errors.append(
                f"Strategy '{strat_id}' has invalid ID format. "
                f"Expected: 'NN_strategy_name' (e.g., '02_reversal_long_4H')"
            )
            validation_y10_errors += 1
            continue
        
        id_parts = strat_id.split('_', 1)
        id_number = id_parts[0]
        
        if len(id_number) != 2:
            errors.append(
                f"Strategy '{strat_id}' numeric prefix must be exactly 2 digits "
                f"(e.g., '02_name', not '2_name' or '002_name')"
            )
            validation_y10_errors += 1
        
        try:
            num_value = int(id_number)
            if num_value < 1 or num_value > 99:
                errors.append(
                    f"Strategy '{strat_id}' numeric prefix must be 01-99 "
                    f"(found: {id_number})"
                )
                validation_y10_errors += 1
        except ValueError:
            errors.append(
                f"Strategy '{strat_id}' numeric prefix is not a valid number"
            )
            validation_y10_errors += 1
    
    if validation_y10_errors == 0:
        logger.info("Val Y10: All IDs have correct prefix format (NN_name)")
    
# ========================================================================
    # Val Y11: regime multipliers (trending, ranging, volatile)
    # ========================================================================
    validation_y11_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        
        # Check regime_trending
        if 'regime_trending' not in strat:
            errors.append(
                f"Strategy '{strat_id}' missing required field 'regime_trending'"
            )
            validation_y11_errors += 1
        else:
            val = strat['regime_trending']
            if not isinstance(val, (int, float)):
                errors.append(
                    f"Strategy '{strat_id}' regime_trending must be numeric, got {type(val)}"
                )
                validation_y11_errors += 1
            elif val < 0:
                errors.append(
                    f"Strategy '{strat_id}' regime_trending = {val} (must be >= 0)"
                )
                validation_y11_errors += 1
            elif val > 5.0:
                warnings.append(
                    f"Strategy '{strat_id}' regime_trending = {val} (>5.0 is very aggressive)"
                )
        
        # Check regime_ranging
        if 'regime_ranging' not in strat:
            errors.append(
                f"Strategy '{strat_id}' missing required field 'regime_ranging'"
            )
            validation_y11_errors += 1
        else:
            val = strat['regime_ranging']
            if not isinstance(val, (int, float)):
                errors.append(
                    f"Strategy '{strat_id}' regime_ranging must be numeric, got {type(val)}"
                )
                validation_y11_errors += 1
            elif val < 0:
                errors.append(
                    f"Strategy '{strat_id}' regime_ranging = {val} (must be >= 0)"
                )
                validation_y11_errors += 1
            elif val > 5.0:
                warnings.append(
                    f"Strategy '{strat_id}' regime_ranging = {val} (>5.0 is very aggressive)"
                )
        
        # Check regime_volatile
        if 'regime_volatile' not in strat:
            errors.append(
                f"Strategy '{strat_id}' missing required field 'regime_volatile'"
            )
            validation_y11_errors += 1
        else:
            val = strat['regime_volatile']
            if not isinstance(val, (int, float)):
                errors.append(
                    f"Strategy '{strat_id}' regime_volatile must be numeric, got {type(val)}"
                )
                validation_y11_errors += 1
            elif val < 0:
                errors.append(
                    f"Strategy '{strat_id}' regime_volatile = {val} (must be >= 0)"
                )
                validation_y11_errors += 1
            elif val > 5.0:
                warnings.append(
                    f"Strategy '{strat_id}' regime_volatile = {val} (>5.0 is very aggressive)"
                )
    
    if validation_y11_errors == 0:
        logger.info("Val Y11: All regime multipliers valid")

    # ========================================================================
# ========================================================================
    # Val Y12: direction_mode field
    # ========================================================================
    validation_y12_errors = 0
    
    valid_dir_modes = {'long_only', 'short_only', 'general'}
    strategies_without_dir_mode = []
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        
        if 'direction_mode' not in strat:
            strategies_without_dir_mode.append(strat_id)
            continue
        
        direction_mode = strat['direction_mode']
        
        # Validate type
        if not isinstance(direction_mode, str):
            errors.append(
                f"Strategy '{strat_id}' direction_mode must be string, "
                f"got {type(direction_mode)}"
            )
            validation_y12_errors += 1
            continue
        
        # Validate value
        if direction_mode not in valid_dir_modes:
            errors.append(
                f"Strategy '{strat_id}' has invalid direction_mode='{direction_mode}' "
                f"(valid: {valid_dir_modes})"
            )
            validation_y12_errors += 1
    
    # Warning for strategies without direction_mode
    if strategies_without_dir_mode:
        warnings.append(
            f"{len(strategies_without_dir_mode)} strategies without 'direction_mode' "
            f"(will default to 'general'): {strategies_without_dir_mode}"
        )
    
    if validation_y12_errors == 0:
        logger.info("Val Y12: All strategy direction_mode values valid")
    

 # ========================================================================
    # Val Y13: Coherencia entre direction y dir_mode
    # ========================================================================
    validation_y13_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        direction = strat.get('direction', '').lower()
        dir_mode = strat.get('dir_mode', 'general')
        
        # Skip if dir_mode is 'general' (always valid)
        if dir_mode == 'general':
            continue
        
        # LONG strategies can only use 'uptrend'
        if direction == 'long' and dir_mode != 'long_only':
            errors.append(
                f"Strategy '{strat_id}' has direction='long' but dir_mode='{dir_mode}'. "
                f"LONG strategies must use dir_mode='uptrend' or 'general'"
            )
            validation_y13_errors += 1
        
        # SHORT strategies can only use 'dwtrend'
        elif direction == 'short' and dir_mode != 'short_only':
            errors.append(
                f"Strategy '{strat_id}' has direction='short' but dir_mode='{dir_mode}'. "
                f"SHORT strategies must use dir_mode='dwtrend' or 'general'"
            )
            validation_y13_errors += 1
    
    if validation_y13_errors == 0:
        logger.info("Val Y13: All direction/dir_mode combinations are coherent")
        
    return errors, warnings