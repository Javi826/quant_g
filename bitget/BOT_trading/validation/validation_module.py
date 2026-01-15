"""
Validation functions for bot configuration
"""
import re
import logging

logger = logging.getLogger('BOT_trading.validation.validation_module')

from config.settings import MIN_ORDER_AMOUNT, MAX_ORDER_AMOUNT, MIN_TP_PCT, MAX_TP_PCT
from config.settings import MIN_SL_PCT, MAX_SL_PCT, MIN_CANDLES, MAX_CANDLES
from config.settings import VALID_TIMEFRAMES, TIMEFRAME_SUFFIXES, STRATEGY_TYPE_REQUIRED_PARAMS
from config.settings import COMMON_REQUIRED_PARAMS, REGIME_FAMILIES, REGIME_FAMILY_SIZING, REGIME_REFERENCE_SYMBOL,REGIME_FAMILY_MATRIX



# ==========================================================================
# SETTINGS VALIDATION
# ==========================================================================

def validate_settings():
    """
    Validates that settings.py is correctly configured.
    
    Returns:
        tuple: (errors, warnings) - lists of validation messages
    """
    from config.settings import (
        ACCOUNTS, ACCOUNT_STRATEGIES,
        BASE_URL
    )
    
    errors = []
    warnings = []
    
    # ========================================================================
    # Val 18: Unique dashboard ports
    # ========================================================================
    validation_18_errors = 0
    ports = [acc['dashboard_port'] for acc in ACCOUNTS.values()]
    if len(ports) != len(set(ports)):
        errors.append("Dashboard ports must be unique across accounts")
        validation_18_errors += 1
    
    if validation_18_errors == 0:
        logger.info("Val 18: Dashboard ports are unique")
    
    # ========================================================================
    # Val 19: Valid timeframes
    # ========================================================================
    validation_19_errors = 0
    if not VALID_TIMEFRAMES:
        errors.append("VALID_TIMEFRAMES cannot be empty")
        validation_19_errors += 1
    
    if validation_19_errors == 0:
        logger.info("Val 19: VALID_TIMEFRAMES configured")
    
    # ========================================================================
    # Val 20: Order amount limits
    # ========================================================================
    validation_20_errors = 0
    if MIN_ORDER_AMOUNT >= MAX_ORDER_AMOUNT:
        errors.append("MIN_ORDER_AMOUNT must be less than MAX_ORDER_AMOUNT")
        validation_20_errors += 1
    
    if validation_20_errors == 0:
        logger.info("Val 20: Order amount limits valid")
    
    # ========================================================================
    # Val 21: TP percentage limits
    # ========================================================================
    validation_21_errors = 0
    if MIN_TP_PCT >= MAX_TP_PCT:
        errors.append("MIN_TP_PCT must be less than MAX_TP_PCT")
        validation_21_errors += 1
    
    if validation_21_errors == 0:
        logger.info("Val 21: TP percentage limits valid")
    
    # ========================================================================
    # Val 22: BASE_URL uses HTTPS
    # ========================================================================
    validation_22_errors = 0
    if not BASE_URL.startswith("https://"):
        errors.append("BASE_URL must use HTTPS")
        validation_22_errors += 1
    
    if validation_22_errors == 0:
        logger.info("Val 22: BASE_URL uses HTTPS")
    
    # ========================================================================
    # Val 23: Account strategies mapping
    # ========================================================================
    validation_23_errors = 0
    for account_num in ACCOUNTS.keys():
        if account_num not in ACCOUNT_STRATEGIES:
            errors.append(f"Account {account_num} missing in ACCOUNT_STRATEGIES")
            validation_23_errors += 1
    
    if validation_23_errors == 0:
        logger.info("Val 23: All accounts mapped in ACCOUNT_STRATEGIES")
    
    return errors, warnings


def validate_regime_configuration():
    """
    Validates market regime configuration.
    
    Returns:
        tuple: (errors, warnings) - lists of validation messages
    """
    errors = []
    warnings = []
    
    # ========================================================================
    # Val 14a: REGIME_FAMILIES required families
    # ========================================================================
    validation_14a_errors = 0
    required_families = {'volatile', 'ranging', 'trending'}
    configured_families = set(REGIME_FAMILIES.keys())
    
    missing_families = required_families - configured_families
    if missing_families:
        errors.append(
            f"REGIME_FAMILIES missing required families: {missing_families}"
        )
        validation_14a_errors += 1
    
    if validation_14a_errors == 0:
        logger.info("Val 14a: REGIME_FAMILIES has all required families")
    
    # ========================================================================
    # Val 14b: REGIME_FAMILIES structure validation
    # ========================================================================
    validation_14b_errors = 0
    
    # Validate structure of each family
    for family, rules in REGIME_FAMILIES.items():
        if not isinstance(rules, dict):
            errors.append(
                f"REGIME_FAMILIES['{family}'] must be a dict, got {type(rules)}"
            )
            validation_14b_errors += 1
            continue
        
        # Validate rule structure: {metric: (operator, threshold)}
        for metric, rule in rules.items():
            if not isinstance(rule, tuple) or len(rule) != 2:
                errors.append(
                    f"REGIME_FAMILIES['{family}']['{metric}'] must be tuple "
                    f"(operator, threshold), got {type(rule)}"
                )
                validation_14b_errors += 1
                continue
            
            operator, threshold = rule
            
            # Validate operator
            if operator not in ['>', '<', '>=', '<=', '==']:
                errors.append(
                    f"REGIME_FAMILIES['{family}']['{metric}'] has invalid operator "
                    f"'{operator}' (valid: >, <, >=, <=, ==)"
                )
                validation_14b_errors += 1
            
            # Validate threshold is numeric
            if not isinstance(threshold, (int, float)):
                errors.append(
                    f"REGIME_FAMILIES['{family}']['{metric}'] threshold must be "
                    f"numeric, got {type(threshold)}"
                )
                validation_14b_errors += 1
            elif threshold < 0:
                errors.append(
                    f"REGIME_FAMILIES['{family}']['{metric}'] threshold = {threshold} "
                    f"(must be >= 0)"
                )
                validation_14b_errors += 1
    
    if validation_14b_errors == 0:
        logger.info("Val 14b: REGIME_FAMILIES structure valid")
    
    # ========================================================================
    # Val 14c: REGIME_FAMILIES recommended metrics
    # ========================================================================
    # Validate that volatile has required metrics
    if 'volatile' in REGIME_FAMILIES and REGIME_FAMILIES['volatile']:
        volatile = REGIME_FAMILIES['volatile']
        if 'atr_pct' not in volatile or 'permutation_entropy' not in volatile:
            warnings.append(
                "REGIME_FAMILIES['volatile'] typically should have 'atr_pct' "
                "and 'permutation_entropy'"
            )
    
    # Validate that trending has required metrics
    if 'trending' in REGIME_FAMILIES and REGIME_FAMILIES['trending']:
        trending = REGIME_FAMILIES['trending']
        if 'hurst' not in trending and 'efficiency_ratio' not in trending:
            warnings.append(
                "REGIME_FAMILIES['trending'] typically should have 'hurst' "
                "or 'efficiency_ratio'"
            )
    
    logger.info("Val 14c: REGIME_FAMILIES recommended metrics checked")
    
    # ========================================================================
    # Val 15a: REGIME_FAMILY_SIZING required families
    # ========================================================================
    validation_15a_errors = 0
    required_families = {'volatile', 'ranging', 'trending'}
    configured_families = set(REGIME_FAMILY_SIZING.keys())
    
    missing_families = required_families - configured_families
    if missing_families:
        warnings.append(
            f"REGIME_FAMILY_SIZING missing families: {missing_families} "
            f"(will use fallback 1.0)"
        )
    
    if validation_15a_errors == 0:
        logger.info("Val 15a: REGIME_FAMILY_SIZING families checked")
    
    # ========================================================================
    # Val 15b: REGIME_FAMILY_SIZING multipliers validation
    # ========================================================================
    validation_15b_errors = 0
    
    # Validate multipliers
    for family, multiplier in REGIME_FAMILY_SIZING.items():
        if not isinstance(multiplier, (int, float)):
            errors.append(
                f"REGIME_FAMILY_SIZING['{family}'] must be numeric, "
                f"got {type(multiplier)}"
            )
            validation_15b_errors += 1
            continue
        
        # Warn about extreme values
        if multiplier < 0:
            errors.append(
                f"REGIME_FAMILY_SIZING['{family}'] = {multiplier} (must be >= 0)"
            )
            validation_15b_errors += 1
        elif multiplier > 5.0:
            warnings.append(
                f"REGIME_FAMILY_SIZING['{family}'] = {multiplier} "
                f"(>5.0 is very aggressive)"
            )
        elif multiplier == 0 and family != 'volatile':
            warnings.append(
                f"REGIME_FAMILY_SIZING['{family}'] = 0 "
                f"(blocks trading in this regime)"
            )
    
    if validation_15b_errors == 0:
        logger.info("Val 15b: REGIME_FAMILY_SIZING multipliers valid")
    
    # ========================================================================
    # Val 15c: REGIME_FAMILY_SIZING coherence check
    # ========================================================================
    # Validate coherence: trending should be >= ranging
    if 'trending' in REGIME_FAMILY_SIZING and 'ranging' in REGIME_FAMILY_SIZING:
        if REGIME_FAMILY_SIZING['trending'] < REGIME_FAMILY_SIZING['ranging']:
            warnings.append(
                f"REGIME_FAMILY_SIZING['trending'] "
                f"({REGIME_FAMILY_SIZING['trending']}) "
                f"is less than 'ranging' ({REGIME_FAMILY_SIZING['ranging']}). "
                f"Usually trending should have higher multiplier."
            )
    
    logger.info("Val 15c: REGIME_FAMILY_SIZING coherence checked")
    
    # ========================================================================
    # Val 16: REGIME thresholds coherence
    # ========================================================================
    validation_16_errors = 0
    
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
                    validation_16_errors += 1
            
            # Validate Efficiency Ratio range (0-1)
            if 'efficiency' in metric.lower():
                if threshold < 0 or threshold > 1:
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] threshold = {threshold} "
                        f"(Efficiency Ratio must be between 0 and 1)"
                    )
                    validation_16_errors += 1
            
            # Validate Permutation Entropy range (0-1)
            if 'entropy' in metric.lower():
                if threshold < 0 or threshold > 1:
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] threshold = {threshold} "
                        f"(Permutation Entropy must be between 0 and 1)"
                    )
                    validation_16_errors += 1
            
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
    
    if validation_16_errors == 0:
        logger.info("Val 16: REGIME thresholds coherent")
    
    # ========================================================================
    # Val 17: REGIME reference symbol
    # ========================================================================
    validation_17_errors = 0
    
    if not isinstance(REGIME_REFERENCE_SYMBOL, str):
        errors.append(
            f"REGIME_REFERENCE_SYMBOL must be string, got {type(REGIME_REFERENCE_SYMBOL)}"
        )
        validation_17_errors += 1
    elif not REGIME_REFERENCE_SYMBOL:
        errors.append("REGIME_REFERENCE_SYMBOL is empty")
        validation_17_errors += 1
    elif not REGIME_REFERENCE_SYMBOL.endswith('USDT'):
        warnings.append(
            f"REGIME_REFERENCE_SYMBOL = '{REGIME_REFERENCE_SYMBOL}' "
            f"doesn't end with 'USDT'. Ensure it's a valid futures symbol."
        )
    
    if validation_17_errors == 0:
        logger.info(f"Val 17: REGIME_REFERENCE_SYMBOL = '{REGIME_REFERENCE_SYMBOL}'")
        
    # ==========================================================================
    # Val 20: REGIME_FAMILY_MATRIX
    # ==========================================================================
    validation_20_errors = 0
    
    # Check if matrix exists
    if not REGIME_FAMILY_MATRIX:
        errors.append("REGIME_FAMILY_MATRIX is empty or not defined")
        validation_20_errors += 1
    else:
        # Required families
        required_families = {'trending', 'ranging', 'volatile'}
        configured_families = set(REGIME_FAMILY_MATRIX.keys())
        
        # Check all families present
        missing_families = required_families - configured_families
        if missing_families:
            errors.append(
                f"REGIME_FAMILY_MATRIX missing families: {missing_families}"
            )
            validation_20_errors += 1
        
        # Validate each family's multipliers
        for family in required_families:
            if family not in REGIME_FAMILY_MATRIX:
                continue
            
            family_mults = REGIME_FAMILY_MATRIX[family]
            
            # Check structure is dict
            if not isinstance(family_mults, dict):
                errors.append(
                    f"REGIME_FAMILY_MATRIX['{family}'] must be dict, "
                    f"got {type(family_mults)}"
                )
                validation_20_errors += 1
                continue
            
            # Check all regimes present
            missing_regimes = required_families - set(family_mults.keys())
            if missing_regimes:
                errors.append(
                    f"REGIME_FAMILY_MATRIX['{family}'] missing regimes: {missing_regimes}"
                )
                validation_20_errors += 1
            
            # Validate multiplier values
            for regime, multiplier in family_mults.items():
                if not isinstance(multiplier, (int, float)):
                    errors.append(
                        f"REGIME_FAMILY_MATRIX['{family}']['{regime}'] must be numeric, "
                        f"got {type(multiplier)}"
                    )
                    validation_20_errors += 1
                elif multiplier < 0:
                    errors.append(
                        f"REGIME_FAMILY_MATRIX['{family}']['{regime}'] = {multiplier} "
                        f"(must be >= 0)"
                    )
                    validation_20_errors += 1
                elif multiplier > 5.0:
                    warnings.append(
                        f"REGIME_FAMILY_MATRIX['{family}']['{regime}'] = {multiplier} "
                        f"(>5.0 is very aggressive)"
                    )
        
        # Coherence check: each family should have max multiplier in its own regime
        for family in required_families:
            if family not in REGIME_FAMILY_MATRIX:
                continue
            
            family_mults = REGIME_FAMILY_MATRIX[family]
            own_regime_mult = family_mults.get(family, 0)
            max_mult = max(family_mults.values())
            
            if own_regime_mult < max_mult and own_regime_mult > 0:
                warnings.append(
                    f"REGIME_FAMILY_MATRIX['{family}'] has max multiplier "
                    f"({max_mult}) in regime other than own '{family}' ({own_regime_mult}). "
                    f"Consider if this is intentional."
                )
    
    if validation_20_errors == 0:
        logger.info("Val 20: REGIME_FAMILY_MATRIX structure valid")
    
    return errors, warnings


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
    # Val 01: Strategy IDs have implementations
    # ========================================================================
    if missing_implementation:
        errors.append(f"Strategies WITHOUT implementation: {missing_implementation}")
    
    if unused_implementation:
        warnings.append(f"Implemented but NOT declared: {unused_implementation}")
    
    if not missing_implementation:
        logger.info("Val 01: All strategy IDs implemented")
    
    # ========================================================================
    # Val 02: Direction coherence with name
    # ========================================================================
    validation_2_errors = 0
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
            validation_2_errors += 1
        
        if name_indicates_short and direction != 'short':
            errors.append(
                f"Strategy '{strat_id}' has name='{name}' (indicates SHORT) "
                f"but direction='{direction}'"
            )
            validation_2_errors += 1
        
        if direction not in ['long', 'short']:
            errors.append(
                f"Strategy '{strat_id}' has invalid direction='{direction}' "
                f"(must be 'long' or 'short')"
            )
            validation_2_errors += 1
    
    if validation_2_errors == 0:
        logger.info("Val 02: All directions coherent with names")
    
    # ========================================================================
    # Val 03: Timeframe coherence with name
    # ========================================================================
    validation_3_errors = 0
    for strat in strategies:
        name = strat.get('name', '')
        timeframe = strat.get('timeframe', '')
        strat_id = strat.get('id', 'UNKNOWN')
        
        if '_4H' in name:
            if timeframe != '4H':
                errors.append(
                    f"Strategy '{strat_id}' has name='{name}' (indicates 4H) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
        
        elif '_1H' in name:
            if timeframe != '1H':
                errors.append(
                    f"Strategy '{strat_id}' has name='{name}' (indicates 1H) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
        
        elif '_6Hutc' in name:
            if timeframe != '6Hutc':
                errors.append(
                    f"Strategy '{strat_id}' has name='{name}' (indicates 6Hutc) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
    
    if validation_3_errors == 0:
        logger.info("Val 03: All timeframes coherent with names")
    
    # ========================================================================
    # Val 04: Order amount within valid range
    # ========================================================================
    validation_4_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        order_amount = strat.get('order_amount', None)
        
        if order_amount is None:
            errors.append(
                f"Strategy '{strat_id}' is missing 'order_amount' parameter"
            )
            validation_4_errors += 1
        elif not isinstance(order_amount, (int, float)):
            errors.append(
                f"Strategy '{strat_id}' has invalid order_amount='{order_amount}' "
                f"(must be a number)"
            )
            validation_4_errors += 1
        elif order_amount < MIN_ORDER_AMOUNT:
            errors.append(
                f"Strategy '{strat_id}' has order_amount={order_amount} "
                f"(minimum is {MIN_ORDER_AMOUNT})"
            )
            validation_4_errors += 1
        elif order_amount > MAX_ORDER_AMOUNT:
            errors.append(
                f"Strategy '{strat_id}' has order_amount={order_amount} "
                f"(maximum is {MAX_ORDER_AMOUNT})"
            )
            validation_4_errors += 1
    
    if validation_4_errors == 0:
        logger.info("Val 04: All order amounts in range (40-100)")
    
    # ========================================================================
    # Val 05: Required parameters for each strategy type
    # ========================================================================
    validation_5_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        strat_name = strat.get('name', '')
        
        # Check common parameters
        for param in COMMON_REQUIRED_PARAMS:
            if param not in strat:
                errors.append(
                    f"Strategy '{strat_id}' is missing required parameter: '{param}'"
                )
                validation_5_errors += 1
        
        # Determine base strategy type
        base_type = None
        for suffix in TIMEFRAME_SUFFIXES:
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
                    validation_5_errors += 1
        elif base_type:
            warnings.append(
                f"Strategy '{strat_id}' base type '{base_type}' has no parameter "
                f"requirements defined. Add it to STRATEGY_TYPE_REQUIRED_PARAMS."
            )
    
    if validation_5_errors == 0:
        logger.info("Val 05: All strategies have required parameters")
    
    # ========================================================================
    # Val 06: TP/SL within valid ranges
    # ========================================================================
    validation_6_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        tp_pct = strat.get('tp_pct', None)
        sl_pct = strat.get('sl_pct', None)
        
        if tp_pct and (tp_pct < MIN_TP_PCT or tp_pct > MAX_TP_PCT):
            errors.append(
                f"Strategy '{strat_id}' has tp_pct={tp_pct} "
                f"(valid range: {MIN_TP_PCT}-{MAX_TP_PCT}%)"
            )
            validation_6_errors += 1
        
        if sl_pct and (sl_pct < MIN_SL_PCT or sl_pct > MAX_SL_PCT):
            errors.append(
                f"Strategy '{strat_id}' has sl_pct={sl_pct} "
                f"(valid range: {MIN_SL_PCT}-{MAX_SL_PCT}%)"
            )
            validation_6_errors += 1
    
    if validation_6_errors == 0:
        logger.info("Val 06: All TP/SL percentages within valid ranges")
    
    # ========================================================================
    # Val 07: Unique strategy IDs
    # ========================================================================
    validation_7_errors = 0
    ids = [s.get('id') for s in strategies]
    duplicates = [id for id in set(ids) if ids.count(id) > 1]
    
    if duplicates:
        errors.append(f"Duplicate strategy IDs found: {duplicates}")
        validation_7_errors += 1
    
    if validation_7_errors == 0:
        logger.info("Val 07: All strategy IDs are unique")
    
    # ========================================================================
    # Val 08: Candles within reasonable range
    # ========================================================================
    validation_8_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        candles = strat.get('sell_after_ncandles', None)
        
        if candles and (candles < MIN_CANDLES or candles > MAX_CANDLES):
            errors.append(
                f"Strategy '{strat_id}' has sell_after_ncandles={candles} "
                f"(typical range: {MIN_CANDLES}-{MAX_CANDLES})"
            )
            validation_8_errors += 1
    
    if validation_8_errors == 0:
        logger.info("Val 08: All sell_after_ncandles within range (45-55)")
    
    # ========================================================================
    # Val 09: No duplicate strategies (name + timeframe)
    # ========================================================================
    validation_9_errors = 0
    seen_combinations = set()
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        name = strat.get('name', '')
        timeframe = strat.get('timeframe', '')
        
        combination = (name, timeframe)
        
        if combination in seen_combinations:
            errors.append(
                f"Duplicate strategy found: name='{name}', timeframe='{timeframe}' "
                f"(strategy '{strat_id}' conflicts with another)"
            )
            validation_9_errors += 1
        else:
            seen_combinations.add(combination)
    
    if validation_9_errors == 0:
        logger.info("Val 09: All strategy name+timeframe combinations are unique")
    
    # ========================================================================
    # Val 10: Unique strategy names
    # ========================================================================
    validation_10_errors = 0
    names = [s.get('name') for s in strategies]
    duplicates = [name for name in set(names) if names.count(name) > 1]
    
    if duplicates:
        errors.append(f"Duplicate strategy names found: {duplicates}")
        validation_10_errors += 1
    
    if validation_10_errors == 0:
        logger.info("Val 10: All strategy names are unique")
    
    # ========================================================================
    # Val 12: Valid timeframes
    # ========================================================================
    validation_12_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        timeframe = strat.get('timeframe', '')
        
        if timeframe not in VALID_TIMEFRAMES:
            errors.append(
                f"Strategy '{strat_id}' has invalid timeframe='{timeframe}' "
                f"(valid: {', '.join(VALID_TIMEFRAMES)})"
            )
            validation_12_errors += 1
    
    if validation_12_errors == 0:
        logger.info("Val 12: All timeframes are valid")
    
    # ========================================================================
    # Val 13: ID format with numeric prefix
    # ========================================================================
    validation_13_errors = 0
    
    for strat in strategies:
        strat_id = strat.get('id', '')
        
        if not re.match(r'^\d{2}_\w+', strat_id):
            errors.append(
                f"Strategy '{strat_id}' has invalid ID format. "
                f"Expected: 'NN_strategy_name' (e.g., '02_reversal_long_4H')"
            )
            validation_13_errors += 1
            continue
        
        id_parts = strat_id.split('_', 1)
        id_number = id_parts[0]
        
        if len(id_number) != 2:
            errors.append(
                f"Strategy '{strat_id}' numeric prefix must be exactly 2 digits "
                f"(e.g., '02_name', not '2_name' or '002_name')"
            )
            validation_13_errors += 1
        
        try:
            num_value = int(id_number)
            if num_value < 1 or num_value > 99:
                errors.append(
                    f"Strategy '{strat_id}' numeric prefix must be 01-99 "
                    f"(found: {id_number})"
                )
                validation_13_errors += 1
        except ValueError:
            errors.append(
                f"Strategy '{strat_id}' numeric prefix is not a valid number"
            )
            validation_13_errors += 1
    
    if validation_13_errors == 0:
        logger.info("Val 13: All IDs have correct prefix format (NN_name)")
        
    # ==========================================================================
    # VALIDATION 21: regime_family field
    # ==========================================================================
    validation_21_errors = 0
    
    valid_families = {'trending', 'ranging', 'volatile'}
    strategies_without_family = []
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        
        if 'regime_family' not in strat:
            strategies_without_family.append(strat_id)
            continue
        
        regime_family = strat['regime_family']
        
        # Validate type
        if not isinstance(regime_family, str):
            errors.append(
                f"Strategy '{strat_id}' regime_family must be string, "
                f"got {type(regime_family)}"
            )
            validation_21_errors += 1
            continue
        
        # Validate value
        if regime_family not in valid_families:
            errors.append(
                f"Strategy '{strat_id}' has invalid regime_family='{regime_family}' "
                f"(valid: {valid_families})"
            )
            validation_21_errors += 1
    
    # Warning for strategies without regime_family
    if strategies_without_family:
        warnings.append(
            f"{len(strategies_without_family)} strategies without 'regime_family' "
            f"(will use REGIME_FAMILY_SIZING fallback): {strategies_without_family}"
        )
    
    if validation_21_errors == 0:
        logger.info("Val 21: All strategy regime_family values valid")
    
    return errors, warnings