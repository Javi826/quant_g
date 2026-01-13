"""
Validation functions for bot strategy configuration
"""
import re
import logging

logger = logging.getLogger('BOT_trading.validation.strategy_validator')

from config.settings import (
    MIN_ORDER_AMOUNT,
    MAX_ORDER_AMOUNT,
    MIN_TP_PCT,
    MAX_TP_PCT,
    MIN_SL_PCT,
    MAX_SL_PCT,
    MIN_CANDLES,
    MAX_CANDLES,
    VALID_TIMEFRAMES,
    TIMEFRAME_SUFFIXES,
    STRATEGY_TYPE_REQUIRED_PARAMS,
    COMMON_REQUIRED_PARAMS,
    REGIME_FAMILIES,
    REGIME_FAMILY_SIZING,
    REGIME_REFERENCE_SYMBOL
)


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
    
    # VALIDATION 1: Strategy IDs have implementations
    if missing_implementation:
        errors.append(f"Strategies WITHOUT implementation: {missing_implementation}")
    
    if unused_implementation:
        warnings.append(f"Implemented but NOT declared: {unused_implementation}")
    
    if not missing_implementation:
        logger.info("Val 01: All strategy IDs implemented")
    
    # VALIDATION 2: Direction coherence with name
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
    
    # VALIDATION 3: Timeframe coherence with name
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
    
    # VALIDATION 4: Order amount within valid range
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
    
    # VALIDATION 5: Required parameters for each strategy type
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
    
    # VALIDATION 6: TP/SL within valid ranges
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
    
    # VALIDATION 7: Unique strategy IDs
    validation_7_errors = 0
    ids = [s.get('id') for s in strategies]
    duplicates = [id for id in set(ids) if ids.count(id) > 1]
    
    if duplicates:
        errors.append(f"Duplicate strategy IDs found: {duplicates}")
        validation_7_errors += 1
    
    if validation_7_errors == 0:
        logger.info("Val 07: All strategy IDs are unique")
    
    # VALIDATION 8: Candles within reasonable range
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
    
    # VALIDATION 9: No duplicate strategies (name + timeframe)
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
    
    # VALIDATION 10: Unique strategy names
    validation_10_errors = 0
    names = [s.get('name') for s in strategies]
    duplicates = [name for name in set(names) if names.count(name) > 1]
    
    if duplicates:
        errors.append(f"Duplicate strategy names found: {duplicates}")
        validation_10_errors += 1
    
    if validation_10_errors == 0:
        logger.info("Val 10: All strategy names are unique")
    
    # VALIDATION 12: Valid timeframes
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
    
    # VALIDATION 13: ID format with numeric prefix
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
    
    # VALIDATIONS 14-17: MARKET REGIME CONFIGURATION
    
    def validate_regime_families():
        """Validates that REGIME_FAMILIES is correctly configured."""
        validation_14_errors = 0
        
        # Validate that basic families exist
        required_families = {'volatile', 'ranging', 'trending'}
        configured_families = set(REGIME_FAMILIES.keys())
        
        missing_families = required_families - configured_families
        if missing_families:
            errors.append(
                f"REGIME_FAMILIES missing required families: {missing_families}"
            )
            validation_14_errors += 1
        
        # Validate structure of each family (can be empty dict for ranging)
        for family, rules in REGIME_FAMILIES.items():
            if not isinstance(rules, dict):
                errors.append(
                    f"REGIME_FAMILIES['{family}'] must be a dict, got {type(rules)}"
                )
                validation_14_errors += 1
                continue
            
            # Validate rule structure: {metric: (operator, threshold)}
            for metric, rule in rules.items():
                if not isinstance(rule, tuple) or len(rule) != 2:
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] must be tuple "
                        f"(operator, threshold), got {type(rule)}"
                    )
                    validation_14_errors += 1
                    continue
                
                operator, threshold = rule
                
                # Validate operator
                if operator not in ['>', '<', '>=', '<=', '==']:
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] has invalid operator "
                        f"'{operator}' (valid: >, <, >=, <=, ==)"
                    )
                    validation_14_errors += 1
                
                # Validate threshold is numeric
                if not isinstance(threshold, (int, float)):
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] threshold must be "
                        f"numeric, got {type(threshold)}"
                    )
                    validation_14_errors += 1
                elif threshold < 0:
                    errors.append(
                        f"REGIME_FAMILIES['{family}']['{metric}'] threshold = {threshold} "
                        f"(must be >= 0)"
                    )
                    validation_14_errors += 1
        
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
        
        if validation_14_errors == 0:
            logger.info("Val 14: REGIME_FAMILIES configuration complete")
        
        return validation_14_errors
    
    def validate_regime_sizing():
        """Validates that REGIME_FAMILY_SIZING is correctly configured."""
        validation_15_errors = 0
        
        # Validate that basic families exist
        required_families = {'volatile', 'ranging', 'trending'}
        configured_families = set(REGIME_FAMILY_SIZING.keys())
        
        missing_families = required_families - configured_families
        if missing_families:
            warnings.append(
                f"REGIME_FAMILY_SIZING missing families: {missing_families} "
                f"(will use fallback 1.0)"
            )
        
        # Validate multipliers
        for family, multiplier in REGIME_FAMILY_SIZING.items():
            if not isinstance(multiplier, (int, float)):
                errors.append(
                    f"REGIME_FAMILY_SIZING['{family}'] must be numeric, "
                    f"got {type(multiplier)}"
                )
                validation_15_errors += 1
                continue
            
            # Warn about extreme values
            if multiplier < 0:
                errors.append(
                    f"REGIME_FAMILY_SIZING['{family}'] = {multiplier} (must be >= 0)"
                )
                validation_15_errors += 1
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
        
        # Validate coherence: trending should be >= ranging
        if 'trending' in REGIME_FAMILY_SIZING and 'ranging' in REGIME_FAMILY_SIZING:
            if REGIME_FAMILY_SIZING['trending'] < REGIME_FAMILY_SIZING['ranging']:
                warnings.append(
                    f"REGIME_FAMILY_SIZING['trending'] "
                    f"({REGIME_FAMILY_SIZING['trending']}) "
                    f"is less than 'ranging' ({REGIME_FAMILY_SIZING['ranging']}). "
                    f"Usually trending should have higher multiplier."
                )
        
        if validation_15_errors == 0:
            logger.info("Val 15: REGIME_FAMILY_SIZING multipliers valid")
        
        return validation_15_errors
    
    def validate_regime_thresholds_coherence():
        """Validates coherence between thresholds of different families."""
        validation_16_errors = 0
        
        # Validate metric ranges
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
        
        return validation_16_errors
    
    def validate_regime_reference_symbol():
        """Validates that REGIME_REFERENCE_SYMBOL is correctly configured."""
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
        
        return validation_17_errors
    
    # EXECUTE MARKET REGIME VALIDATIONS
    try:
        validate_regime_families()
        validate_regime_sizing()
        validate_regime_thresholds_coherence()
        validate_regime_reference_symbol()
    except Exception as e:
        warnings.append(f"Could not validate regime configuration: {e}")
    
    return errors, warnings