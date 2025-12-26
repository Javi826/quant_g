#ZX_BOT_validations.py
"""
Validation functions for bot strategy configuration
"""

def validate_strategy_configuration(strategies, implemented_strategies):

    declared_strategies    = {s['name'] for s in strategies}    
    missing_implementation = declared_strategies - implemented_strategies
    unused_implementation  = implemented_strategies - declared_strategies
    errors   = []
    warnings = []
    
    # --------------------------------------------------------------------
    # VALIDATION 1: Names
    # --------------------------------------------------------------------
    if missing_implementation:
        errors.append(f"❗ Strategies WITHOUT implementation: {missing_implementation}")
    
    if unused_implementation:
        warnings.append(f"❕ Implemented but NOT declared: {unused_implementation}")
    
    if not missing_implementation:
        print("   🆗 Validation 1: All strategy names implemented")
    
    # --------------------------------------------------------------------
    # VALIDATION 2: Coherence direction
    # --------------------------------------------------------------------
    validation_2_errors = 0
    for strat in strategies:
        name      = strat.get('name', '')
        direction = strat.get('direction', '')
        strat_id  = strat.get('id', 'UNKNOWN')

        name_indicates_long  = '_long_' in name.lower()
        name_indicates_short = '_short_' in name.lower()
        
        if name_indicates_long and direction != 'long':
            errors.append(
                f"❌ Strategy '{strat_id}' has name='{name}' (indicates LONG) "
                f"but direction='{direction}'"
            )
            validation_2_errors += 1
        
        if name_indicates_short and direction != 'short':
            errors.append(
                f"❌ Strategy '{strat_id}' has name='{name}' (indicates SHORT) "
                f"but direction='{direction}'"
            )
            validation_2_errors += 1
        
        if direction not in ['long', 'short']:
            errors.append(
                f"❌ Strategy '{strat_id}' has invalid direction='{direction}' "
                f"(must be 'long' or 'short')"
            )
            validation_2_errors += 1
    
    if validation_2_errors == 0:
        print("   🆗 Validation 2: All directions coherent with names")
            
    # --------------------------------------------------------------------
    # VALIDATION 3: Timeframe coherence
    # --------------------------------------------------------------------
    validation_3_errors = 0
    for strat in strategies:
        name      = strat.get('name', '')
        timeframe = strat.get('timeframe', '')
        strat_id  = strat.get('id', 'UNKNOWN')
        
        if '_4H' in name:
            if timeframe != '4H':
                errors.append(
                    f"❌ Strategy '{strat_id}' has name='{name}' (indicates 4H) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
        
        elif '_1H' in name:
            if timeframe != '1H':
                errors.append(
                    f"❌ Strategy '{strat_id}' has name='{name}' (indicates 1H) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
        
        elif '_6Hutc' in name:
            if timeframe != '6Hutc':
                errors.append(
                    f"❌ Strategy '{strat_id}' has name='{name}' (indicates 6Hutc) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
    
    if validation_3_errors == 0:
        print("   🆗 Validation 3: All timeframes coherent with names")
        
        
    # --------------------------------------------------------------------
    # VALIDATION 4: Order amount range
    # --------------------------------------------------------------------
    validation_4_errors = 0
    MIN_ORDER_AMOUNT = 40
    MAX_ORDER_AMOUNT = 100
    
    for strat in strategies:
        strat_id      = strat.get('id', 'UNKNOWN')
        order_amount  = strat.get('order_amount', None)
        
        if order_amount is None:
            errors.append(
                f"❌ Strategy '{strat_id}' is missing 'order_amount' parameter"
            )
            validation_4_errors += 1
        elif not isinstance(order_amount, (int, float)):
            errors.append(
                f"❌ Strategy '{strat_id}' has invalid order_amount='{order_amount}' "
                f"(must be a number)"
            )
            validation_4_errors += 1
        elif order_amount < MIN_ORDER_AMOUNT:
            errors.append(
                f"❌ Strategy '{strat_id}' has order_amount={order_amount} "
                f"(minimum is {MIN_ORDER_AMOUNT})"
            )
            validation_4_errors += 1
        elif order_amount > MAX_ORDER_AMOUNT:
            errors.append(
                f"❌ Strategy '{strat_id}' has order_amount={order_amount} "
                f"(maximum is {MAX_ORDER_AMOUNT})"
            )
            validation_4_errors += 1
    
    if validation_4_errors == 0:
        print("   🆗 Validation 4: All order amounts within valid range (40-100)")
    
    # --------------------------------------------------------------------
    # VALIDATION 5: Required parameters for each strategy type
    # --------------------------------------------------------------------
    validation_5_errors = 0
    
    # Define required parameters by strategy base name (without timeframe)
    STRATEGY_TYPE_REQUIRED_PARAMS = {
        'double_top_long': ['lookback', 'tolerance', 'trend_th'],
        'reversal_long': ['lookback', 'tolerance', 'ma_period'],
        'reversal_short': ['lookback', 'tolerance', 'ma_period'],
        'parity_long': ['lookback', 'tolerance', 'ma_period'],
        'parity_short': ['lookback', 'tolerance', 'ma_period'],
        'orderblocks_long': ['lookback', 'tolerance', 'impulse'],
        'orderblocks_short': ['lookback', 'tolerance', 'impulse'],
    }
    
    # Common required parameters for all strategies
    COMMON_REQUIRED_PARAMS = ['id', 'name', 
                              'timeframe', 'active', 
                              'sell_after_ncandles', 
                              'order_amount', 
                              'tp_pct', 'sl_pct', 'direction']
    
    for strat in strategies:
        strat_id   = strat.get('id', 'UNKNOWN')
        strat_name = strat.get('name', '')
        
        # Check common parameters
        for param in COMMON_REQUIRED_PARAMS:
            if param not in strat:
                errors.append(
                    f"❌ Strategy '{strat_id}' is missing required parameter: '{param}'"
                )
                validation_5_errors += 1
        
        base_type = None
        for suffix in ['_4H', '_1H', '_6Hutc', '_12H', '_8H','30m']:  # Add more timeframes if needed
            if strat_name.endswith(suffix):
                base_type = strat_name[:-len(suffix)]
                break
        
        # Check strategy-specific parameters
        if base_type and base_type in STRATEGY_TYPE_REQUIRED_PARAMS:
            required_params = STRATEGY_TYPE_REQUIRED_PARAMS[base_type]
            for param in required_params:
                if param not in strat:
                    errors.append(
                        f"❌ Strategy '{strat_id}' (type: {base_type}) is missing "
                        f"required parameter: '{param}'"
                    )
                    validation_5_errors += 1
        elif base_type:
            warnings.append(
                f"⚠️  Strategy '{strat_id}' base type '{base_type}' has no parameter "
                f"requirements defined. Add it to STRATEGY_TYPE_REQUIRED_PARAMS."
            )
    
    if validation_5_errors == 0:
        print("   🆗 Validation 5: All strategies have required parameters")
        
    # --------------------------------------------------------------------
    # VALIDATION 6: TP/SL
    # --------------------------------------------------------------------
    validation_6_errors = 0
    MIN_TP_PCT = 1.5
    MAX_TP_PCT = 10
    MIN_SL_PCT = 1.5
    MAX_SL_PCT = 10
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        tp_pct = strat.get('tp_pct', None)
        sl_pct = strat.get('sl_pct', None)
        
        if tp_pct and (tp_pct < MIN_TP_PCT or tp_pct > MAX_TP_PCT):
            errors.append(f"❌ Strategy '{strat_id}' has tp_pct={tp_pct} (valid range: {MIN_TP_PCT}-{MAX_TP_PCT}%)")
            validation_6_errors += 1
        
        if sl_pct and (sl_pct < MIN_SL_PCT or sl_pct > MAX_SL_PCT):
            errors.append(f"❌ Strategy '{strat_id}' has sl_pct={sl_pct} (valid range: {MIN_SL_PCT}-{MAX_SL_PCT}%)")
            validation_6_errors += 1
    
    if validation_6_errors == 0:
        print("   🆗 Validation 6: All TP/SL percentages within valid ranges (1.5 - 10)")

    # --------------------------------------------------------------------
    # VALIDATION 7: IDs
    # --------------------------------------------------------------------        
    validation_7_errors = 0
    ids = [s.get('id') for s in strategies]
    duplicates = [id for id in set(ids) if ids.count(id) > 1]
    
    if duplicates:
        errors.append(f"❌ Duplicate strategy IDs found: {duplicates}")
        validation_7_errors += 1
    
    if validation_7_errors == 0:
        print("   🆗 Validation 7: All strategy IDs are unique")
    # --------------------------------------------------------------------
    # VALIDATION 8: Candles
    # -------------------------------------------------------------------- 
    validation_8_errors = 0
    MIN_CANDLES = 45
    MAX_CANDLES = 55
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        candles = strat.get('sell_after_ncandles', None)
        
        if candles and (candles < MIN_CANDLES or candles > MAX_CANDLES):
            errors.append(f"❌  Strategy '{strat_id}' has sell_after_ncandles={candles} (typical range: {MIN_CANDLES}-{MAX_CANDLES})")
    if validation_8_errors == 0:
        print("   🆗 Validation 8: All sell_after_ncandles checked (45-55)")
    
    # --------------------------------------------------------------------
    # VALIDATION 9: No duplicate strategies (name + timeframe)
    # -------------------------------------------------------------------- 
    validation_9_errors = 0
    seen_combinations = set()
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        name = strat.get('name', '')
        timeframe = strat.get('timeframe', '')
        
        combination = (name, timeframe)
        
        if combination in seen_combinations:
            errors.append(
                f"❌ Duplicate strategy found: name='{name}', timeframe='{timeframe}' "
                f"(strategy '{strat_id}' conflicts with another)"
            )
            validation_9_errors += 1
        else:
            seen_combinations.add(combination)
    
    if validation_9_errors == 0:
        print("   🆗 Validation 9: All strategies names + Timeframe are unique.")
        
    # --------------------------------------------------------------------
    # VALIDATION 10: Unique strategy names
    # -------------------------------------------------------------------- 
    validation_10_errors = 0
    names = [s.get('name') for s in strategies]
    duplicates = [name for name in set(names) if names.count(name) > 1]
    
    if duplicates:
        errors.append(f"❌ Duplicate strategy names found: {duplicates}")
        validation_10_errors += 1
    
    if validation_10_errors == 0:
        print("   🆗 Validation 10: All strategy names are unique")
        
    # --------------------------------------------------------------------
    # VALIDATION 12: Valid timeframes
    # -------------------------------------------------------------------- 
    validation_12_errors = 0
    VALID_TIMEFRAMES = ['1H', '4H', '6Hutc']
    
    for strat in strategies:
        strat_id = strat.get('id', 'UNKNOWN')
        timeframe = strat.get('timeframe', '')
        
        if timeframe not in VALID_TIMEFRAMES:
            errors.append(
                f"❌ Strategy '{strat_id}' has invalid timeframe='{timeframe}' "
                f"(valid: {', '.join(VALID_TIMEFRAMES)})"
            )
            validation_12_errors += 1
    
    if validation_12_errors == 0:
        print("   🆗 Validation 12: All timeframes are valid")
    return errors, warnings