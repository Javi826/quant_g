#!/usr/bin/env python3
"""
Compare YAML vs Python Strategy Configurations

Validates that Python strategy files match their YAML counterparts
field by field, value by value, for all strategies across all accounts.

Usage:
    python compare_yaml_python.py
"""

import os
import sys
import yaml
from typing import Dict, List, Any

# Add parent directory to path to import strategies modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACCOUNTS = ['00', 'E1', '01']

def load_yaml_strategies(account: str) -> List[Dict]:
    """Load strategies from YAML file"""
    yaml_path = f'strategies_{account}.yaml'
    
    if not os.path.exists(yaml_path):
        print(f"⚠️  YAML file not found: {yaml_path}")
        return []
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data.get('strategies', [])


def load_python_strategies(account: str) -> List[Dict]:
    """Load strategies from Python file"""
    module_name = f'strategies_{account}'
    
    try:
        module = __import__(module_name)
        return module.STRATEGIES
    except ImportError:
        print(f"⚠️  Python module not found: {module_name}.py")
        return []


def compare_value(field: str, yaml_val: Any, py_val: Any) -> tuple[bool, str]:
    """
    Compare two values with type awareness
    
    Returns:
        (match: bool, message: str)
    """
    # Handle boolean conversion (YAML: true/false, Python: True/False)
    if isinstance(yaml_val, bool) and isinstance(py_val, bool):
        if yaml_val == py_val:
            return True, ""
        else:
            return False, f"YAML={yaml_val} vs Python={py_val}"
    
    # Handle numeric comparisons (int vs float)
    if isinstance(yaml_val, (int, float)) and isinstance(py_val, (int, float)):
        if yaml_val == py_val:
            return True, ""
        else:
            return False, f"YAML={yaml_val} vs Python={py_val}"
    
    # Handle string comparisons
    if isinstance(yaml_val, str) and isinstance(py_val, str):
        if yaml_val == py_val:
            return True, ""
        else:
            return False, f"YAML='{yaml_val}' vs Python='{py_val}'"
    
    # Type mismatch
    if type(yaml_val) != type(py_val):
        return False, f"Type mismatch: YAML={type(yaml_val).__name__}({yaml_val}) vs Python={type(py_val).__name__}({py_val})"
    
    # Generic comparison
    if yaml_val == py_val:
        return True, ""
    else:
        return False, f"YAML={yaml_val} vs Python={py_val}"


def compare_strategy(yaml_strat: Dict, py_strat: Dict, strat_idx: int) -> tuple[bool, List[str]]:
    """
    Compare single strategy field by field
    
    Returns:
        (all_match: bool, errors: List[str])
    """
    strat_id = yaml_strat.get('id', f'Strategy #{strat_idx+1}')
    errors = []
    
    # Get all unique fields from both
    all_fields = set(yaml_strat.keys()) | set(py_strat.keys())
    
    for field in sorted(all_fields):
        # Check if field exists in both
        if field not in yaml_strat:
            errors.append(f"  ❌ Field '{field}' missing in YAML")
            continue
        
        if field not in py_strat:
            errors.append(f"  ❌ Field '{field}' missing in Python")
            continue
        
        # Compare values
        match, msg = compare_value(field, yaml_strat[field], py_strat[field])
        
        if not match:
            errors.append(f"  ❌ Field '{field}': {msg}")
    
    return len(errors) == 0, errors


def compare_account(account: str) -> bool:
    """
    Compare strategies for a single account
    
    Returns:
        True if all strategies match
    """
    print(f"\n{'=' * 80}")
    print(f"ACCOUNT {account}")
    print(f"{'=' * 80}")
    
    yaml_strategies = load_yaml_strategies(account)
    py_strategies = load_python_strategies(account)
    
    if not yaml_strategies and not py_strategies:
        print(f"⚠️  No strategies found for account {account}")
        return True
    
    if not yaml_strategies:
        print(f"❌ YAML file missing or empty")
        return False
    
    if not py_strategies:
        print(f"❌ Python file missing or empty")
        return False
    
    print(f"\nYAML strategies: {len(yaml_strategies)}")
    print(f"Python strategies: {len(py_strategies)}")
    
    # Check count matches
    if len(yaml_strategies) != len(py_strategies):
        print(f"\n❌ STRATEGY COUNT MISMATCH!")
        print(f"   YAML: {len(yaml_strategies)} strategies")
        print(f"   Python: {len(py_strategies)} strategies")
        return False
    
    # Compare each strategy
    all_match = True
    
    for idx, (yaml_strat, py_strat) in enumerate(zip(yaml_strategies, py_strategies)):
        strat_id = yaml_strat.get('id', f'Strategy #{idx+1}')
        
        # Verify IDs match (strategies should be in same order)
        yaml_id = yaml_strat.get('id')
        py_id = py_strat.get('id')
        
        if yaml_id != py_id:
            print(f"\n❌ Strategy order mismatch at position {idx+1}:")
            print(f"   YAML ID: {yaml_id}")
            print(f"   Python ID: {py_id}")
            all_match = False
            continue
        
        # Compare fields
        match, errors = compare_strategy(yaml_strat, py_strat, idx)
        
        if match:
            print(f"  ✅ {strat_id}: All fields match")
        else:
            print(f"\n  ❌ {strat_id}: MISMATCHES FOUND")
            for error in errors:
                print(error)
            all_match = False
    
    print(f"\n{'=' * 80}")
    if all_match:
        print(f"✅ Account {account}: PERFECT MATCH (all strategies identical)")
    else:
        print(f"❌ Account {account}: MISMATCHES FOUND")
    print(f"{'=' * 80}")
    
    return all_match


def main():
    """Main comparison function"""
    print("=" * 80)
    print("YAML vs PYTHON STRATEGY CONFIGURATION COMPARISON")
    print("=" * 80)
    
    results = {}
    
    for account in ACCOUNTS:
        results[account] = compare_account(account)
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    
    for account, result in results.items():
        status = "✅ MATCH" if result else "❌ MISMATCH"
        print(f"Account {account}: {status}")
    
    print(f"{'=' * 80}")
    
    if all(results.values()):
        print("✅ ALL ACCOUNTS MATCH - Python configs are identical to YAML")
        return 0
    else:
        print("❌ SOME ACCOUNTS HAVE MISMATCHES - Review details above")
        return 1


if __name__ == '__main__':
    sys.exit(main())