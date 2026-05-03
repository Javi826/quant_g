"""
BOT_batch/tools/pys_comp.py

Compares production strategies_E1.py against strategies_E1_from_csv.py
and prints a field-by-field diff to console.
"""

import sys
import os
import importlib.util

# =============================================================================
# PATHS
# =============================================================================
PRODUCTION_PATH = "/home/javi/projects/quant/quant_b/bitget/BOT_trading/config/strategies_00.py"
CANDIDATE_PATH  = "/home/javi/projects/quant/quant_b/bitget/BOT_batch/strategies_files/strategies_BT_00_batch.py"
#CANDIDATE_PATH  = "/home/javi/projects/quant/quant_b/bitget/BOT_batch/strategies_E1/strategies_E1_batch.py"
IGNORE_FIELDS   = {"order_amount", "order_amount_prod"}
# =============================================================================
# LOADER
# =============================================================================
def load_strategies(path: str) -> dict[str, dict]:
    """Load STRATEGIES from a .py file and index by strategy id."""
    spec   = importlib.util.spec_from_file_location("_strategies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategies = getattr(module, "STRATEGIES", None)
    if strategies is None:
        print(f"❌ STRATEGIES not found in {path}")
        sys.exit(1)
    return {s["id"]: s for s in strategies}

# =============================================================================
# COMPARE
# =============================================================================
def compare(prod: dict[str, dict], cand: dict[str, dict]) -> bool:
    """Print diff between production and candidate. Returns True if identical."""
    prod_ids = set(prod.keys())
    cand_ids = set(cand.keys())

    only_in_prod = prod_ids - cand_ids
    only_in_cand = cand_ids - prod_ids
    common_ids   = prod_ids & cand_ids

    has_diff = False

    if only_in_prod:
        has_diff = True
        print("\n⚠️  Strategies only in PRODUCTION (missing in candidate):")
        for sid in sorted(only_in_prod):
            print(f"   - {sid}")

    if only_in_cand:
        has_diff = True
        print("\n⚠️  Strategies only in CANDIDATE (new, not in production):")
        for sid in sorted(only_in_cand):
            print(f"   + {sid}")

    for sid in sorted(common_ids):
        p = prod[sid]
        c = cand[sid]
        all_keys = set(p.keys()) | set(c.keys())
        field_diffs = []
        IGNORE_FIELDS = {"order_amount", "order_amount_prod"}
        for key in sorted(all_keys):
            if key in IGNORE_FIELDS:
                continue
            p_val = p.get(key, "<missing>")
            c_val = c.get(key, "<missing>")
            if p_val != c_val:
                field_diffs.append((key, p_val, c_val))

        if field_diffs:
            has_diff = True
            print(f"\n{'─' * 60}")
            print(f"  {sid}")
            print(f"{'─' * 60}")
            print(f"  {'FIELD':<25} {'PRODUCTION':<20} {'CANDIDATE'}")
            print(f"  {'─'*25} {'─'*20} {'─'*20}")
            for key, p_val, c_val in field_diffs:
                print(f"  {key:<25} {str(p_val):<20} {str(c_val)}")

    return not has_diff

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"  Production : {PRODUCTION_PATH}")
    print(f"  Candidate  : {CANDIDATE_PATH}")

    prod = load_strategies(PRODUCTION_PATH)
    cand = load_strategies(CANDIDATE_PATH)

    print(f"\n  Production strategies : {len(prod)}")
    print(f"  Candidate strategies  : {len(cand)}")

    identical = compare(prod, cand)

    if identical:
        print("\n✅ No differences found — files are identical.")
    else:
        print(f"\n{'═' * 60}")
        print("⚠️  Differences found — review before deploying to production.")
        print(f"{'═' * 60}")


if __name__ == "__main__":
    main()