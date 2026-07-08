#shared/shared_batchs/rule_mining/rule_generator.py
from itertools import combinations

from signals.rule_engine.condition_bank import ConditionBank
from signals.rule_engine.signal_builder import build_signal_fn, describe_rule

MAX_DEPTH = 3
SIDES     = ("long", "short")


def generate_rule_combinations(condition_specs: list, max_depth: int = MAX_DEPTH) -> list:
    rules = []
    for depth in range(1, max_depth + 1):
        for combo in combinations(range(len(condition_specs)), depth):
            rules.append([condition_specs[i] for i in combo])
    return rules


def generate_all_rules(arr_sample: dict, max_depth: int = MAX_DEPTH) -> list:
    bank            = ConditionBank(arr_sample)
    condition_specs = bank.build_condition_specs()
    rule_combos     = generate_rule_combinations(condition_specs, max_depth)
    all_rules = []
    for side in SIDES:
        for rule_specs in rule_combos:
            all_rules.append({
                "side":       side,
                "specs":      rule_specs,
                "label":      describe_rule(bank, rule_specs),
                "signal_fn":  build_signal_fn(rule_specs, side),
            })
    return all_rules