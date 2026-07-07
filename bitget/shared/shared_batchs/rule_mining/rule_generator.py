import numpy as np
from itertools import combinations

from shared_batchs.rule_mining.condition_bank import ConditionBank

MAX_DEPTH = 3
SIDES     = ("long", "short")


def generate_rule_combinations(condition_specs: list, max_depth: int = MAX_DEPTH) -> list:
    rules = []
    for depth in range(1, max_depth + 1):
        for combo in combinations(range(len(condition_specs)), depth):
            rules.append([condition_specs[i] for i in combo])
    return rules


def describe_rule(rule_specs: list) -> str:
    return " AND ".join(ConditionBank.describe(spec) for spec in rule_specs)


def build_signal_fn(rule_specs: list, side: str):

    def signal_fn(arr: dict, live_trading: bool = True) -> np.ndarray:
        bank = ConditionBank(arr)
        mask = np.ones(bank.n, dtype=bool)

        for spec in rule_specs:
            mask &= bank.evaluate(spec)

        signal = np.zeros(bank.n, dtype=np.int32)
        if side == "long":
            signal[mask] = 1
        else:
            signal[mask] = -1

        if not live_trading:
            signal = np.roll(signal, 1)
            signal[0] = 0

        return signal

    return signal_fn


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
                "label":      describe_rule(rule_specs),
                "signal_fn":  build_signal_fn(rule_specs, side),
            })

    return all_rules