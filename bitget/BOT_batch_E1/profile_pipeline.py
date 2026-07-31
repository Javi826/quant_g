#BOT_batch/profile_pipeline.py
import time
import atexit
import functools
import logging

import shared_batchs.pipeline.dsr as dsr
import shared_batchs.pipeline.wfo as wfo
import shared_batchs.pipeline.correlation as correlation
import shared_batchs.pipeline.montecarlo as montecarlo
import shared_batchs.pipeline.multiverse as multiverse

logger = logging.getLogger("BOT_batch.profiling")

_STATS = {}


def _instrument(module, func_name):
    original = getattr(module, func_name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        elapsed = time.perf_counter() - start
        entry = _STATS.setdefault(func_name, {"n_calls": 0, "total_s": 0.0})
        entry["n_calls"] += 1
        entry["total_s"] += elapsed
        return result

    setattr(module, func_name, wrapped)

def print_summary():
    if not _STATS:
        logger.info("PROFILING ── no calls recorded yet")
        return

    logger.info(f"\n{'─' * 100}")
    logger.info("  PIPELINE PROFILING SUMMARY — cumulative time per stage")
    logger.info(f"{'─' * 100}")
    logger.info(f"{'STAGE':<25}{'N_CALLS':<12}{'TOTAL_S':<14}{'AVG_S':<10}")
    logger.info(f"{'─' * 100}")
    for name, stats in sorted(_STATS.items(), key=lambda kv: kv[1]["total_s"], reverse=True):
        n_calls = stats["n_calls"]
        total_s = stats["total_s"]
        avg_s   = (total_s / n_calls) if n_calls else 0.0
        logger.info(f"{name:<25}{n_calls:<12}{total_s:<14.2f}{avg_s:<10.2f}")
    logger.info(f"{'─' * 100}\n")


# atexit only fires when the Python PROCESS exits — unreliable in Spyder/Jupyter,
# where the kernel process stays alive across runs. Kept as a fallback for plain
# `python main_MINER.py` runs; main_MINER.py also calls print_summary() explicitly.
atexit.register(print_summary)

# Each pipe_* is instrumented at its ORIGIN module. rule_runner.py imports
# pipe_dsr / pipe_wfo / pipe_correlation / pipe_montecarlo via
# `from ... import name` at rule_runner's own module-load time, so this
# module must be imported BEFORE rule_runner.py for the patch to take effect.
# pipe_multiverse is imported locally at call-time inside rule_runner, so it
# is unaffected by import ordering.
_instrument(dsr,         "pipe_dsr")
_instrument(wfo,         "pipe_wfo")
_instrument(correlation, "pipe_correlation")
_instrument(montecarlo,  "pipe_montecarlo")
_instrument(multiverse,  "pipe_multiverse")

print("PROFILING ── instrumentation active", flush=True)