#validate_backtest_fix.py
import os
import sys
import time
import hashlib
import importlib.util
import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")))
from shared_batchs.symbols.universe import filter_symbols, select_universe, select_top_n_by_volume
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.rule_mining.rule_runner import _build_rule_dicts
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT

# =============================================================================
# TEST CONFIG — small enough to run fast, large enough to exercise the caches
# (several rules per symbol, several combos) that the fix touches.
# =============================================================================
TIMEFRAME  = "1H"
N_SYMBOLS  = 3
PARAM_GRID = {"SELL_AFTER": [50], "TP_PCT": [6, 8], "SL_PCT": [6, 8]}

OLD_MODULE_PATH = os.path.join(SCRIPT_DIR, "backtest_runner.py")
NEW_MODULE_PATH = os.path.join(SCRIPT_DIR, "backtest_fixed.py")


def _load_module_by_real_name(path: str, real_module_name: str):
    """Load a .py file under a Python-importable module name (not an
    arbitrary alias). joblib/loky workers re-import delayed functions by
    module path when unpickling, so the name registered in sys.modules
    must be resolvable from a fresh worker process, not just from here.
    """
    spec   = importlib.util.spec_from_file_location(real_module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[real_module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_test_inputs() -> tuple:
    ohlcv_is = select_universe(
        data_folder_is    = DATA_FOLDER_IS,
        timeframe         = TIMEFRAME,
        min_price         = MIN_PRICE,
        filter_symbols_fn = filter_symbols,
    )
    ohlcv_is  = select_top_n_by_volume(ohlcv_is, N_SYMBOLS)
    ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)
    rules     = _build_rule_dicts(ohlcv_is, TIMEFRAME, RULE_MAX_DEPTH)
    return rules, ohlcv_arr


def _shm_count() -> int:
    return len(os.listdir("/dev/shm"))


def _hash_matrix(matrix_arr: np.ndarray, col_names) -> str:
    order = np.argsort(np.asarray(col_names, dtype=object))
    return hashlib.sha256(np.ascontiguousarray(matrix_arr[:, order]).tobytes()).hexdigest()


def _run_pipeline(module, rules: list, ohlcv_arr: dict) -> dict:
    shm_before = _shm_count()
    start      = time.time()

    raw_results, n_combos, matrix_arr, col_names = module.pipe_backtesting(
        rules        = rules,
        ohlcv_arr    = ohlcv_arr,
        param_grid   = PARAM_GRID,
        order_amount = ORDER_AMOUNT,
        timeframe    = TIMEFRAME,
    )

    return {
        "matrix_arr": matrix_arr,
        "col_names":  col_names,
        "n_combos":   n_combos,
        "elapsed":    time.time() - start,
        "shm_before": shm_before,
        "shm_after":  _shm_count(),
    }


def main() -> None:
    rules, ohlcv_arr = _build_test_inputs()

    print("Running OLD backtest_runner.py ...")
    old_module = _load_module_by_real_name(OLD_MODULE_PATH, "shared_batchs.pipeline.backtest_runner")
    old_result = _run_pipeline(old_module, rules, ohlcv_arr)

    print("Running NEW backtest_fixed.py ...")
    new_module = _load_module_by_real_name(NEW_MODULE_PATH, "backtest_fixed")
    new_result = _run_pipeline(new_module, rules, ohlcv_arr)

    old_hash  = _hash_matrix(old_result["matrix_arr"], old_result["col_names"])
    new_hash  = _hash_matrix(new_result["matrix_arr"], new_result["col_names"])
    same_cols = sorted(old_result["col_names"]) == sorted(new_result["col_names"])
    same_shape = old_result["matrix_arr"].shape == new_result["matrix_arr"].shape
    same_hash = old_hash == new_hash

    print("\n" + "-" * 70)
    print(f"  OLD  shape={old_result['matrix_arr'].shape}  elapsed={old_result['elapsed']:.1f}s  "
          f"shm_before={old_result['shm_before']}  shm_after={old_result['shm_after']}")
    print(f"  NEW  shape={new_result['matrix_arr'].shape}  elapsed={new_result['elapsed']:.1f}s  "
          f"shm_before={new_result['shm_before']}  shm_after={new_result['shm_after']}")
    print("-" * 70)
    print(f"  same column set    : {same_cols}")
    print(f"  same matrix shape  : {same_shape}")
    print(f"  same matrix hash   : {same_hash}")
    print(f"  shm leaked (old)   : {old_result['shm_after'] - old_result['shm_before']}")
    print(f"  shm leaked (new)   : {new_result['shm_after'] - new_result['shm_before']}")
    print("-" * 70)

    if same_cols and same_shape and same_hash:
        print("  RESULT: IDENTICAL — fix is safe to keep")
    else:
        print("  RESULT: MISMATCH — investigate before keeping the fix")


if __name__ == "__main__":
    main()