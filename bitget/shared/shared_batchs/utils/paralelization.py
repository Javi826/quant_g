#shared/shared_batchs/utils/paralelization.py
import numpy as np
from multiprocessing.shared_memory import SharedMemory


def arrays_to_shared_memory(base_arrays: dict) -> tuple:
    """Write a dict[symbol][field] -> numpy array structure to shared memory
    once, so many tasks can attach to it by name instead of each being
    pickled a full copy. Returns (shm_list, metadata). Non-array values are
    passed through as-is in metadata. Caller owns shm_list and must close()
    + unlink() each block once every worker has finished using it."""
    shm_list = []
    metadata = {}
    for sym, arr_dict in base_arrays.items():
        metadata[sym] = {}
        for key, arr in arr_dict.items():
            if isinstance(arr, np.ndarray):
                shm    = SharedMemory(create=True, size=max(arr.nbytes, 1))
                buf    = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                buf[:] = arr
                shm_list.append(shm)
                metadata[sym][key] = {"name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}
            else:
                metadata[sym][key] = {"value": arr}
    return shm_list, metadata


def arrays_from_shared_memory(metadata: dict) -> tuple:
    """Reconstruct the dict[symbol][field] -> numpy array structure from shared
    memory metadata inside a worker. Returns (base_arrays, shm_handles). Caller
    must close() each handle (not unlink — only the creator unlinks) once done."""
    base_arrays = {}
    shm_handles = []
    for sym, fields in metadata.items():
        base_arrays[sym] = {}
        for key, info in fields.items():
            if "name" in info:
                shm = SharedMemory(name=info["name"], create=False)
                shm_handles.append(shm)
                base_arrays[sym][key] = np.ndarray(info["shape"], dtype=np.dtype(info["dtype"]), buffer=shm.buf)
            else:
                base_arrays[sym][key] = info["value"]
    return base_arrays, shm_handles