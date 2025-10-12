# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def get_price_at_int(long t_int, np.ndarray[np.int64_t, ndim=1] ts_int,
                     np.ndarray[np.float64_t, ndim=1] close):
    """
    Devuelve el precio más cercano <= t_int en ts_int
    """
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = ts_int.shape[0] - 1
    cdef Py_ssize_t mid

    if ts_int.shape[0] == 0:
        return float('nan')

    # binary search
    while left <= right:
        mid = (left + right) // 2
        if ts_int[mid] == t_int:
            return close[mid]
        elif ts_int[mid] < t_int:
            left = mid + 1
        else:
            right = mid - 1

    if right < 0:
        return float('nan')
    else:
        return close[right]
