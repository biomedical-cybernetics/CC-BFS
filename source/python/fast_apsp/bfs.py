"""
BFS-based All-Pairs Shortest Path (ZM-BFS) wrapper
"""

import ctypes
import numpy as np
from .utils import load_dll, prepare_matrix

_lib = None


class BfsCounters(ctypes.Structure):
    """
    Mirrors BFS_PY_COUNTERS from bfs_apsp_l_cpp.cpp.
    Field order and types must stay in sync with the C struct.
    """
    _fields_ = [
        ("estimated_dominant_memory_bytes",      ctypes.c_int64),
        ("distance_matrix_bytes",                ctypes.c_int64),
        ("adjacency_list_bytes",                 ctypes.c_int64),
        ("estimated_total_memory_bytes",         ctypes.c_int64),
        ("process_baseline_working_set_bytes",   ctypes.c_int64),
        ("process_peak_working_set_bytes",       ctypes.c_int64),
        ("process_delta_peak_working_set_bytes", ctypes.c_int64),
        ("process_baseline_private_bytes",       ctypes.c_int64),
        ("process_peak_private_bytes",           ctypes.c_int64),
        ("process_delta_peak_private_bytes",     ctypes.c_int64),
    ]


def _get_lib():
    global _lib
    if _lib is None:
        _lib = load_dll('bfs_apsp.dll')
        _lib.bfs_apsp_py.argtypes = [
            ctypes.POINTER(ctypes.c_double),   # adj_matrix_data
            ctypes.c_int64,                     # n
            ctypes.POINTER(ctypes.c_double),   # output_distances
            ctypes.POINTER(ctypes.c_double),   # elapsed_time
            ctypes.POINTER(BfsCounters),        # out_counters (may be NULL)
        ]
        _lib.bfs_apsp_py.restype = ctypes.c_int
    return _lib


def bfs_apsp(adj_matrix, return_distances=True):
    """
    Args:
        adj_matrix: Adjacency matrix (can be dense or scipy sparse)
        return_distances: If False, skip allocating and copying the n x n
            distance matrix back to Python.

    Returns:
        dict: Dictionary containing:
            - 'distances': n x n distance matrix, or None when return_distances=False
            - 'elapsed_time': Execution time in seconds
            - 'counters': BfsCounters with memory and algorithm statistics
    """
    lib = _get_lib()

    adj_matrix = prepare_matrix(adj_matrix)
    n = adj_matrix.shape[0]

    distances = np.zeros((n, n), dtype=np.float64) if return_distances else None
    elapsed_time = ctypes.c_double()
    counters = BfsCounters()
    output_ptr = (distances.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                  if distances is not None else None)

    result_code = lib.bfs_apsp_py(
        adj_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        output_ptr,
        ctypes.byref(elapsed_time),
        ctypes.byref(counters),
    )

    if result_code != 0:
        raise RuntimeError(f"BFS APSP failed with error code {result_code}")

    return {
        'distances': distances,
        'elapsed_time': elapsed_time.value,
        'estimated_memory_bytes': counters.estimated_dominant_memory_bytes,
        'estimated_total_memory_bytes': counters.estimated_total_memory_bytes,
        'adjacency_list_bytes': counters.adjacency_list_bytes,
        'process_delta_peak_bytes': counters.process_delta_peak_private_bytes,
        'counters': counters,
    }
