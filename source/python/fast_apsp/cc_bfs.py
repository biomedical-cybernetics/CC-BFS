"""
CC-BFS All-Pairs Shortest Path wrapper
"""

import ctypes
from ._distance_output import DistanceAllocator, NativeDistanceOutput
from .utils import load_dll, prepare_matrix

_lib = None


class CcBfsCounters(ctypes.Structure):
    """
    Mirrors CC_BFS_PY_COUNTERS from cc_bfs_apsp_l_cpp.cpp.
    Field order and types must stay in sync with the C struct.
    Note: the std::string data_type field is omitted (not C-compatible).
    """
    _fields_ = [
        ("vec_comp_count",                       ctypes.c_int64),
        ("source_with_cs_count",                 ctypes.c_int64),
        ("source_without_cs_count",              ctypes.c_int64),
        ("hn_count",                             ctypes.c_int64),
        ("graph_expansion_node_count",           ctypes.c_int64),
        ("spdag_expansion_node_count",           ctypes.c_int64),
        ("spill_expansion_node_count",           ctypes.c_int64),
        ("sptree_expansion_node_count",          ctypes.c_int64),
        ("cst_visited_count",                    ctypes.c_int64),
        ("top_down_node_count",                  ctypes.c_int64),
        ("bottom_up_node_count",                 ctypes.c_int64),
        ("time_secs_comp_adj",                   ctypes.c_double),
        ("time_secs_diam_upperbound",            ctypes.c_double),
        ("time_secs_hn_sss",                     ctypes.c_double),
        ("time_secs_non_hn_sss",                 ctypes.c_double),
        ("time_secs_convert",                    ctypes.c_double),
        ("time_secs_overall",                    ctypes.c_double),
        ("diam_upper",                           ctypes.c_int64),
        # data_type omitted (std::string, not C-compatible)
        ("estimated_dominant_memory_bytes",      ctypes.c_int64),
        ("distance_matrix_bytes",                ctypes.c_int64),
        ("sp_dag_bytes",                         ctypes.c_int64),
        ("sp_shortcut_bytes",                    ctypes.c_int64),
        ("sp_tree_bytes",                        ctypes.c_int64),
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
        library = load_dll('cc_bfs_apsp.dll')
        try:
            typed_call = library.cc_bfs_apsp_typed_py
        except AttributeError as error:
            raise RuntimeError("Rebuild the CC-BFS native library for integer Python outputs.") from error
        typed_call.argtypes = [
            ctypes.POINTER(ctypes.c_double),   # adj_matrix_data
            ctypes.c_int64,                     # n
            DistanceAllocator,                 # NumPy allocation callback, or null
            ctypes.POINTER(ctypes.c_double),   # elapsed_time
            ctypes.c_char_p,                   # hub_constraint (NULL => C++ default)
            ctypes.POINTER(CcBfsCounters),      # out_counters (may be NULL)
        ]
        typed_call.restype = ctypes.c_int
        _lib = library
    return _lib


def cc_bfs_apsp(adj_matrix, return_distances=True, hub_constraint=None):
    """
    Args:
        adj_matrix: Adjacency matrix of a connected, unweighted, undirected
            graph (can be dense or scipy sparse).
        return_distances: If False, skip allocating and copying the n x n
            distance matrix back to Python.
        hub_constraint: Optional hub-selection constraint string. None uses
            the C++ default, "bounded-avg-deg".

    Returns:
        dict: Dictionary containing:
            - 'distances': NumPy-owned n x n matrix in the native unsigned
              integer dtype, or None when return_distances=False
            - 'elapsed_time': Execution time in seconds
            - 'counters': CcBfsCounters with memory and algorithm statistics
    """
    lib = _get_lib()

    adj_matrix = prepare_matrix(adj_matrix)
    n = adj_matrix.shape[0]

    output = NativeDistanceOutput(return_distances)
    elapsed_time = ctypes.c_double()
    counters = CcBfsCounters()
    if hub_constraint is None:
        hub_constraint_arg = None
    elif isinstance(hub_constraint, bytes):
        hub_constraint_arg = hub_constraint
    elif isinstance(hub_constraint, str):
        hub_constraint_arg = hub_constraint.encode('utf-8')
    else:
        raise TypeError("hub_constraint must be str, bytes, or None")

    result_code = lib.cc_bfs_apsp_typed_py(
        adj_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        output.callback,
        ctypes.byref(elapsed_time),
        hub_constraint_arg,
        ctypes.byref(counters),
    )

    output.raise_if_failed()
    if result_code != 0:
        raise RuntimeError(f"CC-BFS APSP failed with error code {result_code}")

    return {
        'distances': output.array,
        'elapsed_time': elapsed_time.value,
        'estimated_memory_bytes': counters.estimated_dominant_memory_bytes,
        'estimated_total_memory_bytes': counters.estimated_total_memory_bytes,
        'adjacency_list_bytes': counters.adjacency_list_bytes,
        'process_delta_peak_bytes': counters.process_delta_peak_private_bytes,
        'counters': counters,
    }
