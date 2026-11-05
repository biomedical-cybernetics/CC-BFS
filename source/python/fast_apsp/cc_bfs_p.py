"""Python interface for parallel CC-BFS."""

import ctypes
import sys

import numpy as np
from scipy import sparse

from .utils import load_library, prepare_matrix
from ._distance_output import DistanceAllocator, NativeDistanceOutput


_lib = None


class CcBfsPCounters(ctypes.Structure):
    """C-compatible numeric counters returned by CC-BFS-P."""

    _fields_ = [
        ("vec_comp_count", ctypes.c_int64),
        ("source_with_cs_count", ctypes.c_int64),
        ("source_without_cs_count", ctypes.c_int64),
        ("hn_count", ctypes.c_int64),
        ("graph_expansion_node_count", ctypes.c_int64),
        ("spdag_expansion_node_count", ctypes.c_int64),
        ("spill_expansion_node_count", ctypes.c_int64),
        ("sptree_expansion_node_count", ctypes.c_int64),
        ("cst_visited_count", ctypes.c_int64),
        ("top_down_node_count", ctypes.c_int64),
        ("bottom_up_node_count", ctypes.c_int64),
        ("time_secs_comp_adj", ctypes.c_double),
        ("time_secs_diam_upperbound", ctypes.c_double),
        ("time_secs_hn_sss", ctypes.c_double),
        ("time_secs_non_hn_sss", ctypes.c_double),
        ("time_secs_convert", ctypes.c_double),
        ("time_secs_overall", ctypes.c_double),
        ("diam_upper", ctypes.c_int64),
        ("estimated_dominant_memory_bytes", ctypes.c_int64),
        ("distance_matrix_bytes", ctypes.c_int64),
        ("sp_dag_bytes", ctypes.c_int64),
        ("sp_shortcut_bytes", ctypes.c_int64),
        ("sp_tree_bytes", ctypes.c_int64),
        ("adjacency_list_bytes", ctypes.c_int64),
        ("estimated_total_memory_bytes", ctypes.c_int64),
        ("process_baseline_working_set_bytes", ctypes.c_int64),
        ("process_peak_working_set_bytes", ctypes.c_int64),
        ("process_delta_peak_working_set_bytes", ctypes.c_int64),
        ("process_baseline_private_bytes", ctypes.c_int64),
        ("process_peak_private_bytes", ctypes.c_int64),
        ("process_delta_peak_private_bytes", ctypes.c_int64),
        ("num_threads", ctypes.c_int64),
        ("time_secs_adjacency", ctypes.c_double),
        ("time_secs_hub", ctypes.c_double),
        ("time_secs_non_hub", ctypes.c_double),
        ("time_secs_commit_sync", ctypes.c_double),
        ("time_secs_conversion", ctypes.c_double),
    ]


def _get_lib():
    global _lib
    if _lib is None:
        library = load_library("cc_bfs_p")
        try:
            dense_call = library.cc_bfs_p_apsp_typed_py
            sparse_call = library.cc_bfs_p_apsp_csr_typed_py
        except AttributeError as error:
            raise RuntimeError("Rebuild the CC-BFS-P native library for integer Python outputs.") from error
        dense_call.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_char_p,
            DistanceAllocator,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(CcBfsPCounters),
        ]
        dense_call.restype = ctypes.c_int
        sparse_call.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_char_p,
            DistanceAllocator,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(CcBfsPCounters),
        ]
        sparse_call.restype = ctypes.c_int
        library.cc_bfs_p_last_error.argtypes = []
        library.cc_bfs_p_last_error.restype = ctypes.c_char_p
        _lib = library
    return _lib


def _prepare_sparse_adjacency(adj_matrix):
    """Prepare binary CSR rows without allocating a dense adjacency matrix."""
    if np.iscomplexobj(adj_matrix):
        raise ValueError("Adjacency matrix must contain real binary values 0 or 1")
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {adj_matrix.shape}")
    if adj_matrix.shape[0] < 1:
        raise ValueError("Expected a nonempty adjacency matrix")
    matrix = sparse.csr_matrix(adj_matrix, dtype=np.float64, copy=True)
    matrix.check_format(full_check=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)) or not np.all(matrix.data == 1.0):
        raise ValueError("Adjacency matrix must contain only finite binary values 0 or 1")
    if (matrix != matrix.T).nnz:
        raise ValueError("Adjacency matrix must be symmetric (undirected)")
    return (
        np.ascontiguousarray(matrix.indptr, dtype=np.int64),
        np.ascontiguousarray(matrix.indices, dtype=np.int64),
    )


def cc_bfs_p_apsp(
    adj_matrix,
    num_threads=1,
    return_distances=True,
    hub_constraint=None,
):
    """Compute exact APSP distances with shared-memory parallel CC-BFS.

    Parameters
    ----------
    adj_matrix : array-like or scipy sparse matrix
        Nonempty square, symmetric binary adjacency matrix of a connected, undirected
        graph. SciPy sparse matrices/arrays are passed as CSR rows without
        densifying the adjacency matrix.
    num_threads : int, default=1
        Number of OpenMP worker threads.
    return_distances : bool, default=True
        If false, complete the same computation but omit copying the distance
        matrix back to Python.
    hub_constraint : str or bytes, optional
        Hub-selection constraint. The default is ``bounded-avg-deg``.

    Returns
    -------
    dict
        ``distances`` is a NumPy-owned ``n``-by-``n`` hop-distance matrix in
        the native unsigned integer dtype (uint8/16/32/64), or
        ``None`` when ``return_distances`` is false. ``elapsed_time`` is the
        native algorithm time in seconds, ``num_threads`` is the applied
        OpenMP thread count, and ``counters`` contains the detailed work,
        phase-timing, and memory fields. ``process_delta_peak_bytes`` is
        unavailable (``None``) outside Windows.

    Notes
    -----
    The returned distance matrix is dense even when the adjacency is sparse.
    ``return_distances=False`` suppresses only the Python copy; it does not
    avoid the native distance matrix or return a path-length summary.
    """
    if isinstance(num_threads, bool) or not isinstance(num_threads, (int, np.integer)):
        raise TypeError("num_threads must be an integer")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if num_threads > np.iinfo(np.int64).max:
        raise ValueError("num_threads exceeds the signed 64-bit integer range")

    is_sparse = sparse.issparse(adj_matrix)
    if is_sparse:
        row_offsets, column_indices = _prepare_sparse_adjacency(adj_matrix)
        n = adj_matrix.shape[0]
    else:
        matrix = prepare_matrix(adj_matrix)
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("Adjacency matrix must be symmetric (undirected)")
        n = matrix.shape[0]
    output = NativeDistanceOutput(return_distances)

    if hub_constraint is None:
        constraint = None
    elif isinstance(hub_constraint, bytes):
        constraint = hub_constraint
    elif isinstance(hub_constraint, str):
        constraint = hub_constraint.encode("utf-8")
    else:
        raise TypeError("hub_constraint must be str, bytes, or None")

    library = _get_lib()
    elapsed_time = ctypes.c_double()
    counters = CcBfsPCounters()
    common_args = (
        ctypes.c_int64(int(num_threads)),
        constraint,
        output.callback,
        ctypes.byref(elapsed_time),
        ctypes.byref(counters),
    )
    if is_sparse:
        result_code = library.cc_bfs_p_apsp_csr_typed_py(
            row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            column_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(n),
            ctypes.c_int64(column_indices.size),
            *common_args,
        )
    else:
        result_code = library.cc_bfs_p_apsp_typed_py(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(n),
            *common_args,
        )
    output.raise_if_failed()
    if result_code != 0:
        encoded_error = library.cc_bfs_p_last_error()
        message = encoded_error.decode("utf-8", errors="replace") if encoded_error else "unknown error"
        raise RuntimeError(f"CC-BFS-P failed: {message}")

    return {
        "distances": output.array,
        "elapsed_time": elapsed_time.value,
        "num_threads": counters.num_threads,
        "estimated_memory_bytes": counters.estimated_dominant_memory_bytes,
        "estimated_total_memory_bytes": counters.estimated_total_memory_bytes,
        "adjacency_list_bytes": counters.adjacency_list_bytes,
        "process_delta_peak_bytes": (
            counters.process_delta_peak_private_bytes
            if sys.platform == "win32"
            else None
        ),
        "counters": counters,
    }
