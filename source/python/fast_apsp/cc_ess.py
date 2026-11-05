"""
CC-ESS wrapper for diameter and radius computation
"""

import ctypes
import numpy as np
from .utils import load_dll, prepare_matrix

# Load the DLL once
_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = load_dll('cc_ess.dll')
        # Define the C function signature
        _lib.cc_ess_py.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # adj_matrix_data
            ctypes.c_int64,                    # n
            ctypes.c_int64,                    # k (number of initial sweeps)
            ctypes.POINTER(ctypes.c_int64),   # diameter output
            ctypes.POINTER(ctypes.c_int64),   # radius output
            ctypes.POINTER(ctypes.c_double)   # elapsed_time
        ]
        _lib.cc_ess_py.restype = ctypes.c_int
    return _lib


def cc_ess(adj_matrix, k=4):
    """
    Args:
        adj_matrix: Adjacency matrix (can be dense or scipy sparse)
        k (int): Number of initial sweeps (default: 4)
        
    Returns:
        dict: Dictionary containing:
            - 'diameter': Network diameter
            - 'radius': Network radius
            - 'elapsed_time': Execution time in seconds
    """
    lib = _get_lib()
    
    # Prepare input matrix
    adj_matrix = prepare_matrix(adj_matrix)
    n = adj_matrix.shape[0]
    
    # Allocate output values
    diameter = ctypes.c_int64()
    radius = ctypes.c_int64()
    elapsed_time = ctypes.c_double()
    
    # Call the C function
    result_code = lib.cc_ess_py(
        adj_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        ctypes.c_int64(k),
        ctypes.byref(diameter),
        ctypes.byref(radius),
        ctypes.byref(elapsed_time)
    )
    
    if result_code != 0:
        raise RuntimeError(f"CC-ESS failed with error code {result_code}")
    
    return {
        'diameter': diameter.value,
        'radius': radius.value,
        'elapsed_time': elapsed_time.value
    }
