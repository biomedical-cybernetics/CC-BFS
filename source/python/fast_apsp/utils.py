"""
Utility functions for loading DLLs and handling ctypes operations
"""

import ctypes
import os
import sys
import numpy as np


def _library_filename(library_name):
    """Return the native shared-library filename for the current platform."""
    stem, extension = os.path.splitext(library_name)
    if extension.lower() in {".dll", ".so", ".dylib"}:
        library_name = stem
    if sys.platform == "win32":
        return f"{library_name}.dll"
    if sys.platform.startswith("linux"):
        return f"{library_name}.so"
    if sys.platform == "darwin":
        return f"{library_name}.dylib"
    raise OSError(f"Unsupported platform: {sys.platform}")


def load_library(library_name):
    """
    Load a DLL from the lib/ directory
    
    Args:
        library_name (str): Base name or platform-specific library filename.
        
    Returns:
        ctypes.CDLL: Loaded library
        
    Raises:
        OSError: If the DLL file is not found
    """
    # Get the directory where this package is located
    package_dir = os.path.dirname(__file__)
    filename = _library_filename(library_name)
    dll_path = os.path.join(package_dir, '..', 'lib', filename)
    
    if not os.path.exists(dll_path):
        raise OSError(
            f"Shared library not found: {dll_path}. "
            "Compile the native libraries using the platform build utility."
        )
    
    return ctypes.CDLL(dll_path)


def load_dll(dll_name):
    """Backward-compatible alias for the original Windows-oriented wrappers."""
    return load_library(dll_name)


def prepare_matrix(adj_matrix):
    """
    Convert adjacency matrix to a C-contiguous float64 numpy array
    
    Args:
        adj_matrix: Input adjacency matrix (dense or sparse)
        
    Returns:
        numpy.ndarray: C-contiguous float64 array
        
    Raises:
        ValueError: If the matrix is not 2D
    """
    # Validate before casting: float64 conversion would discard imaginary data.
    if np.iscomplexobj(adj_matrix):
        raise ValueError("Adjacency matrix must contain real binary values 0 or 1")

    # Handle scipy sparse matrices
    try:
        from scipy import sparse
        if sparse.issparse(adj_matrix):
            adj_matrix = adj_matrix.toarray()
    except ImportError:
        pass
    
    # Convert to numpy array if not already
    adj_matrix = np.asarray(adj_matrix, dtype=np.float64)
    
    # Ensure it is a nonempty square matrix.
    if adj_matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {adj_matrix.shape}")
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {adj_matrix.shape}")
    if adj_matrix.shape[0] < 1:
        raise ValueError("Expected a nonempty adjacency matrix")

    if not np.all(np.isfinite(adj_matrix)) or not np.all(
        (adj_matrix == 0.0) | (adj_matrix == 1.0)
    ):
        raise ValueError("Adjacency matrix must contain only finite binary values 0 or 1")
    
    # Ensure it's C-contiguous
    if not adj_matrix.flags['C_CONTIGUOUS']:
        adj_matrix = np.ascontiguousarray(adj_matrix)
    
    return adj_matrix


def call_algorithm(dll, func_name, adj_matrix, *args, **kwargs):
    """
    Generic function to call a C algorithm with proper ctypes handling
    
    This is a helper function used internally by the algorithm wrappers.
    """
    # Prepare input matrix
    adj_matrix = prepare_matrix(adj_matrix)
    n = adj_matrix.shape[0]
    
    if adj_matrix.shape[1] != n:
        raise ValueError(f"Expected square matrix, got shape {adj_matrix.shape}")
    
    return adj_matrix, n
