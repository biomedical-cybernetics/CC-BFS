"""Utilities for loading network data from MATLAB .mat files"""

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import h5py


def _reconstruct_sparse_from_group(obj):
    """Reconstruct sparse matrix from h5py Group containing data, ir, jc fields"""
    data = np.array(obj['data']).flatten()
    ir = np.array(obj['ir']).flatten()
    jc = np.array(obj['jc']).flatten()
    # Get matrix dimensions
    if 'dims' in obj:
        dims = np.array(obj['dims']).flatten()
        shape = (int(dims[0]), int(dims[1]))
    else:
        # Infer shape from jc length and max ir value
        shape = (int(ir.max()) + 1 if len(ir) > 0 else 0, len(jc) - 1)
    # Create sparse matrix and convert to dense
    sparse_mat = csc_matrix((data, ir, jc), shape=shape)
    return sparse_mat.toarray()


def load_mat_variable(filepath, var_names, expected_shape=None, dtype=np.float64):
    """Load a variable from a .mat file, supporting v7 and v7.3 formats
    
    Args:
        filepath: Path to the .mat file
        var_names: Variable name(s) to load. Can be a string or list of strings.
                  If a list, tries each name in order until one is found.
        expected_shape: Optional tuple (n, m) to validate matrix shape. If None, no validation.
        dtype: Data type to convert to (default: np.float64)
        
    Returns:
        tuple: (matrix, var_name) where:
               - matrix: The loaded matrix (numpy array)
               - var_name: Name of the variable found
    
    Raises:
        ValueError: If variable not found or shape mismatch (when expected_shape is provided)
    """
    # Normalize var_names to a list
    if isinstance(var_names, str):
        var_names = [var_names]
    
    # Try loading with scipy.io.loadmat first (v7 format)
    try:
        data = loadmat(filepath)
        
        for var_name in var_names:
            if var_name in data:
                matrix = data[var_name]
                
                # Handle cell arrays: extract first element
                if isinstance(matrix, np.ndarray) and matrix.dtype == 'object':
                    matrix = matrix[0, 0]
                
                # Convert to dense if sparse
                if hasattr(matrix, 'toarray'):
                    matrix = matrix.toarray()
                
                # Ensure correct dtype and shape
                matrix = np.asarray(matrix, dtype=dtype)
                
                # Validate shape if expected_shape is provided
                if expected_shape is not None and matrix.shape != expected_shape:
                    raise ValueError(f"{var_name} shape {matrix.shape} doesn't match expected shape {expected_shape}")
                
                return matrix, var_name
        
        # None of the variables found
        raise ValueError(f"None of the variables {var_names} found in {filepath}")
        
    except NotImplementedError:
        # Fallback for v7.3 .mat files (HDF5 format)
        with h5py.File(filepath, 'r') as f:
            for var_name in var_names:
                if var_name in f:
                    obj = f[var_name]
                    
                    # Handle cell arrays in v7.3: dataset of HDF5 object references
                    if isinstance(obj, h5py.Dataset) and obj.dtype == 'object':
                        first_ref = obj[0, 0]
                        obj = f[first_ref]
                    
                    # Check if it's a sparse matrix (HDF5 group) or dense array (dataset)
                    if isinstance(obj, h5py.Group) and all(k in obj for k in ['data', 'ir', 'jc']):
                        matrix = _reconstruct_sparse_from_group(obj)
                    else:
                        matrix = np.array(obj)
                    
                    # Ensure correct dtype and shape
                    matrix = np.asarray(matrix, dtype=dtype)
                    
                    # Validate shape if expected_shape is provided
                    if expected_shape is not None and matrix.shape != expected_shape:
                        raise ValueError(f"{var_name} shape {matrix.shape} doesn't match expected shape {expected_shape}")
                    
                    return matrix, var_name
            
            # None of the variables found
            raise ValueError(f"None of the variables {var_names} found in v7.3 file {filepath}")

