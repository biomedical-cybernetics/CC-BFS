"""Allocate a NumPy-owned result after C++ selects its native unsigned type."""

import ctypes

import numpy as np


DistanceAllocator = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32)
_DTYPES = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


class NativeDistanceOutput:
    """Keep the allocation callback and its output alive for one native call."""

    def __init__(self, enabled):
        self.array = None
        self.error = None
        self.enabled = enabled

    @property
    def callback(self):
        # ctypes keeps the argument alive through the synchronous call. Do not
        # store a bound callback on self: that cycle would retain large arrays
        # until Python's cyclic garbage collector next runs.
        return DistanceAllocator(self._allocate) if self.enabled else DistanceAllocator()

    def _allocate(self, n, itemsize):
        try:
            self.array = np.empty((n, n), dtype=_DTYPES[itemsize], order="C")
            return self.array.ctypes.data
        except BaseException as error:
            # ctypes cannot propagate callback exceptions through C++ safely.
            # Signal failure without exposing an invalid pointer, then re-raise
            # the original Python error after the native stack has unwound.
            self.error = error
            return None

    def raise_if_failed(self):
        if self.error is not None:
            raise self.error
