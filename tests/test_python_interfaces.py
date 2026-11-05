"""Focused correctness and lifecycle tests for the released Python APIs."""

from __future__ import annotations

import ctypes
import gc
import os
import subprocess
import sys
import textwrap
import unittest
import weakref
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "python"))

from fast_apsp import (  # noqa: E402
    bfs_apsp,
    cc_bfs_apsp,
    cc_bfs_p_apsp,
    cc_ess,
    do_bfs_apsp,
    ess,
)


def reference_apsp(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    distances = np.full((n, n), np.inf, dtype=np.float64)
    for source in range(n):
        distances[source, source] = 0.0
        queue = deque([source])
        while queue:
            vertex = queue.popleft()
            for neighbor in np.flatnonzero(adjacency[vertex]):
                if np.isinf(distances[source, neighbor]):
                    distances[source, neighbor] = distances[source, vertex] + 1.0
                    queue.append(int(neighbor))
    return distances


def connected_test_graph(n: int = 24) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=np.float64)
    for vertex in range(n):
        for offset in (1, 3):
            neighbor = (vertex + offset) % n
            adjacency[vertex, neighbor] = 1.0
            adjacency[neighbor, vertex] = 1.0
    return adjacency


class PythonInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adjacency = connected_test_graph()
        cls.expected = reference_apsp(cls.adjacency)

    def test_sequential_and_parallel_cc_bfs_are_exact(self):
        sequential = cc_bfs_apsp(self.adjacency)["distances"]
        np.testing.assert_array_equal(sequential, self.expected)
        for threads in (1, min(4, os.cpu_count() or 1)):
            parallel = cc_bfs_p_apsp(
                self.adjacency, num_threads=threads
            )["distances"]
            np.testing.assert_array_equal(parallel, self.expected)

    def test_zm_bfs_reference_is_exact(self):
        np.testing.assert_array_equal(
            bfs_apsp(self.adjacency)["distances"], self.expected
        )

    def test_disconnected_apsp_sentinels(self):
        adjacency = np.zeros((6, 6), dtype=np.float64)
        for a, b in ((0, 1), (1, 2), (2, 0), (3, 4)):
            adjacency[a, b] = adjacency[b, a] = 1
        expected = reference_apsp(adjacency)
        for method in (bfs_apsp, do_bfs_apsp, cc_bfs_apsp, cc_bfs_p_apsp):
            with self.subTest(method=method.__name__):
                target = expected.copy()
                if method in (cc_bfs_apsp, cc_bfs_p_apsp):
                    target[np.isinf(target)] = adjacency.shape[0]
                np.testing.assert_array_equal(method(adjacency)["distances"], target)

    def test_cc_bfs_unsigned_storage_boundary(self):
        # The native type reserves twice the computed diameter upper bound,
        # which need not equal the true diameter even on a path.
        observed_widths = {method: set() for method in (cc_bfs_apsp, cc_bfs_p_apsp)}
        for n in (127, 128, 129, 256):
            adjacency = np.zeros((n, n), dtype=np.float64)
            edges = np.arange(n - 1)
            adjacency[edges, edges + 1] = 1
            adjacency[edges + 1, edges] = 1
            expected = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            for method in (cc_bfs_apsp, cc_bfs_p_apsp):
                with self.subTest(n=n, method=method.__name__):
                    result = method(adjacency)
                    width = 1 if 2 * result["counters"].diam_upper <= 255 else 2
                    observed_widths[method].add(width)
                    self.assertEqual(result["distances"].dtype, np.dtype(f"uint{8 * width}"))
                    np.testing.assert_array_equal(result["distances"], expected)
                    self.assertTrue(result["distances"].flags.c_contiguous)
                    self.assertTrue(result["distances"].flags.owndata)
                    self.assertEqual(result["distances"].nbytes, n * n * width)
                    self.assertEqual(result["counters"].distance_matrix_bytes, n * n * width)
        for method, widths in observed_widths.items():
            self.assertEqual(widths, {1, 2}, method.__name__)

    def test_cc_outputs_own_their_storage(self):
        from scipy.sparse import csr_matrix

        for method in (cc_bfs_apsp, cc_bfs_p_apsp):
            for adjacency in (self.adjacency, csr_matrix(self.adjacency)):
                with self.subTest(method=method.__name__, input_type=type(adjacency).__name__):
                    first = method(adjacency)["distances"]
                    self.assertIs(type(first), np.ndarray)
                    self.assertIsNone(first.base)
                    self.assertTrue(first.flags.owndata)
                    self.assertTrue(first.flags.c_contiguous)
                    self.assertEqual(first.dtype, np.dtype(np.uint8))
                    second = method(adjacency)["distances"]
                    self.assertFalse(np.shares_memory(first, second))
                    second[0, 1] = 99
                    del second
                    gc.collect()
                    np.testing.assert_array_equal(first, self.expected)

    def test_cc_no_return_does_not_allocate_python_distances(self):
        for method in (cc_bfs_apsp, cc_bfs_p_apsp):
            full = method(self.adjacency)
            with self.subTest(method=method.__name__), patch(
                "fast_apsp._distance_output.NativeDistanceOutput._allocate",
                side_effect=AssertionError("No output allocation expected"),
            ) as allocate:
                result = method(self.adjacency, return_distances=False)
                self.assertIsNone(result["distances"])
                allocate.assert_not_called()
                self.assertEqual(
                    result["counters"].distance_matrix_bytes,
                    full["counters"].distance_matrix_bytes,
                )

    def test_cc_outputs_are_reclaimed_without_cyclic_gc(self):
        from scipy.sparse import csr_matrix

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for method in (cc_bfs_apsp, cc_bfs_p_apsp):
                for adjacency in (self.adjacency, csr_matrix(self.adjacency)):
                    with self.subTest(method=method.__name__, input_type=type(adjacency).__name__):
                        distances = method(adjacency)["distances"]
                        reference = weakref.ref(distances)
                        del distances
                        self.assertIsNone(reference(), "Callback retained the output array")
        finally:
            if was_enabled:
                gc.enable()

    def test_legacy_double_c_abis_remain_compatible(self):
        from scipy.sparse import csr_matrix
        from fast_apsp.cc_bfs import CcBfsCounters, _get_lib as sequential_library
        from fast_apsp.cc_bfs_p import (
            CcBfsPCounters,
            _get_lib as parallel_library,
            _prepare_sparse_adjacency,
        )

        double_pointer = ctypes.POINTER(ctypes.c_double)
        index_pointer = ctypes.POINTER(ctypes.c_int64)
        n = self.adjacency.shape[0]
        adjacency_pointer = self.adjacency.ctypes.data_as(double_pointer)
        stable_fields = (
            "hn_count", "diam_upper", "distance_matrix_bytes", "sp_dag_bytes",
            "sp_shortcut_bytes", "sp_tree_bytes", "adjacency_list_bytes",
            "estimated_dominant_memory_bytes", "estimated_total_memory_bytes",
        )

        def check(code, distances, elapsed, counters, typed):
            self.assertEqual(code, 0)
            self.assertGreaterEqual(elapsed.value, 0)
            np.testing.assert_array_equal(distances, typed["distances"])
            for field in stable_fields:
                self.assertEqual(getattr(counters, field), getattr(typed["counters"], field), field)

        sequential = sequential_library().cc_bfs_apsp_py
        sequential.argtypes = [
            double_pointer, ctypes.c_int64, double_pointer, double_pointer,
            ctypes.c_char_p, ctypes.POINTER(CcBfsCounters),
        ]
        sequential.restype = ctypes.c_int
        distances = np.empty((n, n), dtype=np.float64)
        elapsed, counters = ctypes.c_double(), CcBfsCounters()
        code = sequential(
            adjacency_pointer, n, distances.ctypes.data_as(double_pointer),
            ctypes.byref(elapsed), None, ctypes.byref(counters),
        )
        check(code, distances, elapsed, counters, cc_bfs_apsp(self.adjacency))

        library = parallel_library()
        dense = library.cc_bfs_p_apsp_py
        dense.argtypes = [
            double_pointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p,
            double_pointer, double_pointer, ctypes.POINTER(CcBfsPCounters),
        ]
        dense.restype = ctypes.c_int
        sparse = library.cc_bfs_p_apsp_csr_py
        sparse.argtypes = [
            index_pointer, index_pointer, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_char_p, double_pointer, double_pointer,
            ctypes.POINTER(CcBfsPCounters),
        ]
        sparse.restype = ctypes.c_int
        matrix = csr_matrix(self.adjacency)
        offsets, indices = _prepare_sparse_adjacency(matrix)
        cases = (
            (dense, (adjacency_pointer, n), self.adjacency),
            (sparse, (offsets.ctypes.data_as(index_pointer),
                      indices.ctypes.data_as(index_pointer), n, indices.size), matrix),
        )
        for call, input_args, adjacency in cases:
            with self.subTest(symbol=call.__name__):
                distances = np.empty((n, n), dtype=np.float64)
                elapsed, counters = ctypes.c_double(), CcBfsPCounters()
                code = call(
                    *input_args, 1, None, distances.ctypes.data_as(double_pointer),
                    ctypes.byref(elapsed), ctypes.byref(counters),
                )
                check(code, distances, elapsed, counters, cc_bfs_p_apsp(adjacency, num_threads=1))

    def test_cc_output_allocation_failure_is_preserved(self):
        for method in (cc_bfs_apsp, cc_bfs_p_apsp):
            with self.subTest(method=method.__name__):
                failure = MemoryError("deliberate output allocation failure")
                with patch("fast_apsp._distance_output.np.empty", side_effect=failure):
                    with self.assertRaises(MemoryError) as raised:
                        method(self.adjacency)
                self.assertIs(raised.exception, failure)
                # A failed callback must not leave native or Python state stuck.
                np.testing.assert_array_equal(
                    method(self.adjacency)["distances"], self.expected
                )

    def test_other_released_algorithms_are_exact(self):
        np.testing.assert_array_equal(
            do_bfs_apsp(self.adjacency)["distances"], self.expected
        )
        eccentricities = np.max(self.expected, axis=1)
        expected_diameter = int(np.max(eccentricities))
        expected_radius = int(np.min(eccentricities))
        for result in (ess(self.adjacency), cc_ess(self.adjacency)):
            self.assertEqual(result["diameter"], expected_diameter)
            self.assertEqual(result["radius"], expected_radius)

    def test_native_calls_are_quiet_by_default(self):
        # A subprocess also captures native output flushed only at process exit.
        script = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, "tests")
            from test_python_interfaces import connected_test_graph
            from fast_apsp import (
                bfs_apsp, cc_bfs_apsp, cc_bfs_p_apsp, do_bfs_apsp, ess, cc_ess
            )
            from scipy.sparse import csr_matrix

            adjacency = connected_test_graph()
            bfs_apsp(adjacency)
            cc_bfs_apsp(adjacency)
            cc_bfs_p_apsp(adjacency, num_threads=1)
            cc_bfs_p_apsp(csr_matrix(adjacency), num_threads=1)
            do_bfs_apsp(adjacency)
            ess(adjacency)
            cc_ess(adjacency)
            # Fewer than four vertices can exhaust diameter-sweep candidates.
            # This is normal, not a user-facing error.
            import numpy as np
            pair = np.array([[0., 1.], [1., 0.]])
            cc_bfs_p_apsp(pair, num_threads=1)
            cc_bfs_p_apsp(csr_matrix(pair), num_threads=1)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode, 0, completed.stdout + completed.stderr
        )
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_parallel_no_return_preserves_full_computation(self):
        result = cc_bfs_p_apsp(
            self.adjacency,
            num_threads=min(4, os.cpu_count() or 1),
            return_distances=False,
        )
        self.assertIsNone(result["distances"])
        self.assertGreaterEqual(result["elapsed_time"], 0.0)
        self.assertEqual(result["num_threads"], min(4, os.cpu_count() or 1))
        self.assertGreater(result["estimated_memory_bytes"], 0)

    def test_parallel_repeated_calls_are_exact(self):
        thread_count = min(2, os.cpu_count() or 1)
        for _ in range(12):
            result = cc_bfs_p_apsp(self.adjacency, num_threads=thread_count)
            np.testing.assert_array_equal(result["distances"], self.expected)

    def test_parallel_rejects_invalid_thread_count(self):
        with self.assertRaisesRegex(RuntimeError, "logical-processor count"):
            cc_bfs_p_apsp(self.adjacency, num_threads=1_000_000)
        with self.assertRaisesRegex(ValueError, "64-bit"):
            cc_bfs_p_apsp(self.adjacency, num_threads=2**64 + 1)

    def test_common_input_validation(self):
        with self.assertRaisesRegex(ValueError, "square"):
            cc_bfs_p_apsp(np.zeros((2, 3)), num_threads=1)
        invalid = self.adjacency.copy()
        invalid[0, 1] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            cc_bfs_p_apsp(invalid, num_threads=1)

    def test_singleton_graph(self):
        singleton = np.zeros((1, 1), dtype=np.float64)
        expected = np.zeros((1, 1), dtype=np.float64)
        for method in (cc_bfs_apsp, cc_bfs_p_apsp):
            with self.subTest(method=method.__name__):
                distances = method(singleton)["distances"]
                self.assertEqual(distances.dtype, np.dtype(np.uint8))
                np.testing.assert_array_equal(distances, expected)

    def test_complex_adjacency_is_rejected_before_conversion(self):
        from scipy.sparse import csr_matrix

        adjacency = self.adjacency.astype(np.complex128)
        adjacency[0, 0] = 1j
        for method in (bfs_apsp, cc_bfs_apsp, cc_bfs_p_apsp,
                       do_bfs_apsp, ess, cc_ess):
            for value in (adjacency, csr_matrix(adjacency)):
                with self.subTest(method=method.__name__, sparse=not isinstance(value, np.ndarray)):
                    with self.assertRaisesRegex(ValueError, "real binary"):
                        method(value)


if __name__ == "__main__":
    unittest.main()
