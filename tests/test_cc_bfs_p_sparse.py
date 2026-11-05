"""Sparse-input and independent correctness checks for CC-BFS-P."""

from __future__ import annotations

import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "python"))

from fast_apsp import cc_bfs_p_apsp  # noqa: E402


THREAD_COUNTS = sorted({1, min(4, os.cpu_count() or 1)})
SPARSE_TYPES = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.coo_matrix,
]
SPARSE_TYPES += [
    getattr(sparse, name)
    for name in ("csr_array", "csc_array", "coo_array")
    if hasattr(sparse, name)
]


def path_graph(n):
    edges = np.arange(n - 1)
    return sparse.csr_matrix(
        (
            np.ones(2 * (n - 1)),
            (np.r_[edges, edges + 1], np.r_[edges + 1, edges]),
        ),
        shape=(n, n),
    )


def scipy_distances(adjacency):
    return shortest_path(adjacency, directed=False, unweighted=True)


class CcBfsPSparseTests(unittest.TestCase):
    def assert_distances(self, result, expected):
        actual = result["distances"]
        bound = 2 * result["counters"].diam_upper
        dtype = next(dtype for dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
                     if bound <= np.iinfo(dtype).max)
        self.assertEqual(actual.dtype, np.dtype(dtype))
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertTrue(actual.flags.owndata)
        np.testing.assert_array_equal(actual, expected)

    def test_sparse_formats_without_densification(self):
        adjacency = path_graph(12)
        expected = scipy_distances(adjacency)
        for sparse_type in SPARSE_TYPES:
            matrix = sparse_type(adjacency)
            with self.subTest(format=sparse_type.__name__), ExitStack() as stack:
                for cls in SPARSE_TYPES:
                    for method in ("toarray", "todense"):
                        stack.enter_context(
                            patch.object(
                                cls,
                                method,
                                side_effect=AssertionError(
                                    "CC-BFS-P must not densify sparse adjacency"
                                ),
                            )
                        )
                self.assert_distances(
                    cc_bfs_p_apsp(matrix, num_threads=THREAD_COUNTS[-1]),
                    expected,
                )

    def test_noncanonical_sparse_storage_is_preserved(self):
        adjacency = path_graph(9)
        expected = scipy_distances(adjacency)
        data, indices, indptr = [], [], [0]
        for row in range(adjacency.shape[0]):
            neighbors = adjacency.indices[
                adjacency.indptr[row] : adjacency.indptr[row + 1]
            ]
            # Reverse column order; duplicate entries sum to a binary edge.
            for neighbor in neighbors[::-1]:
                indices.extend((neighbor, neighbor))
                data.extend((0.25, 0.75))
            indices.append(row)
            data.append(0.0)  # Explicitly stored zero, not a self-loop.
            indptr.append(len(data))
        matrix = sparse.csr_matrix(
            (np.array(data), np.array(indices), np.array(indptr)),
            shape=adjacency.shape,
        )
        original = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
        original_flags = (matrix.has_sorted_indices, matrix.has_canonical_format)
        for threads in THREAD_COUNTS:
            self.assert_distances(
                cc_bfs_p_apsp(matrix, num_threads=threads), expected
            )
            for current, before in zip(
                (matrix.data, matrix.indices, matrix.indptr), original
            ):
                np.testing.assert_array_equal(current, before)
            self.assertEqual(
                (matrix.has_sorted_indices, matrix.has_canonical_format),
                original_flags,
            )

    def test_connected_graph_families_match_scipy(self):
        star = np.zeros((33, 33), dtype=np.float64)
        star[0, 1:] = 1
        star[1:, 0] = 1
        clique = np.ones((21, 21)) - np.eye(21)
        random = np.random.default_rng(20261020).random((48, 48)) < 0.12
        random = np.triu(random, 1)
        random |= random.T
        edges = np.arange(47)
        random[edges, edges + 1] = True
        random[edges + 1, edges] = True
        graphs = {
            "path_300": path_graph(300),
            "star": sparse.csr_matrix(star),
            "clique": sparse.csr_matrix(clique),
            "seeded_connected_random": sparse.csr_matrix(random),
        }
        for name, matrix in graphs.items():
            expected = scipy_distances(matrix)
            for threads in THREAD_COUNTS:
                with self.subTest(graph=name, threads=threads):
                    self.assert_distances(
                        cc_bfs_p_apsp(matrix, num_threads=threads), expected
                    )
            dense_result = cc_bfs_p_apsp(matrix.toarray(), num_threads=1)
            self.assert_distances(dense_result, expected)

    def test_repeated_outputs_have_independent_ownership(self):
        matrix = path_graph(17)
        expected = scipy_distances(matrix)
        first = cc_bfs_p_apsp(matrix, num_threads=THREAD_COUNTS[-1])["distances"]
        for _ in range(3):
            result = cc_bfs_p_apsp(matrix, num_threads=THREAD_COUNTS[-1])
            self.assert_distances(result, expected)
            self.assertFalse(np.shares_memory(first, result["distances"]))
            result["distances"][0, 1] = 100
            np.testing.assert_array_equal(first, expected)

    def test_invalid_sparse_shapes_are_rejected(self):
        for matrix, message in (
            (sparse.csr_matrix((2, 3)), "square"),
            (sparse.csr_matrix((0, 0)), "nonempty"),
        ):
            with self.subTest(shape=matrix.shape):
                with self.assertRaisesRegex(ValueError, message):
                    cc_bfs_p_apsp(matrix)

    def test_nonbinary_sparse_values_are_rejected(self):
        for value in (0.5, -1.0, 2.0, np.nan, np.inf):
            matrix = sparse.csr_matrix(
                ([value, value], ([0, 1], [1, 0])), shape=(2, 2)
            )
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "binary"):
                    cc_bfs_p_apsp(matrix)
        duplicate_ones = sparse.coo_matrix(
            ([1, 1, 1, 1], ([0, 0, 1, 1], [1, 1, 0, 0])),
            shape=(2, 2),
        )
        with self.assertRaisesRegex(ValueError, "binary"):
            cc_bfs_p_apsp(duplicate_ones)

    def test_asymmetric_inputs_are_rejected(self):
        directed = np.array([[0.0, 1.0], [0.0, 0.0]])
        for matrix in (directed, sparse.csr_matrix(directed)):
            with self.subTest(input_type=type(matrix).__name__):
                with self.assertRaisesRegex(ValueError, "symmetric"):
                    cc_bfs_p_apsp(matrix)

    def test_disconnected_graphs_preserve_native_sentinel(self):
        graphs = (
            sparse.block_diag(
                (path_graph(5), path_graph(3), sparse.csr_matrix((1, 1))),
                format="csr",
            ),
            sparse.csr_matrix((4, 4)),
        )
        for matrix in graphs:
            expected = scipy_distances(matrix)
            unreachable = ~np.isfinite(expected)
            expected[unreachable] = matrix.shape[0]
            for threads in THREAD_COUNTS:
                with self.subTest(n=matrix.shape[0], threads=threads):
                    result = cc_bfs_p_apsp(matrix, num_threads=threads)
                    self.assert_distances(result, expected)
                    self.assertTrue(
                        np.all(result["distances"][unreachable] == matrix.shape[0])
                    )
            self.assert_distances(
                cc_bfs_p_apsp(matrix.toarray(), num_threads=1), expected
            )

    def test_sparse_singleton_returns_zero(self):
        self.assert_distances(
            cc_bfs_p_apsp(sparse.csr_matrix((1, 1))),
            np.zeros((1, 1)),
        )

    def test_sparse_no_return_keeps_native_computation(self):
        matrix = path_graph(20)
        full = cc_bfs_p_apsp(matrix, num_threads=THREAD_COUNTS[-1])
        result = cc_bfs_p_apsp(
            matrix, num_threads=THREAD_COUNTS[-1], return_distances=False
        )
        self.assertIsNone(result["distances"])
        self.assertGreaterEqual(result["elapsed_time"], 0)
        self.assertEqual(result["num_threads"], THREAD_COUNTS[-1])
        self.assertEqual(
            result["counters"].distance_matrix_bytes,
            full["counters"].distance_matrix_bytes,
        )


if __name__ == "__main__":
    unittest.main()
