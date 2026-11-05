# C++ Interface

The algorithms can be called directly from C++. Compilation is described in [INSTALL.md](INSTALL.md#c), and the [README quick start](../README.md#c) contains a complete CC-BFS example.

## Entry points

| Algorithm | Header | Entry point |
|-----------|--------|-------------|
| ZM-BFS | `source/cpp/bfs_apsp_l.h` | `RESULT BFS_APSP_L(MatrixType& adj_matrix, bool nd, bool mem_test_only, bool verbose = true)` |
| CC-BFS | `source/cpp/cc_bfs_apsp_l.h` | `RESULT CC_BFS_APSP_L(MatrixType& adj_matrix, const std::string hub_constraint, bool nd, bool verbose = true)` |
| ESS | `source/cpp/ess.h` | `RESULT ExactSumSweep(const MatrixType& adj_matrix, const int64_t k, const bool verbose = true)` |
| CC-ESS | `source/cpp/cc_ess.h` | `RESULT CC_ExactSumSweep(const MatrixType& adj_matrix, const int64_t k, const bool verbose = true)` |
| DO-BFS | `third_party/gapbs-do-bfs/do_bfs_gapbs_apsp_l.h` | `RESULT DO_BFS_APSP_L(MatrixType& adj_matrix, bool nd, bool mem_test_only, bool verbose = true)` |
| CC-BFS-P | `source/cpp/cc_bfs_p/cc_bfs_p.h` | `parallel_cc::cc_bfs_p::Run(...)`; see [CC-BFS-P](#cc-bfs-p) |

## Input

- `MatrixType`, defined in `source/cpp/include/graph_utils.h`, is a `std::variant` of `Eigen::MatrixXd`, a dense `bool` Eigen matrix and `Eigen::SparseMatrix<bool>`.
- The matrix must describe a connected, unweighted, undirected graph: nonempty, square, symmetric, with entries 0 or 1. The C++ functions do not check these requirements, so make sure your matrix meets them.
- `BFS_APSP_L`, `CC_BFS_APSP_L` and `DO_BFS_APSP_L` release the input matrix after building their adjacency list, so `adj_matrix` is empty after the call. `ExactSumSweep` and `CC_ExactSumSweep` take the matrix by `const` reference and leave it unchanged.
- `hub_constraint` has no default in C++; pass `"bounded-avg-deg"` for the standard rule. The accepted values are listed in [PYTHON_INTERFACE.md](PYTHON_INTERFACE.md#hub-constraint). An unrecognized value throws `std::runtime_error`.
- `nd = true` runs without producing the full distance matrix. `mem_test_only = true` (ZM-BFS and DO-BFS) stops after building the adjacency list, to measure memory. Pass `false` for both in normal use.
- `verbose = true` prints progress and timing to standard output; pass `false` for a quiet call.

## Output

The sequential APSP functions (`BFS_APSP_L`, `CC_BFS_APSP_L` and `DO_BFS_APSP_L`) return a `RESULT` with the distance matrix `D`, a `counters` structure and `computation_time`: the seconds from the start of adjacency-list construction until the native computation finishes. CC-BFS-P returns a different structure; see [CC-BFS-P](#cc-bfs-p).

- **ZM-BFS and DO-BFS**: `D` is an `Eigen::MatrixXd`.
- **CC-BFS**: `D` is a `ResultMatrix` (`source/cpp/include/dtype_utils.h`), a `std::variant` holding an Eigen matrix of `uint8_t`, `uint16_t`, `uint32_t` or `uint64_t`, whichever is the smallest type that can hold twice the 4-SumSweep diameter upper bound. `counters.data_type` names that type. Use `std::visit` to work with the matrix in its own type, or `ResultMatrixToDouble(result.D)` to get an `Eigen::MatrixXd` copy. With `nd = true`, `D` holds no matrix (`std::monostate`).
- Column `s` of `D` holds the distances from vertex `s`; for an undirected graph the matrix is symmetric.
- With `mem_test_only = true`, ZM-BFS and DO-BFS return no distance matrix and `computation_time` is `NaN`.

`ExactSumSweep` and `CC_ExactSumSweep` return a `RESULT` with `diameter`, `radius`, `counters` and `computation_time`.

## CC-BFS-P

```cpp
namespace parallel_cc::cc_bfs_p {
using AdjacencyList = std::vector<std::vector<int64_t>>;

void ValidateThreadCount(int64_t num_threads);
int64_t DiamUpperBound(const AdjacencyList& A, const Eigen::VectorXd& DEG);
Result Run(const AdjacencyList& A, const Eigen::VectorXd& DEG,
           const std::string& hub_constraint, int64_t num_threads,
           bool no_distances, bool memory_test_only, bool verbose,
           double adjacency_time, int64_t diameter, double diameter_time);
}
```

`Run` does not build its own graph structures. The caller supplies:

- `A`: zero-based, sorted neighbour lists, one per vertex;
- `DEG`: the vertex degrees;
- `diameter`: the bound from `DiamUpperBound(A, DEG)`, or 0 for a single vertex;
- `adjacency_time` and `diameter_time`: the seconds spent preparing `A` and the bound, which are added to the reported total.

Call `ValidateThreadCount` first; it throws if `num_threads` is below 1 or above the logical-processor count. `Result` contains `distances` (a `ResultMatrix`, as for CC-BFS), `counters`, the parallel phase `timings` and the applied `num_threads`. `source/python/wrappers/cc_bfs_p_python.cpp` shows the complete call sequence.

## Linking

- Include the algorithm's header and compile and link the corresponding `.cpp` file, for example `cc_bfs_apsp_l.h` with `source/cpp/cc_bfs_apsp_l_cpp.cpp`.
- Link at most one of the sequential algorithm sources (ZM-BFS, CC-BFS, ESS, CC-ESS, DO-BFS) into one program. Each defines its own global `COUNTERS` and `RESULT` types and some identically named helper functions, so combining two of them breaks the one-definition rule.
- CC-BFS-P includes its own copy of the distance-storage helpers in `source/cpp/cc_bfs_p/include/dtype_utils.h`, which differs from the sequential copy. Do not link CC-BFS-P and a sequential algorithm into the same program either.
- To use several algorithms, build them as separate programs or shared libraries, as the Python build does.
