# Python Interface

The `fast_apsp` package in `source/python/` calls the native libraries through `ctypes`. Installation and import setup are described in [INSTALL.md](INSTALL.md#python).

## Functions

```python
bfs_apsp(adj_matrix, return_distances=True)                                   # ZM-BFS
cc_bfs_apsp(adj_matrix, return_distances=True, hub_constraint=None)            # CC-BFS
cc_bfs_p_apsp(adj_matrix, num_threads=1, return_distances=True,
              hub_constraint=None)                                            # CC-BFS-P
do_bfs_apsp(adj_matrix, return_distances=True)                                # DO-BFS (third-party)
ess(adj_matrix, k=4)                                                          # ESS
cc_ess(adj_matrix, k=4)                                                       # CC-ESS
```

`k` is the number of initial sweeps used by ESS and CC-ESS.

## Input

Pass the adjacency matrix of a connected, unweighted, undirected graph: nonempty, square, symmetric, with entries 0 or 1. Not all of these requirements are checked automatically, so make sure your matrix meets them.

- **Dense input** (a NumPy array or nested lists) is converted to a C-contiguous `float64` array. This makes a copy unless the input already has that type and layout.
- **SciPy sparse input to `cc_bfs_p_apsp`** (CSR, CSC or COO, as a sparse matrix or sparse array) stays sparse: no dense adjacency matrix is created, and your matrix is not modified.
- **SciPy sparse input to the other functions** is converted to a dense `float64` array first, which needs `8*n*n` bytes for `n` vertices.

## Output

The four APSP functions return a dictionary:

| Key | Meaning |
|-----|---------|
| `distances` | `n`-by-`n` NumPy array of hop distances, or `None` when `return_distances=False`. Its type is described below. |
| `elapsed_time` | Native computation time in seconds; see [Timing](#timing) |
| `estimated_memory_bytes` | Estimated storage for the distances, plus the hub shortest-path structures for CC-BFS and CC-BFS-P; see [Memory](#memory) |
| `estimated_total_memory_bytes` | `estimated_memory_bytes` plus the adjacency list |
| `adjacency_list_bytes` | Estimated adjacency-list size |
| `process_delta_peak_bytes` | Measured increase in process private memory during the native call (Windows only) |
| `num_threads` | Number of threads used (CC-BFS-P only) |
| `counters` | `ctypes` structure with all native counters for that function, for example `counters.distance_matrix_bytes`; CC-BFS and CC-BFS-P also have fields such as `counters.diam_upper` and phase timings |

`distances[s, t]` is the hop distance between vertices `s` and `t`, with zeros on the diagonal. The type of the `distances` array depends on the function:

- `cc_bfs_apsp` and `cc_bfs_p_apsp`: the unsigned integer type the algorithm used internally, `uint8`, `uint16`, `uint32` or `uint64` (see [Memory](#memory)).
- `bfs_apsp` and `do_bfs_apsp`: `float64`.

If your code needs floating-point distances, convert explicitly, for example with `D.astype(np.float64)`. Unsigned subtraction wraps around instead of going negative, so convert to a signed or floating-point type before any arithmetic that can produce negative values.

`ess` and `cc_ess` return a dictionary with `diameter`, `radius` and `elapsed_time`.

## Hub constraint

`hub_constraint` sets the maximum number of hub vertices for `cc_bfs_apsp` and `cc_bfs_p_apsp`. Pass a `str` (or `bytes`):

| Value | Hub limit (rounded to the nearest integer) |
|-------|--------------------------------------------|
| `None` or `"bounded-avg-deg"` (default) | `min(avg_deg, n / avg_deg)`, where `avg_deg` is the mean degree |
| `"avg-deg"` | `avg_deg` |
| `"sqrt-deg"` | `sqrt(avg_deg)` |
| `"inv-den"` | `ceil(1 / density)` |
| `"sqrt-inv-den"` | `sqrt(ceil(1 / density))` |
| `"none"` | `n` (no limit) |
| A number given as text, for example `"25"` | That number |

An unrecognized value raises `RuntimeError`.

## Memory

**Distance storage.** CC-BFS and CC-BFS-P store distances in the smallest of `uint8`, `uint16`, `uint32` or `uint64` that can hold twice the 4-SumSweep diameter upper bound (`counters.diam_upper`). ZM-BFS and DO-BFS store `float64` distances.

**Returned array.** `cc_bfs_apsp` and `cc_bfs_p_apsp` copy their distance matrix once into a new NumPy array of the same type, after timing and memory sampling have finished. During the copy, both matrices exist, each taking `w*n*n` bytes, where `w` is the entry size (1 to 8 bytes). For example, 10,000 vertices with `uint8` distances need 0.1 GB for each. `bfs_apsp` and `do_bfs_apsp` return a `float64` array of `8*n*n` bytes in addition to their own `8*n*n`-byte matrix.

**`return_distances=False`** skips the returned array only. The full distance matrix is still computed and stored natively, as with the MATLAB `'ndr'` option, so native memory use is unchanged.

**Input copies.** Dense input takes `8*n*n` bytes as a `float64` array. All functions except `cc_bfs_p_apsp` make one more native copy before building the adjacency list. For large graphs, use `cc_bfs_p_apsp` with sparse input to avoid both copies.

**Estimated memory** (`estimated_memory_bytes`, `estimated_total_memory_bytes`, `adjacency_list_bytes`) is computed from the sizes of the algorithm's main data structures. It is not the total memory of the process: Python arrays, input copies, temporary buffers and runtime overhead are not included.

**Measured process memory** is sampled on Windows during the native call. In `counters`, `process_baseline_*` and `process_peak_*` are absolute values at the start and at the sampled peak, and `process_delta_peak_*` is their difference. The returned `process_delta_peak_bytes` is `counters.process_delta_peak_private_bytes`. It does not include the returned `distances` array. ZM-BFS, CC-BFS and DO-BFS start measuring after building the adjacency list; CC-BFS-P starts before it, so its value includes the adjacency list. On Linux, process memory is not measured: `cc_bfs_p_apsp` returns `None` for `process_delta_peak_bytes`, and the other functions return `0`.

## Timing

`elapsed_time` is measured inside the native code, from the start of adjacency-list construction until the native computation finishes. For CC-BFS and CC-BFS-P it also includes the 4-SumSweep diameter bound. It excludes Python-side input checks and conversion, and creating the returned array.

Building the adjacency list scans all `n*n` entries of dense input, but only the stored entries of sparse input to `cc_bfs_p_apsp`.

## Threads (CC-BFS-P)

- `num_threads` must be an integer of at least 1; the default is 1.
- The maximum is the number of logical processors the system reports; larger values raise `RuntimeError`. This limit ignores CPU-affinity masks and the cores a container or job scheduler allocates to you, so set `num_threads` to the cores actually available to your job.
- The requested count is used regardless of `OMP_NUM_THREADS`. CC-BFS-P does not pin threads to cores; placement is left to the OpenMP runtime and the operating system.

## Errors

- `ValueError` or `TypeError`: an input problem detected in Python, such as a non-square or non-binary matrix, or an invalid `num_threads`.
- `RuntimeError`: a failure in the native code. For `cc_bfs_p_apsp` the message includes the native error text; the other functions report only an error code.
- If the `cc_bfs_apsp` or `cc_bfs_p` library does not match the Python package, a `RuntimeError` asks you to rebuild it; see [Troubleshooting](INSTALL.md#troubleshooting).
