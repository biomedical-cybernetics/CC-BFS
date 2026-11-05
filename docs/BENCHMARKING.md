# Benchmarking

Two scripts time the four APSP methods on every `.mat` network in a folder: `scripts/benchmark_apsp_on_examples.m` for MATLAB and `scripts/benchmark_apsp_on_examples.py` for Python. Both write a CSV table of running time and memory per method and network.

> **Scope**: these scripts demonstrate the released interfaces and a comparison workflow. They do not reproduce the complete server-side experimental protocol of the article.

## Methods compared

| CSV label | MATLAB function | Python function |
|-----------|-----------------|-----------------|
| ZM-BFS | `bfs_apsp_l_cpp` | `bfs_apsp` |
| CC-BFS | `cc_bfs_apsp_l_cpp` | `cc_bfs_apsp` |
| CC-BFS-P (`N` threads) | `cc_bfs_p` | `cc_bfs_p_apsp` |
| DO-BFS | `do_bfs_gapbs_apsp_l_cpp` | `do_bfs_apsp` |

DO-BFS is the bundled sequential baseline adapted from [GAPBS](https://github.com/sbeamer/gapbs). CC-BFS and CC-BFS-P use their default bounded-average-degree hub constraint, and CC-BFS-P uses `N` OpenMP threads. Each method runs once per network.

## Running the benchmark

Set the options near the top of the script, then run it.

| Setting | MATLAB variable | Python variable | Default |
|---------|-----------------|-----------------|---------|
| Network folder | `DATA_FOLDER_RELATIVE` | `DATA_FOLDER` | `data/example_networks` |
| CC-BFS-P threads | `cc_bfs_p_threads` | `CC_BFS_P_THREADS` | `4` |

Set the CC-BFS-P thread count no higher than the machine's number of logical processors; otherwise every CC-BFS-P run fails and its columns are left empty.

### MATLAB

Requires the MEX files ([INSTALL.md](INSTALL.md#matlab)). The script adds the MEX folder to the path itself.

```matlab
% From the repository root
addpath(fullfile(pwd, 'scripts'));
benchmark_apsp_on_examples
```

Results are written to `results/comparison/comparison_results_matlab.csv`.

### Python

Requires the native libraries, NumPy, SciPy, pandas and h5py ([INSTALL.md](INSTALL.md#python)).

```bash
# From the repository root
python scripts/benchmark_apsp_on_examples.py
```

Results are written to `results/comparison/comparison_results_python.csv`.

## Network data

Each `.mat` file must contain the adjacency matrix in a variable named `x`, `adjMatrix` or `x_matrices`; for `x_matrices`, a cell array, the first cell is used. The Python script reads MATLAB v7 and v7.3 files.

| Folder | Contents |
|--------|----------|
| `data/example_networks/` | Bundled example networks: nPSO networks with 10,000 vertices and several small real-world networks |
| `data/benchmark_dataset/` | Artificial nPSO and Watts–Strogatz benchmark networks, generated locally (see below) |
| Full article benchmark collection (not bundled) | The complete set of networks used in the article is too large to ship here; please refer to the article for the datasets. Save them as `.mat` adjacency matrices and point the folder setting at them. |

### Generating the benchmark dataset

```matlab
% From the repository root
addpath(fullfile(pwd, 'scripts'));
generate_benchmark_artificial_networks
```

The script adds the nPSO and Watts–Strogatz generators to the path and writes the networks to `data/benchmark_dataset/`. nPSO generation needs the Statistics and Machine Learning Toolbox and the Symbolic Math Toolbox ([INSTALL.md](INSTALL.md#matlab)).

The full grid is large. It includes 100,000-vertex networks with densities up to 0.9; the densest of these has about 4.5 billion edges. Generating, storing and benchmarking the grid needs a large-memory machine and substantial time and disk space. To use it, set the folder setting to `data/benchmark_dataset`.

## How each script calls the methods

**MATLAB.** ZM-BFS, CC-BFS and DO-BFS return their full distance matrices to MATLAB. CC-BFS-P is called with `'ndr'`, so it computes the full matrix but does not return it. Networks are passed as stored in the file, so sparse matrices stay sparse.

**Python.** Each method runs on each network in a fresh Python process, so earlier runs do not affect the memory measurement. All methods are called with `return_distances=False`, which skips the returned array but not the computation. The script loads every network as a dense `float64` array, so CC-BFS-P receives dense input here.

## Memory needed

Most of the memory scales with `n*n` for `n` vertices; `8*n*n` bytes is 0.8 GB at 10,000 vertices and 80 GB at 100,000 vertices.

- ZM-BFS and DO-BFS store `float64` distances (`8*n*n` bytes). In the MATLAB script, their returned matrices need the same amount again.
- CC-BFS and CC-BFS-P store distances in 1 to 8 bytes per entry, plus the hub shortest-path structures.
- The Python script holds the dense `float64` adjacency matrix (`8*n*n` bytes) in the main process and again in each run's process. ZM-BFS, CC-BFS and DO-BFS make one more copy of it before building their adjacency lists.

For 100,000-vertex networks, use the MATLAB script on a large-memory machine; in the Python script, loading the dense adjacency matrix alone needs 80 GB per process.

## Reported quantities

Each CSV row describes one network: its name, vertex and edge counts and density, then three columns per method. The Python script names the columns after the labels above, for example `ZM-BFS_time_s`; the MATLAB script uses valid MATLAB names, for example `ZM_BFS_time_s`.

| Column suffix | Source | Meaning |
|---------------|--------|---------|
| `_time_s` | returned time | Native computation time, from the start of adjacency-list construction until the native computation finishes. It excludes file loading and creating the MATLAB or Python output. |
| `_theory_bytes` | `estimated_dominant_memory_bytes` | Estimated storage for the distances, plus the hub shortest-path structures for CC-BFS and CC-BFS-P. It excludes the adjacency list and is not the total memory of the process. |
| `_process_bytes` | `process_delta_peak_private_bytes` | Measured peak increase in process private memory during the native call (Windows only). |

ZM-BFS, CC-BFS and DO-BFS start measuring process memory after building the adjacency list; CC-BFS-P starts before it, so its `_process_bytes` includes the adjacency list. On Linux, process memory is not measured: the Python script records `0` for ZM-BFS, CC-BFS and DO-BFS and leaves CC-BFS-P empty, and prints both as `N/A`.

If you add repetitions, average the repeated runs within each network first, then average those per-network means, so that every network contributes equally.
