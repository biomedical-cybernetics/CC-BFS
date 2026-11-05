# MATLAB MEX Interface

This reference covers the six MATLAB entry points: the five first-class algorithms and the bundled third-party DO-BFS baseline. Setup and rebuilding are described in [INSTALL.md](INSTALL.md#matlab).

Add the MEX folder to the path before calling them:

```matlab
addpath(fullfile(pwd, 'source', 'matlab', 'mex_functions'));   % from the repository root
```

## Calling conventions

**Adjacency matrix.** The first input is the adjacency matrix, dense or sparse, `double` or `logical`. It must describe a connected, unweighted, undirected graph: nonempty, square, symmetric, with entries 0 or 1. Not all of these requirements are checked automatically, so make sure your matrix meets them.

**Options are character vectors.** Write options in single quotes, for example `'nv'` or `'sqrt-deg'`. Double quotes create a MATLAB string, which these functions do not recognize: `cc_bfs_p` stops with an error, and the other functions ignore the option.

**Console output.** The functions print progress and timing information by default. CC-BFS and CC-BFS-P accept `'nv'` to turn it off. ZM-BFS, DO-BFS, ESS and CC-ESS have no such option; wrap the call in `evalc` if needed, as the benchmark script does.

**Hub constraint.** `cc_bfs_apsp_l_cpp` and `cc_bfs_p` accept an option that sets the maximum number of hub vertices. The limit is rounded to the nearest integer.

| Option | Hub limit |
|--------|-----------|
| `'bounded-avg-deg'` (default) | `min(avg_deg, n / avg_deg)`, where `avg_deg` is the mean degree |
| `'avg-deg'` | `avg_deg` |
| `'sqrt-deg'` | `sqrt(avg_deg)` |
| `'inv-den'` | `ceil(1 / density)` |
| `'sqrt-inv-den'` | `sqrt(ceil(1 / density))` |
| `'none'` | `n` (no limit) |
| A number given as text, for example `'25'` | That number |

Any character vector that is not one of the flags below is taken as the hub constraint; if several are given, the last one is used. An unrecognized constraint stops with an error.

**Flags.** The APSP functions accept these flags, in any combination and order after the adjacency matrix:

| Flag | Effect |
|------|--------|
| `'nd'` | Runs without keeping or returning the full distance matrix. CC-BFS and CC-BFS-P still keep the distance vectors they need for memoization. |
| `'ndr'` | Computes and stores the full distance matrix as usual, but returns a dummy first output |
| `'mem_test_only'` | Stops early to measure memory: after the adjacency list (ZM-BFS, DO-BFS) or after the hub shortest-path structures (CC-BFS, CC-BFS-P). `t` is `NaN`. |
| `'nv'` | Turns off console output (`cc_bfs_apsp_l_cpp` and `cc_bfs_p` only) |

With `'nd'`, `'ndr'` or `'mem_test_only'`, the first output is the `double` scalar `0`.

## Entry points

### `bfs_apsp_l_cpp` (ZM-BFS)

```matlab
[D, t, counters] = bfs_apsp_l_cpp(A)
[D, t, counters] = bfs_apsp_l_cpp(A, flags...)   % 'nd', 'ndr', 'mem_test_only'
```

`D` is an `n`-by-`n` `double` matrix. Unrecognized options are ignored.

### `cc_bfs_apsp_l_cpp` (CC-BFS)

```matlab
[D, t, counters] = cc_bfs_apsp_l_cpp(A)
[D, t, counters] = cc_bfs_apsp_l_cpp(A, options...)   % hub constraint, 'nd', 'ndr', 'nv', 'mem_test_only'
```

`D` has the smallest of the classes `uint8`, `uint16`, `uint32` or `uint64` that can hold twice the 4-SumSweep diameter upper bound (`counters.diam_upper`); `counters.data_type` names the class.

### `cc_bfs_p` (CC-BFS-P)

```matlab
[D, t, counters] = cc_bfs_p(A)
[D, t, counters] = cc_bfs_p(A, num_threads)
[D, t, counters] = cc_bfs_p(A, num_threads, options...)   % hub constraint, 'nd', 'ndr', 'nv', 'mem_test_only'
```

- `num_threads` is a positive integer; the default is 1. Give it at most once, in any position after `A`.
- The maximum is the number of logical processors the system reports; larger values stop with an error. This limit ignores CPU-affinity masks and the cores a job scheduler allocates to you. CC-BFS-P does not pin threads to cores.
- Each optional input after `A` must be a numeric scalar (`num_threads`) or a character vector; anything else stops with an error.
- `D` has the same class as for `cc_bfs_apsp_l_cpp`.
- `counters` also contains `variant` (`'cc_bfs_p'`), `num_threads` and the parallel phase timings `time_secs_adjacency`, `time_secs_hub`, `time_secs_non_hub`, `time_secs_commit_sync` and `time_secs_conversion`.

### `do_bfs_gapbs_apsp_l_cpp` (DO-BFS, third-party)

```matlab
[D, t, counters] = do_bfs_gapbs_apsp_l_cpp(A)
[D, t, counters] = do_bfs_gapbs_apsp_l_cpp(A, flags...)   % 'nd', 'ndr', 'mem_test_only'
```

Adapted from GAPBS; see `third_party/gapbs-do-bfs/`. `D` is an `n`-by-`n` `double` matrix. The direction-switch parameters are fixed at `alpha = 14` and `beta = 24`. Unrecognized options are ignored. `counters` also contains `top_down_step_count` and `bottom_up_step_count`.

### `ess_cpp` and `cc_ess_cpp` (ESS, CC-ESS)

```matlab
[diameter, radius, t, counters] = ess_cpp(A, k)
[diameter, radius, t, counters] = cc_ess_cpp(A, k)
```

Exactly two inputs are required. `k` is the number of initial sweeps, normally 4. `diameter`, `radius` and `t` are `double` scalars. `counters` contains `processed_count`, `processed_ratio` and `time_secs_overall`.

## Timing

`t` is measured inside the native code, from the start of adjacency-list construction until the native computation finishes. For CC-BFS and CC-BFS-P it also includes the 4-SumSweep diameter bound. It excludes creating the MATLAB output. `cc_bfs_p` reads the MATLAB array directly; the other functions first copy it into an internal matrix, and that copy is not timed.

## Memory counters

The APSP functions return these memory fields in `counters`:

| Field | Meaning |
|-------|---------|
| `estimated_dominant_memory_bytes` | Estimated storage for the distances, plus the hub shortest-path structures (`sp_dag_bytes`, `sp_shortcut_bytes`, `sp_tree_bytes`) for CC-BFS and CC-BFS-P |
| `distance_matrix_bytes`, `adjacency_list_bytes` | Estimated sizes of those two structures |
| `estimated_total_memory_bytes` | `estimated_dominant_memory_bytes` plus `adjacency_list_bytes` |
| `process_baseline_*`, `process_peak_*` | Absolute process memory measured on Windows at the start and at the sampled peak, as `working_set_bytes` and `private_bytes` |
| `process_delta_peak_*` | Peak minus baseline |

The estimates are computed from the sizes of the algorithm's main data structures. They are not the total memory of the process: MATLAB arrays, input copies, temporary buffers and runtime overhead are not included. The measured values do not include the MATLAB output array, which is created after the last sample. ZM-BFS, CC-BFS and DO-BFS start measuring after building the adjacency list; CC-BFS-P starts before it, so its values include the adjacency list.
