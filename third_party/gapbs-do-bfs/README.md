# GAPBS-derived DO-BFS (bundled)

This directory contains an adapted **DO-BFS** implementation from the [GAP Benchmark Suite (GAPBS)](https://github.com/sbeamer/gapbs), redistributed here for benchmark reproducibility and convenience.

The original algorithms and supporting code are due to the GAPBS project. This repository’s adaptation adds **MATLAB MEX** and **Python** (`ctypes`) entry points, memory counters, and integration with this project's benchmark scripts.

Please read **`LICENSE.txt`** for the full license terms that apply to this third-party code.

To rebuild the MEX binary for this component from MATLAB, use **`compile_mex_do_bfs.m`** in this folder (after setting the Eigen include path as described in that script).
