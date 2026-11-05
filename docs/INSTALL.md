# Installation Guide

This guide covers Python, MATLAB and C++ setup and common problems. For how to call the functions once installed, see the [C++ interface](CPP_INTERFACE.md), [Python interface](PYTHON_INTERFACE.md) and [MATLAB MEX interface](MATLAB_MEX_INTERFACE.md) references.

## Requirements

- **Operating system**: Windows x64 or Linux. Prebuilt binaries are included for Windows x64 only. macOS is untested and has no build script.
- **Memory**: the distance matrix has `n*n` entries for `n` vertices. ZM-BFS and DO-BFS use 8 bytes per entry (0.8 GB for 10,000 vertices, 80 GB for 100,000); CC-BFS and CC-BFS-P use 1 to 8 bytes. Size RAM for the largest network you plan to run; [PYTHON_INTERFACE.md](PYTHON_INTERFACE.md#memory) and [BENCHMARKING.md](BENCHMARKING.md#memory-needed) describe the additional copies each route makes.
- **Disk**: the bundled example networks take about 150 MB. Generating the full artificial benchmark dataset needs far more; see [BENCHMARKING.md](BENCHMARKING.md#generating-the-benchmark-dataset).
- **Eigen 3.3 or later**: needed only to rebuild binaries or use the C++ sources directly. Build scripts expect the directory that directly contains the `Eigen` folder.

## Python

### Requirements

- Python 3.9 or later.
- NumPy and SciPy. Both are needed to import `fast_apsp`.
- pandas and h5py, only for the benchmark script `scripts/benchmark_apsp_on_examples.py`.
- A C++ compiler, only if you build the native libraries yourself:
  - **Windows**: MSVC, from Visual Studio or the Visual Studio Build Tools with the "Desktop development with C++" workload.
  - **Linux**: GCC 9 or later with OpenMP.

### 1. Install the Python packages

In a virtual or Conda environment, if you use one:

```bash
python -m pip install numpy scipy
python -m pip install pandas h5py   # only for the benchmark script
```

### 2. Get the native libraries

The Python package loads one shared library per algorithm from `source/python/lib/`: `bfs_apsp`, `cc_bfs_apsp`, `cc_bfs_p`, `do_bfs_gapbs_apsp`, `ess` and `cc_ess`, with the extension `.dll` on Windows and `.so` on Linux.

**Windows x64**: prebuilt DLLs are included, so no build is needed. To rebuild them, open an "x64 Native Tools Command Prompt for VS 2022" (or another MSVC developer prompt) at the repository root and run:

```cmd
source\python\compile_windows_msvc.bat C:\path\to\eigen-3.4.0
```

**Linux**: build the `.so` libraries from the repository root:

```bash
EIGEN_PATH=/path/to/eigen-3.4.0 bash source/python/compile_linux.sh
```

On Debian or Ubuntu, `sudo apt-get install libeigen3-dev` provides Eigen at `/usr/include/eigen3`. The script uses `g++` unless the `CXX` environment variable names another compiler.

Both scripts write to `source/python/lib/`, replacing the libraries there. The Windows script puts its intermediate build files in `source/python/build/windows/`.

### 3. Verify the installation

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests need NumPy, SciPy and all six native libraries. They check every APSP method against an independent reference and test sparse input to CC-BFS-P.

### 4. Import the package

`fast_apsp` is not pip-installable. Add the checkout's `source/python` directory to the import path, either in code or through `PYTHONPATH`:

```python
import sys
sys.path.insert(0, "/absolute/path/to/CC-BFS/source/python")

from fast_apsp import bfs_apsp, cc_bfs_apsp, cc_bfs_p_apsp, do_bfs_apsp, ess, cc_ess
```

The package loads its libraries from `source/python/lib/`, so keep the `fast_apsp` and `lib` folders together. How to call the functions is described in [PYTHON_INTERFACE.md](PYTHON_INTERFACE.md).

## MATLAB

### Requirements

- MATLAB R2022a or later.
- To rebuild the MEX files: Windows, a MATLAB-supported MSVC compiler selected with `mex -setup C++`, and Eigen.
- To generate the artificial nPSO benchmark networks: the Statistics and Machine Learning Toolbox and the Symbolic Math Toolbox. The Parallel Computing Toolbox is optional; it is used only if you turn on `par_realizations` or `par_nets` in the generation scripts.

No toolbox is needed to run the algorithms on existing `.mat` networks.

### 1. Add the MEX folder to the path

Prebuilt Windows x64 MEX files for all six entry points are included in `source/matlab/mex_functions/`. From the repository root:

```matlab
addpath(fullfile(pwd, 'source', 'matlab', 'mex_functions'));
addpath(fullfile(pwd, 'scripts'));   % only for the benchmark and generation scripts
```

Run `savepath` if you want MATLAB to keep these folders on the path in later sessions.

### 2. Rebuild the MEX files (optional)

The MEX build scripts are written for MSVC and tested only on Windows; on other platforms they stop with an error.

```matlab
addpath(fullfile(pwd, 'source', 'matlab'));
compile_mex_all('C:/path/to/eigen-3.4.0');
```

`compile_mex_all` builds all six entry points into `source/matlab/mex_functions/`, replacing the prebuilt files. To rebuild only DO-BFS, set `eigen_include_dir` inside `third_party/gapbs-do-bfs/compile_mex_do_bfs.m` and run that script.

### 3. Verify the installation

Run the MATLAB example in the [README](../README.md#matlab), or time the four APSP methods on the bundled networks with `benchmark_apsp_on_examples` (see [BENCHMARKING.md](BENCHMARKING.md)).

## C++

There is no separate C++ build script or prebuilt C++ library: compile the algorithm source together with your own program. The entry points, input and output types and linking rules are described in [CPP_INTERFACE.md](CPP_INTERFACE.md).

### Requirements

- A C++17 compiler: MSVC on Windows or GCC on Linux.
- Eigen 3.3 or later.
- For CC-BFS-P only: OpenMP, and on Linux GCC 9 or later.

### Compiling

Keep the repository's directory layout, because headers include their helpers through relative paths such as `include/graph_utils.h`. Put `source/cpp` on the include path (`third_party/gapbs-do-bfs` for DO-BFS) and compile the algorithm's `.cpp` file with your program. Link only one sequential algorithm into a program; see [Linking](CPP_INTERFACE.md#linking).

To build the [README example](../README.md#c), saved as `main.cpp`, run from the repository root:

```bash
# GCC
g++ -std=c++17 -O2 -I/path/to/eigen-3.4.0 -Isource/cpp \
    main.cpp source/cpp/cc_bfs_apsp_l_cpp.cpp -o cc_bfs_example
```

```cmd
:: MSVC developer prompt
cl /std:c++17 /O2 /EHsc /I C:\path\to\eigen-3.4.0 /I source\cpp main.cpp source\cpp\cc_bfs_apsp_l_cpp.cpp
```

For CC-BFS-P, compile `source/cpp/cc_bfs_p/cc_bfs_p.cpp` with OpenMP enabled (`-fopenmp` or `/openmp`) and with `source/cpp/cc_bfs_p` and `source/cpp/cc_bfs_p/include` on the include path.

## Troubleshooting

- **`OSError: Shared library not found` (Python)**: the library for that algorithm is missing from `source/python/lib/`. On Linux, run `compile_linux.sh`; the prebuilt files are Windows DLLs.
- **A Windows DLL is found but fails to load**: the OpenMP-enabled libraries need the Microsoft Visual C++ OpenMP runtime (`vcomp140.dll`). Installing the latest Microsoft Visual C++ Redistributable for x64 provides it.
- **`RuntimeError: Rebuild the CC-BFS native library for integer Python outputs`** (or the CC-BFS-P equivalent): `cc_bfs_apsp` or `cc_bfs_p` in `source/python/lib/` does not match the Python package. Rebuild the libraries, or replace the DLLs with the ones from the same checkout as `source/python`.
- **`RuntimeError: CC-BFS-P failed: num_threads exceeds the available logical-processor count`**: lower `num_threads`; see [Threads](PYTHON_INTERFACE.md#threads-cc-bfs-p). The benchmark scripts use 4 threads by default; see [Running the benchmark](BENCHMARKING.md#running-the-benchmark).
- **`cl.exe` not found (Windows)**: run `compile_windows_msvc.bat` from an MSVC developer prompt such as "x64 Native Tools Command Prompt for VS 2022". A plain `cmd` or Conda prompt does not have `cl.exe` on its path.
- **`<memory_resource>` not found (Linux)**: the compiler is too old for CC-BFS-P; use GCC 9 or later.
- **MEX compiler not found (MATLAB)**: run `mex -setup C++` and select a supported MSVC compiler.
- **Invalid Eigen include directory (MATLAB)**: pass the directory that directly contains the `Eigen` folder, so that `<dir>/Eigen/Dense` exists.
- **MATLAB option seems to be ignored, or `cc_bfs_p` reports "Options must be a numeric thread count or character vectors"**: options must be single-quoted character vectors such as `'nv'`; double-quoted `"nv"` is a MATLAB string. See [calling conventions](MATLAB_MEX_INTERFACE.md#calling-conventions).
