#!/usr/bin/env python3
"""
Benchmark APSP algorithms on a folder of networks.

Benchmarks four APSP algorithms on every .mat file in the configured folder:
  - ZM-BFS  : Standard BFS-based APSP
  - CC-BFS  : The CC-BFS APSP algorithm
  - CC-BFS-P: Shared-memory parallel CC-BFS
  - DO-BFS  : Direction-Optimizing BFS baseline (adapted from GAPBS)

Metrics recorded per algorithm per network:
  - Running time (seconds)
  - Theoretical dominant memory estimate (bytes)
  - Empirical peak private memory delta (bytes)

Implementation note:
    To keep the empirical memory metric stable and order-independent, each
    algorithm/network pair is benchmarked in a fresh child Python process.

Configuration:
    Edit DATA_FOLDER below to point at any folder of .mat networks.
    Use "data/example_networks"   for the bundled example networks.
    Use "data/benchmark_dataset"  for the generated benchmark networks
        (run scripts/generate_benchmark_artificial_networks.m first).

Usage:
    python scripts/benchmark_apsp_on_examples.py

Results are saved to results/comparison/comparison_results_python.csv.
"""

import argparse
import json
import os
import sys
import glob
import subprocess
import numpy as np
from pathlib import Path

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source', 'python'))

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'python_utils'))

from fast_apsp import bfs_apsp, cc_bfs_apsp, cc_bfs_p_apsp, do_bfs_apsp
from io_utils import suppress_c_stdout_stderr, format_number
from network_loader import load_mat_variable


# ============================================================
# CONFIGURATION — edit this to switch between network folders
# ============================================================
DATA_FOLDER = "data/example_networks"
# DATA_FOLDER = "data/benchmark_dataset"  # generated benchmark networks
CC_BFS_P_THREADS = 4
# ============================================================

ALGORITHMS = [
    {'name': 'ZM-BFS', 'func': bfs_apsp, 'kwargs': {}},
    {'name': 'CC-BFS', 'func': cc_bfs_apsp, 'kwargs': {}},
    {
        'name': f'CC-BFS-P ({CC_BFS_P_THREADS} threads)',
        'func': cc_bfs_p_apsp,
        'kwargs': {'num_threads': CC_BFS_P_THREADS},
    },
    {'name': 'DO-BFS', 'func': do_bfs_apsp, 'kwargs': {}},
]

ALGORITHM_CONFIGS = {algo['name']: algo for algo in ALGORITHMS}


def _fmt_bytes(n_bytes):
    """Return a human-readable string for a byte count."""
    if n_bytes is None or n_bytes <= 0:
        return "N/A"
    for unit, threshold in [('GiB', 1 << 30), ('MiB', 1 << 20), ('KiB', 1 << 10)]:
        if n_bytes >= threshold:
            return f"{n_bytes / threshold:.2f} {unit}"
    return f"{n_bytes} B"


def _last_nonempty_line(text):
    """Return the last non-empty line from a subprocess output string."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _run_algorithm_in_subprocess(algo_name, network_path):
    """Run one algorithm on one network in a fresh child Python process."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--algorithm", algo_name,
        "--network", str(network_path),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    line = _last_nonempty_line(completed.stdout or "")
    if not line:
        stderr_text = (completed.stderr or "").strip()
        raise RuntimeError(
            stderr_text or f"{algo_name} benchmark subprocess produced no output"
        )

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{algo_name} benchmark subprocess returned invalid JSON: {line}"
        ) from exc

    if completed.returncode != 0 or payload.get("status") != "ok":
        error_text = payload.get("error")
        if not error_text:
            error_text = (completed.stderr or "").strip()
        raise RuntimeError(
            error_text or f"{algo_name} benchmark subprocess failed with exit code {completed.returncode}"
        )

    return payload


def _run_child_mode(algo_name, network_path):
    """Child-process entry point for a single algorithm/network benchmark."""
    try:
        algorithm = ALGORITHM_CONFIGS[algo_name]
    except KeyError as exc:
        print(json.dumps({
            "status": "error",
            "error": f"Unknown algorithm: {algo_name}",
        }))
        return 1

    try:
        x, _ = load_mat_variable(network_path, ['x', 'adjMatrix', 'x_matrices'])
        with suppress_c_stdout_stderr():
            result = algorithm['func'](
                x,
                return_distances=False,
                **algorithm['kwargs'],
            )

        process_bytes = result['process_delta_peak_bytes']

        print(json.dumps({
            "status": "ok",
            "elapsed_time": float(result['elapsed_time']),
            "estimated_memory_bytes": int(result['estimated_memory_bytes']),
            "process_delta_peak_bytes": (
                None if process_bytes is None else int(process_bytes)
            ),
        }))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "error": str(exc),
        }))
        return 1


def run_on_folder(data_folder, result_folder, csv_filename):
    data_folder = Path(data_folder)
    result_folder = Path(result_folder)
    result_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading networks from: {data_folder}\n")

    if not data_folder.exists():
        print(f"Error: Networks folder not found: {data_folder}")
        return

    mat_files = sorted(glob.glob(str(data_folder / "*.mat")))
    if not mat_files:
        print(f"Error: No .mat files found in {data_folder}")
        return

    print(f"   Found {len(mat_files)} network(s)\n")

    results_data = []

    for file_idx, filepath in enumerate(mat_files, 1):
        network_name = Path(filepath).stem

        print("=" * 70)
        print(f"Network {file_idx}/{len(mat_files)}: {network_name}")
        print("=" * 70)

        try:
            x, _ = load_mat_variable(filepath, ['x', 'adjMatrix', 'x_matrices'])

            n = x.shape[0]
            m = int(np.count_nonzero(x)) // 2
            density = (2 * m) / (n * (n - 1)) if n > 1 else 0

            print(f"   Nodes:   {format_number(n)}")
            print(f"   Edges:   {format_number(m)}")
            print(f"   Density: {density * 100:.4f}%\n")

            row = {'Network': network_name, 'Nodes': n, 'Edges': m, 'Density': density}

            for algo in ALGORITHMS:
                algo_name = algo['name']
                print(f"   Algorithm: {algo_name}")
                try:
                    result = _run_algorithm_in_subprocess(algo_name, filepath)

                    elapsed   = result['elapsed_time']
                    mem_th    = result['estimated_memory_bytes']
                    mem_proc  = result['process_delta_peak_bytes']

                    print(f"   OK  {elapsed:.6f} s"
                          f"  |  Dominant theory: {_fmt_bytes(mem_th)}"
                          f"  |  Empirical: {_fmt_bytes(mem_proc)}\n")

                    row[f"{algo_name}_time_s"]        = elapsed
                    row[f"{algo_name}_theory_bytes"]  = mem_th
                    row[f"{algo_name}_process_bytes"] = mem_proc

                except Exception as exc:
                    print(f"   FAIL  {exc}\n")
                    row[f"{algo_name}_time_s"]        = None
                    row[f"{algo_name}_theory_bytes"]  = None
                    row[f"{algo_name}_process_bytes"] = None

            results_data.append(row)

        except Exception as exc:
            print(f"   Error loading network: {exc}\n")
            continue

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70 + "\n")

    if results_data:
        import pandas as pd

        df = pd.DataFrame(results_data)

        # Runtime summary columns only (memory columns remain in the CSV)
        time_cols = ['Network'] + [f"{a['name']}_time_s" for a in ALGORITHMS]
        print("Runtime (seconds):")
        print(df[time_cols].to_string(index=False))
        print()

        mem_th_cols   = ['Network'] + [f"{a['name']}_theory_bytes"  for a in ALGORITHMS]
        mem_proc_cols = ['Network'] + [f"{a['name']}_process_bytes" for a in ALGORITHMS]

        def _fmt_col(col):
            return df[col].apply(lambda v: _fmt_bytes(int(v)) if pd.notna(v) else "N/A")

        mem_display = df[['Network']].copy()
        for a in ALGORITHMS:
            mem_display[f"{a['name']} theory"]   = _fmt_col(f"{a['name']}_theory_bytes")
            mem_display[f"{a['name']} empirical"] = _fmt_col(f"{a['name']}_process_bytes")

        print("Memory usage:")
        print(mem_display.to_string(index=False))
        print()

        csv_file = result_folder / csv_filename
        df.to_csv(csv_file, index=False)
        print(f"Results saved to: {csv_file}\n")

    print("=" * 70)
    print("  DONE")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark APSP algorithms on a folder of networks.")
    parser.add_argument("--child-run", action="store_true",
                        help="Internal mode: benchmark one algorithm on one network in a fresh subprocess.")
    parser.add_argument("--algorithm", type=str,
                        help="Internal mode: algorithm display name.")
    parser.add_argument("--network", type=str,
                        help="Internal mode: path to the input .mat network file.")
    args = parser.parse_args()

    if args.child_run:
        if not args.algorithm or not args.network:
            parser.error("--child-run requires --algorithm and --network")
        sys.exit(_run_child_mode(args.algorithm, args.network))

    print("\n" + "=" * 70)
    print("  APSP ALGORITHM BENCHMARK")
    print("=" * 70 + "\n")

    script_dir = Path(__file__).parent
    run_on_folder(
        data_folder=script_dir.parent / DATA_FOLDER,
        result_folder=script_dir.parent / "results" / "comparison",
        csv_filename="comparison_results_python.csv",
    )


if __name__ == "__main__":
    main()
