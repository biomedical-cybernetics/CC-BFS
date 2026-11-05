#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
output_dir="${script_dir}/lib"
source_dir="${project_root}/source/cpp"
do_bfs_dir="${project_root}/third_party/gapbs-do-bfs"
wrapper_dir="${script_dir}/wrappers"

eigen_path="${EIGEN_PATH:-${1:-}}"
if [[ -z "${eigen_path}" || ! -f "${eigen_path}/Eigen/Dense" ]]; then
    echo "Usage: EIGEN_PATH=/path/to/eigen-3.4.0 $0" >&2
    echo "The path must directly contain Eigen/Dense." >&2
    exit 2
fi

cxx="${CXX:-g++}"
common_flags=(-std=c++17 -O2 -fPIC -shared -I"${eigen_path}")
mkdir -p "${output_dir}"

build_python_library() {
    local output_name="$1"
    shift
    echo "Compiling ${output_name}.so"
    "${cxx}" "${common_flags[@]}" -DPYTHON_BUILD "$@" \
        -o "${output_dir}/${output_name}.so"
}

build_python_library bfs_apsp "${source_dir}/bfs_apsp_l_cpp.cpp"
build_python_library cc_bfs_apsp -I"${wrapper_dir}" "${source_dir}/cc_bfs_apsp_l_cpp.cpp"
build_python_library ess "${source_dir}/ess_cpp.cpp"
build_python_library cc_ess "${source_dir}/cc_ess_cpp.cpp"
build_python_library do_bfs_gapbs_apsp \
    -I"${do_bfs_dir}" -fopenmp \
    "${do_bfs_dir}/do_bfs_gapbs_apsp_l_cpp.cpp"

echo "Compiling cc_bfs_p.so"
"${cxx}" "${common_flags[@]}" -fopenmp \
    -I"${source_dir}/cc_bfs_p" \
    -I"${source_dir}/cc_bfs_p/include" \
    "${source_dir}/cc_bfs_p/cc_bfs_p.cpp" \
    "${wrapper_dir}/cc_bfs_p_python.cpp" \
    -o "${output_dir}/cc_bfs_p.so"

echo "All Linux shared libraries were written to ${output_dir}."
