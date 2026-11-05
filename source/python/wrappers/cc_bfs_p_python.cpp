// Release-only C ABI for the canonical CC-BFS-P C++ implementation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#include "distance_output.h"

#include "../../cpp/cc_bfs_p/cc_bfs_p.h"
#include "../../cpp/cc_bfs_p/include/process_memory_utils.h"
#include "../../cpp/cc_bfs_p/include/timer.h"

#ifdef _WIN32
#define FAST_APSP_EXPORT __declspec(dllexport)
#else
#define FAST_APSP_EXPORT __attribute__((visibility("default")))
#endif

namespace {

using parallel_cc::cc_bfs_p::AdjacencyList;
using parallel_cc::cc_bfs_p::Counters;
using parallel_cc::cc_bfs_p::Result;

thread_local std::string last_error;

struct CcBfsPValues {
  int64_t vec_comp_count;
  int64_t source_with_cs_count;
  int64_t source_without_cs_count;
  int64_t hn_count;
  int64_t graph_expansion_node_count;
  int64_t spdag_expansion_node_count;
  int64_t spill_expansion_node_count;
  int64_t sptree_expansion_node_count;
  int64_t cst_visited_count;
  int64_t top_down_node_count;
  int64_t bottom_up_node_count;
  double time_secs_comp_adj;
  double time_secs_diam_upperbound;
  double time_secs_hn_sss;
  double time_secs_non_hn_sss;
  double time_secs_convert;
  double time_secs_overall;
  int64_t diam_upper;
  int64_t estimated_dominant_memory_bytes;
  int64_t distance_matrix_bytes;
  int64_t sp_dag_bytes;
  int64_t sp_shortcut_bytes;
  int64_t sp_tree_bytes;
  int64_t adjacency_list_bytes;
  int64_t estimated_total_memory_bytes;
  int64_t process_baseline_working_set_bytes;
  int64_t process_peak_working_set_bytes;
  int64_t process_delta_peak_working_set_bytes;
  int64_t process_baseline_private_bytes;
  int64_t process_peak_private_bytes;
  int64_t process_delta_peak_private_bytes;
  int64_t num_threads;
  double time_secs_adjacency;
  double time_secs_hub;
  double time_secs_non_hub;
  double time_secs_commit_sync;
  double time_secs_conversion;
};

void CopyCounters(const Result& result, CcBfsPValues* output) {
  if (output == nullptr) return;
  const Counters& counters = result.counters;
  output->vec_comp_count = counters.vec_comp_count;
  output->source_with_cs_count = counters.source_with_cs_count;
  output->source_without_cs_count = counters.source_without_cs_count;
  output->hn_count = counters.hn_count;
  output->graph_expansion_node_count = counters.graph_expansion_node_count;
  output->spdag_expansion_node_count = counters.spdag_expansion_node_count;
  output->spill_expansion_node_count = counters.spill_expansion_node_count;
  output->sptree_expansion_node_count = counters.sptree_expansion_node_count;
  output->cst_visited_count = counters.cst_visited_count;
  output->top_down_node_count = counters.top_down_node_count;
  output->bottom_up_node_count = counters.bottom_up_node_count;
  output->time_secs_comp_adj = counters.time_secs_comp_adj;
  output->time_secs_diam_upperbound = counters.time_secs_diam_upperbound;
  output->time_secs_hn_sss = counters.time_secs_hn_sss;
  output->time_secs_non_hn_sss = counters.time_secs_non_hn_sss;
  output->time_secs_convert = counters.time_secs_convert;
  output->time_secs_overall = counters.time_secs_overall;
  output->diam_upper = counters.diam_upper;
  output->estimated_dominant_memory_bytes =
      counters.estimated_dominant_memory_bytes;
  output->distance_matrix_bytes = counters.distance_matrix_bytes;
  output->sp_dag_bytes = counters.sp_dag_bytes;
  output->sp_shortcut_bytes = counters.sp_shortcut_bytes;
  output->sp_tree_bytes = counters.sp_tree_bytes;
  output->adjacency_list_bytes = counters.adjacency_list_bytes;
  output->estimated_total_memory_bytes = counters.estimated_total_memory_bytes;
  output->process_baseline_working_set_bytes =
      counters.process_baseline_working_set_bytes;
  output->process_peak_working_set_bytes =
      counters.process_peak_working_set_bytes;
  output->process_delta_peak_working_set_bytes =
      counters.process_delta_peak_working_set_bytes;
  output->process_baseline_private_bytes =
      counters.process_baseline_private_bytes;
  output->process_peak_private_bytes = counters.process_peak_private_bytes;
  output->process_delta_peak_private_bytes =
      counters.process_delta_peak_private_bytes;
  output->num_threads = result.num_threads;
  output->time_secs_adjacency = result.timings.adjacency;
  output->time_secs_hub = result.timings.hub;
  output->time_secs_non_hub = result.timings.non_hub;
  output->time_secs_commit_sync = result.timings.commit_sync;
  output->time_secs_conversion = result.timings.conversion;
}

void CopyDistances(const ResultMatrix& distances,
                   int64_t n,
                   double* output) {
  if (output == nullptr) return;
  std::visit(
      [=](const auto& matrix) {
        using Matrix = std::decay_t<decltype(matrix)>;
        if constexpr (std::is_same_v<Matrix, std::monostate>) {
          throw std::runtime_error("CC-BFS-P distance matrix is unavailable.");
        } else {
          for (int64_t source = 0; source < n; ++source) {
            for (int64_t target = 0; target < n; ++target) {
              output[source * n + target] =
                  static_cast<double>(matrix(target, source));
            }
          }
        }
      },
      distances);
}

// Both input encodings feed the unchanged native algorithm and share timing,
// output ownership and error handling. CSR preparation costs O(n + nnz).
template <typename BuildAdjacency>
int RunPython(
    int64_t n,
    int64_t num_threads,
    const char* hub_constraint,
    double* output_distances,
    double* elapsed_time,
    CcBfsPValues* output_values,
    BuildAdjacency build_adjacency,
    fast_apsp::DistanceAllocator allocate_distances) {
  try {
    last_error.clear();
    if (elapsed_time == nullptr) {
      throw std::runtime_error("Elapsed-time output is required.");
    }
    if (n < 1) {
      throw std::runtime_error("Adjacency matrix must be nonempty.");
    }
    if (static_cast<uint64_t>(n) >
        (std::numeric_limits<size_t>::max)() / static_cast<uint64_t>(n)) {
      throw std::runtime_error("Adjacency matrix dimensions are too large.");
    }

    namespace Target = parallel_cc::cc_bfs_p;
    Target::ValidateThreadCount(num_threads);

    ProcessMemoryPeakTracker memory_tracker;
    memory_tracker.CaptureBaseline();
    Result result;
    {
      ProcessMemoryTrackerScope tracker_scope(&memory_tracker);
      Timer adjacency_timer;
      adjacency_timer.Start();
      AdjacencyList adjacency(static_cast<size_t>(n));
      Eigen::VectorXd degree(n);
      build_adjacency(adjacency);
      for (int64_t vertex = 0; vertex < n; ++vertex) {
        degree(vertex) = static_cast<double>(
            adjacency[static_cast<size_t>(vertex)].size());
      }
      adjacency_timer.Stop();
      UpdateProcessMemoryPeakCheckpoint();

      Timer diameter_timer;
      diameter_timer.Start();
      const int64_t diameter =
          n <= 1 ? 0 : Target::DiamUpperBound(adjacency, degree);
      diameter_timer.Stop();

      const std::string constraint =
          hub_constraint == nullptr || hub_constraint[0] == '\0'
              ? "bounded-avg-deg"
              : std::string(hub_constraint);
      result = Target::Run(
          adjacency, degree, constraint, num_threads,
          false, false, false, adjacency_timer.Seconds(), diameter,
          diameter_timer.Seconds());
    }
    memory_tracker.UpdatePeak();
    FillProcessMemoryCounters(result.counters, memory_tracker);

    fast_apsp::CopyNativeDistances(result.distances, allocate_distances);
    CopyDistances(result.distances, n, output_distances);
    *elapsed_time = result.timings.total;
    CopyCounters(result, output_values);
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  } catch (...) {
    last_error = "Unknown C++ exception.";
    return -1;
  }
}

}  // namespace

extern "C" {

FAST_APSP_EXPORT const char* cc_bfs_p_last_error() {
  return last_error.c_str();
}

// Dense input is a row-major n-by-n array of binary doubles. Output, when
// requested, is a caller-owned row-major n-by-n array of hop distances.
static int RunDensePython(
    const double* adjacency_matrix,
    int64_t n,
    int64_t num_threads,
    const char* hub_constraint,
    double* output_distances,
    double* elapsed_time,
    CcBfsPValues* output_values,
    fast_apsp::DistanceAllocator allocate_distances) {
  return RunPython(
      n, num_threads, hub_constraint, output_distances, elapsed_time,
      output_values, [=](AdjacencyList& adjacency) {
        if (adjacency_matrix == nullptr) {
          throw std::runtime_error("Input matrix is required.");
        }
        for (int64_t vertex = 0; vertex < n; ++vertex) {
          auto& neighbors = adjacency[static_cast<size_t>(vertex)];
          for (int64_t neighbor = 0; neighbor < n; ++neighbor) {
            const double value = adjacency_matrix[vertex * n + neighbor];
            if (!std::isfinite(value) || (value != 0.0 && value != 1.0)) {
              throw std::runtime_error(
                  "Adjacency matrix must contain only finite binary values 0 or 1.");
            }
            if (value != 0.0) neighbors.push_back(neighbor);
          }
        }
      }, allocate_distances);
}

// Sparse input has n+1 CSR offsets and nnz sorted, unique zero-based column
// indices. All stored edges have weight 1; Python validates values/symmetry.
// No dense adjacency matrix is constructed, and output matches the dense ABI.
static int RunSparsePython(
    const int64_t* row_offsets,
    const int64_t* column_indices,
    int64_t n,
    int64_t nnz,
    int64_t num_threads,
    const char* hub_constraint,
    double* output_distances,
    double* elapsed_time,
    CcBfsPValues* output_values,
    fast_apsp::DistanceAllocator allocate_distances) {
  return RunPython(
      n, num_threads, hub_constraint, output_distances, elapsed_time,
      output_values, [=](AdjacencyList& adjacency) {
        if (row_offsets == nullptr || nnz < 0 ||
            (nnz > 0 && column_indices == nullptr) ||
            row_offsets[0] != 0 || row_offsets[n] != nnz) {
          throw std::runtime_error("Invalid CSR adjacency arrays.");
        }
        for (int64_t vertex = 0; vertex < n; ++vertex) {
          const int64_t begin = row_offsets[vertex];
          const int64_t end = row_offsets[vertex + 1];
          if (begin < 0 || end < begin || end > nnz) {
            throw std::runtime_error("Invalid CSR row offsets.");
          }
          auto& neighbors = adjacency[static_cast<size_t>(vertex)];
          neighbors.reserve(static_cast<size_t>(end - begin));
          int64_t previous = -1;
          for (int64_t entry = begin; entry < end; ++entry) {
            const int64_t neighbor = column_indices[entry];
            if (neighbor <= previous || neighbor >= n) {
              throw std::runtime_error(
                  "CSR column indices must be sorted, unique and in range.");
            }
            neighbors.push_back(neighbor);
            previous = neighbor;
          }
        }
      }, allocate_distances);
}

// Retain the previous double-output symbols for existing C ABI callers.
FAST_APSP_EXPORT int cc_bfs_p_apsp_py(
    const double* adjacency, int64_t n, int64_t threads, const char* constraint,
    double* output, double* elapsed, CcBfsPValues* counters) {
  return RunDensePython(adjacency, n, threads, constraint, output, elapsed,
                        counters, nullptr);
}

FAST_APSP_EXPORT int cc_bfs_p_apsp_csr_py(
    const int64_t* offsets, const int64_t* indices, int64_t n, int64_t nnz,
    int64_t threads, const char* constraint, double* output, double* elapsed,
    CcBfsPValues* counters) {
  return RunSparsePython(offsets, indices, n, nnz, threads, constraint, output,
                         elapsed, counters, nullptr);
}

// The allocator is called once after the unchanged native computation; null
// skips only the Python output. No graph conversion or search is repeated.
FAST_APSP_EXPORT int cc_bfs_p_apsp_typed_py(
    const double* adjacency, int64_t n, int64_t threads, const char* constraint,
    fast_apsp::DistanceAllocator allocate, double* elapsed, CcBfsPValues* counters) {
  return RunDensePython(adjacency, n, threads, constraint, nullptr, elapsed,
                        counters, allocate);
}

FAST_APSP_EXPORT int cc_bfs_p_apsp_csr_typed_py(
    const int64_t* offsets, const int64_t* indices, int64_t n, int64_t nnz,
    int64_t threads, const char* constraint, fast_apsp::DistanceAllocator allocate,
    double* elapsed, CcBfsPValues* counters) {
  return RunSparsePython(offsets, indices, n, nnz, threads, constraint, nullptr,
                         elapsed, counters, allocate);
}

}  // extern "C"
