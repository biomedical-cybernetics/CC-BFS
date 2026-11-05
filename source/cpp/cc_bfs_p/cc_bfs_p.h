#ifndef FAST_UNWEIGHTED_APSP_CC_BFS_P_H_
#define FAST_UNWEIGHTED_APSP_CC_BFS_P_H_

// Public C++ contract for CC-BFS-P. The implementation follows canonical
// CC-BFS terminology while adding parallel hub and non-hub execution.

#include <cstdint>
#include <string>
#include <vector>

#include "include/dtype_utils.h"

namespace parallel_cc::cc_bfs_p {

// Zero-based, sorted neighbor rows produced by the shared MATLAB parser.
using AdjacencyList = std::vector<std::vector<int64_t>>;

// Canonical CC-BFS work counters plus explicit memory-accounting fields. The
// counters are accumulated privately by OpenMP workers and merged after the
// batch barrier; no worker writes this shared aggregate.
struct Counters {
  // Search work and source-path selection.
  int64_t vec_comp_count = 0;
  int64_t source_with_cs_count = 0;
  int64_t source_without_cs_count = 0;
  int64_t hn_count = 0;
  // Vertices expanded through ordinary adjacency, the hub SP-DAG, same-level
  // spill shortcuts, or the hub discovery tree. cst_visited is retained from
  // the canonical counter contract.
  int64_t graph_expansion_node_count = 0;
  int64_t spdag_expansion_node_count = 0;
  int64_t spill_expansion_node_count = 0;
  int64_t sptree_expansion_node_count = 0;
  int64_t cst_visited_count = 0;
  // Number of hub-guided frontier decisions executed in each direction.
  int64_t top_down_node_count = 0;
  int64_t bottom_up_node_count = 0;
  // Canonical phase timings exposed for compatibility with the baseline.
  double time_secs_comp_adj = 0.0;
  double time_secs_diam_upperbound = 0.0;
  double time_secs_hn_sss = 0.0;
  double time_secs_non_hn_sss = 0.0;
  double time_secs_convert = 0.0;
  double time_secs_overall = 0.0;
  int64_t diam_upper = 0;
  std::string data_type;
  // Component-based estimates describe algorithm-owned live allocations;
  // process fields below are independent OS working-set/private-byte peaks.
  int64_t estimated_dominant_memory_bytes = 0;
  int64_t distance_matrix_bytes = 0;
  int64_t sp_dag_bytes = 0;
  int64_t sp_shortcut_bytes = 0;
  int64_t sp_tree_bytes = 0;
  int64_t adjacency_list_bytes = 0;
  int64_t estimated_total_memory_bytes = 0;
  int64_t process_baseline_working_set_bytes = 0;
  int64_t process_peak_working_set_bytes = 0;
  int64_t process_delta_peak_working_set_bytes = 0;
  int64_t process_baseline_private_bytes = 0;
  int64_t process_peak_private_bytes = 0;
  int64_t process_delta_peak_private_bytes = 0;
};

// Parallel phase breakdown. commit_sync measures only the deterministic
// availability publication after workers have copied their distinct columns.
struct Timings {
  double adjacency = 0.0;
  double hub = 0.0;
  double non_hub = 0.0;
  double commit_sync = 0.0;
  double conversion = 0.0;
  double total = 0.0;
};

// `distances` is monostate when `nd` or `memory_test_only` suppresses native
// result materialization.
struct Result {
  ResultMatrix distances;
  Counters counters;
  Timings timings;
  int64_t num_threads = 1;
};

// Canonical 4-sweep upper bound used to choose a lossless distance scalar and
// the finite disconnected sentinel; it is not a separate APSP computation.
int64_t DiamUpperBound(const AdjacencyList& A,
                       const Eigen::VectorXd& DEG);

// Rejects thread counts outside the logical-processor range supported by the
// current process. Kept public so non-MATLAB interfaces use the same contract.
void ValidateThreadCount(int64_t num_threads);

// Runs parallel hub preprocessing followed by deterministic non-hub batches.
// MATLAB parsing and output creation remain outside this function so worker
// threads never call the MATLAB API. `memory_test_only` stops after hub
// preprocessing so experiments can measure the canonical hub-stage footprint.
Result Run(const AdjacencyList& A,
           const Eigen::VectorXd& DEG,
           const std::string& hub_constraint,
           int64_t num_threads,
           bool no_distances,
           bool memory_test_only,
           bool verbose,
           double adjacency_time,
           int64_t diameter,
           double diameter_time);

}  // namespace parallel_cc::cc_bfs_p

#endif  // FAST_UNWEIGHTED_APSP_CC_BFS_P_H_
