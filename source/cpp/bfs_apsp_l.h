#ifndef BFS_APSP_L_H
#define BFS_APSP_L_H

#include <Eigen/Dense>
#include <cstdint>
#include "include/graph_utils.h"

struct COUNTERS {
    int64_t estimated_dominant_memory_bytes = 0;
    int64_t distance_matrix_bytes = 0;
    int64_t adjacency_list_bytes = 0;
    int64_t estimated_total_memory_bytes = 0;
    int64_t process_baseline_working_set_bytes = 0;
    int64_t process_peak_working_set_bytes = 0;
    int64_t process_delta_peak_working_set_bytes = 0;
    int64_t process_baseline_private_bytes = 0;
    int64_t process_peak_private_bytes = 0;
    int64_t process_delta_peak_private_bytes = 0;
};

struct RESULT {
    Eigen::MatrixXd D;
    COUNTERS counters;
    double computation_time;
};

/**
 * @brief ZM-BFS: Standard BFS-based All-Pairs Shortest Path algorithm
 * @param mem_test_only If true, skips APSP work after building the adjacency list; fills
 *        counters with theoretical memory estimates without allocating D.
 * @param verbose If true, prints progress and timing summaries.
 */
RESULT BFS_APSP_L(MatrixType &adj_matrix, bool nd, bool mem_test_only, bool verbose = true);

#endif // BFS_APSP_L_H

