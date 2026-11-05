#ifndef DO_BFS_GAPBS_APSP_L_H
#define DO_BFS_GAPBS_APSP_L_H

#include <Eigen/Dense>
#include <cstdint>
#include "include/graph_utils.h"

struct COUNTERS {
    int64_t top_down_step_count = 0;
    int64_t bottom_up_step_count = 0;
    double time_secs_overall = 0;
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

RESULT DO_BFS_APSP_L(MatrixType &adj_matrix, bool nd, bool mem_test_only, bool verbose = true);

#endif // DO_BFS_GAPBS_APSP_L_H
