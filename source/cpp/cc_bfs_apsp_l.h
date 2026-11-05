#ifndef CC_BFS_APSP_L_H
#define CC_BFS_APSP_L_H

#include <string>
#include <cstdint>
#include "include/dtype_utils.h"
#include "include/graph_utils.h"

struct COUNTERS {
    int64_t vec_comp_count = 0;
    int64_t source_with_cs_count = 0;
    int64_t source_without_cs_count = 0;
    int64_t hn_count = 0;
    int64_t graph_expansion_node_count = 0;
    int64_t spdag_expansion_node_count = 0;
    int64_t spill_expansion_node_count = 0;
    int64_t sptree_expansion_node_count = 0;
    int64_t cst_visited_count = 0;
    int64_t top_down_node_count = 0;
    int64_t bottom_up_node_count = 0;
    double time_secs_comp_adj = 0;
    double time_secs_diam_upperbound = 0;
    double time_secs_hn_sss = 0;
    double time_secs_non_hn_sss = 0;
    double time_secs_convert = 0;
    double time_secs_overall = 0;
    
    int64_t diam_upper = 0;
    std::string data_type = "";

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

struct RESULT {
    ResultMatrix D;
    COUNTERS counters;
    double computation_time;
};

RESULT CC_BFS_APSP_L(
    MatrixType &adj_matrix,
    const std::string hub_constraint,
    bool nd,
    bool verbose = true);

#endif // CC_BFS_APSP_L_H

