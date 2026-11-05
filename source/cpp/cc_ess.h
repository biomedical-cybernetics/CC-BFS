#ifndef CC_ESS_H
#define CC_ESS_H

#include <Eigen/Dense>
#include <cstdint>
#include "include/graph_utils.h"

struct COUNTERS {
    int64_t processed_count = 0;
    double  processed_ratio = 0.;
    int64_t sel_hn_count = 0;
    int64_t proc_hn_count = 0;
    int64_t source_with_cs_count = 0;
    int64_t vec_comp_count = 0;
    int64_t graph_expansion_node_count = 0;
    int64_t spdag_expansion_node_count = 0;
    int64_t spill_expansion_node_count = 0;
    int64_t sptree_expansion_node_count = 0;
    int64_t top_down_node_count = 0;
    int64_t bottom_up_node_count = 0;
    double time_secs_comp_adj = 0;
    double time_secs_buildspstructs = 0;
    double time_secs_memoizedsss = 0;
    double time_secs_hubguidedsss = 0;
    double time_secs_generalizedcc = 0;
    double time_secs_overall = 0;
};

struct RESULT {
    int64_t diameter;
    int64_t radius;
    COUNTERS counters;
    double computation_time;
};

/**
 * @brief CC-ExactSumSweep: Accelerated diameter and radius computation
 */
RESULT CC_ExactSumSweep(
    const MatrixType& adj_matrix,
    const int64_t k,
    const bool verbose = true);

#endif // CC_ESS_H

