#ifndef ESS_H
#define ESS_H

#include <Eigen/Dense>
#include <cstdint>
#include "include/graph_utils.h"

struct COUNTERS {
    int64_t processed_count = 0;
    double  processed_ratio = 0.;
    double  time_secs_overall = 0.;
};

struct RESULT {
    int64_t diameter;
    int64_t radius;
    COUNTERS counters;
    double computation_time;
};

/**
 * @brief ExactSumSweep: Efficient algorithm for computing network diameter and radius
 */
RESULT ExactSumSweep(
    const MatrixType& adj_matrix,
    const int64_t k,
    const bool verbose = true);

#endif // ESS_H

