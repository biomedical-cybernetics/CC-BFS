
#include "bfs_apsp_l.h"
#include <Eigen/Sparse>
#include <vector>
#include <limits>
#include <chrono>
#include <queue>
#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <variant>
#include <type_traits>
#include "include/timer.h"
#include "include/progress_and_time_utils.h"
#include "include/process_memory_utils.h"
#ifdef MATLAB_BUILD
#include "mex.h"
#include "include/mex_utils.h"
#endif

inline void ReleaseAdjacencyMatrix(MatrixType& adj_matrix) {
    std::visit([](auto& m) {
        using Mat = std::decay_t<decltype(m)>;
        Mat empty;
        m.swap(empty);
    }, adj_matrix);
}

static void PrintTheoreticalMemorySummary(const COUNTERS& counters, const int64_t n) {
    std::cout << "@@ Theoretical memory usage (GiB): n=" << n
              << " estimated_total_memory_bytes=" << BytesToGiB(counters.estimated_total_memory_bytes)
              << " estimated_dominant_memory_bytes=" << BytesToGiB(counters.estimated_dominant_memory_bytes)
              << " adjacency_list_bytes=" << BytesToGiB(counters.adjacency_list_bytes)
              << " distance_matrix_bytes=" << BytesToGiB(counters.distance_matrix_bytes)
              << std::endl;
}


// BFS algorithm to find the shortest path from a single source to all other vertices in the graph
void BFSSSS(const std::vector<std::vector<int64_t>>& A,
            Eigen::MatrixXd *distances_out,
            int64_t N,
            int64_t s) {
    Eigen::VectorXd distance = Eigen::VectorXd::Constant(N, std::numeric_limits<double>::infinity());
    distance(s) = 0;

    std::queue<int64_t> queue;
    queue.push(s);

    while (!queue.empty()) {
        int64_t curr = queue.front();
        queue.pop();

        double alt_dst = distance(curr) + 1;
        for (int64_t t: A[curr]) {
            if (alt_dst < distance(t)) {
                queue.push(t);
                distance(t) = alt_dst;
            }
        }
    }
    if (distances_out != nullptr) {
        distances_out->row(s) = distance;
    }
}


static int64_t DistanceMatrixBytes(int64_t n) {
    if (n <= 0) {
        return 0;
    }
    const size_t prod = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = prod * sizeof(double);
    return static_cast<int64_t>(bytes);
}

static int64_t EstimateAdjacencyListBytes(const std::vector<std::vector<int64_t>>& A) {
    int64_t bytes = static_cast<int64_t>(A.capacity() * sizeof(std::vector<int64_t>));
    for (const auto& neighbors : A) {
        bytes += static_cast<int64_t>(neighbors.capacity() * sizeof(int64_t));
    }
    return bytes;
}

static void UpdateTotalMemoryCounter(COUNTERS& counters) {
    counters.estimated_total_memory_bytes =
        counters.estimated_dominant_memory_bytes + counters.adjacency_list_bytes;
}

// The actual algorithm implementation
RESULT BFS_APSP_L(MatrixType &adj_matrix, bool nd, bool mem_test_only, bool verbose) {

    const int64_t n = matrix_ext::rows(adj_matrix);
    COUNTERS counters{};
    counters.distance_matrix_bytes = DistanceMatrixBytes(n);
    counters.estimated_dominant_memory_bytes = counters.distance_matrix_bytes;
    UpdateTotalMemoryCounter(counters);
    UpdateProcessMemoryPeakCheckpoint();

    /// computation starts
    if (verbose) {
        std::cout << "computation starts: " << NowLocalDateTimeString() << std::endl;
    }
    Timer timer_global; // timer for performance benchmark
    timer_global.Start();
    Timer timer; // timer for debug

    /// compute adjacency list
    timer.Start();
    auto degree_and_adj = ComputeDegreeAndAdjList(adj_matrix, n);
    std::vector<std::vector<int64_t>> A = std::move(degree_and_adj.second);
    counters.adjacency_list_bytes = EstimateAdjacencyListBytes(A);
    UpdateTotalMemoryCounter(counters);

    ReleaseAdjacencyMatrix(adj_matrix);
    ResetProcessMemoryPeakBaseline();
    timer.Stop();
    UpdateProcessMemoryPeakCheckpoint();
    if (verbose) {
        std::cout << "@@ Time to compute adjacency list: " << timer.Seconds() << " seconds" << std::endl;
    }

    if (mem_test_only) {
        Eigen::MatrixXd D;
        if (verbose) {
            PrintTheoreticalMemorySummary(counters, n);
        }
        return RESULT{std::move(D), counters, std::numeric_limits<double>::quiet_NaN()};
    }

    Eigen::MatrixXd D;
    Eigen::MatrixXd* D_out = nullptr;
    if (!nd) {
        timer.Start();
        D = Eigen::MatrixXd::Constant(n, n, std::numeric_limits<double>::infinity());
        D_out = &D;
        timer.Stop();
        UpdateProcessMemoryPeakCheckpoint();
        if (verbose) {
            std::cout << "@@ Time to initialize D: " << timer.Seconds() << " seconds" << std::endl;
        }
    }
    /// main loop
    timer.Start();
    const int progress_update_interval_pct = 1;
    int next_report_pct = progress_update_interval_pct;
    if (verbose && n > 0) {
        std::cout << "Progress:  0%" << std::flush;
    }
    for (int64_t idx = 0; idx < n; ++idx) {
        BFSSSS(A, D_out, n, idx);
        // update process memory peak checkpoint
        if (((idx + 1) & 63) == 0 || (idx + 1) == n) {
            UpdateProcessMemoryPeakCheckpoint();
        }
        if (verbose) {
            UpdateProgressPercent(idx + 1, n, progress_update_interval_pct, next_report_pct);
        }
    }
    timer.Stop();
    UpdateProcessMemoryPeakCheckpoint();
    if (verbose) {
        std::cout << "@@ Time of main loop: " << timer.Seconds() << " seconds" << std::endl;
    }

    /// computation ends
    timer_global.Stop();
    double durationInSeconds = timer_global.Seconds();
    UpdateProcessMemoryPeakCheckpoint();
    if (verbose) {
        std::cout << "end datetime: " << NowLocalDateTimeString() << std::endl;
        std::cout << "@@ Total computation time: " << durationInSeconds << " seconds" << std::endl;
        PrintTheoreticalMemorySummary(counters, n);
        PrintProcessMemorySummary();
    }

    return RESULT{std::move(D), counters, durationInSeconds};
}



// ============================================================================
// Python-callable interface (for ctypes)
// ============================================================================
#ifdef PYTHON_BUILD

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

/**
 * C-compatible POD mirror of BFS COUNTERS, for ctypes access from Python.
 * Field order and types must stay in sync with the Python BfsCounters Structure.
 */
struct BFS_PY_COUNTERS {
    int64_t estimated_dominant_memory_bytes;
    int64_t distance_matrix_bytes;
    int64_t adjacency_list_bytes;
    int64_t estimated_total_memory_bytes;
    int64_t process_baseline_working_set_bytes;
    int64_t process_peak_working_set_bytes;
    int64_t process_delta_peak_working_set_bytes;
    int64_t process_baseline_private_bytes;
    int64_t process_peak_private_bytes;
    int64_t process_delta_peak_private_bytes;
};

/**
 * @brief Python-callable wrapper for BFS APSP algorithm
 *
 * @param adj_matrix_data Flat array of adjacency matrix (row-major, n*n elements)
 * @param n Number of nodes
 * @param output_distances Output array for distance matrix (n*n elements, allocated by caller).
 *                         May be NULL to skip copying distances back to Python.
 * @param elapsed_time Output parameter for computation time
 * @param out_counters Output struct for all algorithm counters; may be NULL
 * @return 0 on success, -1 on error
 */
EXPORT int bfs_apsp_py(
    const double* adj_matrix_data,
    int64_t n,
    double* output_distances,
    double* elapsed_time,
    BFS_PY_COUNTERS* out_counters)
{
    try {
        // Convert input array to Eigen matrix
        Eigen::Map<const Eigen::MatrixXd> adj_matrix_map(adj_matrix_data, n, n);
        MatrixType adj_matrix = Eigen::MatrixXd(adj_matrix_map);
        
        ProcessMemoryPeakTracker mem_tracker;
        mem_tracker.CaptureBaseline();

        // Call the algorithm
        RESULT result;
        {
            ProcessMemoryTrackerScope tracker_scope(&mem_tracker);
            result = BFS_APSP_L(adj_matrix, false, false, false);
        }
        mem_tracker.UpdatePeak();
        FillProcessMemoryCounters(result.counters, mem_tracker);
        
        if (output_distances != nullptr) {
            // Copy output distance matrix (row-major) only when requested by the caller.
            Eigen::Map<Eigen::MatrixXd>(output_distances, n, n) = result.D;
        }
        
        *elapsed_time = result.computation_time;
        if (out_counters) {
            out_counters->estimated_dominant_memory_bytes      = result.counters.estimated_dominant_memory_bytes;
            out_counters->distance_matrix_bytes                = result.counters.distance_matrix_bytes;
            out_counters->adjacency_list_bytes                 = result.counters.adjacency_list_bytes;
            out_counters->estimated_total_memory_bytes         = result.counters.estimated_total_memory_bytes;
            out_counters->process_baseline_working_set_bytes   = result.counters.process_baseline_working_set_bytes;
            out_counters->process_peak_working_set_bytes       = result.counters.process_peak_working_set_bytes;
            out_counters->process_delta_peak_working_set_bytes = result.counters.process_delta_peak_working_set_bytes;
            out_counters->process_baseline_private_bytes       = result.counters.process_baseline_private_bytes;
            out_counters->process_peak_private_bytes           = result.counters.process_peak_private_bytes;
            out_counters->process_delta_peak_private_bytes     = result.counters.process_delta_peak_private_bytes;
        }
        
        return 0;  // Success
    }
    catch (...) {
        return -1;  // Error
    }
}

#ifdef __cplusplus
}
#endif
#endif // PYTHON_BUILD


// ============================================================================
// MATLAB MEX Interface
// ============================================================================

#ifdef MATLAB_BUILD

static mxArray* createCountersStruct(const COUNTERS& counters) {
    const char* fieldnames[] = {
        "estimated_dominant_memory_bytes",
        "distance_matrix_bytes",
        "adjacency_list_bytes",
        "estimated_total_memory_bytes",
        "process_baseline_working_set_bytes",
        "process_peak_working_set_bytes",
        "process_delta_peak_working_set_bytes",
        "process_baseline_private_bytes",
        "process_peak_private_bytes",
        "process_delta_peak_private_bytes",
    };
    constexpr size_t numFields = sizeof(fieldnames) / sizeof(fieldnames[0]);
    mxArray* matlabStruct = mxCreateStructMatrix(1, 1, static_cast<int>(numFields), fieldnames);
    mxSetFieldByNumber(matlabStruct, 0, 0,
                       mxCreateDoubleScalar(static_cast<double>(counters.estimated_dominant_memory_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 1,
                       mxCreateDoubleScalar(static_cast<double>(counters.distance_matrix_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 2,
                       mxCreateDoubleScalar(static_cast<double>(counters.adjacency_list_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 3,
                       mxCreateDoubleScalar(static_cast<double>(counters.estimated_total_memory_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 4,
                       mxCreateDoubleScalar(static_cast<double>(counters.process_baseline_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 5,
                       mxCreateDoubleScalar(static_cast<double>(counters.process_peak_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 6,
                       mxCreateDoubleScalar(static_cast<double>(counters.process_delta_peak_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 7,
                       mxCreateDoubleScalar(static_cast<double>(counters.process_baseline_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 8,
                       mxCreateDoubleScalar(static_cast<double>(counters.process_peak_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 9,
                       mxCreateDoubleScalar(static_cast<double>(counters.process_delta_peak_private_bytes)));
    return matlabStruct;
}

/**
 * MATLAB MEX entry point for baseline BFS APSP on an undirected, unweighted graph.
 *
 * Usage:
 *   [D, t, counters] = bfs_apsp_l_cpp(A)
 *   [D, t, counters] = bfs_apsp_l_cpp(A, flag, ...)
 *
 * Inputs:
 *   A                Real, square adjacency matrix supplied as dense or sparse `double`
 *                    or `logical`. Nonzero entries are treated as unweighted edges. A is
 *                    expected to be symmetric because the algorithm assumes an undirected
 *                    graph.
 *   flag             Optional MATLAB character-vector flag. Flags may be combined in any
 *                    order:
 *                    - "nd": do not allocate or populate the distance matrix.
 *                    - "ndr": compute distances normally, but return a dummy first output.
 *                    - "mem_test_only": stop after constructing the adjacency list and
 *                      memory estimates; no searches or distance matrix are performed.
 *
 * Outputs:
 *   D         n-by-n `double` all-pairs shortest-path distance matrix. With "nd", "ndr",
 *             or "mem_test_only", D is instead the `double` scalar 0 (a 1-by-1 matrix).
 *   t         Algorithm computation time in seconds; NaN with "mem_test_only".
 *   counters  Scalar struct containing theoretical distance/adjacency/total memory
 *             estimates and sampled process working-set/private-memory baseline, peak,
 *             and delta-peak values, all reported as `double` scalars.
 *
 * Notes:
 *   Between zero and three outputs may be requested. Only MATLAB character vectors are
 *   parsed as flags (string scalars are not); unrecognized character vectors and non-char
 *   optional inputs are ignored. "ndr" changes only the returned first output, whereas
 *   "nd" changes the algorithm's storage mode.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check the number of input arguments.
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                          "Usage: [D,t,counters] = bfs_apsp_l_cpp(A, ['nd'], ['ndr'], ['mem_test_only']).");
    }

    ProcessMemoryPeakTracker mem_tracker;
    mem_tracker.CaptureBaseline();

    // Parse adjacency matrix
    MatrixType adj_matrix = parseAdjacencyMatrix(prhs[0]);
    int64_t numVertices = matrix_ext::rows(adj_matrix);
    mem_tracker.UpdatePeak();

    bool nd = false;
    bool ndr = false;
    bool mem_test_only = false;

    // Check for additional parameters
    for (int i = 1; i < nrhs; i++) {
        if (mxIsChar(prhs[i])) {
            // Convert MATLAB string to C++ string
            char* str = mxArrayToString(prhs[i]);
            std::string param(str);
            mxFree(str);

            if (param == "nd") {
                nd = true;
            } else if (param == "ndr") {
                ndr = true;
            } else if (param == "mem_test_only") {
                mem_test_only = true;
            }
        }
    }

    RESULT result;
    {
        ProcessMemoryTrackerScope tracker_scope(&mem_tracker);
        result = BFS_APSP_L(adj_matrix, nd, mem_test_only);
    }
    mem_tracker.UpdatePeak();
    FillProcessMemoryCounters(result.counters, mem_tracker);

    // Create first output: distances, dummy matrix, or NaN matrix (mem_test_only)
    if (nlhs > 0) {
        if (nd || ndr || mem_test_only) {
            // Return a dummy 1x1 matrix with value 0
            plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
            *mxGetPr(plhs[0]) = 0.0;
        } else {
            // Return the actual distance matrix
            plhs[0] = mxCreateDoubleMatrix(static_cast<mwSize>(numVertices),
                                            static_cast<mwSize>(numVertices), mxREAL);
            Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), numVertices, numVertices) = result.D;
        }
    }

    // Second output: elapsed time (NaN when mem_test_only; set in BFS_APSP_L)
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar(result.computation_time);
    }

    if (nlhs > 2) {
        plhs[2] = createCountersStruct(result.counters);
    }
}

#endif // MATLAB_BUILD
