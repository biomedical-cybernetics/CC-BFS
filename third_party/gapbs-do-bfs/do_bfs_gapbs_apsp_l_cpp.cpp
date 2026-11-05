#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <limits>
#include <chrono>
#include <iostream>
#include <memory>
#include <utility>
#include "do_bfs_gapbs_apsp_l.h"
#include "string"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <variant>
#include <type_traits>
#include "include/bitmap.h"
#include "include/sliding_queue.h"
#include "include/platform_atomics.h"
#include "include/pvector.h"
#include "include/graph.h"
#include "include/timer.h"
#include "include/progress_and_time_utils.h"
#include "include/process_memory_utils.h"
#ifdef MATLAB_BUILD
#include "mex.h"
#include "include/mex_utils.h"
#endif

typedef int64_t NodeID;
typedef CSRGraph<NodeID> Graph;

inline void ReleaseAdjacencyMatrix(MatrixType& adj_matrix) {
    std::visit([](auto& m) {
        using Mat = std::decay_t<decltype(m)>;
        Mat empty;
        m.swap(empty);
    }, adj_matrix);
}

static int64_t DistanceMatrixBytes(int64_t n) {
    if (n <= 0) {
        return 0;
    }
    const size_t prod = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = prod * sizeof(double);
    return static_cast<int64_t>(bytes);
}

static int64_t EstimateCSRGraphBytes(const Graph& g) {
    if (g.num_nodes() < 0) {
        return 0;
    }
    return static_cast<int64_t>((g.num_nodes() + 1) * sizeof(NodeID*))
        + static_cast<int64_t>(g.num_edges_directed() * sizeof(NodeID));
}

static void UpdateTotalMemoryCounter(COUNTERS& counters) {
    counters.estimated_total_memory_bytes =
        counters.estimated_dominant_memory_bytes + counters.adjacency_list_bytes;
}

static void PrintTheoreticalMemorySummary(const COUNTERS &counters, const int64_t N) {
    std::cout << "@@ Theoretical memory usage (GiB): n=" << N
              << " estimated_total_memory_bytes=" << BytesToGiB(counters.estimated_total_memory_bytes)
              << " estimated_dominant_memory_bytes=" << BytesToGiB(counters.estimated_dominant_memory_bytes)
              << " adjacency_list_bytes=" << BytesToGiB(counters.adjacency_list_bytes)
              << " distance_matrix_bytes=" << BytesToGiB(counters.distance_matrix_bytes)
              << std::endl;
}

void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
        NodeID u = *q_iter;
        bm.set_bit_atomic(u);
    }
}

void BitmapToQueue(const int64_t N,
                   const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
    {
        QueueBuffer<NodeID> lqueue(queue);
        for (NodeID n=0; n < N; n++)
            if (bm.get_bit(n))
                lqueue.push_back(n);
        lqueue.flush();
    }
    queue.slide_window();
}


int64_t BUStep(const Graph &g,
               Eigen::VectorXd &D_s,
               pvector<NodeID> &parent,
               Bitmap &front,
               Bitmap &next,
               COUNTERS &counters) {
    counters.bottom_up_step_count++;
    int64_t awake_count = 0;
    next.reset();
    for (NodeID u=0; u < g.num_nodes(); u++) {
        if (parent[u] < 0) {
            for (NodeID v : g.in_neigh(u)) {
                if (front.get_bit(v)) {
                    parent[u] = v;
                    D_s(u) = D_s(v) + 1;
                    awake_count++;
                    next.set_bit(u);
                    break;
                }
            }
        }
    }
    return awake_count;
}


int64_t TDStep(const Graph &g,
               Eigen::VectorXd &D_s,
               pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue,
               COUNTERS &counters) {
    counters.top_down_step_count++;
    int64_t scout_count = 0;
    {
        QueueBuffer<NodeID> lqueue(queue);
        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
            NodeID u = *q_iter;
            for (NodeID v : g.out_neigh(u)) {
                NodeID curr_val = parent[v];
                if (curr_val < 0) {  // v does not have a parent
                    if (compare_and_swap(parent[v], curr_val, u)) {
                        D_s(v) = D_s(u) + 1;
                        lqueue.push_back(v);
                        scout_count += -curr_val;
                    }
                }
            }
        }
        lqueue.flush();
    }
    return scout_count;
}

pvector<NodeID> InitParent(const Graph &g) {
    pvector<NodeID> parent(g.num_nodes());
    for (NodeID n=0; n < g.num_nodes(); n++)
        parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
    return parent;
}

// BFS algorithm to find the shortest path from a single source to all other vertices in the graph
void single_source_do_bfs(const Graph &g,
                       Eigen::MatrixXd *D_out,
                       int64_t N,
                       int64_t s,
                       COUNTERS &counters,
                       int64_t alpha = 14,
                       int64_t beta = 24
                       ) {



    Eigen::VectorXd D_s = Eigen::VectorXd::Constant(N, std::numeric_limits<double>::infinity());
    D_s(s) = 0;

    pvector<NodeID> parent = InitParent(g);
    parent[s] = s;

    SlidingQueue<NodeID> queue(N);
    queue.push_back(s);
    queue.slide_window();

    Bitmap curr(N);
    curr.reset();
    Bitmap front(N);
    front.reset();
    int64_t edges_to_check = g.num_edges_directed();
    int64_t scout_count = g.out_degree(s);
    while (!queue.empty()) {
        if (scout_count > edges_to_check / alpha) {
            int64_t awake_count, old_awake_count;
            QueueToBitmap(queue, front);
            awake_count = queue.size();
            queue.slide_window();
            do {
                old_awake_count = awake_count;
                awake_count = BUStep(g, D_s, parent, front, curr, counters);
                front.swap(curr);
            } while ((awake_count >= old_awake_count) ||
                     (awake_count > N / beta));
            BitmapToQueue(N, front, queue);
            scout_count = 1;
        } else {
            edges_to_check -= scout_count;
            scout_count = TDStep(g, D_s, parent, queue, counters);
            queue.slide_window();
        }
    }
    if (D_out != nullptr) {
        D_out->row(s) = D_s.transpose();
    }
}

Graph BuildCSRGraph(const MatrixType &adj_matrix) {
    const int64_t N = matrix_ext::rows(adj_matrix);

    // Count edges per source (row) to build CSR offsets
    pvector<SGOffset> degree(N);
    for (int64_t i = 0; i < N; ++i) {
        degree[i] = 0;
    }

    if (std::holds_alternative<Eigen::SparseMatrix<bool>>(adj_matrix)) {
        const auto &sp = std::get<Eigen::SparseMatrix<bool>>(adj_matrix);
        for (int col = 0; col < sp.outerSize(); ++col) {
            for (Eigen::SparseMatrix<bool>::InnerIterator it(sp, col); it; ++it) {
                if (it.value()) {
                    degree[static_cast<int64_t>(it.row())] += 1;
                }
            }
        }
    } else if (std::holds_alternative<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(adj_matrix)) {
        const auto &m = std::get<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(adj_matrix);
        for (int64_t i = 0; i < N; ++i) {
            int64_t cnt = 0;
            for (int64_t j = 0; j < N; ++j) {
                if (m(i, j)) {
                    ++cnt;
                }
            }
            degree[i] = cnt;
        }
    } else {
        const auto &m = std::get<Eigen::MatrixXd>(adj_matrix);
        for (int64_t i = 0; i < N; ++i) {
            int64_t cnt = 0;
            for (int64_t j = 0; j < N; ++j) {
                if (m(i, j) > 0) {
                    ++cnt;
                }
            }
            degree[i] = cnt;
        }
    }

    pvector<SGOffset> offsets(N + 1);
    offsets[0] = 0;
    for (int64_t i = 0; i < N; ++i) {
        offsets[i + 1] = offsets[i] + degree[i];
    }

    const int64_t total_edges = offsets[N];
    NodeID *out_neighbors = new NodeID[total_edges];
    NodeID **out_index = CSRGraph<NodeID>::GenIndex(offsets, out_neighbors);

    // Fill neighbor lists (row -> col)
    pvector<SGOffset> pos(N);
    for (int64_t i = 0; i < N; ++i) {
        pos[i] = offsets[i];
    }

    if (std::holds_alternative<Eigen::SparseMatrix<bool>>(adj_matrix)) {
        const auto &sp = std::get<Eigen::SparseMatrix<bool>>(adj_matrix);
        for (int col = 0; col < sp.outerSize(); ++col) {
            for (Eigen::SparseMatrix<bool>::InnerIterator it(sp, col); it; ++it) {
                if (it.value()) {
                    const int64_t row = static_cast<int64_t>(it.row());
                    out_neighbors[pos[row]++] = static_cast<NodeID>(col);
                }
            }
        }
    } else if (std::holds_alternative<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(adj_matrix)) {
        const auto &m = std::get<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(adj_matrix);
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                if (m(i, j)) {
                    out_neighbors[pos[i]++] = static_cast<NodeID>(j);
                }
            }
        }
    } else {
        const auto &m = std::get<Eigen::MatrixXd>(adj_matrix);
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                if (m(i, j) > 0) {
                    out_neighbors[pos[i]++] = static_cast<NodeID>(j);
                }
            }
        }
    }

    return CSRGraph<NodeID>(N, out_index, out_neighbors);
}

// Direction-optimizing BFS for All-Pairs Shortest Path Distance
RESULT DO_BFS_APSP_L(MatrixType &adj_matrix, const bool nd, const bool mem_test_only, const bool verbose) {

    const int64_t N = matrix_ext::rows(adj_matrix);
    COUNTERS counters{};
    counters.distance_matrix_bytes = DistanceMatrixBytes(N);
    counters.estimated_dominant_memory_bytes = counters.distance_matrix_bytes;
    UpdateTotalMemoryCounter(counters);
    UpdateProcessMemoryPeakCheckpoint();

    /// computation starts
    if (verbose) {
        std::cout << "Computation starts: " << NowLocalDateTimeString() << std::endl;
    }

    Timer timer_global; // timer for performance benchmark
    timer_global.Start();
    Timer timer; // timer for debug

    int64_t alpha = 14;
    int64_t beta = 24;

    // Construct CSR graph from adjacency matrix
    timer.Start();
    Graph g = BuildCSRGraph(adj_matrix);
    counters.adjacency_list_bytes = EstimateCSRGraphBytes(g);
    UpdateTotalMemoryCounter(counters);

    ReleaseAdjacencyMatrix(adj_matrix);
    ResetProcessMemoryPeakBaseline();
    timer.Stop();
    UpdateProcessMemoryPeakCheckpoint();
    if (verbose) {
        std::cout << "@@ Time to build CSR graph: " << timer.Seconds() << " seconds" << std::endl;
    }

    if (mem_test_only) {
        Eigen::MatrixXd D;
        if (verbose) {
            PrintTheoreticalMemorySummary(counters, N);
        }
        return RESULT{std::move(D), counters, std::numeric_limits<double>::quiet_NaN()};
    }

    Eigen::MatrixXd D;
    Eigen::MatrixXd *D_ptr = nullptr;
    if (!nd) {
        timer.Start();
        D = Eigen::MatrixXd::Constant(N, N, std::numeric_limits<double>::infinity());
        D_ptr = &D;
        timer.Stop();
        UpdateProcessMemoryPeakCheckpoint();
        if (verbose) {
            std::cout << "@@ Time to initialize D: " << timer.Seconds() << " seconds" << std::endl;
        }
    }


    /// main loop
    const int progress_update_interval_pct = 5;
    int next_report_pct = progress_update_interval_pct;
    if (verbose && N > 0) {
        std::cout << "Progress:  0%" << std::flush;
    }
    for (int64_t idx = 0; idx < N; ++idx) {
        int64_t s = idx;
        single_source_do_bfs(g, D_ptr, N, s, counters, alpha, beta);
        if (((idx + 1) & 63) == 0 || (idx + 1) == N) {
            UpdateProcessMemoryPeakCheckpoint();
        }
        if (verbose) {
            UpdateProgressPercent(idx + 1, N, progress_update_interval_pct, next_report_pct);
        }
    }

    /// computation ends
    timer_global.Stop();
    counters.time_secs_overall = timer_global.Seconds();
    UpdateProcessMemoryPeakCheckpoint();
    if (verbose) {
        std::cout << "end datetime: " << NowLocalDateTimeString() << std::endl;
        std::cout << "@@ Total computation time: " << counters.time_secs_overall << " seconds" << std::endl;
        PrintTheoreticalMemorySummary(counters, N);
        PrintProcessMemorySummary();

        std::cout << "top_down_step_count: " << counters.top_down_step_count << std::endl;
        std::cout << "bottom_up_step_count: " << counters.bottom_up_step_count << std::endl;
        std::cout << "ratio: " << (double)counters.top_down_step_count / counters.bottom_up_step_count << std::endl;
    }

    return RESULT{std::move(D), std::move(counters), timer_global.Seconds()};
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
 * C-compatible POD mirror of DO-BFS COUNTERS, for ctypes access from Python.
 * Field order and types must stay in sync with the Python DoBfsCounters Structure.
 */
struct DO_BFS_PY_COUNTERS {
    int64_t top_down_step_count;
    int64_t bottom_up_step_count;
    double  time_secs_overall;
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
 * @brief Python-callable wrapper for DO-BFS APSP algorithm
 *
 * @param adj_matrix_data Flat array of adjacency matrix (row-major, n*n elements)
 * @param n Number of nodes
 * @param output_distances Output array for distance matrix (n*n elements, allocated by caller).
 *                         May be NULL to skip copying distances back to Python.
 * @param elapsed_time Output parameter for computation time
 * @param out_counters Output struct for all algorithm counters; may be NULL
 * @return 0 on success, -1 on error
 */
EXPORT int do_bfs_gapbs_apsp_py(
    const double* adj_matrix_data,
    int64_t n,
    double* output_distances,
    double* elapsed_time,
    DO_BFS_PY_COUNTERS* out_counters)
{
    try {
        Eigen::Map<const Eigen::MatrixXd> adj_matrix_map(adj_matrix_data, n, n);
        MatrixType adj_matrix = Eigen::MatrixXd(adj_matrix_map);

        ProcessMemoryPeakTracker mem_tracker;
        mem_tracker.CaptureBaseline();

        RESULT result;
        {
            ProcessMemoryTrackerScope tracker_scope(&mem_tracker);
            result = DO_BFS_APSP_L(adj_matrix, false, false, false);
        }
        mem_tracker.UpdatePeak();
        FillProcessMemoryCounters(result.counters, mem_tracker);

        if (output_distances != nullptr) {
            Eigen::Map<Eigen::MatrixXd>(output_distances, n, n) = result.D;
        }
        *elapsed_time = result.computation_time;
        if (out_counters) {
            out_counters->top_down_step_count                  = result.counters.top_down_step_count;
            out_counters->bottom_up_step_count                 = result.counters.bottom_up_step_count;
            out_counters->time_secs_overall                    = result.counters.time_secs_overall;
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

        return 0;
    }
    catch (...) {
        return -1;
    }
}

#ifdef __cplusplus
}
#endif
#endif // PYTHON_BUILD


// ============================================================================
// MATLAB MEX interface
// ============================================================================
#ifdef MATLAB_BUILD

mxArray* createCountersStruct(const COUNTERS& counters) {
    const char* fieldnames[] = {
            "top_down_step_count",
            "bottom_up_step_count",
            "time_secs_overall",
            "estimated_dominant_memory_bytes",
            "distance_matrix_bytes",
            "adjacency_list_bytes",
            "estimated_total_memory_bytes",
            "process_baseline_working_set_bytes",
            "process_peak_working_set_bytes",
            "process_delta_peak_working_set_bytes",
            "process_baseline_private_bytes",
            "process_peak_private_bytes",
            "process_delta_peak_private_bytes"
    };
    constexpr size_t numFields = sizeof(fieldnames) / sizeof(fieldnames[0]);

    mxArray* matlabStruct = mxCreateStructMatrix(1, 1, static_cast<int>(numFields), fieldnames);

    mxSetFieldByNumber(matlabStruct, 0, 0, mxCreateDoubleScalar(static_cast<double>(counters.top_down_step_count)));
    mxSetFieldByNumber(matlabStruct, 0, 1, mxCreateDoubleScalar(static_cast<double>(counters.bottom_up_step_count)));
    mxSetFieldByNumber(matlabStruct, 0, 2, mxCreateDoubleScalar(counters.time_secs_overall));
    mxSetFieldByNumber(matlabStruct, 0, 3, mxCreateDoubleScalar(static_cast<double>(counters.estimated_dominant_memory_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 4, mxCreateDoubleScalar(static_cast<double>(counters.distance_matrix_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 5, mxCreateDoubleScalar(static_cast<double>(counters.adjacency_list_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 6, mxCreateDoubleScalar(static_cast<double>(counters.estimated_total_memory_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 7, mxCreateDoubleScalar(static_cast<double>(counters.process_baseline_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 8, mxCreateDoubleScalar(static_cast<double>(counters.process_peak_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 9, mxCreateDoubleScalar(static_cast<double>(counters.process_delta_peak_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 10, mxCreateDoubleScalar(static_cast<double>(counters.process_baseline_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 11, mxCreateDoubleScalar(static_cast<double>(counters.process_peak_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 12, mxCreateDoubleScalar(static_cast<double>(counters.process_delta_peak_private_bytes)));

    return matlabStruct;
}

/**
 * MATLAB MEX entry point for direction-optimizing BFS APSP on an undirected,
 * unweighted graph.
 *
 * Usage:
 *   [D, t, counters] = do_bfs_gapbs_apsp_l_cpp(A)
 *   [D, t, counters] = do_bfs_gapbs_apsp_l_cpp(A, flag, ...)
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
 *                    - "mem_test_only": stop after constructing the CSR graph and memory
 *                      estimates; no searches or distance matrix are performed.
 *
 * Outputs:
 *   D         n-by-n `double` all-pairs shortest-path distance matrix. With "nd", "ndr",
 *             or "mem_test_only", D is instead the `double` scalar 0 (a 1-by-1 matrix).
 *   t         Algorithm computation time in seconds; NaN with "mem_test_only".
 *   counters  Scalar struct containing top-down and bottom-up step counts, overall time,
 *             theoretical distance/CSR/total memory estimates, and sampled process
 *             working-set/private-memory baseline, peak, and delta-peak values.
 *
 * Notes:
 *   Between zero and three outputs may be requested. Only MATLAB character vectors are
 *   parsed as flags (string scalars are not); unrecognized character vectors and non-char
 *   optional inputs are ignored. "ndr" changes only the returned first output, whereas
 *   "nd" changes the algorithm's storage mode. The direction-switch parameters are fixed
 *   internally at alpha=14 and beta=24.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check the number of input arguments.
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                          "At least one inputs required (algorithm type and adjacency matrix).");
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
        result = DO_BFS_APSP_L(adj_matrix, nd, mem_test_only);
    }
    mem_tracker.UpdatePeak();
    FillProcessMemoryCounters(result.counters, mem_tracker);

    const double nanVal = std::numeric_limits<double>::quiet_NaN();

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

    // Second output: elapsed time (NaN when mem_test_only; set in DO_BFS_APSP_L)
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar(result.computation_time);
    }
    if (nlhs > 2) {
        plhs[2] = createCountersStruct(result.counters);
    }
}
#endif // MATLAB_BUILD
