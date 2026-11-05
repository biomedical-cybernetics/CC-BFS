//
// Exact Sum Sweep (undirected, connected, unweighted)
// MEX front-end   — BFSSSS()  and  ComputeDegreeAndAdjList() UNCHANGED
//
#include "ess.h"
#include <queue>
#include <vector>
#include <limits>
#include <iostream>
#include <chrono>
#include <array>
#include <numeric>
#ifdef MATLAB_BUILD
#include "include/mex_utils.h"
#include "mex.h"
#endif
#include "include/timer.h"
#include "include/graph_utils.h"

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */
inline bool isOpenForDiameter(double  U_v,
                              int64_t DL,
                              bool    closed)
{   return !closed && U_v > DL; }

inline bool isOpenForRadius  (double  L_v,
                              int64_t RU,
                              bool    closed)
{   return !closed && L_v < RU; }

size_t countOpen(const Eigen::VectorXd& L,
                 const Eigen::VectorXd& U,
                 const std::vector<char>& closed,
                 int64_t DL,
                 int64_t RU)
{
    size_t c = 0;
    const int64_t n = L.size();
    for (int64_t v = 0; v < n; ++v)
        if (isOpenForDiameter(U[v], DL, closed[v]) ||
            isOpenForRadius  (L[v], RU, closed[v]))
            ++c;
    return c;
}

/* ------------------------------------------------------------------ */
/*  Vertex selection                                                  */
/* ------------------------------------------------------------------ */
int64_t selectMaxU(const Eigen::VectorXd& U,
                   const std::vector<char>& closed,
                   const std::vector<double>& S_F,
                   int64_t DL)
{
    double bestU   = -std::numeric_limits<double>::infinity();
    double bestSF  = -std::numeric_limits<double>::infinity();
    int64_t best   = -1;

    const int64_t n = U.size();
    for (int64_t v = 0; v < n; ++v)
        if (isOpenForDiameter(U[v], DL, closed[v])) {
            if (U[v] > bestU ||
               (U[v] == bestU && S_F[v] > bestSF)) {
                bestU  = U[v];
                bestSF = S_F[v];
                best   = v;
            }
        }
    if (best == -1) std::cerr << "selectMaxU(): no candidate" << std::endl;
    return best;
}

int64_t selectMinL(const Eigen::VectorXd& L,
                   const std::vector<char>& closed,
                   const std::vector<double>& S_F,
                   int64_t RU)
{
    double bestL   =  std::numeric_limits<double>::infinity();
    double bestSF  =  std::numeric_limits<double>::infinity();
    int64_t best   = -1;

    const int64_t n = L.size();
    for (int64_t v = 0; v < n; ++v)
        if (isOpenForRadius(L[v], RU, closed[v])) {
            if (L[v] < bestL ||
               (L[v] == bestL && S_F[v] < bestSF)) {
                bestL  = L[v];
                bestSF = S_F[v];
                best   = v;
            }
        }
    if (best == -1) std::cerr << "selectMinL(): no candidate" << std::endl;
    return best;
}

int64_t selectMaxSF(const std::vector<double>& S_F,
                    const std::vector<char>& closed,
                    const bool verbose)
{
    double bestSF  = -std::numeric_limits<double>::infinity();
    int64_t best   = -1;

    const int64_t n = static_cast<int64_t>(S_F.size());
    for (int64_t v = 0; v < n; ++v)
        if (!closed[v]) {
            if (S_F[v] > bestSF) {
                bestSF = S_F[v];
                best   = v;
            }
        }
    if (verbose && best == -1) std::cerr << "selectMaxSF(): no candidate" << std::endl;
    return best;
}

inline bool haveVU(const Eigen::VectorXd& U,const std::vector<char>& closed,
                   int64_t DL)
{
    for (int64_t v=0,n=U.size(); v<n; ++v)
        if (isOpenForDiameter(U[v],DL,closed[v])) return true;
    return false;
}
inline bool haveVL(const Eigen::VectorXd& L,const std::vector<char>& closed,
                   int64_t RU)
{
    for (int64_t v=0,n=L.size(); v<n; ++v)
        if (isOpenForRadius(L[v],RU,closed[v]))   return true;
    return false;
}

/* ------------------------------------------------------------------ */
/*  BFSSSS — UNMODIFIED (keep your own version)                       */
/* ------------------------------------------------------------------ */

// BFS algorithm to find the shortest path from a single source to all other vertices in the graph
Eigen::VectorXd BFSSSS(const std::vector<std::vector<int64_t>>& A,
                       const int64_t n,
                       const int64_t s) {
    Eigen::VectorXd distance = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
    distance(s) = 0;

    std::queue<int64_t> queue;
    queue.push(s);

    while (!queue.empty()) {
        int64_t curr = queue.front();
        queue.pop();

        double alt_dst = distance(curr) + 1;
        for (int64_t t : A[curr]) {
            if (alt_dst < distance(t)) {
                queue.push(t);
                distance(t) = alt_dst;
            }
        }
    }
    return distance;
}


/* ------------------------------------------------------------------ */
/*  Sum-Sweep (unchanged except for closing the source)               */
/* ------------------------------------------------------------------ */
void sumSweep(const std::vector<std::vector<int64_t>>& A,
              const int64_t n,
              int64_t s,
              const int64_t k,
              int64_t& DL,
              int64_t& RU,
              Eigen::VectorXd& L,
              Eigen::VectorXd& U,
              std::vector<char>& closed,
              std::vector<double>& S_F,
              const bool verbose)
{
    Eigen::VectorXd D_s = BFSSSS(A, n, s);
    int64_t ecc_s       = static_cast<int64_t>(D_s.maxCoeff());

    L[s] = U[s] = ecc_s;
    closed[s]   = true;
    DL = RU     = ecc_s;
    L  = L.cwiseMax(D_s);
    U  = U.cwiseMin(Eigen::VectorXd::Constant(n, ecc_s) + D_s);
    S_F = std::vector<double>(n);
    for (int64_t v=0; v<n; ++v) S_F[v] = D_s[v];

    for (int64_t i = 2; i <= k - 1; ++i) {
        s = selectMaxSF(S_F, closed, verbose);
        if (s == -1) {
            break;
        }
        D_s   = BFSSSS(A,n,s);
        ecc_s = static_cast<int64_t>(D_s.maxCoeff());

        L[s] = U[s] = ecc_s;
        closed[s]   = true;
        DL = std::max<int64_t>(DL, ecc_s);
        RU = std::min<int64_t>(RU, ecc_s);

        L = L.cwiseMax(D_s);
        U = U.cwiseMin(Eigen::VectorXd::Constant(n, ecc_s) + D_s);
        for (int64_t v=0; v<n; ++v) S_F[v] += D_s[v];
    }

    // final sweep from the current argmin S_F
    s = static_cast<int64_t>(std::distance(S_F.begin(),
         std::min_element(S_F.begin(), S_F.end())));
    D_s   = BFSSSS(A,n,s);
    ecc_s = static_cast<int64_t>(D_s.maxCoeff());

    L[s] = U[s] = ecc_s;
    closed[s]   = true;
    DL = std::max<int64_t>(DL, ecc_s);
    RU = std::min<int64_t>(RU, ecc_s);

    L = L.cwiseMax(D_s);
    U = U.cwiseMin(Eigen::VectorXd::Constant(n, ecc_s) + D_s);
    for (int64_t v=0; v<n; ++v) S_F[v] += D_s[v];
}

/* ------------------------------------------------------------------ */
/*  One generic Step-Forward                                          */
/* ------------------------------------------------------------------ */
void stepForward(const std::vector<std::vector<int64_t>>& A,
                 const int64_t n,
                 int64_t s,
                 int64_t& DL,
                 int64_t& RU,
                 Eigen::VectorXd& L,
                 Eigen::VectorXd& U,
                 std::vector<char>& closed,
                 std::vector<double>& S_F)
{
    Eigen::VectorXd D_s = BFSSSS(A,n,s);
    int64_t ecc_s       = static_cast<int64_t>(D_s.maxCoeff());

    closed[s] = true;
    L[s] = U[s] = ecc_s;

    L = L.cwiseMax(D_s);
    U = U.cwiseMin(Eigen::VectorXd::Constant(n, ecc_s) + D_s);

    DL = std::max<int64_t>(DL, ecc_s);
    RU = std::min<int64_t>(RU, ecc_s);

    for (int64_t v=0; v<n; ++v)
        if (!closed[v]) S_F[v] += D_s[v];
}

/* ------------------------------------------------------------------ */
/*  Adjacency list construction — UNMODIFIED                          */
/* ------------------------------------------------------------------ */

std::pair<Eigen::VectorXd, std::vector<std::vector<int64_t>>> ComputeDegreeAndAdjList(
        const Eigen::MatrixXd &adj_matrix,
        const int64_t n) {

    Eigen::VectorXd DEG(n);
    for (int64_t i = 0; i < n; ++i) {
        DEG(i) = adj_matrix.col(i).sum();
    }

    std::vector<std::vector<int64_t>> A(n);
    for (int64_t i = 0; i < n; ++i) {
        A[i].reserve((int64_t)DEG(i));
    }

    for (int64_t i = 0; i < n; ++i) {
        auto col = adj_matrix.col(i);
        for (int64_t j = 0; j < n; ++j) {
            if (col(j) > 0) {
                A[i].push_back(j);
            }
        }
    }

    return {DEG, A};
}

/* ------------------------------------------------------------------ */
/*  Exact Sum Sweep main                                             */
/* ------------------------------------------------------------------ */
RESULT ExactSumSweep(
                const MatrixType& adj_matrix,
                const int64_t k,
                const bool verbose)
{
    Timer timer_global; timer_global.Start();

    const int64_t n = matrix_ext::rows(adj_matrix);
    auto [DEG, A] = ComputeDegreeAndAdjList(adj_matrix, n);

    /* ---------- initial Sum-Sweep (k sources) -------------------- */
    int64_t s0; DEG.maxCoeff(&s0);

    int64_t DL = 0, RU = n;
    Eigen::VectorXd L = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd U = Eigen::VectorXd::Constant(n, n);
    std::vector<char> closed(n, false);
    std::vector<double> S_F;

    sumSweep(A, n, s0, k, DL, RU, L, U, closed, S_F, verbose);

    if (verbose) {
        std::cout << "Initial sweep finished. DL = " << DL
                  << ", RU = " << RU << '\n';
    }
    /* ---------- refinement loop ---------------------------------- */

    enum Tech { STEP_D = 0, STEP_R = 1, SINGLE = 2 };
    std::array<double,3> UE = {0.,0.,0.};
    size_t iter = 0;

    auto choose_tech = [&]() -> Tech {
        if (UE[STEP_D] >= UE[STEP_R] && UE[STEP_D] >= UE[SINGLE])
            return STEP_D;
        else if (UE[STEP_R] >= UE[SINGLE])
            return STEP_R;
        else
            return SINGLE;
    };

    while (DL < U.maxCoeff() || RU > L.minCoeff())
    {
        ++iter;
        size_t open_before = countOpen(L,U,closed,DL,RU);

        /* --- pick a technique that still has at least one candidate ---- */
        Tech t;
        for (;;) {
            t = choose_tech();
            bool ok =
                 (t==STEP_D && haveVU(U,closed,DL)) ||
                 (t!=STEP_D && haveVL(L,closed,RU));   // STEP_R or SINGLE
            if (ok) {
                break;
            } else if (verbose) {
                std::cout << "No candidates for technique " << t << '\n';
            }
            /* no candidate: penalise this technique and try again */
            UE[t] = -std::numeric_limits<double>::infinity();
        }

        int64_t s;
        if (t == STEP_D)
            s = selectMaxU(U, closed, S_F, DL);
        else            /* STEP_R or SINGLE share the same root rule */
            s = selectMinL(L, closed, S_F, RU);

        stepForward(A,n,s,DL,RU,L,U,closed,S_F);

        size_t open_after  = countOpen(L,U,closed,DL,RU);
        double util        = static_cast<double>(open_before - open_after);

        UE[t] = util;
        double bonus = 2.0 / static_cast<double>(iter);
        for (int i=0;i<3;++i) if (i!=t) UE[i] += bonus;
    }

    timer_global.Stop();

    COUNTERS cnt;
    cnt.processed_count  = static_cast<int64_t>(
           std::count(closed.begin(), closed.end(), true));
    cnt.processed_ratio  = static_cast<double>(cnt.processed_count)/n;
    cnt.time_secs_overall = timer_global.Seconds();

    if (verbose) {
        std::cout << "Processed " << cnt.processed_count
                  << " vertices (" << cnt.processed_ratio*100.0 << "%). "
                  << "Diameter = " << DL << ", Radius = " << RU << '\n';
    }

    return RESULT{DL, RU, cnt, timer_global.Seconds()};
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
 * @brief Python-callable wrapper for ExactSumSweep algorithm
 * 
 * @param adj_matrix_data Flat array of adjacency matrix (row-major, n*n elements)
 * @param n Number of nodes
 * @param k Number of initial sweep sources (typically 4)
 * @param diameter Output parameter for network diameter
 * @param radius Output parameter for network radius
 * @param elapsed_time Output parameter for computation time
 * @return 0 on success, -1 on error
 */
EXPORT int ess_py(
    const double* adj_matrix_data,
    int64_t n,
    int64_t k,
    int64_t* diameter,
    int64_t* radius,
    double* elapsed_time)
{
    try {
        // Convert input array to Eigen matrix
        Eigen::Map<const Eigen::MatrixXd> adj_matrix_map(adj_matrix_data, n, n);
        Eigen::MatrixXd adj_matrix = adj_matrix_map;
        
        // Call the algorithm
        RESULT result = ExactSumSweep(adj_matrix, k, false);
        
        // Set output parameters
        *diameter = result.diameter;
        *radius = result.radius;
        *elapsed_time = result.computation_time;
        
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

mxArray* createCountersStruct(const COUNTERS& ctr)
{
    const char* names[] = {"processed_count","processed_ratio",
                           "time_secs_overall"};
    mxArray* S = mxCreateStructMatrix(1,1,3,names);

    mxSetFieldByNumber(S,0,0,mxCreateDoubleScalar(ctr.processed_count));
    mxSetFieldByNumber(S,0,1,mxCreateDoubleScalar(ctr.processed_ratio));
    mxSetFieldByNumber(S,0,2,mxCreateDoubleScalar(ctr.time_secs_overall));
    return S;
}

/**
 * MATLAB MEX entry point for ExactSumSweep on an undirected, unweighted graph.
 *
 * Usage:
 *   [diameter, radius, t, counters] = ess_cpp(A, k)
 *
 * Inputs:
 *   A  Nonempty square adjacency matrix of a connected, undirected, unweighted
 *      graph. Dense or sparse logical and double matrices are supported;
 *      nonzero values are interpreted as edges.
 *   k  Real numeric scalar giving the number of initial SumSweep sources.
 *      The value is converted to int64 before use (normally k = 4).
 *
 * Outputs:
 *   diameter  Double scalar containing the exact graph diameter.
 *   radius    Double scalar containing the exact graph radius.
 *   t         Double scalar containing overall computation time in seconds.
 *   counters  Scalar struct with fields `processed_count` (number of vertices
 *             closed by the algorithm), `processed_ratio` (that count divided
 *             by the number of vertices), and `time_secs_overall` (seconds).
 *
 * Interface notes:
 *   Exactly two inputs are required. Outputs are optional from left to right.
 *   The graph preconditions and scalar/integer range of k are not validated by
 *   this MEX wrapper; callers must supply inputs satisfying them.
 */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    if (nrhs != 2)
        mexErrMsgTxt("Usage: [diameter,radius,time,counters] = ess_cpp(A,k)");

    int64_t kSweeps =
        static_cast<int64_t>(mxGetScalar(prhs[1]));
    MatrixType adj = parseAdjacencyMatrix(prhs[0]);

    RESULT res = ExactSumSweep(adj, kSweeps);

    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(res.diameter);
    if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(res.radius);
    if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(res.computation_time);
    if (nlhs > 3) plhs[3] = createCountersStruct(res.counters);
}
#endif // MATLAB_BUILD
