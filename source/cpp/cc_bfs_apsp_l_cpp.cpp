//
// Created by Junda on 2024/4/4.
//


#include "cc_bfs_apsp_l.h"
#include <Eigen/Sparse>
#include <vector>
#include <limits>
#include <algorithm>
#include <queue>
#include <chrono>
#include <numeric>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <utility>
#include "include/timer.h"
#include "include/progress_and_time_utils.h"
#include "include/process_memory_utils.h"
#ifdef MATLAB_BUILD
#include "mex.h"
#include "include/mex_utils.h"
#endif

struct GraphNode {
    std::vector<int64_t> children;
};

struct Level {
    std::vector<int64_t> nodes;
};

static int64_t EstimateGraphNodeGridBytes(const std::vector<std::vector<GraphNode>>& grid) {
    int64_t bytes = static_cast<int64_t>(grid.capacity() * sizeof(std::vector<GraphNode>));
    for (const auto& row : grid) {
        bytes += static_cast<int64_t>(row.capacity() * sizeof(GraphNode));
        for (const auto& gn : row) {
            bytes += static_cast<int64_t>(gn.children.capacity() * sizeof(int64_t));
        }
    }
    return bytes;
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

static void PrintTheoreticalMemorySummary(const COUNTERS& counters, bool verbose) {
    if (!verbose) {
        return;
    }
    std::cout << "@@ Theoretical memory usage (GiB): estimated_total_memory_bytes="
              << BytesToGiB(counters.estimated_total_memory_bytes)
              << " estimated_dominant_memory_bytes=" << BytesToGiB(counters.estimated_dominant_memory_bytes)
              << " adjacency_list_bytes=" << BytesToGiB(counters.adjacency_list_bytes)
              << " distance_matrix_bytes=" << BytesToGiB(counters.distance_matrix_bytes)
              << " sp_dag_bytes=" << BytesToGiB(counters.sp_dag_bytes)
              << " sp_shortcut_bytes=" << BytesToGiB(counters.sp_shortcut_bytes)
              << " sp_tree_bytes=" << BytesToGiB(counters.sp_tree_bytes)
              << std::endl;
}

inline void ReleaseAdjacencyMatrix(MatrixType& adj_matrix) {
    std::visit([](auto& m) {
        using Mat = std::decay_t<decltype(m)>;
        Mat empty;
        m.swap(empty);
    }, adj_matrix);
}


static Eigen::VectorXd BFSSSS(const std::vector<std::vector<int64_t>>& A,
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

static int64_t selectMaxSF(const std::vector<double>& S_F,
                           const std::vector<char>& closed) {
    double best_sf = -std::numeric_limits<double>::infinity();
    int64_t best = -1;

    const int64_t n = static_cast<int64_t>(S_F.size());
    for (int64_t v = 0; v < n; ++v) {
        if (closed[v]) {
            continue;
        }
        if (S_F[v] > best_sf) {
            best_sf = S_F[v];
            best = v;
        }
    }
    if (best == -1) std::cerr << "selectMaxSF(): no candidate" << std::endl;
    return best;
}

static int64_t selectMinSF(const std::vector<double>& S_F,
                           const std::vector<char>& closed) {
    double best_sf = std::numeric_limits<double>::infinity();
    int64_t best = -1;

    const int64_t n = static_cast<int64_t>(S_F.size());
    for (int64_t v = 0; v < n; ++v) {
        if (closed[v]) {
            continue;
        }
        if (S_F[v] < best_sf) {
            best_sf = S_F[v];
            best = v;
        }
    }
    if (best == -1) std::cerr << "selectMinSF(): no candidate" << std::endl;
    return best;
}

static int64_t FourSumSweep(const std::vector<std::vector<int64_t>>& A,
                             const int64_t n,
                             const int64_t s0) {
    Eigen::VectorXd U = Eigen::VectorXd::Constant(n, static_cast<double>(n));
    std::vector<double> S_F(n, 0.0);
    std::vector<char> closed(n, false);

    auto do_sweep = [&](int64_t s) {
        Eigen::VectorXd D = BFSSSS(A, n, s);
        double ecc = D.maxCoeff();
        closed[s] = true;
        U = U.cwiseMin(Eigen::VectorXd::Constant(n, ecc) + D);
        for (int64_t v = 0; v < n; ++v) S_F[v] += D(v);
    };

    // Sweep 1: highest-degree node
    do_sweep(s0);
    // Sweeps 2-3: argmax S_F
    for (int i = 2; i <= 3; ++i) {
        int64_t s = selectMaxSF(S_F, closed);
        if (s == -1) {
            break;
        }
        do_sweep(s);
    }
    // Sweep 4: argmin S_F
    {
        int64_t s = static_cast<int64_t>(std::min_element(S_F.begin(), S_F.end()) - S_F.begin());
        do_sweep(s);
    }

    return static_cast<int64_t>(U.maxCoeff());
}

int64_t DiamUpperBound(const std::vector<std::vector<int64_t>>& A,
                       const int64_t n,
                       const Eigen::VectorXd& DEG) {
    int64_t s0;
    DEG.maxCoeff(&s0);
    return FourSumSweep(A, n, s0);
}

template <typename DistScalar>
static void FillDominantMemoryCounters(
    COUNTERS& counters,
    const std::vector<std::vector<int64_t>>& A,
    const DistVecCache<DistScalar>& D_mem,
    const std::vector<std::vector<GraphNode>>& sp_dags,
    const std::vector<std::vector<GraphNode>>& sp_scs,
    const std::vector<std::vector<GraphNode>>& sp_trees) {
    const int64_t dist_b = DistVecCacheAllocatedBytes(D_mem);
    const int64_t dag_b = EstimateGraphNodeGridBytes(sp_dags);
    const int64_t sc_b = EstimateGraphNodeGridBytes(sp_scs);
    const int64_t tr_b = EstimateGraphNodeGridBytes(sp_trees);
    counters.distance_matrix_bytes = dist_b;
    counters.sp_dag_bytes = dag_b;
    counters.sp_shortcut_bytes = sc_b;
    counters.sp_tree_bytes = tr_b;
    counters.estimated_dominant_memory_bytes = dist_b + dag_b + sc_b + tr_b;
    UpdateTotalMemoryCounter(counters);
}

template <typename DistScalar>
static void FillDominantMemoryCounters(
    COUNTERS& counters,
    const std::vector<std::vector<int64_t>>& A,
    const DistMat<DistScalar>& D,
    const std::vector<std::vector<GraphNode>>& sp_dags,
    const std::vector<std::vector<GraphNode>>& sp_scs,
    const std::vector<std::vector<GraphNode>>& sp_trees) {
    const int64_t dist_b = DistMatAllocatedBytes(D);
    const int64_t dag_b = EstimateGraphNodeGridBytes(sp_dags);
    const int64_t sc_b = EstimateGraphNodeGridBytes(sp_scs);
    const int64_t tr_b = EstimateGraphNodeGridBytes(sp_trees);
    counters.distance_matrix_bytes = dist_b;
    counters.sp_dag_bytes = dag_b;
    counters.sp_shortcut_bytes = sc_b;
    counters.sp_tree_bytes = tr_b;
    counters.estimated_dominant_memory_bytes = dist_b + dag_b + sc_b + tr_b;
    UpdateTotalMemoryCounter(counters);
}

std::vector<int64_t> SelectHubNodes(const std::vector<std::vector<int64_t>> &A,
                                    const int64_t n,
                                    const Eigen::VectorXd &DEG,
                                    const int64_t mu) {

    std::vector<int64_t> C;
    std::vector<char> active(n, true);
    Eigen::VectorXd scores = DEG;

    // prioritize higher degree nodes
    int64_t remaining = n;
    while ((remaining > 0) && (C.size() < mu)) {
        int64_t cn;
        scores.maxCoeff(&cn);
        C.push_back(cn);
        scores(cn) = -std::numeric_limits<double>::infinity();
        // remove its degree contribution
        if (active[cn]) {
            active[cn] = false;
            remaining--;
            for (int64_t neighbor: A[cn]) {
                scores(neighbor)--;
            }
        }
        for (int64_t t: A[cn]) {
            // mask node t
            if (active[t]) {
                // remove its degree contribution
                for (int64_t neighbor: A[t]) {
                    scores(neighbor)--;
                }
                active[t] = false;
                remaining--;
            }
        }
    }
    return C;
}

int64_t FindHighestDegreeNeighborHub(const std::vector<std::vector<int64_t>> &A,
                                     const std::vector<char> &hn_mask,
                                     const Eigen::VectorXd &DEG,
                                     const int64_t s) {
    /**
     * \brief Finds the highest-degree neighbor hub for a given node.
     *
     * This function selects the corresponding hub node for a given node `s`
     * using a degree-based greedy strategy. It iterates through the neighbors of `s`
     * and selects the neighbor with the highest degree as the hub node. If multiple
     * neighbors have the same degree, the first one encountered is selected.
     *
     * \param A The adjacency list of the graph, where A[i] contains the neighbors of node `i`.
     * \param hn_mask A mask indicating which nodes are hub nodes.
     *                `hn_mask[i]` is true if node `i` is a hub node, false otherwise.
     * \param DEG The degree vector of the nodes, where `DEG[i]` represents the degree of node `i`.
     * \param s The node for which the corresponding hub node is to be selected.
     * \return The index of the corresponding hub node with the highest degree,
     *         or -1 if no suitable hub node is found.
     */
    int64_t c_s = -1;
    for (int64_t t: A[s]) {
        if (hn_mask[t]) {
            if ((c_s == -1) || (DEG[t] > DEG[c_s])) {
                c_s = t;
            }
        }
    }
    return c_s;
}

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


template <typename DistScalar>
double EstimateBackwardCost(
    const int64_t& curr_depth,
    const std::vector<Level>& c_s_sp_level,
    const std::vector<int64_t>& eq_dist_fr,
    const std::vector<int64_t>& inc_dist_fr,
    const DistVec<DistScalar>& D_s,
    const Eigen::VectorXd& DEG
) {
    size_t pop_size = 0;
    if (curr_depth > 0) pop_size += c_s_sp_level[curr_depth - 1].nodes.size();
    pop_size += c_s_sp_level[curr_depth].nodes.size();
    if (curr_depth + 1 < c_s_sp_level.size()) pop_size += c_s_sp_level[curr_depth + 1].nodes.size();
    double target_node_num = eq_dist_fr.size() + inc_dist_fr.size();
    double K = pop_size - target_node_num;
    double est_bu_cost = K / (target_node_num + 1) + 1;

    double backward_cost = 0;
    for (int64_t curr: c_s_sp_level[curr_depth].nodes){
        if (static_cast<int64_t>(D_s[curr]) > curr_depth) {
            backward_cost += std::min<double>(est_bu_cost, DEG[curr]);
        }
    }

    return backward_cost;
}

double ComputeForwardCost(
    const std::vector<int64_t>& eq_dist_fr,
    const std::vector<int64_t>& inc_dist_fr,
    const std::vector<GraphNode>& c_s_sp_dag,
    const std::vector<GraphNode>& c_s_sp_sc
) {
    double forward_cost = 0;

    // top_down_cost
    for (int64_t curr: eq_dist_fr) {
        forward_cost += c_s_sp_dag[curr].children.size();
    }
    for (int64_t curr: inc_dist_fr) {
        forward_cost += c_s_sp_sc[curr].children.size();
    }

    return forward_cost;
}

template <typename DistScalar>
void BuildSPStructures(const std::vector<std::vector<int64_t>> &A,
                        const int64_t n,
                        const int64_t s,
                        DistVecCache<DistScalar> &D_mem,
                        const DistScalar dist_inf,
                        std::vector<std::vector<GraphNode>> &sp_dags,
                        std::vector<std::vector<GraphNode>> &sp_shortcuts,
                        std::vector<std::vector<GraphNode>> &sp_trees,
                        std::vector<std::vector<Level>> &sp_levels,
                        std::vector<double> &eccs
) {

    std::vector<GraphNode> sp_dag(n);
    std::vector<GraphNode> sp_sc(n);  // records intra-level links of the SP_DAG
    std::vector<GraphNode> sp_tree(n);
    std::vector<Level> sp_level(n);


    DistVec<DistScalar> D_s = DistVec<DistScalar>::Constant(n, dist_inf);
    D_s(s) = static_cast<DistScalar>(0);
    sp_level[0].nodes.push_back(s);

    std::vector<int64_t> frontier;
    frontier.push_back(s);
    std::vector<int64_t> next;


    int64_t curr_depth = 0;
    while (!frontier.empty()) {
        // explore at a higher depth
        curr_depth++;
        Level curr_level;
        for (int64_t curr: frontier) {
            for (int64_t t: A[curr]) {
                if (curr_depth < D_s[t]) {
                    next.push_back(t);
                    D_s(t) = static_cast<DistScalar>(curr_depth);
                    sp_dag[curr].children.push_back(t);
                    sp_tree[curr].children.push_back(t);
                    curr_level.nodes.push_back(t);
                } else if (curr_depth == D_s[t]) {
                    sp_dag[curr].children.push_back(t);
                } else if ((curr_depth > 2) && (curr_depth - 1 == D_s[t])) {
                    sp_sc[curr].children.push_back(t);
                }
            }
        }
        frontier.swap(next);
        next.clear();
        if (!curr_level.nodes.empty()) {
            sp_level[curr_depth] = std::move(curr_level);
        }
    }

    eccs[s] = static_cast<double>(D_s.maxCoeff());
    D_mem.Store(s, D_s);
    sp_dags[s] = std::move(sp_dag);
    sp_shortcuts[s] = std::move(sp_sc);
    sp_trees[s] = std::move(sp_tree);
    sp_levels[s] = std::move(sp_level);
}


template <typename DistScalar>
void NodeProp(const DistScalar curr_depth,
              const std::vector<int64_t>& neighborhood,
              DistVec<DistScalar>& D_s,
              const DistVecCache<DistScalar>& D_mem,
              double& vcc,
              const double eta,
              std::vector<int64_t>& next,
              const bool use_vc) {
    if (neighborhood.size() == 0) return;

    // when vector comparison is disabled
    if (!use_vc || vcc >= eta) {
        for (const int64_t t : neighborhood) {
            if (curr_depth < D_s[t]) {
                D_s(t) = curr_depth;
                next.push_back(t);
            }
        }
        return;
    }

    // Vector comparison enabled
    for (const int64_t t : neighborhood) {
        if (curr_depth < D_s[t]) {
            D_s(t) = curr_depth;
            if (D_mem.Has(t) && vcc < eta) {
                D_s = D_s.cwiseMin(
                    (D_mem.Get(t).array() + curr_depth).matrix()
                );
                vcc++;
            } else {
                next.push_back(t);
            }
        }
    }
}

template <typename DistScalar>
void MemoizedSSS(const std::vector<std::vector<int64_t>> &A,
                 const int64_t n,
                 const int64_t s,
                 const double eta,
                 DistVecCache<DistScalar> &D_mem,
                 const DistScalar dist_inf,
                 COUNTERS& counters) {

    counters.source_without_cs_count++;
    // Memoization-enabled Single-source search
    DistVec<DistScalar> D_s = DistVec<DistScalar>::Constant(n, dist_inf);
    D_s(s) = static_cast<DistScalar>(0);
    double vcc = 0;  // VC count

    std::vector<int64_t> frontier;
    frontier.push_back(s);
    int64_t curr_depth = 0;

    while (!frontier.empty()) {
        // explore at a higher depth
        curr_depth++;
        std::vector<int64_t> next = {};

        // visit nodes in frontier
        if (!frontier.empty()) {
            for (int64_t curr: frontier) {
                NodeProp(static_cast<DistScalar>(curr_depth), A[curr], D_s, D_mem, vcc, eta, next, true);
                counters.graph_expansion_node_count++;
            }
        }
        // swap frontier lists
        frontier.swap(next);
    }
    D_mem.Store(s, D_s);
    counters.vec_comp_count += static_cast<int64_t>(vcc);
}


template <typename DistScalar>
void HubGuidedSSS(const std::vector<std::vector<int64_t>> &A,
                  const int64_t n,
                  const int64_t s,
                  const double eta,
                  const int64_t c_s,
                  DistVecCache<DistScalar> &D_mem,
                  const DistScalar dist_inf,
                  const std::vector<std::vector<GraphNode>> &sp_dags,
                  const std::vector<std::vector<GraphNode>> &sp_scs,
                  const std::vector<std::vector<GraphNode>> &sp_trees,
                  const std::vector<std::vector<Level>> &sp_levels,
                  const std::vector<double> &eccs,
                  const Eigen::VectorXd &DEG,
                  COUNTERS& counters) {
    counters.source_with_cs_count++;
    const std::vector<GraphNode>& c_s_sp_dag = sp_dags[c_s];
    const std::vector<GraphNode>& c_s_sp_sc = sp_scs[c_s];
    const std::vector<GraphNode>& c_s_sp_tree = sp_trees[c_s];
    const std::vector<Level>& c_s_sp_level = sp_levels[c_s];

    DistVec<DistScalar> D_s = DistVec<DistScalar>::Constant(n, dist_inf);
    D_s(s) = static_cast<DistScalar>(0);

    double ecc_c_s = eccs[c_s];
    double vcc = 0;

    std::vector<int64_t> inc_dist_fr;
    std::vector<int64_t> eq_dist_fr;

    int64_t curr_depth = 0;
    // build the first level
    curr_depth++;
    std::vector<char> c_s_neighbor_mask(A.size(), false);
    for (int64_t t: A[c_s]) {
        c_s_neighbor_mask[t] = true;
    }
    for (int64_t t: A[s]) {
        if (D_s[t] > curr_depth) {
            D_s(t) = static_cast<DistScalar>(curr_depth);
            if (c_s_neighbor_mask[t]) { // t is a common neighbor of s and c_s
                eq_dist_fr.push_back(t);
            } else if (t == c_s) {  // make sure c_s is absorbed
                D_s = D_s.cwiseMin(
                    (D_mem.Get(t).array() + static_cast<DistScalar>(curr_depth)).matrix()
                );
                vcc++;
            } else if ((D_mem.Has(t)) && (vcc < eta) && (curr_depth < ecc_c_s - 1)) {  // t is a valid prior source node
                // this is equivalent to incorporating the SP tree of t all at once
                D_s = D_s.cwiseMin(
                    (D_mem.Get(t).array() + static_cast<DistScalar>(curr_depth)).matrix()
                );
                vcc++;
            } else {  // t is a source exclusive node
                inc_dist_fr.push_back(t);
            }
        }
    }


    while (!inc_dist_fr.empty() || !eq_dist_fr.empty()) {
        // move to the next level
        curr_depth++;

        // disable VC upon reaching the last level of c_s
        bool use_vc = false;
        if (curr_depth < ecc_c_s - 1) {
            use_vc = true;
        }

        const Level& curr_level = c_s_sp_level[curr_depth];
        std::vector<int64_t> next_eq_dist_fr = {};
        std::vector<int64_t> next_inc_dist_fr = {};

        // first of all, expand spd_fr node from SP-tree space to narrow down the search space
        if (!eq_dist_fr.empty()) {
            for (int64_t curr: eq_dist_fr) {
                // it often doesn't matter if we use VC here
                NodeProp(static_cast<DistScalar>(curr_depth), c_s_sp_tree[curr].children, D_s, D_mem,
                         vcc, eta, next_eq_dist_fr, false);
                counters.sptree_expansion_node_count++;
            }
        }
        // expand sc_fr nodes in SP-DAG space
        if (!inc_dist_fr.empty() && (curr_depth < ecc_c_s)) {
            for (int64_t curr: inc_dist_fr) {
                NodeProp(static_cast<DistScalar>(curr_depth), c_s_sp_dag[curr].children, D_s, D_mem, vcc,
                         eta, next_inc_dist_fr, use_vc);
                counters.spdag_expansion_node_count++;
            }
        }

        // compute backward propagation cost
        double backward_cost = EstimateBackwardCost<DistScalar>(curr_depth, c_s_sp_level, eq_dist_fr, inc_dist_fr, D_s, DEG);
        double forward_cost = ComputeForwardCost(eq_dist_fr, inc_dist_fr, c_s_sp_dag, c_s_sp_sc);

        if (forward_cost < backward_cost) {
            // top-down is preferred
            // visit nodes in spd_fr
            if (!eq_dist_fr.empty()) {
                for (int64_t curr: eq_dist_fr) {
                    // must not use VC here
                    NodeProp(static_cast<DistScalar>(curr_depth), c_s_sp_dag[curr].children, D_s, D_mem,
                             vcc, eta, next_eq_dist_fr, false);
                    counters.top_down_node_count++;
                    counters.spdag_expansion_node_count++;
                }
            }

            // visit nodes in inc_dist_fr
            if (!inc_dist_fr.empty()) {
                for (int64_t curr: inc_dist_fr) {
                    // intra-links are special because the ending node can expand from SP-DAG space
                    NodeProp(static_cast<DistScalar>(curr_depth), c_s_sp_sc[curr].children, D_s, D_mem,
                             vcc, eta, next_eq_dist_fr, use_vc);
                    counters.top_down_node_count++;
                    counters.spill_expansion_node_count++;
                }
            }
        } else {
            // bottom-up is preferred
            for (int64_t curr: curr_level.nodes) {
                if (D_s[curr] > curr_depth) {
                    for (int64_t t: A[curr]) {
                        if (D_s[t] == curr_depth - 1) {
                            D_s[curr] = static_cast<DistScalar>(curr_depth);
                            next_eq_dist_fr.push_back(curr);
                            break;
                        }
                    }
                    counters.bottom_up_node_count++;
                }
            }
        }


        // swap frontier lists
        inc_dist_fr.swap(next_inc_dist_fr);
        eq_dist_fr.swap(next_eq_dist_fr);
    }

    D_mem.Store(s, D_s);
    counters.vec_comp_count += static_cast<int64_t>(vcc);
}

int64_t CompHubConstraint(
        int64_t n,
        int64_t m,
        double density,
        const std::string &opt) {
    double avg_deg = static_cast<double>(m) / n;
    double hn_con = 0;

    // Check if opt is a number using istringstream (more efficient than try/catch)
    std::istringstream iss(opt);
    double value;
    if (iss >> value && iss.eof()) {
        // If opt is a valid number, use it directly
        hn_con = value;
    } else {
        // Not a number, check other options
        if (opt == "bounded-avg-deg") {
            hn_con = (avg_deg > 0.0)
                    ? std::min(avg_deg, static_cast<double>(n) / avg_deg)
                    : 0.0;
        } else if (opt == "avg-deg") {
            hn_con = avg_deg;
        } else if (opt == "sqrt-deg") {
            hn_con = std::pow(avg_deg, 0.5);
        } else if (opt == "sqrt-inv-den") {
            hn_con = std::sqrt(std::ceil(1.0 / density));
        } else if (opt == "inv-den") {
            hn_con = std::ceil(1.0 / density);
        } else if (opt == "none") {
            hn_con = n; // means no limitation
        } else {
            std::ostringstream oss;
            oss << "Unrecognized hub node constraint option: " << opt << ".";
            throw std::runtime_error(oss.str());
        }
    }

    // Convert the floating-point constraint to its nearest integer.
    return static_cast<int64_t>(std::round(hn_con));
}

static RESULT CC_BFS_APSP_L_mem(MatrixType &adj_matrix,
                                const std::string hub_constraint,
                                bool nd,
                                bool mem_test_only,
                                bool verbose) {
    const int64_t n = matrix_ext::rows(adj_matrix);
    UpdateProcessMemoryPeakCheckpoint();

    if (verbose) {
        std::cout << "computation starts: " << NowLocalDateTimeString() << std::endl;
    }

    Timer timer_global;
    timer_global.Start();
    Timer timer;
    COUNTERS counters;

    if (n <= 1) {
        ReleaseAdjacencyMatrix(adj_matrix);
        UpdateProcessMemoryPeakCheckpoint();

        ResultMatrix D;
        if (!nd) {
            DistMat<uint8_t> D_uint8 = DistMat<uint8_t>::Zero(n, n);
            D = ResultMatrix{std::move(D_uint8)};
        }

        counters.diam_upper = 0;
        counters.data_type = "uint8";
        timer_global.Stop();
        counters.time_secs_overall = timer_global.Seconds();
        UpdateProcessMemoryPeakCheckpoint();
        return RESULT{std::move(D), std::move(counters), counters.time_secs_overall};
    }

    // compute degree and adjacency list
    timer.Start();
    auto deg_adj = ComputeDegreeAndAdjList(adj_matrix, n);
    Eigen::VectorXd DEG = std::move(deg_adj.first);
    std::vector<std::vector<int64_t>> A = std::move(deg_adj.second);
    counters.adjacency_list_bytes = EstimateAdjacencyListBytes(A);

    ReleaseAdjacencyMatrix(adj_matrix);
    ResetProcessMemoryPeakBaseline();
    timer.Stop();
    counters.time_secs_comp_adj = timer.Seconds();
    UpdateProcessMemoryPeakCheckpoint();
    if (verbose) {
        std::cout << "@@ Time to compute adjacency list: " << counters.time_secs_comp_adj << " seconds" << std::endl;
    }

    // compute diameter upper bound
    timer.Start();
    const int64_t diam_upper = DiamUpperBound(A, n, DEG);
    timer.Stop();
    counters.time_secs_diam_upperbound = timer.Seconds();
    counters.diam_upper = diam_upper;
    UpdateProcessMemoryPeakCheckpoint();
    if (verbose) {
        std::cout << "@@ Time to run 4-SumSweep: " << counters.time_secs_diam_upperbound << " seconds" << std::endl;
        std::cout << "4-SumSweep diameter upper bound: " << diam_upper << std::endl;
    }

    // Reserve the exact range for sums of two bounded distances.
    if (diam_upper < 0) {
        throw std::invalid_argument("Diameter upper bound must be nonnegative");
    }
    const uint64_t unsigned_diam_upper = static_cast<uint64_t>(diam_upper);
    if (unsigned_diam_upper > (std::numeric_limits<uint64_t>::max)() / 2) {
        throw std::overflow_error("Twice the diameter upper bound exceeds uint64 capacity");
    }
    const uint64_t dtype_max_required = 2 * unsigned_diam_upper;

    auto run_with_dtype = [&](auto dist_scalar_tag) -> RESULT {
        using DistScalar = decltype(dist_scalar_tag);

        counters.data_type = DistTypeName<DistScalar>();
        const DistScalar dist_inf = static_cast<DistScalar>(diam_upper);

        int64_t deg1_count = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (static_cast<int64_t>(DEG(i)) == 1) {
                deg1_count++;
            }
        }
        int64_t max_dist_vec = nd ? (n - deg1_count) : n;
        if (max_dist_vec < 0) {
            max_dist_vec = 0;
        }

        if (verbose) {
            std::cout << "D_mem dtype: " << DistTypeName<DistScalar>()
                      << ", diam_upper: " << diam_upper
                      << ", inf distance: " << diam_upper << std::endl;
        }

        const double est_bytes = static_cast<double>(n) * static_cast<double>(max_dist_vec) * static_cast<double>(sizeof(DistScalar));
        const double est_gib = est_bytes / (1024.0 * 1024.0 * 1024.0);
        if (nd) {
            if (verbose) {
                std::cout << "D_mem capacity = n - (#deg==1) = " << max_dist_vec
                          << " (n=" << n << ", deg1=" << deg1_count << ")" << std::endl;
            }
        } else {
            if (verbose) {
                std::cout << "D_mem capacity = n = " << max_dist_vec << std::endl;
            }
        }
        if (verbose) {
            std::cout << "Estimated D_mem buffer size: " << est_bytes << " bytes (" << est_gib << " GiB)" << std::endl;
        }

        std::vector<char> hn_mask(n, false);
        std::vector<std::vector<GraphNode>> sp_dags(n);
        std::vector<std::vector<GraphNode>> sp_scs(n);
        std::vector<std::vector<GraphNode>> sp_trees(n);
        std::vector<std::vector<Level>> sp_levels(n);
        std::vector<int64_t> source_list(n);
        std::iota(source_list.begin(), source_list.end(), 0);
        std::vector<double> eccs(n, n);

        timer.Start();

        // prioritize sources
        std::sort(source_list.begin(), source_list.end(), [&](int64_t a, int64_t b) {
            return DEG[a] > DEG[b];
        });
        int64_t m = DEG.sum();
        double density = (double)m / (n * (n - 1));
        double avg_deg = (double)m / n;
        // compute memoization constraints
        double eta = std::pow(avg_deg, 0.5);
        int64_t mu = CompHubConstraint(n, m, density, hub_constraint);

        // select hub nodes
        std::vector<int64_t> hub_nodes = SelectHubNodes(A, n, DEG, mu);
        counters.hn_count = hub_nodes.size();

        if (counters.hn_count > max_dist_vec) {
            const int64_t prev = max_dist_vec;
            max_dist_vec = counters.hn_count;
            if (verbose) {
                std::cout << "Warning: hub nodes (" << counters.hn_count
                          << ") exceeds computed capacity (" << prev << "). "
                          << "Increasing capacity to " << max_dist_vec << std::endl;
            }
        }

        DistVecCache<DistScalar> D_mem(n, max_dist_vec, !nd);
        if (verbose) {
            std::cout << "Created D_mem: capacity=" << max_dist_vec << " vectors" << std::endl;
        }

        timer.Stop();
        if (verbose) {
            std::cout << "@@ prioritize sources and compute hub nodes costs: " << timer.Seconds() << " seconds" << std::endl;
            std::cout << "hub nodes num: " << counters.hn_count << std::endl;
            std::cout << "avg_deg: " << avg_deg << std::endl;
            std::cout << "density: " << density * 100 << "%" << std::endl;
            std::cout << "Using hub constraint: " << hub_constraint << "; mu: " << mu << std::endl;
            std::cout << "eta: " << eta << std::endl;
        }

        timer.Start();

        // Process hub nodes
        for (int64_t i = 0; i < hub_nodes.size(); ++i) {
            int64_t s = hub_nodes[i];
            BuildSPStructures<DistScalar>(A, n, s, D_mem, dist_inf, sp_dags, sp_scs, sp_trees, sp_levels, eccs);
            hn_mask[s] = true;
            if (((i + 1) & 63) == 0 || (i + 1) == static_cast<int64_t>(hub_nodes.size())) {
                UpdateProcessMemoryPeakCheckpoint();
            }
        }

        timer.Stop();
        counters.time_secs_hn_sss = timer.Seconds();
        UpdateProcessMemoryPeakCheckpoint();
        if (verbose) {
            std::cout << "@@ Hub nodes SSS computation costs: " << timer.Seconds() << " seconds" << std::endl;
        }

        if (mem_test_only) {
            timer_global.Stop();
            counters.time_secs_overall = timer_global.Seconds();
            UpdateProcessMemoryPeakCheckpoint();
            FillDominantMemoryCounters(counters, A, D_mem, sp_dags, sp_scs, sp_trees);
            PrintTheoreticalMemorySummary(counters, verbose);
            ResultMatrix D;
            return RESULT{std::move(D), std::move(counters), std::numeric_limits<double>::quiet_NaN()};
        }

        timer.Start();

        int64_t total_non_hub = n - hub_nodes.size();
        int64_t processed_non_hub = 0;
        const int progress_update_interval_pct = 1;
        int next_report_pct = progress_update_interval_pct;
        if (verbose && total_non_hub > 0) {
            std::cout << "Progress:  0%" << std::flush;
        }

        for (int64_t idx = 0; idx < n; ++idx) {
            int64_t s = source_list[idx];
            if (hn_mask[s]) {
                continue;
            }
            int64_t c_s = FindHighestDegreeNeighborHub(A, hn_mask, DEG, s);
            if (c_s != -1) {
                // Process spoke nodes
                HubGuidedSSS<DistScalar>(A, n, s, eta, c_s, D_mem, dist_inf, sp_dags, sp_scs, sp_trees, sp_levels, eccs, DEG, counters);
            } else {
                // Process peripheral nodes
                MemoizedSSS<DistScalar>(A, n, s, eta, D_mem, dist_inf, counters);
            }
            processed_non_hub++;
            if ((processed_non_hub & 63) == 0 || processed_non_hub == total_non_hub) {
                UpdateProcessMemoryPeakCheckpoint();
            }
            if (verbose) {
                UpdateProgressPercent(processed_non_hub,
                                      total_non_hub,
                                      progress_update_interval_pct,
                                      next_report_pct);
            }
        }

        timer.Stop();
        counters.time_secs_non_hn_sss = timer.Seconds();
        UpdateProcessMemoryPeakCheckpoint();
        if (verbose) {
            std::cout << "@@ Non-hub nodes SSS computation costs: " << timer.Seconds() << " seconds" << std::endl;
        }


        ResultMatrix D;
        if (!nd) {
            timer.Start();
            DistMat<DistScalar> D_typed;
            ConvertDistVecCacheToMatrix(D_mem, D_typed);
            timer.Stop();
            counters.time_secs_convert = timer.Seconds();
            if (verbose) {
                std::cout << "@@ Convert D_mem to D costs: " << counters.time_secs_convert << " seconds" << std::endl;
            }
            timer_global.Stop();
            counters.time_secs_overall = timer_global.Seconds();
            UpdateProcessMemoryPeakCheckpoint();
            FillDominantMemoryCounters(counters, A, D_typed, sp_dags, sp_scs, sp_trees);
            D = ResultMatrix{std::move(D_typed)};
        } else {
            timer_global.Stop();
            counters.time_secs_overall = timer_global.Seconds();
            UpdateProcessMemoryPeakCheckpoint();
            FillDominantMemoryCounters(counters, A, D_mem, sp_dags, sp_scs, sp_trees);
        }

        if (verbose) {
            std::cout << "Vector-comparison count: " << counters.vec_comp_count << "Average per node: " << (double)counters.vec_comp_count / n << std::endl;
            std::cout << "source with corresponding hub node portion: " << (double)counters.source_with_cs_count /
                                                                              (counters.source_with_cs_count + counters.source_without_cs_count) * 100 << "%" << std::endl;
            std::cout << "graph expansion node count: " << counters.graph_expansion_node_count << " Proportion: " << (double)counters.graph_expansion_node_count / (n * (n - 1)) * 100 << "%" << std::endl;
            std::cout << "SP-DAG expansion node count: " << counters.spdag_expansion_node_count << " Proportion: " << (double)counters.spdag_expansion_node_count / (n * (n - 1)) * 100 << "%" << std::endl;
            std::cout << "SP-tree expansion node count: " << counters.sptree_expansion_node_count << " Proportion: " << (double)counters.sptree_expansion_node_count / (n * (n - 1)) * 100 << "%" << std::endl;
            std::cout << "Intra-level expansion node count: " << counters.spill_expansion_node_count << " Proportion: " << (double)counters.spill_expansion_node_count / (n * (n - 1)) * 100 << "%" << std::endl;

            std::cout << "top-down node count: " << counters.top_down_node_count << " Proportion: " << (double)counters.top_down_node_count / (n * counters.source_with_cs_count) * 100 << "%" << std::endl;
            std::cout << "bottom-up node count: " << counters.bottom_up_node_count << " Proportion: " << (double)counters.bottom_up_node_count / (n * counters.source_with_cs_count) * 100 << "%" << std::endl;

            std::cout << "Number of Distance vectors cached: " << D_mem.num_stored << " / " << n << std::endl;
            std::cout << "end datetime: " << NowLocalDateTimeString() << std::endl;
            std::cout << "@@ Total computation time: " << counters.time_secs_overall << " seconds" << std::endl;
            PrintTheoreticalMemorySummary(counters, verbose);
            PrintProcessMemorySummary();
        }
        const double total_time = counters.time_secs_overall;
        return RESULT{std::move(D), std::move(counters), total_time};
    };

    if (dtype_max_required <= (std::numeric_limits<uint8_t>::max)()) {
        return run_with_dtype(uint8_t{});
    }
    if (dtype_max_required <= (std::numeric_limits<uint16_t>::max)()) {
        return run_with_dtype(uint16_t{});
    }
    if (dtype_max_required <= (std::numeric_limits<uint32_t>::max)()) {
        return run_with_dtype(uint32_t{});
    }
    return run_with_dtype(uint64_t{});
}

RESULT CC_BFS_APSP_L(MatrixType &adj_matrix,
                     const std::string hub_constraint,
                     bool nd,
                     bool verbose) {
    return CC_BFS_APSP_L_mem(adj_matrix, hub_constraint, nd, false, verbose);
}

// ============================================================================
// Python-callable interface (for ctypes)
// ============================================================================
#ifdef PYTHON_BUILD
#include "distance_output.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

/**
 * C-compatible POD mirror of CC-BFS COUNTERS, for ctypes access from Python.
 * The std::string field (data_type) is omitted as it is not C-compatible.
 * Field order and types must stay in sync with the Python CcBfsCounters Structure.
 */
struct CC_BFS_PY_COUNTERS {
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
    double  time_secs_comp_adj;
    double  time_secs_diam_upperbound;
    double  time_secs_hn_sss;
    double  time_secs_non_hn_sss;
    double  time_secs_convert;
    double  time_secs_overall;
    int64_t diam_upper;
    // data_type omitted (std::string, not C-compatible)
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
};

/**
 * @brief Python-callable wrapper for CC-BFS APSP algorithm
 *
 * @param adj_matrix_data Flat array of adjacency matrix (n*n elements)
 * @param n Number of nodes
 * @param output_distances Output array for distance matrix (n*n elements, allocated by caller).
 *                         May be NULL to skip copying distances back to Python.
 * @param elapsed_time Output parameter for computation time
 * @param hub_constraint Hub selection constraint string (e.g., "avg-deg"); NULL defaults to "bounded-avg-deg"
 * @param out_counters Output struct for all algorithm counters; may be NULL
 * @return 0 on success, -1 on error
 */
static int RunCcBfsPython(
    const double* adj_matrix_data,
    int64_t n,
    double* output_distances,
    double* elapsed_time,
    const char* hub_constraint,
    CC_BFS_PY_COUNTERS* out_counters,
    fast_apsp::DistanceAllocator allocate_distances)
{
    try {
        // Convert input array to Eigen matrix
        Eigen::Map<const Eigen::MatrixXd> adj_matrix_map(adj_matrix_data, n, n);
        MatrixType adj_matrix = Eigen::MatrixXd(adj_matrix_map);
        
        // Determine hub constraint string
        std::string hub_str = ((hub_constraint == nullptr) || (hub_constraint[0] == '\0'))
            ? "bounded-avg-deg"
            : std::string(hub_constraint);
        
        ProcessMemoryPeakTracker mem_tracker;
        mem_tracker.CaptureBaseline();

        // Call the algorithm
        RESULT result;
        {
            ProcessMemoryTrackerScope tracker_scope(&mem_tracker);
            result = CC_BFS_APSP_L(adj_matrix, hub_str, false, false);
        }
        mem_tracker.UpdatePeak();
        FillProcessMemoryCounters(result.counters, mem_tracker);

        fast_apsp::CopyNativeDistances(result.D, allocate_distances);
        
        if (output_distances != nullptr) {
            // For undirected graphs, D is symmetric so transposition does not affect values.
            Eigen::Map<Eigen::MatrixXd>(output_distances, n, n) = ResultMatrixToDouble(result.D);
        }
        
        *elapsed_time = result.computation_time;
        if (out_counters) {
            out_counters->vec_comp_count                       = result.counters.vec_comp_count;
            out_counters->source_with_cs_count                 = result.counters.source_with_cs_count;
            out_counters->source_without_cs_count              = result.counters.source_without_cs_count;
            out_counters->hn_count                             = result.counters.hn_count;
            out_counters->graph_expansion_node_count           = result.counters.graph_expansion_node_count;
            out_counters->spdag_expansion_node_count           = result.counters.spdag_expansion_node_count;
            out_counters->spill_expansion_node_count           = result.counters.spill_expansion_node_count;
            out_counters->sptree_expansion_node_count          = result.counters.sptree_expansion_node_count;
            out_counters->cst_visited_count                    = result.counters.cst_visited_count;
            out_counters->top_down_node_count                  = result.counters.top_down_node_count;
            out_counters->bottom_up_node_count                 = result.counters.bottom_up_node_count;
            out_counters->time_secs_comp_adj                   = result.counters.time_secs_comp_adj;
            out_counters->time_secs_diam_upperbound            = result.counters.time_secs_diam_upperbound;
            out_counters->time_secs_hn_sss                     = result.counters.time_secs_hn_sss;
            out_counters->time_secs_non_hn_sss                 = result.counters.time_secs_non_hn_sss;
            out_counters->time_secs_convert                    = result.counters.time_secs_convert;
            out_counters->time_secs_overall                    = result.counters.time_secs_overall;
            out_counters->diam_upper                           = result.counters.diam_upper;
            out_counters->estimated_dominant_memory_bytes      = result.counters.estimated_dominant_memory_bytes;
            out_counters->distance_matrix_bytes                = result.counters.distance_matrix_bytes;
            out_counters->sp_dag_bytes                         = result.counters.sp_dag_bytes;
            out_counters->sp_shortcut_bytes                    = result.counters.sp_shortcut_bytes;
            out_counters->sp_tree_bytes                        = result.counters.sp_tree_bytes;
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

// Preserve the previous double-output ABI for existing native callers.
EXPORT int cc_bfs_apsp_py(
    const double* adjacency, int64_t n, double* output, double* elapsed,
    const char* constraint, CC_BFS_PY_COUNTERS* counters) {
    return RunCcBfsPython(adjacency, n, output, elapsed, constraint, counters, nullptr);
}

// Python allocates exactly the selected unsigned type, once the result is ready.
// A null allocator suppresses only the output copy, not the APSP computation.
EXPORT int cc_bfs_apsp_typed_py(
    const double* adjacency, int64_t n, fast_apsp::DistanceAllocator allocate,
    double* elapsed, const char* constraint, CC_BFS_PY_COUNTERS* counters) {
    return RunCcBfsPython(adjacency, n, nullptr, elapsed, constraint, counters, allocate);
}

#ifdef __cplusplus
}
#endif
#endif // PYTHON_BUILD

// ============================================================================
// MATLAB MEX Interface
// ============================================================================

#ifdef MATLAB_BUILD
// Entry point for the MEX-file

mxArray* createCountersStruct(const COUNTERS& counters) {
    // Create field names array
    const char* fieldnames[] = {
            "vec_comp_count",
            "source_with_cs_count",
            "source_without_cs_count",
            "hn_count",
            "graph_expansion_node_count",
            "spdag_expansion_node_count",
            "spill_expansion_node_count",
            "sptree_expansion_node_count",
            "top_down_node_count",
            "bottom_up_node_count",
            "time_secs_comp_adj",
            "time_secs_diam_upperbound",
            "time_secs_hn_sss",
            "time_secs_non_hn_sss",
            "time_secs_convert",
            "time_secs_overall",
            "diam_upper",
            "data_type",
            "estimated_dominant_memory_bytes",
            "distance_matrix_bytes",
            "sp_dag_bytes",
            "sp_shortcut_bytes",
            "sp_tree_bytes",
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

    // Create the struct
    mxArray* matlabStruct = mxCreateStructMatrix(1, 1, static_cast<int>(numFields), fieldnames);

    // Set each field manually
    mxSetFieldByNumber(matlabStruct, 0, 0, mxCreateDoubleScalar(counters.vec_comp_count));
    mxSetFieldByNumber(matlabStruct, 0, 1, mxCreateDoubleScalar(counters.source_with_cs_count));
    mxSetFieldByNumber(matlabStruct, 0, 2, mxCreateDoubleScalar(counters.source_without_cs_count));
    mxSetFieldByNumber(matlabStruct, 0, 3, mxCreateDoubleScalar(counters.hn_count));
    mxSetFieldByNumber(matlabStruct, 0, 4, mxCreateDoubleScalar(counters.graph_expansion_node_count));
    mxSetFieldByNumber(matlabStruct, 0, 5, mxCreateDoubleScalar(counters.spdag_expansion_node_count));
    mxSetFieldByNumber(matlabStruct, 0, 6, mxCreateDoubleScalar(counters.spill_expansion_node_count));
    mxSetFieldByNumber(matlabStruct, 0, 7, mxCreateDoubleScalar(counters.sptree_expansion_node_count));
    mxSetFieldByNumber(matlabStruct, 0, 8, mxCreateDoubleScalar(counters.top_down_node_count));
    mxSetFieldByNumber(matlabStruct, 0, 9, mxCreateDoubleScalar(counters.bottom_up_node_count));
    mxSetFieldByNumber(matlabStruct, 0, 10, mxCreateDoubleScalar(counters.time_secs_comp_adj));
    mxSetFieldByNumber(matlabStruct, 0, 11, mxCreateDoubleScalar(counters.time_secs_diam_upperbound));
    mxSetFieldByNumber(matlabStruct, 0, 12, mxCreateDoubleScalar(counters.time_secs_hn_sss));
    mxSetFieldByNumber(matlabStruct, 0, 13, mxCreateDoubleScalar(counters.time_secs_non_hn_sss));
    mxSetFieldByNumber(matlabStruct, 0, 14, mxCreateDoubleScalar(counters.time_secs_convert));
    mxSetFieldByNumber(matlabStruct, 0, 15, mxCreateDoubleScalar(counters.time_secs_overall));
    mxSetFieldByNumber(matlabStruct, 0, 16, mxCreateDoubleScalar(counters.diam_upper));
    mxSetFieldByNumber(matlabStruct, 0, 17, mxCreateString(counters.data_type.c_str()));
    mxSetFieldByNumber(matlabStruct, 0, 18, mxCreateDoubleScalar(static_cast<double>(counters.estimated_dominant_memory_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 19, mxCreateDoubleScalar(static_cast<double>(counters.distance_matrix_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 20, mxCreateDoubleScalar(static_cast<double>(counters.sp_dag_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 21, mxCreateDoubleScalar(static_cast<double>(counters.sp_shortcut_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 22, mxCreateDoubleScalar(static_cast<double>(counters.sp_tree_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 23, mxCreateDoubleScalar(static_cast<double>(counters.adjacency_list_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 24, mxCreateDoubleScalar(static_cast<double>(counters.estimated_total_memory_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 25, mxCreateDoubleScalar(static_cast<double>(counters.process_baseline_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 26, mxCreateDoubleScalar(static_cast<double>(counters.process_peak_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 27, mxCreateDoubleScalar(static_cast<double>(counters.process_delta_peak_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 28, mxCreateDoubleScalar(static_cast<double>(counters.process_baseline_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 29, mxCreateDoubleScalar(static_cast<double>(counters.process_peak_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 30, mxCreateDoubleScalar(static_cast<double>(counters.process_delta_peak_private_bytes)));

    return matlabStruct;
}

/**
 * MATLAB MEX entry point for CC-BFS APSP on an undirected, unweighted graph.
 *
 * Usage:
 *   [D, t, counters] = cc_bfs_apsp_l_cpp(A)
 *   [D, t, counters] = cc_bfs_apsp_l_cpp(A, option, ...)
 *
 * Inputs:
 *   A                Real, square adjacency matrix supplied as dense or sparse `double`
 *                    or `logical`. Nonzero entries are treated as unweighted edges. A is
 *                    expected to be symmetric because CC-BFS assumes an undirected graph.
 *   option           Optional MATLAB character vector; options may appear in any order:
 *                    - hub constraint: "bounded-avg-deg" (default), "avg-deg",
 *                      "sqrt-deg", "sqrt-inv-den", "inv-den", "none", or a
 *                      character vector containing a numeric value accepted by
 *                      `CompHubConstraint()`. "bounded-avg-deg" computes
 *                      min(avg_deg, n / avg_deg). The resulting floating-point
 *                      constraint is rounded to the nearest integer hub limit.
 *                    - "nd": do not retain or return the complete distance matrix.
 *                    - "ndr": compute distances normally, but return a dummy first output.
 *                    - "nv": suppress CC-BFS progress and diagnostic output.
 *                    - "mem_test_only": stop after constructing the hub shortest-path
 *                      structures and memory estimates; skip non-hub searches.
 *
 * Outputs:
 *   D         All-pairs shortest-path distance matrix using the selected adaptive numeric
 *             class (`uint8`, `uint16`, `uint32`, or `uint64`). With "nd", "ndr", or
 *             "mem_test_only", D is instead the `double` scalar 0 (a 1-by-1 matrix).
 *   t         Algorithm computation time in seconds; NaN with "mem_test_only".
 *   counters  Scalar struct of CC-BFS search/expansion counts, phase timings, diameter and
 *             selected data type, theoretical memory estimates, and sampled process
 *             working-set/private-memory baseline, peak, and delta-peak values.
 *
 * Notes:
 *   Between zero and three outputs may be requested. Only MATLAB character vectors are
 *   parsed as options (string scalars are not); non-char optional inputs are ignored.
 *   Any character vector that is not a reserved flag is interpreted as the hub constraint,
 *   and the last such option wins. "ndr" changes only the returned first output, whereas
 *   "nd" changes the algorithm's storage mode.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check the number of input arguments.
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                          "Usage: [D,t,counters] = cc_bfs_apsp_l_cpp(A, [hub_constraint], ['nd'], ['ndr'], ['nv'], ['mem_test_only']).");
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
    bool verbose = true;
    std::string hub_constraint = "bounded-avg-deg";

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
            } else if (param == "nv") {
                verbose = false;
            } else if (param == "mem_test_only") {
                mem_test_only = true;
            } else {
                hub_constraint = param;
            }
        }
    }

    RESULT result;
    {
        ProcessMemoryTrackerScope tracker_scope(&mem_tracker);
        result = CC_BFS_APSP_L_mem(adj_matrix, hub_constraint, nd, mem_test_only, verbose);
    }
    mem_tracker.UpdatePeak();
    FillProcessMemoryCounters(result.counters, mem_tracker);
    if (verbose) {
        std::cout << "Returning result to MATLAB" << std::endl;
    }

    if (nlhs > 0) {
        if (nd || ndr || mem_test_only) {
            // Return a dummy 1x1 matrix with value 0
            plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
            *mxGetPr(plhs[0]) = 0.0;
        } else {
            // Return the actual distance matrix
            plhs[0] = CreateMatlabMatrixFromResult(result.D);
        }
    }

    // Second output: elapsed time (NaN when mem_test_only; set in CC_BFS_APSP_L)
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar(result.computation_time);
    }
    if (nlhs > 2) {
        plhs[2] = createCountersStruct(result.counters);
    }
    if (verbose) {
        std::cout << "Returning result to MATLAB completed" << std::endl;
    }
}
#endif // MATLAB_BUILD
