//
// CC-Exact Sum Sweep (undirected, connected, unweighted)
// MEX front-end
//
#include "cc_ess.h"
#include <queue>
#include <vector>
#include <limits>
#include <iostream>
#include <chrono>
#include <array>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#ifdef MATLAB_BUILD
#include "mex.h"
#include "include/mex_utils.h"
#endif
#include "include/timer.h"
#include "include/graph_utils.h"

struct GraphNode {
    std::vector<int64_t> children;
};

struct Level {
    std::vector<int64_t> nodes;
};
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
/*  Generalized CC-BFS (for SSSP)                  */
/* ------------------------------------------------------------------ */



// Helper functions
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
                                     const std::vector<char> &cn_mask,
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
     * \param cn_mask A mask indicating which nodes are hub nodes.
     *                `cn_mask[i]` is true if node `i` is a hub node, false otherwise.
     * \param DEG The degree vector of the nodes, where `DEG[i]` represents the degree of node `i`.
     * \param s The node for which the corresponding hub node is to be selected.
     * \return The index of the corresponding hub node with the highest degree,
     *         or -1 if no suitable hub node is found.
     */
    int64_t c_s = -1;
    for (int64_t t: A[s]) {
        if (cn_mask[t]) {
            if ((c_s == -1) || (DEG[t] > DEG[c_s])) {
                c_s = t;
            }
        }
    }
    return c_s;
}


void BuildSPStructures(const std::vector<std::vector<int64_t>> &A,
                        const int64_t n,
                        const int64_t s,
                        std::vector<Eigen::VectorXd> &D,
                        std::vector<std::vector<GraphNode>> &sp_dags,
                        std::vector<std::vector<GraphNode>> &sp_shortcuts,
                        std::vector<std::vector<GraphNode>> &sp_trees,
                        std::vector<std::vector<Level>> &sp_levels,
                        std::vector<double> &eccs,
                        COUNTERS& counters
) {
    auto start_time_ = std::chrono::high_resolution_clock::now();

    std::vector<GraphNode> sp_dag(n);
    std::vector<GraphNode> sp_sc(n);  // records intra-level links of the SP_DAG
    std::vector<GraphNode> sp_tree(n);
    std::vector<Level> sp_level(n);


    Eigen::VectorXd D_s = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
    D_s(s) = 0;
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
                    D_s(t) = curr_depth;
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

    eccs[s] = D_s.maxCoeff();
    D[s] = std::move(D_s);
    sp_dags[s] = std::move(sp_dag);
    sp_shortcuts[s] = std::move(sp_sc);
    sp_trees[s] = std::move(sp_tree);
    sp_levels[s] = std::move(sp_level);

    // Calculate elapsed time
    auto end_time_ = std::chrono::high_resolution_clock::now();
    counters.time_secs_buildspstructs += std::chrono::duration<double>(end_time_ - start_time_).count();
}

void NodeProp(const int64_t curr_depth,
              const std::vector<int64_t>& neighborhood,
              Eigen::VectorXd& D_s,
              const std::vector<Eigen::VectorXd> &D,
              const std::vector<char>& processed,
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
            if (processed[t] && vcc < eta) {
                D_s = D_s.cwiseMin((D[t].array() + curr_depth).matrix());
                vcc++;
            } else {
                next.push_back(t);
            }
        }
    }
}

void MemoizedSSS(const std::vector<std::vector<int64_t>> &A,
                 const int64_t s,
                 const double eta,
                 std::vector<Eigen::VectorXd> &D,
                 const std::vector<char> &processed,
                 COUNTERS& counters) {
    auto start_time_ = std::chrono::high_resolution_clock::now();

    // Memoization-enabled Single-source search
    Eigen::VectorXd D_s = Eigen::VectorXd::Constant(D.size(), std::numeric_limits<double>::infinity());
    D_s(s) = 0;
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
                NodeProp(curr_depth, A[curr], D_s, D, processed, vcc, eta, next, true);
                counters.graph_expansion_node_count++;
            }
        }
        // swap frontier lists
        frontier.swap(next);
    }
    D[s] = D_s;
    counters.vec_comp_count += vcc;

    // Calculate elapsed time
    auto end_time_ = std::chrono::high_resolution_clock::now();
    counters.time_secs_memoizedsss += std::chrono::duration<double>(end_time_ - start_time_).count();
}


double EstimateBackwardCost(
    const int64_t curr_depth,
    const std::vector<Level>& c_s_sp_level,
    const std::vector<int64_t>& eq_dist_fr,
    const std::vector<int64_t>& inc_dist_fr,
    const Eigen::VectorXd& D_s,
    const Eigen::VectorXd& DEG) {
    size_t pop_size = 0;
    if (curr_depth > 0) pop_size += c_s_sp_level[curr_depth - 1].nodes.size();
    pop_size += c_s_sp_level[curr_depth].nodes.size();
    if (curr_depth + 1 < static_cast<int64_t>(c_s_sp_level.size())) {
        pop_size += c_s_sp_level[curr_depth + 1].nodes.size();
    }

    double target_node_num = eq_dist_fr.size() + inc_dist_fr.size();
    double K = pop_size - target_node_num;
    double est_bu_cost = K / (target_node_num + 1) + 1;

    double backward_cost = 0;
    for (int64_t curr: c_s_sp_level[curr_depth].nodes) {
        if (D_s[curr] > curr_depth) {
            backward_cost += std::min<double>(est_bu_cost, DEG[curr]);
        }
    }

    return backward_cost;
}


double ComputeForwardCost(
    const std::vector<int64_t>& eq_dist_fr,
    const std::vector<int64_t>& inc_dist_fr,
    const std::vector<GraphNode>& c_s_sp_dag,
    const std::vector<GraphNode>& c_s_sp_sc) {
    double forward_cost = 0;

    for (int64_t curr: eq_dist_fr) {
        forward_cost += c_s_sp_dag[curr].children.size();
    }
    for (int64_t curr: inc_dist_fr) {
        forward_cost += c_s_sp_sc[curr].children.size();
    }

    return forward_cost;
}


void HubGuidedSSS(const std::vector<std::vector<int64_t>> &A,
                  const int64_t s,
                  const double eta,
                  const int64_t c_s,
                  std::vector<Eigen::VectorXd> &D,
                  const std::vector<std::vector<GraphNode>> &sp_dags,
                  const std::vector<std::vector<GraphNode>> &sp_scs,
                  const std::vector<std::vector<GraphNode>> &sp_trees,
                  const std::vector<std::vector<Level>> &sp_levels,
                  const std::vector<double> &eccs,
                  const std::vector<char> &processed,
                  const Eigen::VectorXd &DEG,
                  COUNTERS& counters) {
    auto start_time_ = std::chrono::high_resolution_clock::now();

    counters.source_with_cs_count++;
    const std::vector<GraphNode>& c_s_sp_dag = sp_dags[c_s];
    const std::vector<GraphNode>& c_s_sp_sc = sp_scs[c_s];
    const std::vector<GraphNode>& c_s_sp_tree = sp_trees[c_s];
    const std::vector<Level>& c_s_sp_level = sp_levels[c_s];

    Eigen::VectorXd D_s = Eigen::VectorXd::Constant(D.size(), std::numeric_limits<double>::infinity());
    D_s(s) = 0;

    double ecc_c_s = eccs[c_s];
    double vcc = 0;

    std::vector<int64_t> inc_dist_fr;
    std::vector<int64_t> eq_dist_fr;

    int64_t curr_depth = 0;
    // build the first level
    curr_depth++;
    std::vector<bool> c_s_neighbor_mask(D.size(), false);
    for (int64_t t: A[c_s]) {
        c_s_neighbor_mask[t] = true;
    }
    for (int64_t t: A[s]) {
        if (D_s[t] > curr_depth) {
            D_s(t) = curr_depth;
            if (c_s_neighbor_mask[t]) {  // t is a common neighbor of s and c_s
                eq_dist_fr.push_back(t);
            } else if (t == c_s) {  // make sure c_s is absorbed
                D_s = D_s.cwiseMin((D[t].array() + curr_depth).matrix());
                vcc++;
            } else if ((processed[t]) && (vcc < eta) && (curr_depth < ecc_c_s - 1)) {  // t is a valid prior source node
                // this is equivalent to incorporating the SP tree of t all at once
                D_s = D_s.cwiseMin((D[t].array() + curr_depth).matrix());
                vcc++;
            } else {  // t is a source exclusive node
                inc_dist_fr.push_back(t);
            }
        }
    }


    while (!inc_dist_fr.empty() || !eq_dist_fr.empty()) {
        // move to the next level
        curr_depth++;

        // reset
        double top_down_cost = 0;
        double bottom_up_cost = 0;

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
                NodeProp(curr_depth, c_s_sp_tree[curr].children, D_s, D, processed,
                         vcc, eta,
                         next_eq_dist_fr, false);
                counters.sptree_expansion_node_count++;
            }
        }
        // expand sc_fr nodes in SP-DAG space
        if (!inc_dist_fr.empty() && (curr_depth < ecc_c_s)) {
            for (int64_t curr: inc_dist_fr) {
                NodeProp(curr_depth, c_s_sp_dag[curr].children, D_s, D, processed, vcc,
                         eta,
                         next_inc_dist_fr, use_vc);
                counters.spdag_expansion_node_count++;
            }
        }

        top_down_cost = ComputeForwardCost(eq_dist_fr, inc_dist_fr, c_s_sp_dag, c_s_sp_sc);
        bottom_up_cost = EstimateBackwardCost(curr_depth, c_s_sp_level, eq_dist_fr, inc_dist_fr, D_s, DEG);

        if (top_down_cost < bottom_up_cost) {
            // top-down is preferred
            // visit nodes in spd_fr
            if (!eq_dist_fr.empty()) {
                for (int64_t curr: eq_dist_fr) {
                    // must not use VC here
                    NodeProp(curr_depth, c_s_sp_dag[curr].children, D_s, D, processed,
                             vcc,
                             eta,
                             next_eq_dist_fr, false);
                    counters.top_down_node_count++;
                    counters.spdag_expansion_node_count++;
                }
            }

            // visit nodes in inc_dist_fr
            if (!inc_dist_fr.empty()) {
                for (int64_t curr: inc_dist_fr) {
                    // intra-links are special because the ending node can expand from SP-DAG space
                    NodeProp(curr_depth,
                             c_s_sp_sc[curr].children, D_s, D, processed,
                             vcc,
                             eta,
                             next_eq_dist_fr, use_vc);
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
                            D_s[curr] = curr_depth;
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

    D[s] = D_s;
    counters.vec_comp_count += vcc;

    // Calculate elapsed time
    auto end_time_ = std::chrono::high_resolution_clock::now();
    counters.time_secs_hubguidedsss += std::chrono::duration<double>(end_time_ - start_time_).count();
}

// A generalized CC_BFS for single-source shortest-path computation - Designed to replace single-source BFS
Eigen::VectorXd GeneralizedCCBFS(const std::vector<std::vector<int64_t>> &A,
                                 const Eigen::VectorXd &DEG,
                                 const int64_t n,
                                 const int64_t s,
                                 const int64_t eta,
                                 std::vector<Eigen::VectorXd> &D,
                                 std::vector<double> &eccs,
                                 std::vector<char> &sel_hn_mask,
                                 std::vector<char> &processed,
                                 std::vector<std::vector<GraphNode>> &sp_dags,
                                 std::vector<std::vector<GraphNode>> &sp_trees,
                                 std::vector<std::vector<GraphNode>> &sp_scs,
                                 std::vector<std::vector<Level>> &sp_levels,
                                 COUNTERS &counters) {
    auto start_time_ = std::chrono::high_resolution_clock::now();

    if (processed[s]) {
        return D[s];
    }
    // s is not processed
    if (sel_hn_mask[s]) {  // s is a unprocessed hub node, compute the SP structures of s
        BuildSPStructures(A, n, s, D, sp_dags, sp_scs, sp_trees, sp_levels, eccs, counters);
        counters.proc_hn_count++;
    } else {
        int64_t c_s = FindHighestDegreeNeighborHub(A, sel_hn_mask, DEG, s);
        if (c_s == -1) {  // s is a peripheral node
            MemoizedSSS(A, s, eta, D, processed, counters);
        } else {  // s is a spoke node
            // if c_s is not processed, we need to process it first
            if (!processed[c_s]) {
                // process c_s first
                BuildSPStructures(A, n, c_s, D, sp_dags, sp_scs, sp_trees, sp_levels, eccs, counters);
                processed[c_s] = true;
                counters.proc_hn_count++;
            }
            HubGuidedSSS(A,  s, eta, c_s, D, sp_dags, sp_scs, sp_trees, sp_levels, eccs, processed, DEG, counters);
        }
    }
    processed[s] = true;

    // Calculate elapsed time
    auto end_time_ = std::chrono::high_resolution_clock::now();
    counters.time_secs_generalizedcc += std::chrono::duration<double>(end_time_ - start_time_).count();

    return D[s];
}


/* ------------------------------------------------------------------ */
/*  Sum-Sweep (unchanged except for closing the source)               */
/* ------------------------------------------------------------------ */
void sumSweep(const std::vector<std::vector<int64_t>>& A,
              const int64_t n,
              int64_t s,
              std::vector<Eigen::VectorXd> &D,
              const Eigen::VectorXd &DEG,
              const int64_t k,
              const int64_t eta,
              int64_t& DL,
              int64_t& RU,
              Eigen::VectorXd& L,
              Eigen::VectorXd& U,
              std::vector<char> &closed,
              std::vector<char> &sel_hn_mask,
              std::vector<char> &processed,
              std::vector<std::vector<GraphNode>> &sp_dags,
              std::vector<std::vector<GraphNode>> &sp_trees,
              std::vector<std::vector<GraphNode>> &sp_scs,
              std::vector<std::vector<Level>> &sp_levels,
              std::vector<double> &eccs,
              std::vector<double>& S_F,
              COUNTERS &counters,
              const bool verbose)
{
    Eigen::VectorXd D_s = GeneralizedCCBFS(A, DEG, n, s, eta,
                                          D,  eccs, sel_hn_mask,
                                          processed, sp_dags, sp_trees, sp_scs,
                                          sp_levels, counters);
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
        D_s   = GeneralizedCCBFS(A, DEG, n, s, eta,
                                          D,  eccs, sel_hn_mask,
                                          processed, sp_dags, sp_trees, sp_scs,
                                          sp_levels, counters);
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
    D_s   = GeneralizedCCBFS(A, DEG, n, s, eta,
                                          D,  eccs, sel_hn_mask,
                                          processed, sp_dags, sp_trees, sp_scs,
                                          sp_levels, counters);
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
                 const Eigen::VectorXd &DEG,
                 const int64_t n,
                 const int64_t eta,
                 std::vector<Eigen::VectorXd> &D,
                 int64_t s,
                 int64_t& DL,
                 int64_t& RU,
                 Eigen::VectorXd& L,
                 Eigen::VectorXd& U,
                 std::vector<char>& closed,
                 std::vector<char> &sel_hn_mask,
                 std::vector<char> &processed,
                 std::vector<std::vector<GraphNode>> &sp_dags,
                 std::vector<std::vector<GraphNode>> &sp_trees,
                 std::vector<std::vector<GraphNode>> &sp_scs,
                 std::vector<std::vector<Level>> &sp_levels,
                 std::vector<double> &eccs,
                 std::vector<double>& S_F,
                 COUNTERS &counters)
{
    Eigen::VectorXd D_s = GeneralizedCCBFS(A, DEG, n, s, eta,
                                          D,  eccs, sel_hn_mask,
                                          processed, sp_dags, sp_trees, sp_scs,
                                          sp_levels, counters);
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
/*  Exact Sum Sweep main                                             */
/* ------------------------------------------------------------------ */
RESULT CC_ExactSumSweep(
        const MatrixType& adj_matrix,
        const int64_t k,
        const bool verbose)
{
    Timer timer_global; timer_global.Start();
    Timer timer; // timer for debug
    COUNTERS counters;

    /// compute deg and adjacency list
    timer.Start();

    const int64_t n = matrix_ext::rows(adj_matrix);
    auto [DEG, A] = ComputeDegreeAndAdjList(adj_matrix, n);

    counters.time_secs_comp_adj = timer.Seconds();
    if (verbose) {
        std::cout << "@@ Time to compute adjacency list: " << timer.Seconds() << " seconds" << std::endl;
    }

    /// initialization
    timer.Start();

    std::vector<Eigen::VectorXd> D = std::vector<Eigen::VectorXd>(n);
    std::vector<char> sel_hn_mask(n, false); // selected hub nodes
    std::vector<char> checked(n, false); // those whose sssps are used to update the bounds
    std::vector<char> processed(n, false); // those whose sssps are computed
    std::vector<std::vector<GraphNode>> sp_dags(n);
    std::vector<std::vector<GraphNode>> sp_scs(n);
    std::vector<std::vector<GraphNode>> sp_trees(n);
    std::vector<std::vector<Level>> sp_levels(n);
    std::vector<double> eccs(n, n);  // eccentricity of each node

    timer.Stop();
    if (verbose) {
        std::cout << "@@ Initialization costs: " << timer.Seconds() << " seconds" << std::endl;
    }


    /// select hub nodes
    timer.Start();

    int64_t m = DEG.sum(); // total number of directed edges
    double density = (double)m / (n * (n - 1));
    // compute memoization constraints
    double avg_deg = (double)m / n;
    double eta = std::pow(avg_deg, 0.5);                                 // constraint for Vector Comparison
    const double bounded_avg_deg = (avg_deg > 0.0)
            ? std::min(avg_deg, static_cast<double>(n) / avg_deg)
            : 0.0;
    int64_t mu = static_cast<int64_t>(std::round(bounded_avg_deg)); // constraint for hub nodes
    // select hub nodes
    std::vector<int64_t> hub_nodes = SelectHubNodes(A, n, DEG, mu);
    counters.sel_hn_count = hub_nodes.size();
    for (int64_t i: hub_nodes) {
        sel_hn_mask[i] = true;
    }

    timer.Stop();
    if (verbose) {
        std::cout << "@@ Select hub nodes costs: " << timer.Seconds() << " seconds" << std::endl;
        std::cout << "Number of selected hub nodes: " << counters.sel_hn_count << std::endl;
        std::cout << "avg-deg: " << avg_deg << std::endl;
        std::cout << "density: " << density * 100 << "%" << std::endl;
        std::cout << "mu: " << mu << std::endl;
        std::cout << "eta: " << eta << std::endl;
    }


    /* ---------- initial Sum-Sweep (k sources) -------------------- */
    int64_t s0; DEG.maxCoeff(&s0);

    int64_t DL = 0, RU = n;
    Eigen::VectorXd L = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd U = Eigen::VectorXd::Constant(n, n);
    std::vector<char> closed(n, false);
    std::vector<double> S_F;

    sumSweep(A, n, s0, D, DEG, k, eta, DL, RU, L, U, closed, sel_hn_mask, processed, sp_dags, sp_trees,
             sp_scs, sp_levels, eccs, S_F, counters, verbose);

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

        stepForward(A, DEG, n, eta, D, s,
                     DL, RU, L, U, closed,
                     sel_hn_mask, processed,
                     sp_dags, sp_trees, sp_scs,
                     sp_levels, eccs, S_F, counters);

        size_t open_after  = countOpen(L,U,closed,DL,RU);
        double util        = static_cast<double>(open_before - open_after);

        UE[t] = util;
        double bonus = 2.0 / static_cast<double>(iter);
        for (int i=0;i<3;++i) if (i!=t) UE[i] += bonus;
    }
    /// computation ends
    timer_global.Stop();
    counters.time_secs_overall = timer_global.Seconds();

    int64_t numProcessed = std::count(processed.begin(), processed.end(), true);
    double ratio = (double)numProcessed / (double)n;
    counters.processed_count = numProcessed;
    counters.processed_ratio = ratio;

    // print statistics
    if (verbose) {
        std::cout << "@@ Time to compute the whole graph: " << timer_global.Seconds() << " seconds" << std::endl;
        std::cout << "@@ Time to run BuildSPStructures: " << counters.time_secs_buildspstructs << " seconds" << "; Ratio: " << (double)counters.time_secs_buildspstructs / (double)counters.time_secs_overall << std::endl;
        std::cout << "@@ Time to run MemoizedSSS: " << counters.time_secs_memoizedsss << " seconds" << "; Ratio: " << (double)counters.time_secs_memoizedsss / (double)counters.time_secs_overall << std::endl;
        std::cout << "@@ Time to run HubGuidedSSS: " << counters.time_secs_hubguidedsss << " seconds" << "; Ratio: " << (double)counters.time_secs_hubguidedsss / (double)counters.time_secs_overall << std::endl;
        std::cout << "@@ Time to run GeneralizedCCBFS: " << counters.time_secs_generalizedcc << " seconds" << "; Ratio: " << (double)counters.time_secs_generalizedcc / (double)counters.time_secs_overall << std::endl;
        std::cout << "Number of processed nodes: " << numProcessed << "; Ratio: " << ratio << "; D and R: " << DL << ", " << RU << std::endl;
        std::cout << "Number of processed hub nodes: " << counters.proc_hn_count << "; Ratio to all hub nodes: " << (double)counters.proc_hn_count / (double)counters.sel_hn_count << std::endl;
        std::cout << "Number of VCs: " << counters.vec_comp_count << "; Average per node: " << (double)counters.vec_comp_count / (double)n << std::endl;
        std::cout << "Number of processed nodes with hub neighbors: " << counters.source_with_cs_count << std::endl;
    }
    return RESULT{DL, RU, counters, timer_global.Seconds()};
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
 * @brief Python-callable wrapper for CC-ExactSumSweep algorithm
 * 
 * @param adj_matrix_data Flat array of adjacency matrix (row-major, n*n elements)
 * @param n Number of nodes
 * @param k Number of initial sweep sources (typically 4)
 * @param diameter Output parameter for network diameter
 * @param radius Output parameter for network radius
 * @param elapsed_time Output parameter for computation time
 * @return 0 on success, -1 on error
 */
EXPORT int cc_ess_py(
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
        RESULT result = CC_ExactSumSweep(adj_matrix, k, false);
        
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
 * MATLAB MEX entry point for CC-ExactSumSweep on an undirected, unweighted graph.
 *
 * Usage:
 *   [diameter, radius, t, counters] = cc_ess_cpp(A, k)
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
 *             whose SSSP was computed), `processed_ratio` (that count divided
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
        mexErrMsgTxt("Usage: [diameter,radius,time,counters] = cc_ess_cpp(A,k)");

    int64_t kSweeps = static_cast<int64_t>(mxGetScalar(prhs[1]));
    
    // Parse the MATLAB matrix into our MatrixType and call CC_ExactSumSweep
    MatrixType adj = parseAdjacencyMatrix(prhs[0]);
    
    RESULT res = CC_ExactSumSweep(adj, kSweeps);

    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(res.diameter);
    if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(res.radius);
    if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(res.computation_time);
    if (nlhs > 3) plhs[3] = createCountersStruct(res.counters);
}
#endif // MATLAB_BUILD
