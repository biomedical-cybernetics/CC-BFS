// CC-BFS-P: parallel CC-BFS for exact unweighted APSP.
//
// The source follows the terminology and search logic of the sequential
// CC-BFS implementation. Hub preprocessing and non-hub searches are parallel;
// each worker retains one private scratch workspace, and completed non-hub
// results become visible only at deterministic batch boundaries.

#include "cc_bfs_p.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "parallel_utils.hpp"
#include "process_memory_utils.h"
#include "timer.h"

#ifdef MATLAB_BUILD
#include "matlab_adjacency_list.h"
#include "mex.h"
#endif

namespace parallel_cc::cc_bfs_p {
namespace {

// Canonical hub structures store outgoing vertex IDs only. A GraphNode is
// reused for shortest-path DAG, shortcut/spill, and one-parent tree grids.
class WorkerArenaResource final : public std::pmr::memory_resource {
 public:
  WorkerArenaResource() {
#ifdef _WIN32
    heap_ = HeapCreate(HEAP_NO_SERIALIZE, 0, 0);
    if (heap_ == nullptr) throw std::bad_alloc();
#endif
  }

  ~WorkerArenaResource() override {
#ifdef _WIN32
    if (heap_ != nullptr) HeapDestroy(heap_);
#endif
  }

  void BeginBulkRelease() noexcept { bulk_release_ = true; }

  void Release() noexcept {
#ifdef _WIN32
    if (heap_ != nullptr) {
      HANDLE heap_to_destroy = heap_;
      heap_ = nullptr;
      HeapDestroy(heap_to_destroy);
    }
#else
    pool_.release();
#endif
  }

 private:
  void* do_allocate(size_t bytes, size_t alignment) override {
#ifdef _WIN32
    if (bulk_release_ || heap_ == nullptr ||
        alignment > MEMORY_ALLOCATION_ALIGNMENT) {
      throw std::bad_alloc();
    }
    void* pointer = HeapAlloc(heap_, 0, bytes == 0 ? 1 : bytes);
    if (pointer == nullptr) throw std::bad_alloc();
    return pointer;
#else
    if (bulk_release_) throw std::bad_alloc();
    return pool_.allocate(bytes == 0 ? 1 : bytes, alignment);
#endif
  }

  void do_deallocate(void* pointer, size_t bytes,
                     size_t alignment) override {
    static_cast<void>(bytes);
    static_cast<void>(alignment);
#ifdef _WIN32
    // Once every object using this worker's arena is being destroyed, all
    // remaining buffers are discarded together by HeapDestroy.
    if (!bulk_release_ && heap_ != nullptr && pointer != nullptr) {
      HeapFree(heap_, 0, pointer);
    }
#else
    // The standard pool returns superseded buffers for reuse during the run;
    // the final release discards all remaining worker-owned blocks together.
    if (!bulk_release_ && pointer != nullptr) {
      pool_.deallocate(pointer, bytes == 0 ? 1 : bytes, alignment);
    }
#endif
  }

  bool do_is_equal(
      const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

#ifdef _WIN32
  HANDLE heap_ = nullptr;
#else
  std::pmr::unsynchronized_pool_resource pool_;
#endif
  bool bulk_release_ = false;
};

struct GraphNode {
  explicit GraphNode(
      std::pmr::memory_resource* resource =
          std::pmr::get_default_resource())
      : children(resource) {}

  std::pmr::vector<int64_t> children;
};

struct Level {
  explicit Level(
      std::pmr::memory_resource* resource =
          std::pmr::get_default_resource())
      : nodes(resource) {}

  std::pmr::vector<int64_t> nodes;
};

template <typename Element>
void InitializeArenaRow(std::vector<Element>& row,
                        int64_t n,
                        std::pmr::memory_resource* resource) {
  row.reserve(static_cast<size_t>(n));
  for (int64_t index = 0; index < n; ++index) {
    row.emplace_back(resource);
  }
}

template <typename DistScalar>
struct WorkerScratch {
  explicit WorkerScratch(int64_t n)
      : D_s(n),
        hub_neighbor_generation(static_cast<size_t>(n), 0) {}

  DistVec<DistScalar> D_s;
  std::vector<int64_t> frontier;
  std::vector<int64_t> next;
  std::vector<int64_t> inc_dist_fr;
  std::vector<int64_t> eq_dist_fr;
  std::vector<int64_t> next_inc_dist_fr;
  std::vector<int64_t> next_eq_dist_fr;

  // Each worker owns this stamp array and generation counter. Advancing the
  // generation logically clears the mask without writing all n entries for
  // every source. On the extremely rare counter wrap, a physical clear makes
  // generation 1 unambiguous again before any membership query.
  std::vector<uint32_t> hub_neighbor_generation;
  uint32_t current_hub_generation = 0;

  void MarkHubNeighbors(const std::vector<int64_t>& neighbors) {
    if (current_hub_generation ==
        (std::numeric_limits<uint32_t>::max)()) {
      std::fill(hub_neighbor_generation.begin(),
                hub_neighbor_generation.end(), 0);
      current_hub_generation = 1;
    } else {
      ++current_hub_generation;
    }
    for (int64_t neighbor : neighbors) {
      hub_neighbor_generation[static_cast<size_t>(neighbor)] =
          current_hub_generation;
    }
  }

  bool IsHubNeighbor(int64_t vertex) const {
    return hub_neighbor_generation[static_cast<size_t>(vertex)] ==
        current_hub_generation;
  }
};

void DestroyHubStructuresInWorkerArenas(
    const std::vector<int64_t>& hub_nodes,
    const std::vector<int>& hub_owner_workers,
    std::vector<std::unique_ptr<WorkerArenaResource>>& worker_arenas,
    std::vector<std::vector<Level>>& sp_levels,
    std::vector<std::vector<GraphNode>>& sp_trees,
    std::vector<std::vector<GraphNode>>& sp_scs,
    std::vector<std::vector<GraphNode>>& sp_dags) {
  if (hub_nodes.size() != hub_owner_workers.size()) {
    throw std::runtime_error(
        "Hub-owner metadata does not match the selected hub nodes.");
  }

  // Each logical worker first destroys the row headers backed by its private
  // arena. Individual child-buffer deallocations are suppressed during this
  // final pass, then the arena destroys its private heap in one operation.
  const int arena_count = static_cast<int>(worker_arenas.size());
  #pragma omp parallel num_threads(arena_count)
  {
    const int worker = omp_get_thread_num();
    WorkerArenaResource& arena = *worker_arenas[worker];
    arena.BeginBulkRelease();

    for (int64_t slot = 0;
         slot < static_cast<int64_t>(hub_nodes.size()); ++slot) {
      if (hub_owner_workers[slot] == worker) {
        const int64_t hub = hub_nodes[slot];
        std::vector<Level> level_row_to_destroy;
        level_row_to_destroy.swap(sp_levels[hub]);
        std::vector<GraphNode> tree_row_to_destroy;
        tree_row_to_destroy.swap(sp_trees[hub]);
        std::vector<GraphNode> shortcut_row_to_destroy;
        shortcut_row_to_destroy.swap(sp_scs[hub]);
        std::vector<GraphNode> dag_row_to_destroy;
        dag_row_to_destroy.swap(sp_dags[hub]);
      }
    }
    arena.Release();
  }
}

// Capacity-based estimates intentionally account for reserved storage, not
// only elements currently populated. OS process peaks are reported separately.
int64_t EstimateGraphNodeGridBytes(
    const std::vector<std::vector<GraphNode>>& grid) {
  int64_t bytes = static_cast<int64_t>(
      grid.capacity() * sizeof(std::vector<GraphNode>));
  for (const auto& row : grid) {
    bytes += static_cast<int64_t>(row.capacity() * sizeof(GraphNode));
    for (const auto& node : row) {
      bytes += static_cast<int64_t>(
          node.children.capacity() * sizeof(int64_t));
    }
  }
  return bytes;
}

int64_t EstimateAdjacencyListBytes(const AdjacencyList& A) {
  int64_t bytes = static_cast<int64_t>(
      A.capacity() * sizeof(std::vector<int64_t>));
  for (const auto& neighbors : A) {
    bytes += static_cast<int64_t>(
        neighbors.capacity() * sizeof(int64_t));
  }
  return bytes;
}

void UpdateTotalMemoryCounter(Counters& counters) {
  counters.estimated_total_memory_bytes =
      counters.estimated_dominant_memory_bytes +
      counters.adjacency_list_bytes;
}

Eigen::VectorXd BFSSSS(const AdjacencyList& A, int64_t s) {
  // Plain BFS is used only by the canonical 4-sweep diameter-bound phase.
  const int64_t n = static_cast<int64_t>(A.size());
  Eigen::VectorXd D = Eigen::VectorXd::Constant(
      n, std::numeric_limits<double>::infinity());
  D(s) = 0.0;
  std::queue<int64_t> queue;
  queue.push(s);
  while (!queue.empty()) {
    const int64_t curr = queue.front();
    queue.pop();
    const double alt_dst = D(curr) + 1.0;
    for (int64_t t : A[curr]) {
      if (alt_dst < D(t)) {
        D(t) = alt_dst;
        queue.push(t);
      }
    }
  }
  return D;
}

int64_t selectMaxSF(const std::vector<double>& S_F,
                    const std::vector<char>& closed) {
  double best_sf = -std::numeric_limits<double>::infinity();
  int64_t best = -1;
  for (int64_t v = 0;
       v < static_cast<int64_t>(S_F.size()); ++v) {
    if (!closed[v] && S_F[v] > best_sf) {
      best_sf = S_F[v];
      best = v;
    }
  }
  // Exhausting candidates is normal on small graphs. Preserve the existing
  // MATLAB diagnostic while keeping the native/Python interface quiet.
#ifdef MATLAB_BUILD
  if (best == -1) {
    std::cerr << "selectMaxSF(): no candidate" << std::endl;
  }
#endif
  return best;
}

int64_t FourSumSweep(const AdjacencyList& A,
                     int64_t s0) {
  // Preserve the canonical 4-SumSweep upper-bound heuristic. The returned
  // bound controls storage width; the APSP searches still compute distances.
  const int64_t n = static_cast<int64_t>(A.size());
  Eigen::VectorXd U = Eigen::VectorXd::Constant(
      n, static_cast<double>(n));
  std::vector<double> S_F(static_cast<size_t>(n), 0.0);
  std::vector<char> closed(static_cast<size_t>(n), false);

  auto sweep = [&](int64_t s) {
    const Eigen::VectorXd D = BFSSSS(A, s);
    const double ecc = D.maxCoeff();
    closed[s] = true;
    U = U.cwiseMin(
        Eigen::VectorXd::Constant(n, ecc) + D);
    for (int64_t v = 0; v < n; ++v) {
      S_F[v] += D(v);
    }
  };

  sweep(s0);
  for (int pass = 2; pass <= 3; ++pass) {
    const int64_t s = selectMaxSF(S_F, closed);
    if (s == -1) break;
    sweep(s);
  }
  const int64_t s = static_cast<int64_t>(
      std::min_element(S_F.begin(), S_F.end()) - S_F.begin());
  sweep(s);
  return static_cast<int64_t>(U.maxCoeff());
}

int64_t DiamUpperBoundImpl(const AdjacencyList& A,
                           const Eigen::VectorXd& DEG) {
  int64_t s0 = 0;
  DEG.maxCoeff(&s0);
  return FourSumSweep(A, s0);
}

std::vector<int64_t> SelectHubNodes(const AdjacencyList& A,
                                    const Eigen::VectorXd& DEG,
                                    int64_t mu) {
  // Greedily cover each selected candidate node and its one-hop neighborhood,
  // decrementing residual scores so later hubs favor uncovered graph regions.
  const int64_t n = static_cast<int64_t>(A.size());
  std::vector<int64_t> hub_nodes;
  std::vector<char> active(static_cast<size_t>(n), true);
  Eigen::VectorXd scores = DEG;
  int64_t remaining = n;
  while (remaining > 0 &&
         hub_nodes.size() < static_cast<size_t>(std::max<int64_t>(0, mu))) {
    int64_t cn = 0;
    scores.maxCoeff(&cn);
    hub_nodes.push_back(cn);
    scores(cn) = -std::numeric_limits<double>::infinity();
    if (active[cn]) {
      active[cn] = false;
      --remaining;
      for (int64_t neighbor : A[cn]) {
        scores(neighbor) -= 1.0;
      }
    }
    for (int64_t t : A[cn]) {
      if (!active[t]) continue;
      for (int64_t neighbor : A[t]) {
        scores(neighbor) -= 1.0;
      }
      active[t] = false;
      --remaining;
    }
  }
  return hub_nodes;
}

int64_t FindHighestDegreeNeighborHub(const AdjacencyList& A,
                                     const std::vector<char>& hn_mask,
                                     const Eigen::VectorXd& DEG,
                                     int64_t s) {
  int64_t c_s = -1;
  for (int64_t neighbor : A[s]) {
    if (hn_mask[neighbor] &&
        (c_s == -1 || DEG(neighbor) > DEG(c_s))) {
      c_s = neighbor;
    }
  }
  return c_s;
}

template <typename DistScalar>
double EstimateBackwardCost(
    int64_t curr_depth,
    const std::vector<Level>& c_s_sp_level,
    const std::vector<int64_t>& eq_dist_fr,
    const std::vector<int64_t>& inc_dist_fr,
    const DistVec<DistScalar>& D_s,
    const Eigen::VectorXd& DEG) {
  // Estimate bottom-up probes within the adjacent hub's BFS levels.
  size_t pop_size = 0;
  if (curr_depth > 0) {
    pop_size += c_s_sp_level[curr_depth - 1].nodes.size();
  }
  pop_size += c_s_sp_level[curr_depth].nodes.size();
  if (curr_depth + 1 < static_cast<int64_t>(c_s_sp_level.size())) {
    pop_size += c_s_sp_level[curr_depth + 1].nodes.size();
  }
  const double target_node_num = static_cast<double>(
      eq_dist_fr.size() + inc_dist_fr.size());
  const double K = static_cast<double>(pop_size) - target_node_num;
  const double est_bu_cost = K / (target_node_num + 1.0) + 1.0;
  double backward_cost = 0.0;
  for (int64_t curr : c_s_sp_level[curr_depth].nodes) {
    if (static_cast<int64_t>(D_s(curr)) > curr_depth) {
      backward_cost += std::min<double>(est_bu_cost, DEG(curr));
    }
  }
  return backward_cost;
}

double ComputeForwardCost(
    const std::vector<int64_t>& eq_dist_fr,
    const std::vector<int64_t>& inc_dist_fr,
    const std::vector<GraphNode>& c_s_sp_dag,
    const std::vector<GraphNode>& c_s_sp_sc) {
  // Forward cost is the number of stored DAG/shortcut children to traverse.
  double forward_cost = 0.0;
  for (int64_t curr : eq_dist_fr) {
    forward_cost += static_cast<double>(c_s_sp_dag[curr].children.size());
  }
  for (int64_t curr : inc_dist_fr) {
    forward_cost += static_cast<double>(c_s_sp_sc[curr].children.size());
  }
  return forward_cost;
}

template <typename DistScalar>
void BuildSPStructures(
    const AdjacencyList& A,
    int64_t s,
    Eigen::Ref<DistVec<DistScalar>> D_s,
    DistScalar dist_inf,
    std::vector<GraphNode>& sp_dag,
    std::vector<GraphNode>& sp_sc,
    std::vector<GraphNode>& sp_tree,
    std::vector<Level>& sp_level,
    std::pmr::memory_resource* arena_resource,
    double& ecc_s) {
  // Hub preprocessing is canonical CC-BFS: one exact BFS produces D_s, all
  // shortest-path DAG edges, a discovery tree, same-level spill shortcuts,
  // level membership, and ecc_s. Each result occupies the current hub's
  // exclusive slot in shared preallocated storage.
  D_s.setConstant(dist_inf);
  D_s(s) = static_cast<DistScalar>(0);
  sp_level[0].nodes.push_back(s);

  std::vector<int64_t> frontier{s};
  std::vector<int64_t> next;
  int64_t curr_depth = 0;
  while (!frontier.empty()) {
    ++curr_depth;
    Level curr_level(arena_resource);
    for (int64_t curr : frontier) {
      for (int64_t t : A[curr]) {
        if (curr_depth < D_s(t)) {
          next.push_back(t);
          D_s(t) = static_cast<DistScalar>(curr_depth);
          sp_dag[curr].children.push_back(t);
          sp_tree[curr].children.push_back(t);
          curr_level.nodes.push_back(t);
        } else if (curr_depth == D_s(t)) {
          sp_dag[curr].children.push_back(t);
        } else if (curr_depth > 2 && curr_depth - 1 == D_s(t)) {
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

  ecc_s = static_cast<double>(D_s.maxCoeff());
}

template <typename DistScalar, typename Neighborhood>
void NodeProp(DistScalar curr_depth,
              const Neighborhood& neighborhood,
              DistVec<DistScalar>& D_s,
              const DistVecCache<DistScalar>& D_mem_snapshot,
              double& vcc,
              double eta,
              std::vector<int64_t>& next,
              bool use_vc) {
  // A cached vector supplies exact distances from t; adding the established
  // source-to-t distance curr_depth gives valid path lengths.
  // Taking a componentwise minimum cannot discard a shorter path, while the
  // frontier continues ordinary BFS wherever no D_mem vector is usable.
  if (neighborhood.empty()) return;
  if (!use_vc || vcc >= eta) {
    for (int64_t t : neighborhood) {
      if (curr_depth < D_s(t)) {
        D_s(t) = curr_depth;
        next.push_back(t);
      }
    }
    return;
  }
  for (int64_t t : neighborhood) {
    if (curr_depth < D_s(t)) {
      D_s(t) = curr_depth;
      if (D_mem_snapshot.Has(t) &&
          vcc < eta) {
        D_s = D_s.cwiseMin(
            (D_mem_snapshot.Get(t).array() + curr_depth).matrix());
        ++vcc;
      } else {
        next.push_back(t);
      }
    }
  }
}

template <typename DistScalar>
const DistVec<DistScalar>& MemoizedSSS(
    const AdjacencyList& A,
    int64_t s,
    double eta,
    const DistVecCache<DistScalar>& D_mem_snapshot,
    DistScalar dist_inf,
    Counters& counters,
    WorkerScratch<DistScalar>& scratch) {
  // Sources without an adjacent hub retain the canonical memoized BFS. The
  // D_mem_snapshot is immutable throughout the search from s.
  const int64_t n = static_cast<int64_t>(A.size());
  ++counters.source_without_cs_count;
  DistVec<DistScalar>& D_s = scratch.D_s;
  D_s.setConstant(n, dist_inf);
  D_s(s) = static_cast<DistScalar>(0);
  double vcc = 0.0;
  std::vector<int64_t>& frontier = scratch.frontier;
  std::vector<int64_t>& next = scratch.next;
  frontier.clear();
  next.clear();
  frontier.push_back(s);
  int64_t curr_depth = 0;
  while (!frontier.empty()) {
    ++curr_depth;
    next.clear();
    for (int64_t curr : frontier) {
      NodeProp(static_cast<DistScalar>(curr_depth), A[curr],
                    D_s, D_mem_snapshot, vcc,
                    eta, next, true);
      ++counters.graph_expansion_node_count;
    }
    frontier.swap(next);
  }
  counters.vec_comp_count += static_cast<int64_t>(vcc);
  return D_s;
}

template <typename DistScalar>
const DistVec<DistScalar>& HubGuidedSSS(
    const AdjacencyList& A,
    int64_t s,
    double eta,
    int64_t c_s,
    const DistVecCache<DistScalar>& D_mem_snapshot,
    DistScalar dist_inf,
    const std::vector<std::vector<GraphNode>>& sp_dags,
    const std::vector<std::vector<GraphNode>>& sp_scs,
    const std::vector<std::vector<GraphNode>>& sp_trees,
    const std::vector<std::vector<Level>>& sp_levels,
    const std::vector<double>& eccs,
    const Eigen::VectorXd& DEG,
    Counters& counters,
    WorkerScratch<DistScalar>& scratch) {
  // A source adjacent to c_s reuses that hub's exact prebuilt search spaces.
  // eq_dist_fr follows the hub's level, whereas inc_dist_fr reaches vertices
  // whose distance from s is still diverging from it.
  const int64_t n = static_cast<int64_t>(A.size());
  ++counters.source_with_cs_count;
  const auto& c_s_sp_dag = sp_dags[c_s];
  const auto& c_s_sp_sc = sp_scs[c_s];
  const auto& c_s_sp_tree = sp_trees[c_s];
  const auto& c_s_sp_level = sp_levels[c_s];
  const double ecc_c_s = eccs[c_s];
  DistVec<DistScalar>& D_s = scratch.D_s;
  D_s.setConstant(n, dist_inf);
  D_s(s) = static_cast<DistScalar>(0);
  double vcc = 0.0;
  std::vector<int64_t>& inc_dist_fr = scratch.inc_dist_fr;
  std::vector<int64_t>& eq_dist_fr = scratch.eq_dist_fr;
  std::vector<int64_t>& next_inc_dist_fr =
      scratch.next_inc_dist_fr;
  std::vector<int64_t>& next_eq_dist_fr = scratch.next_eq_dist_fr;
  inc_dist_fr.clear();
  eq_dist_fr.clear();
  next_inc_dist_fr.clear();
  next_eq_dist_fr.clear();
  int64_t curr_depth = 1;
  scratch.MarkHubNeighbors(A[c_s]);
  for (int64_t t : A[s]) {
    if (D_s(t) <= curr_depth) continue;
    D_s(t) = static_cast<DistScalar>(curr_depth);
    if (scratch.IsHubNeighbor(t)) {
      eq_dist_fr.push_back(t);
    } else if (t == c_s) {
      D_s = D_s.cwiseMin(
          (D_mem_snapshot.Get(t).array() +
           static_cast<DistScalar>(curr_depth)).matrix());
      ++vcc;
    } else if (D_mem_snapshot.Has(t) &&
               vcc < eta &&
               curr_depth < ecc_c_s - 1) {
      D_s = D_s.cwiseMin(
          (D_mem_snapshot.Get(t).array() +
           static_cast<DistScalar>(curr_depth)).matrix());
      ++vcc;
    } else {
      inc_dist_fr.push_back(t);
    }
  }

  while (!inc_dist_fr.empty() || !eq_dist_fr.empty()) {
    ++curr_depth;
    const bool use_vc = curr_depth < ecc_c_s - 1;
    const Level& curr_level = c_s_sp_level[curr_depth];
    next_eq_dist_fr.clear();
    next_inc_dist_fr.clear();
    for (int64_t curr : eq_dist_fr) {
      NodeProp(static_cast<DistScalar>(curr_depth),
                    c_s_sp_tree[curr].children, D_s, D_mem_snapshot,
                    vcc, eta,
                    next_eq_dist_fr, false);
      ++counters.sptree_expansion_node_count;
    }
    if (curr_depth < ecc_c_s) {
      for (int64_t curr : inc_dist_fr) {
        NodeProp(static_cast<DistScalar>(curr_depth),
                      c_s_sp_dag[curr].children, D_s, D_mem_snapshot,
                      vcc, eta,
                      next_inc_dist_fr, use_vc);
        ++counters.spdag_expansion_node_count;
      }
    }
    const double backward_cost = EstimateBackwardCost(
        curr_depth, c_s_sp_level, eq_dist_fr, inc_dist_fr,
        D_s, DEG);
    const double forward_cost = ComputeForwardCost(
        eq_dist_fr, inc_dist_fr, c_s_sp_dag, c_s_sp_sc);
    // Choose between traversing stored DAG/shortcut edges and scanning the
    // current hub level for a predecessor reached at curr_depth - 1.
    if (forward_cost < backward_cost) {
      for (int64_t curr : eq_dist_fr) {
        NodeProp(static_cast<DistScalar>(curr_depth),
                      c_s_sp_dag[curr].children, D_s, D_mem_snapshot,
                      vcc, eta,
                      next_eq_dist_fr, false);
        ++counters.top_down_node_count;
        ++counters.spdag_expansion_node_count;
      }
      for (int64_t curr : inc_dist_fr) {
        NodeProp(static_cast<DistScalar>(curr_depth),
                      c_s_sp_sc[curr].children, D_s, D_mem_snapshot,
                      vcc, eta,
                      next_eq_dist_fr, use_vc);
        ++counters.top_down_node_count;
        ++counters.spill_expansion_node_count;
      }
    } else {
      for (int64_t curr : curr_level.nodes) {
        if (D_s(curr) <= curr_depth) continue;
        for (int64_t t : A[curr]) {
          if (D_s(t) == curr_depth - 1) {
            D_s(curr) = static_cast<DistScalar>(curr_depth);
            next_eq_dist_fr.push_back(curr);
            break;
          }
        }
        ++counters.bottom_up_node_count;
      }
    }
    inc_dist_fr.swap(next_inc_dist_fr);
    eq_dist_fr.swap(next_eq_dist_fr);
  }
  counters.vec_comp_count += static_cast<int64_t>(vcc);
  return D_s;
}

int64_t CompHubConstraint(int64_t n,
                          int64_t edge_slots,
                          double density,
                          const std::string& opt) {
  // Match canonical CC-BFS's named constraints and numeric conversion.
  const double avg_deg = static_cast<double>(edge_slots) / n;
  std::istringstream iss(opt);
  double hn_con = 0.0;
  if (iss >> hn_con && iss.eof()) {
    // Use the parsed numeric value directly before nearest-integer rounding.
  } else if (opt == "bounded-avg-deg") {
    hn_con = avg_deg > 0.0
        ? (std::min)(avg_deg, static_cast<double>(n) / avg_deg)
        : 0.0;
  } else if (opt == "avg-deg") {
    hn_con = avg_deg;
  } else if (opt == "sqrt-deg") {
    hn_con = std::sqrt(avg_deg);
  } else if (opt == "sqrt-inv-den") {
    hn_con = std::sqrt(std::ceil(1.0 / density));
  } else if (opt == "inv-den") {
    hn_con = std::ceil(1.0 / density);
  } else if (opt == "none") {
    hn_con = static_cast<double>(n);
  } else {
    throw std::runtime_error(
        "Unrecognized hub node constraint option: " + opt + ".");
  }
  return static_cast<int64_t>(std::round(hn_con));
}

void MergeWorkerCounters(Counters& output, const Counters& worker) {
  // Reduction is deliberately serial after the OpenMP join, avoiding atomics
  // in the search hot path and making aggregate publication deterministic.
  output.vec_comp_count += worker.vec_comp_count;
  output.source_with_cs_count += worker.source_with_cs_count;
  output.source_without_cs_count += worker.source_without_cs_count;
  output.graph_expansion_node_count += worker.graph_expansion_node_count;
  output.spdag_expansion_node_count += worker.spdag_expansion_node_count;
  output.spill_expansion_node_count += worker.spill_expansion_node_count;
  output.sptree_expansion_node_count += worker.sptree_expansion_node_count;
  output.cst_visited_count += worker.cst_visited_count;
  output.top_down_node_count += worker.top_down_node_count;
  output.bottom_up_node_count += worker.bottom_up_node_count;
}

template <typename DistScalar>
Result RunTyped(const AdjacencyList& A,
                const Eigen::VectorXd& DEG,
                const std::string& hub_constraint,
                int64_t num_threads,
                bool no_distances,
                bool memory_test_only,
                bool verbose,
                double adjacency_time,
                int64_t diameter,
                double diameter_time) {
  // Phase 0: initialize compatibility counters and handle trivial graphs.
  const int64_t n = static_cast<int64_t>(A.size());
  Result result;
  result.num_threads = num_threads;
  result.timings.adjacency = adjacency_time;
  Counters& counters = result.counters;
  counters.data_type = DistTypeName<DistScalar>();
  counters.time_secs_comp_adj = adjacency_time;
  counters.time_secs_diam_upperbound = diameter_time;
  counters.diam_upper = diameter;
  counters.adjacency_list_bytes = EstimateAdjacencyListBytes(A);

  Timer total_timer;
  total_timer.Start();
  if (n <= 1) {
    if (!no_distances && !memory_test_only) {
      DistMat<DistScalar> zero = DistMat<DistScalar>::Zero(n, n);
      counters.distance_matrix_bytes = DistMatAllocatedBytes(zero);
      result.distances = std::move(zero);
    }
    // Stop as soon as the native APSP result is ready; accounting and
    // destruction below are deliberately outside the common total.
    total_timer.Stop();
    result.timings.total = adjacency_time + diameter_time +
        total_timer.Seconds();
    counters.time_secs_overall = result.timings.total;
    counters.estimated_dominant_memory_bytes =
        counters.distance_matrix_bytes;
    UpdateTotalMemoryCounter(counters);
    return result;
  }

  // Phase 1: reproduce canonical hub/search parameters and deterministic
  // degree-descending source priority (vertex ID breaks degree ties).
  const int64_t edge_slots = static_cast<int64_t>(DEG.sum());
  const double density = static_cast<double>(edge_slots) /
      static_cast<double>(n * (n - 1));
  const double eta =
      std::sqrt(static_cast<double>(edge_slots) / n);
  const DistScalar dist_inf =
      static_cast<DistScalar>(diameter);
  const int64_t mu = CompHubConstraint(
      n, edge_slots, density, hub_constraint);
  const std::vector<int64_t> hub_nodes = SelectHubNodes(
      A, DEG, mu);
  counters.hn_count = static_cast<int64_t>(hub_nodes.size());

  std::vector<char> hn_mask(static_cast<size_t>(n), false);
  for (int64_t s : hub_nodes) hn_mask[s] = true;
  std::vector<int64_t> source_list(static_cast<size_t>(n));
  std::iota(source_list.begin(), source_list.end(), 0);
  std::sort(source_list.begin(), source_list.end(),
            [&](int64_t left, int64_t right) {
              if (DEG(left) != DEG(right)) {
                return DEG(left) > DEG(right);
              }
              return left < right;
            });

  // The D_mem cache is the only cross-source memoization state. With output
  // enabled, source-indexed fixed slots make final conversion a move. With
  // `nd`, match sequential CC-BFS by retaining n - (# degree-1 vertices)
  // vectors, raised when necessary to retain every selected hub. Slots are
  // preassigned in the same hub-first, degree-descending processing order so
  // workers still write distinct columns without locks.
  // Sources beyond that compact capacity are searched normally but are not
  // published as memoization state, exactly as when the sequential cache fills.
  // Hub child buffers use one private heap resource per logical worker. The
  // arenas outlive the grids below, including on exception unwinding.
  std::vector<std::unique_ptr<WorkerArenaResource>> worker_arenas;
  worker_arenas.reserve(static_cast<size_t>(num_threads));
  for (int64_t worker = 0; worker < num_threads; ++worker) {
    worker_arenas.emplace_back(std::make_unique<WorkerArenaResource>());
  }
  std::vector<int64_t> nd_fixed_slot_sources;
  if (no_distances) {
    const int64_t degree_one_count = static_cast<int64_t>(std::count_if(
        DEG.data(), DEG.data() + n,
        [](double degree) { return degree == 1.0; }));
    const int64_t nd_capacity = std::max<int64_t>(
        n - degree_one_count, static_cast<int64_t>(hub_nodes.size()));
    nd_fixed_slot_sources.reserve(static_cast<size_t>(nd_capacity));
    for (int64_t s : hub_nodes) {
      if (static_cast<int64_t>(nd_fixed_slot_sources.size()) == nd_capacity) {
        break;
      }
      nd_fixed_slot_sources.push_back(s);
    }
    for (int64_t s : source_list) {
      if (static_cast<int64_t>(nd_fixed_slot_sources.size()) == nd_capacity) {
        break;
      }
      if (!hn_mask[s]) nd_fixed_slot_sources.push_back(s);
    }
    if (static_cast<int64_t>(nd_fixed_slot_sources.size()) != nd_capacity) {
      throw std::runtime_error(
          "CC-BFS-P could not assign the compact nd distance cache.");
    }
  }
  DistVecCache<DistScalar> D_mem = no_distances
      ? DistVecCache<DistScalar>(n, nd_fixed_slot_sources)
      : DistVecCache<DistScalar>(n, n, true);
  std::vector<std::vector<GraphNode>> sp_dags(static_cast<size_t>(n));
  std::vector<std::vector<GraphNode>> sp_scs(static_cast<size_t>(n));
  std::vector<std::vector<GraphNode>> sp_trees(static_cast<size_t>(n));
  std::vector<std::vector<Level>> sp_levels(static_cast<size_t>(n));
  std::vector<double> eccs(static_cast<size_t>(n),
                                     static_cast<double>(n));

  // Phase 2: finish all hub searches before parallel non-hub work. Source IDs
  // are unique, so each worker writes directly to one fixed distance column
  // and that hub's DAG/shortcut/tree/level/eccentricity slots. Outer storage
  // is never resized by workers, and no result is copied through a private
  // full-size hub object. Availability remains false until after the join.
  // setupOpenMP configures the requested size. This hub team finishes before
  // the unchanged non-hub team starts; neither region is nested, and no
  // worker-reachable MATLAB API exists.
  setupOpenMP(true, num_threads, verbose);
  Timer hub_timer;
  hub_timer.Start();
  std::vector<int> hub_owner_workers(hub_nodes.size(), -1);
  std::vector<std::exception_ptr> hub_failures(hub_nodes.size());
  #pragma omp parallel for
  for (int64_t slot = 0;
       slot < static_cast<int64_t>(hub_nodes.size()); ++slot) {
    try {
      const int64_t s = hub_nodes[slot];
      const int worker = omp_get_thread_num();
      hub_owner_workers[slot] = worker;
      WorkerArenaResource* arena = worker_arenas[worker].get();
      // The fixed row headers remain ordinary vectors. Their nested integer
      // buffers allocate only from the private arena of the building worker.
      InitializeArenaRow(sp_dags[s], n, arena);
      InitializeArenaRow(sp_scs[s], n, arena);
      InitializeArenaRow(sp_trees[s], n, arena);
      InitializeArenaRow(sp_levels[s], n, arena);
      BuildSPStructures<DistScalar>(
          A, s, D_mem.GetFixedSlotForWrite(s), dist_inf,
          sp_dags[s], sp_scs[s], sp_trees[s], sp_levels[s],
          arena, eccs[s]);
      const int64_t completed_hubs = slot + 1;
      if ((completed_hubs & 63) == 0 ||
          completed_hubs == static_cast<int64_t>(hub_nodes.size())) {
        #pragma omp critical(cc_bfs_p_memory_checkpoint)
        {
          UpdateProcessMemoryPeakCheckpoint();
        }
      }
    } catch (...) {
      hub_failures[slot] = std::current_exception();
    }
  }

  // The implicit join completes every hub slot. A failure is rethrown before
  // any distance column is published; otherwise the original hub order is the
  // sole deterministic visibility step.
  for (const auto& failure : hub_failures) {
    if (failure) std::rethrow_exception(failure);
  }
  for (int64_t s : hub_nodes) {
    D_mem.CommitFixedSlot(s);
  }
  hub_timer.Stop();
  result.timings.hub = hub_timer.Seconds();
  counters.time_secs_hn_sss = result.timings.hub;
  UpdateProcessMemoryPeakCheckpoint();

  // Match sequential CC-BFS's memory-test boundary: retain the complete hub
  // distance cache and shortest-path structures, but do not allocate worker
  // scratch, run non-hub searches, or materialize a final result matrix.
  if (memory_test_only) {
    total_timer.Stop();
    result.timings.total = adjacency_time + diameter_time +
        total_timer.Seconds();
    counters.time_secs_overall = result.timings.total;
    UpdateProcessMemoryPeakCheckpoint();

    counters.distance_matrix_bytes =
        DistVecCacheAllocatedBytes(D_mem);
    counters.sp_dag_bytes = EstimateGraphNodeGridBytes(sp_dags);
    counters.sp_shortcut_bytes = EstimateGraphNodeGridBytes(sp_scs);
    counters.sp_tree_bytes = EstimateGraphNodeGridBytes(sp_trees);
    counters.estimated_dominant_memory_bytes =
        counters.distance_matrix_bytes + counters.sp_dag_bytes +
        counters.sp_shortcut_bytes + counters.sp_tree_bytes;
    UpdateTotalMemoryCounter(counters);

    if (verbose) {
      std::cout << "CC-BFS-P memory test threads="
                << num_threads << " hub=" << result.timings.hub
                << std::endl;
    }
    DestroyHubStructuresInWorkerArenas(
        hub_nodes, hub_owner_workers, worker_arenas,
        sp_levels, sp_trees, sp_scs, sp_dags);
    return result;
  }

  std::vector<int64_t> non_hub_sources;
  non_hub_sources.reserve(
      static_cast<size_t>(n - static_cast<int64_t>(hub_nodes.size())));
  for (int64_t s : source_list) {
    if (!hn_mask[s]) non_hub_sources.push_back(s);
  }

  // Phase 3: process a small multiple of the team size per batch. One OpenMP
  // team persists across all batches. Dynamic scheduling balances uneven
  // sources, while fixed source columns preserve deterministic output layout.
  const int64_t batch_size = std::max<int64_t>(1, num_threads * 4);
  const int64_t peak_batch_sources = std::min<int64_t>(
      batch_size, static_cast<int64_t>(non_hub_sources.size()));
  Timer non_hub_timer;
  non_hub_timer.Start();
  std::vector<std::exception_ptr> failures(
      static_cast<size_t>(peak_batch_sources));
  std::exception_ptr batch_failure;
  const int worker_count = omp_get_max_threads();
  std::vector<Counters> worker_counters(
      static_cast<size_t>(worker_count));
  std::vector<WorkerScratch<DistScalar>> worker_scratch;
  worker_scratch.reserve(static_cast<size_t>(worker_count));
  for (int worker = 0; worker < worker_count; ++worker) {
    worker_scratch.emplace_back(n);
  }

  #pragma omp parallel shared(batch_failure, failures, D_mem)
  {
    const int worker = omp_get_thread_num();
    Counters& local_counters = worker_counters[worker];
    // Dynamic scheduling may give a worker many sources over many batches.
    // Its scratch is indexed by thread ID, never shared, and lives until the
    // persistent team finishes, so clear()/setConstant() reset logical state
    // while vector capacity is retained for the next assigned source.
    WorkerScratch<DistScalar>& scratch = worker_scratch[worker];
    for (int64_t begin = 0;
         begin < static_cast<int64_t>(non_hub_sources.size());
         begin += batch_size) {
      const int64_t end = std::min<int64_t>(
          static_cast<int64_t>(non_hub_sources.size()), begin + batch_size);
      const int64_t current_batch_size = end - begin;
      const DistVecCache<DistScalar>& D_mem_snapshot = D_mem;

      // Each slot has one writer even though execution order is dynamic. A
      // completed vector is copied to its distinct fixed column, but remains
      // invisible because its availability flag is still false.
      #pragma omp for schedule(dynamic, 1)
      for (int64_t slot = 0; slot < current_batch_size; ++slot) {
        failures[static_cast<size_t>(slot)] = nullptr;
        try {
          const int64_t s = non_hub_sources[begin + slot];
          const int64_t c_s = FindHighestDegreeNeighborHub(
              A, hn_mask, DEG, s);
          if (c_s != -1) {
            HubGuidedSSS(
                A, s, eta, c_s,
                D_mem_snapshot, dist_inf, sp_dags, sp_scs,
                sp_trees, sp_levels, eccs, DEG, local_counters,
                scratch);
          } else {
            MemoizedSSS(
                A, s, eta, D_mem_snapshot,
                dist_inf, local_counters, scratch);
          }
          D_mem.StoreUncommittedFixedSlot(s, scratch.D_s);
          const int64_t completed_non_hubs = begin + slot + 1;
          if ((completed_non_hubs & 63) == 0 ||
              completed_non_hubs ==
                  static_cast<int64_t>(non_hub_sources.size())) {
            #pragma omp critical(cc_bfs_p_memory_checkpoint)
            {
              UpdateProcessMemoryPeakCheckpoint();
            }
          }
        } catch (...) {
          failures[static_cast<size_t>(slot)] = std::current_exception();
        }
      }

      // The omp-for barrier completes every column copy. A single thread first
      // checks the whole batch, then publishes availability in deterministic
      // source order. A failed batch publishes no availability at all.
      #pragma omp single
      {
        for (int64_t slot = 0; slot < current_batch_size; ++slot) {
          if (failures[static_cast<size_t>(slot)]) {
            batch_failure = failures[static_cast<size_t>(slot)];
            break;
          }
        }
        if (!batch_failure) {
          Timer commit_timer;
          commit_timer.Start();
          for (int64_t slot = 0; slot < current_batch_size; ++slot) {
            D_mem.CommitFixedSlot(non_hub_sources[begin + slot]);
          }
          commit_timer.Stop();
          result.timings.commit_sync += commit_timer.Seconds();
        }
      }
      // The single-region barrier makes the availability commit visible to
      // every worker before the next batch snapshots the cache.
      if (batch_failure) break;
    }
  }
  if (batch_failure) std::rethrow_exception(batch_failure);
  for (const Counters& worker : worker_counters) {
    MergeWorkerCounters(counters, worker);
  }
  UpdateProcessMemoryPeakCheckpoint();
  non_hub_timer.Stop();
  result.timings.non_hub = non_hub_timer.Seconds();
  counters.time_secs_non_hn_sss = result.timings.non_hub;
  UpdateProcessMemoryPeakCheckpoint();

  // Phase 4: materialize the source-per-column result unless `nd` requested
  // computation/counters without a distance return.
  if (!no_distances) {
    Timer conversion_timer;
    conversion_timer.Start();
    DistMat<DistScalar> distances;
    ConvertDistVecCacheToMatrix(D_mem, distances);
    conversion_timer.Stop();
    result.timings.conversion = conversion_timer.Seconds();
    counters.time_secs_convert = result.timings.conversion;
    counters.distance_matrix_bytes = DistMatAllocatedBytes(distances);
    result.distances = ResultMatrix{std::move(distances)};
    UpdateProcessMemoryPeakCheckpoint();
  } else {
    counters.distance_matrix_bytes =
        DistVecCacheAllocatedBytes(D_mem);
  }

  // The common total ends at native result completion. MATLAB conversion,
  // memory-accounting passes, and later object destruction are excluded.
  total_timer.Stop();
  result.timings.total = adjacency_time + diameter_time +
      total_timer.Seconds();
  counters.time_secs_overall = result.timings.total;

  // Dominant-live estimate: D_mem result/cache + all hub structures + the
  // fixed reusable distance and generation arrays owned by every worker.
  // Frontier capacity follows the same bounded per-source search storage as
  // batched execution and remains excluded from this estimate. Adjacency is
  // added separately; process peaks are filled by the MEX-scoped OS tracker.
  counters.sp_dag_bytes = EstimateGraphNodeGridBytes(sp_dags);
  counters.sp_shortcut_bytes = EstimateGraphNodeGridBytes(sp_scs);
  counters.sp_tree_bytes = EstimateGraphNodeGridBytes(sp_trees);
  const int64_t private_worker_bytes =
      static_cast<int64_t>(worker_count) * n *
      static_cast<int64_t>(sizeof(DistScalar) + sizeof(uint32_t));
  counters.estimated_dominant_memory_bytes =
      counters.distance_matrix_bytes + counters.sp_dag_bytes +
      counters.sp_shortcut_bytes + counters.sp_tree_bytes +
      private_worker_bytes;
  UpdateTotalMemoryCounter(counters);

  if (verbose) {
    std::cout << "CC-BFS-P threads=" << num_threads
              << " hub=" << result.timings.hub
              << " non_hub=" << result.timings.non_hub
              << " commit_sync=" << result.timings.commit_sync
              << " conversion=" << result.timings.conversion
              << std::endl;
  }
  // Runtime and memory accounting have ended, and no later search reads the
  // hub structures. Each logical owner now releases its private arena.
  DestroyHubStructuresInWorkerArenas(
      hub_nodes, hub_owner_workers, worker_arenas,
      sp_levels, sp_trees, sp_scs, sp_dags);
  return result;
}

}  // namespace

int64_t DiamUpperBound(const AdjacencyList& A,
                       const Eigen::VectorXd& DEG) {
  return DiamUpperBoundImpl(A, DEG);
}

void ValidateThreadCount(int64_t num_threads) {
  if (num_threads < 1) {
    throw std::runtime_error("num_threads must be a positive integer.");
  }
  if (static_cast<size_t>(num_threads) > GetLogicalProcessorCount()) {
    throw std::runtime_error(
        "num_threads exceeds the available logical-processor count.");
  }
}

Result Run(const AdjacencyList& A,
           const Eigen::VectorXd& DEG,
           const std::string& hub_constraint,
           int64_t num_threads,
           bool no_distances,
           bool memory_test_only,
           bool verbose,
           double adjacency_time,
           int64_t diameter,
           double diameter_time) {
  // Reserve the exact range for sums of two bounded distances.
  if (diameter < 0) {
    throw std::invalid_argument("Diameter upper bound must be nonnegative");
  }
  const uint64_t unsigned_diam_upper = static_cast<uint64_t>(diameter);
  if (unsigned_diam_upper > (std::numeric_limits<uint64_t>::max)() / 2) {
    throw std::overflow_error("Twice the diameter upper bound exceeds uint64 capacity");
  }
  const uint64_t dtype_max_required = 2 * unsigned_diam_upper;
  if (dtype_max_required <= (std::numeric_limits<uint8_t>::max)()) {
    return RunTyped<uint8_t>(
        A, DEG, hub_constraint, num_threads, no_distances,
        memory_test_only, verbose, adjacency_time, diameter, diameter_time);
  }
  if (dtype_max_required <= (std::numeric_limits<uint16_t>::max)()) {
    return RunTyped<uint16_t>(
        A, DEG, hub_constraint, num_threads, no_distances,
        memory_test_only, verbose, adjacency_time, diameter, diameter_time);
  }
  if (dtype_max_required <= (std::numeric_limits<uint32_t>::max)()) {
    return RunTyped<uint32_t>(
        A, DEG, hub_constraint, num_threads, no_distances,
        memory_test_only, verbose, adjacency_time, diameter, diameter_time);
  }
  return RunTyped<uint64_t>(
      A, DEG, hub_constraint, num_threads, no_distances,
      memory_test_only, verbose, adjacency_time, diameter, diameter_time);
}

#ifdef MATLAB_BUILD
mxArray* CreateCountersStruct(const Result& result) {
  // Retain canonical field names and append explicit variant/thread/phase
  // metadata so MATLAB experiments can consume all targets uniformly.
  const char* fields[] = {
      "vec_comp_count",
      "source_with_cs_count",
      "source_without_cs_count",
      "hn_count",
      "graph_expansion_node_count",
      "spdag_expansion_node_count",
      "spill_expansion_node_count",
      "sptree_expansion_node_count",
      "cst_visited_count",
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
      "process_delta_peak_private_bytes",
      "variant",
      "num_threads",
      "time_secs_adjacency",
      "time_secs_hub",
      "time_secs_non_hub",
      "time_secs_commit_sync",
      "time_secs_conversion"};
  mxArray* output = mxCreateStructMatrix(
      1, 1, static_cast<int>(sizeof(fields) / sizeof(fields[0])), fields);
  const Counters& counters = result.counters;
  auto set_number = [&](const char* field, double value) {
    mxSetField(output, 0, field, mxCreateDoubleScalar(value));
  };
  set_number("vec_comp_count", counters.vec_comp_count);
  set_number("source_with_cs_count", counters.source_with_cs_count);
  set_number("source_without_cs_count", counters.source_without_cs_count);
  set_number("hn_count", counters.hn_count);
  set_number("graph_expansion_node_count",
             counters.graph_expansion_node_count);
  set_number("spdag_expansion_node_count",
             counters.spdag_expansion_node_count);
  set_number("spill_expansion_node_count",
             counters.spill_expansion_node_count);
  set_number("sptree_expansion_node_count",
             counters.sptree_expansion_node_count);
  set_number("cst_visited_count", counters.cst_visited_count);
  set_number("top_down_node_count", counters.top_down_node_count);
  set_number("bottom_up_node_count", counters.bottom_up_node_count);
  set_number("time_secs_comp_adj", counters.time_secs_comp_adj);
  set_number("time_secs_diam_upperbound",
             counters.time_secs_diam_upperbound);
  set_number("time_secs_hn_sss", counters.time_secs_hn_sss);
  set_number("time_secs_non_hn_sss", counters.time_secs_non_hn_sss);
  set_number("time_secs_convert", counters.time_secs_convert);
  set_number("time_secs_overall", counters.time_secs_overall);
  set_number("diam_upper", counters.diam_upper);
  mxSetField(output, 0, "data_type",
             mxCreateString(counters.data_type.c_str()));
  set_number("estimated_dominant_memory_bytes",
             static_cast<double>(counters.estimated_dominant_memory_bytes));
  set_number("distance_matrix_bytes",
             static_cast<double>(counters.distance_matrix_bytes));
  set_number("sp_dag_bytes", static_cast<double>(counters.sp_dag_bytes));
  set_number("sp_shortcut_bytes",
             static_cast<double>(counters.sp_shortcut_bytes));
  set_number("sp_tree_bytes", static_cast<double>(counters.sp_tree_bytes));
  set_number("adjacency_list_bytes",
             static_cast<double>(counters.adjacency_list_bytes));
  set_number("estimated_total_memory_bytes",
             static_cast<double>(counters.estimated_total_memory_bytes));
  set_number("process_baseline_working_set_bytes",
             static_cast<double>(
                 counters.process_baseline_working_set_bytes));
  set_number("process_peak_working_set_bytes",
             static_cast<double>(counters.process_peak_working_set_bytes));
  set_number("process_delta_peak_working_set_bytes",
             static_cast<double>(
                 counters.process_delta_peak_working_set_bytes));
  set_number("process_baseline_private_bytes",
             static_cast<double>(counters.process_baseline_private_bytes));
  set_number("process_peak_private_bytes",
             static_cast<double>(counters.process_peak_private_bytes));
  set_number("process_delta_peak_private_bytes",
             static_cast<double>(
                 counters.process_delta_peak_private_bytes));
  mxSetField(output, 0, "variant", mxCreateString("cc_bfs_p"));
  set_number("num_threads", result.num_threads);
  set_number("time_secs_adjacency", result.timings.adjacency);
  set_number("time_secs_hub", result.timings.hub);
  set_number("time_secs_non_hub", result.timings.non_hub);
  set_number("time_secs_commit_sync", result.timings.commit_sync);
  set_number("time_secs_conversion", result.timings.conversion);
  return output;
}
#endif

}  // namespace parallel_cc::cc_bfs_p

#ifdef MATLAB_BUILD
namespace {

struct MexOptions {
  int64_t num_threads = 1;
  std::string hub_constraint = "bounded-avg-deg";
  bool no_distances = false;
  bool no_distance_return = false;
  bool memory_test_only = false;
  bool verbose = true;
};

// MEX contract:
//   [D,t,c] = cc_bfs_p(A, [num_threads],
//       [hub_constraint], ['nd'], ['ndr'], ['nv'], ['mem_test_only'])
// A is a nonempty square binary adjacency matrix accepted by the shared
// parser. `nd` skips distance materialization and uses sequential CC-BFS's
// compact cache; `ndr` performs full conversion but suppresses only the MATLAB
// return; `mem_test_only` stops after hub preprocessing; `nv` disables timing.
// Other strings select the hub constraint.
MexOptions ParseMexOptions(int nrhs, const mxArray* prhs[]) {
  MexOptions options;
  bool thread_count_seen = false;
  for (int index = 1; index < nrhs; ++index) {
    if (mxIsNumeric(prhs[index]) && !mxIsComplex(prhs[index]) &&
        mxGetNumberOfElements(prhs[index]) == 1) {
      if (thread_count_seen) {
        throw std::runtime_error("num_threads may be specified only once.");
      }
      const double raw = mxGetScalar(prhs[index]);
      if (!std::isfinite(raw) || raw != std::floor(raw) || raw < 1.0 ||
          raw > static_cast<double>(
                    (std::numeric_limits<int64_t>::max)())) {
        throw std::runtime_error(
            "num_threads must be a positive finite integer scalar.");
      }
      options.num_threads = static_cast<int64_t>(raw);
      thread_count_seen = true;
      continue;
    }
    if (!mxIsChar(prhs[index])) {
      throw std::runtime_error(
          "Options must be a numeric thread count or character vectors.");
    }
    char* raw = mxArrayToString(prhs[index]);
    if (raw == nullptr) {
      throw std::runtime_error("Failed to read a character option.");
    }
    const std::string option(raw);
    mxFree(raw);
    if (option == "nd") {
      options.no_distances = true;
    } else if (option == "ndr") {
      options.no_distance_return = true;
    } else if (option == "mem_test_only") {
      options.memory_test_only = true;
    } else if (option == "nv") {
      options.verbose = false;
    } else if (!option.empty()) {
      options.hub_constraint = option;
    }
  }
  return options;
}

}  // namespace

/**
 * MATLAB MEX entry point for parallel CC-BFS APSP on an undirected,
 * unweighted graph.
 *
 * Usage:
 *   [D, t, counters] = cc_bfs_p(A)
 *   [D, t, counters] = cc_bfs_p(A, num_threads)
 *   [D, t, counters] = cc_bfs_p(A, hub_constraint)
 *   [D, t, counters] = cc_bfs_p(A, num_threads, hub_constraint, flags...)
 *
 * Inputs:
 *   A                Nonempty square binary adjacency matrix. Real double or
 *                    logical matrices, either dense or sparse, are accepted;
 *                    double entries must be finite zeros or ones. The caller
 *                    must supply a symmetric matrix for the documented
 *                    undirected-graph contract.
 *   num_threads      Optional positive finite integer numeric scalar not
 *                    exceeding the detected logical-processor count. Defaults
 *                    to 1 and may be specified at most once.
 *   hub_constraint   Optional character vector passed to the CC-BFS hub
 *                    constraint selector. Defaults to "bounded-avg-deg",
 *                    which computes min(avg_deg, n / avg_deg). The resulting
 *                    floating-point constraint is rounded to the nearest
 *                    integer. "avg-deg", "sqrt-deg", "sqrt-inv-den",
 *                    "inv-den", "none", and numeric character vectors are
 *                    also accepted. Any nonempty character option other than
 *                    a recognized flag replaces the constraint.
 *   "nd"             Optional character-vector flag that suppresses final
 *                    result-matrix materialization and uses the same compact
 *                    cache as sequential CC-BFS: n - (# degree-1 vertices),
 *                    raised when necessary to retain all selected hubs.
 *   "ndr"            Optional character-vector flag that performs the normal
 *                    distance computation but suppresses its MATLAB return.
 *   "mem_test_only"  Optional character-vector flag that stops after all hub
 *                    shortest-path structures have been built. Non-hub
 *                    searches and final result materialization are skipped.
 *   "nv"             Optional character-vector flag that suppresses console
 *                    progress and timing output.
 *
 * Outputs:
 *   D         All-pairs shortest-path distance matrix in the adaptive class
 *             selected by the algorithm (`uint8`, `uint16`, `uint32`, or
 *             `uint64`). With `nd`, `ndr`, or `mem_test_only`, returns a dummy
 *             double scalar zero.
 *   t         Overall computation time in seconds, including adjacency
 *             conversion and diameter-bound computation. Returns NaN with
 *             `mem_test_only`; measured phase times remain in `counters`.
 *   counters  Scalar struct containing algorithm, phase-timing, theoretical-
 *             memory, sampled process-memory, variant, and thread counters.
 *
 * Notes:
 *   Optional arguments may be supplied in any order after `A`. `nd` skips the
 *   final result object and uses sequential CC-BFS's compact cache; `ndr`
 *   preserves the full normal cache and changes only the MATLAB return.
 *   `mem_test_only` measures the live hub-stage structures relative to
 *   the process-memory baseline captured before adjacency-list and
 *   degree-vector construction. MATLAB API access remains on the MEX entry
 *   thread.
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs < 1) {
    mexErrMsgIdAndTxt(
        "FastUnweightedAPSP:cc_bfs_p_args",
        "Usage: [D,t,c] = cc_bfs_p(A, num_threads, "
        "[hub_constraint], ['nd'], ['ndr'], ['nv'], "
        "['mem_test_only']).");
  }
  try {
    namespace Target = parallel_cc::cc_bfs_p;
    const MexOptions options = ParseMexOptions(nrhs, prhs);
    Target::ValidateThreadCount(options.num_threads);

    // All MATLAB API work stays on the entry thread: validate/parse first,
    // then call the MATLAB-independent kernel, then create outputs afterward.
    ProcessMemoryPeakTracker memory_tracker;
    memory_tracker.CaptureBaseline();
    Target::Result result;
    {
      ProcessMemoryTrackerScope tracker_scope(&memory_tracker);
      parallel_cc::matlab::ValidateBinaryAdjacencyMatrix(
          prhs[0], (std::numeric_limits<int64_t>::max)());
      Timer adjacency_timer;
      adjacency_timer.Start();
      Target::AdjacencyList A =
          parallel_cc::matlab::ParseValidatedBinaryAdjacencyList(prhs[0]);
      const int64_t n = static_cast<int64_t>(A.size());
      Eigen::VectorXd DEG(n);
      for (int64_t vertex = 0; vertex < n; ++vertex) {
        DEG(vertex) = static_cast<double>(A[vertex].size());
      }
      adjacency_timer.Stop();
      UpdateProcessMemoryPeakCheckpoint();

      Timer diameter_timer;
      diameter_timer.Start();
      const int64_t diameter = n <= 1
          ? 0
          : Target::DiamUpperBound(A, DEG);
      diameter_timer.Stop();
      result = Target::Run(
          A, DEG, options.hub_constraint, options.num_threads,
          options.no_distances, options.memory_test_only, options.verbose,
          adjacency_timer.Seconds(), diameter, diameter_timer.Seconds());
    }
    memory_tracker.UpdatePeak();
    FillProcessMemoryCounters(result.counters, memory_tracker);

    // Outputs are distance matrix (or scalar zero when suppressed), total
    // seconds including parse/bound phases, and the counters/phase struct.
    if (nlhs > 0) {
      if (options.no_distances || options.no_distance_return ||
          options.memory_test_only) {
        plhs[0] = mxCreateDoubleScalar(0.0);
      } else {
        plhs[0] = CreateMatlabMatrixFromResult(result.distances);
      }
    }
    if (nlhs > 1) {
      const double output_time = options.memory_test_only
          ? std::numeric_limits<double>::quiet_NaN()
          : result.timings.total;
      plhs[1] = mxCreateDoubleScalar(output_time);
    }
    if (nlhs > 2) {
      plhs[2] = Target::CreateCountersStruct(result);
    }
  } catch (const std::exception& error) {
    mexErrMsgIdAndTxt(
        "FastUnweightedAPSP:cc_bfs_p_runtime",
        "%s", error.what());
  } catch (...) {
    mexErrMsgIdAndTxt(
        "FastUnweightedAPSP:cc_bfs_p_runtime",
        "Unknown C++ exception.");
  }
}
#endif
