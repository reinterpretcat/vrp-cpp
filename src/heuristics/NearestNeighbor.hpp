#ifndef VRP_HEURISTICS_NEARESTNEIGHBOR_HPP
#define VRP_HEURISTICS_NEARESTNEIGHBOR_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace heuristics {

/// Implements algorithm of cheapest insertion heuristic.
struct NearestNeighbor final {
  using TransitionCost = thrust::pair<vrp::models::Transition, float>;

  vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;

  /// Finds the "nearest" transition for given task
  __host__ __device__
  TransitionCost operator()(int task);
};

}
}

#endif //VRP_HEURISTICS_NEARESTNEIGHBOR_HPP
