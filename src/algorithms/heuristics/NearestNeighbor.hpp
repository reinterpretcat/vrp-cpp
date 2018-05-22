#ifndef VRP_HEURISTICS_NEARESTNEIGHBOR_HPP
#define VRP_HEURISTICS_NEARESTNEIGHBOR_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Implements algorithm of nearest neighbor heuristic.
struct nearest_neighbor final {
  const vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;

  __host__ __device__ nearest_neighbor(const vrp::models::Problem::Shadow problem,
                                       vrp::models::Tasks::Shadow tasks) :
    problem(problem),
    tasks(tasks) {}

  /// Finds the "nearest" transition for given task and vehicle
  __host__ __device__ vrp::models::TransitionCost operator()(int fromTask, int toTask, int vehicle);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_NEARESTNEIGHBOR_HPP
