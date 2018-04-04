#ifndef VRP_HEURISTICS_NEARESTNEIGHBOR_HPP
#define VRP_HEURISTICS_NEARESTNEIGHBOR_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace heuristics {

/// Implements algorithm of cheapest insertion heuristic.
struct NearestNeighbor final {

  const vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;

  __host__ __device__
  NearestNeighbor(const vrp::models::Problem::Shadow problem,
                  vrp::models::Tasks::Shadow tasks) :
    problem(problem), tasks(tasks) {}

  /// Finds the "nearest" transition for given task and vehicle
  __host__ __device__
  vrp::models::TransitionCost operator()(int fromTask, int toTask, int vehicle);
};

}
}

#endif //VRP_HEURISTICS_NEARESTNEIGHBOR_HPP
