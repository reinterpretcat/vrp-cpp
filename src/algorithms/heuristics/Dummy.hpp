#ifndef VRP_HEURISTICS_DUMMY_HPP
#define VRP_HEURISTICS_DUMMY_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// A dummy implementation of heuristic which returns an invalid transition and cost.
struct dummy final {
  __host__ __device__ dummy(const vrp::models::Problem::Shadow problem,
                            vrp::models::Tasks::Shadow tasks) {}

  __host__ __device__ vrp::models::Transition operator()(int fromTask, int toTask, int vehicle);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_DUMMY_HPP
