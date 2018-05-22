#ifndef VRP_ALGORITHMS_TRANSITIONS_EXECUTORS_HPP
#define VRP_ALGORITHMS_TRANSITIONS_EXECUTORS_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {
namespace transitions {

/// Performs transition with a cost.
struct perform_transition final {
  explicit perform_transition(const vrp::models::Tasks::Shadow& tasks) : tasks(tasks) {}

  __host__ __device__ void operator()(const vrp::models::TransitionCost& transitionCost) const;

private:
  vrp::models::Tasks::Shadow tasks;
};

}  // namespace transitions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_TRANSITIONS_EXECUTORS_HPP
