#ifndef VRP_ALGORITHMS_TRANSITIONS_FACTORIES_HPP
#define VRP_ALGORITHMS_TRANSITIONS_FACTORIES_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

#include <thrust/execution_policy.h>

namespace vrp {
namespace algorithms {
namespace transitions {

/// Creates transition between customers.
struct create_transition final {

  const vrp::models::Problem::Shadow problem;
  const vrp::models::Tasks::Shadow tasks;

  // TODO remove this method to avoid confusion
  __host__ __device__ vrp::models::Transition operator()(
    const vrp::models::Transition::Details& details) const;

  __host__ __device__ vrp::models::Transition operator()(
    const vrp::models::Transition::Details& details,
    const vrp::models::Transition::State& state) const;

};

}  // namespace transitions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_TRANSITIONS_FACTORIES_HPP
