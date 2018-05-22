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
  __host__ __device__ explicit create_transition(const vrp::models::Problem::Shadow& problem,
                                                 const vrp::models::Tasks::Shadow tasks) :
    problem(problem),
    tasks(tasks) {}

  __host__ __device__ vrp::models::Transition operator()(
    const vrp::models::Transition::Details& details) const;

private:
  const vrp::models::Problem::Shadow problem;
  const vrp::models::Tasks::Shadow tasks;
};

}  // namespace transitions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_TRANSITIONS_FACTORIES_HPP
