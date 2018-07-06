#ifndef VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP
#define VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP

#include "models/Resources.hpp"
#include "models/Transition.hpp"

#include <models/Problem.hpp>

namespace vrp {
namespace algorithms {
namespace costs {

/// Calculates cost of transition.
struct calculate_transition_cost final {
  __host__ __device__ explicit calculate_transition_cost(const vrp::models::Problem::Shadow problem,
                                                         const vrp::models::Tasks::Shadow tasks) :
    problem(problem) {}

  __host__ __device__ float operator()(const vrp::models::Transition& transition) const;

private:
  const vrp::models::Problem::Shadow problem;
};

}  // namespace costs
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP
