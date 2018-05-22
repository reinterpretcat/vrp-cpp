#ifndef VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP
#define VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP

#include "models/Resources.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {
namespace costs {

/// Calculates cost of transition.
struct calculate_transition_cost final {
  const vrp::models::Resources::Shadow resources;

  __host__ __device__ explicit calculate_transition_cost(
    const vrp::models::Resources::Shadow& resources) :
    resources(resources) {}

  __host__ __device__ float operator()(const vrp::models::Transition& transition) const;
};

}  // namespace costs
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP
