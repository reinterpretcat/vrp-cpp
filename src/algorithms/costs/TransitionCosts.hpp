#ifndef VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP
#define VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP

#include "models/Transition.hpp"
#include "models/Resources.hpp"

namespace vrp {
namespace algorithms {
namespace costs {

/// Calculates cost of transition.
struct calculate_transition_cost final {
  const vrp::models::Resources::Shadow resources;

  __host__ __device__
  explicit calculate_transition_cost(const vrp::models::Resources::Shadow &resources) :
      resources(resources) {}

  __host__ __device__
  float operator()(const vrp::models::Transition &transition) const;

};

}
}
}

#endif //VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP
