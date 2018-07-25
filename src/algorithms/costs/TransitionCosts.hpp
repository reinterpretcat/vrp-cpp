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
  const vrp::models::Problem::Shadow problem;
  const vrp::models::Tasks::Shadow tasks;

  ANY_EXEC_UNIT float operator()(const vrp::models::Transition& transition) const;
};

}  // namespace costs
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_COSTS_TRANSITIONCOSTS_HPP
