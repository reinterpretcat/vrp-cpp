#ifndef VRP_HEURISTICS_MODELS_HPP
#define VRP_HEURISTICS_MODELS_HPP

#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Factories.hpp"

#include <models/Transition.hpp>
#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Stores transition and its cost as single model.
using TransitionCostModel = thrust::pair<vrp::models::Transition, float>;

/// Stores transition and cost operators as one.
using TransitionCostOp = thrust::pair<vrp::algorithms::transitions::create_transition,
                                      vrp::algorithms::costs::calculate_transition_cost>;

/// Specifies information about tasks for heuristic step.
struct Step {
  /// Base index.
  int base;
  /// Task to start from.
  int from;
  /// Task to finish.
  int to;
  /// Vehicle index.
  int vehicle;
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_MODELS_HPP
