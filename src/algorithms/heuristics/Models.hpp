#ifndef VRP_HEURISTICS_MODELS_HPP
#define VRP_HEURISTICS_MODELS_HPP

#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "models/Convolution.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

#include <thrust/device_ptr.h>
#include <thrust/pair.h>

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Heuristic context.
struct Context final {
  vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;
  vrp::runtime::vector_ptr<vrp::models::Convolution> convolutions;
};

/// Aggregates basic actions on transition.
template<typename TransitionFactory, typename CostFactory, typename TransitionExecutor>
struct TransitionDelegate final {
  TransitionFactory transitionFactory;
  CostFactory costFactory;
  TransitionExecutor transitionExecutor;

  ANY_EXEC_UNIT TransitionDelegate(const vrp::models::Problem::Shadow problem,
                                   vrp::models::Tasks::Shadow tasks) :
    transitionFactory{problem, tasks},
    costFactory{problem, tasks}, transitionExecutor{problem, tasks} {}

  /// Creates transition from details.
  ANY_EXEC_UNIT vrp::models::Transition create(
    const vrp::models::Transition::Details& details) const {
    return transitionFactory(details);
  }

  /// Creates transition from details and state.
  ANY_EXEC_UNIT vrp::models::Transition create(const vrp::models::Transition::Details& details,
                                               const vrp::models::Transition::State& state) const {
    return transitionFactory(details, state);
  }

  /// Estimates cost of performing transition.
  ANY_EXEC_UNIT float estimate(const vrp::models::Transition& transition) const {
    return costFactory(transition);
  }

  /// Performs transition within cost and returns last task.
  ANY_EXEC_UNIT int perform(const vrp::models::Transition& transition, float cost) const {
    return transitionExecutor(transition, cost);
  }
};

/// Transition operator alias.
using TransitionOperator = TransitionDelegate<vrp::algorithms::transitions::create_transition,
                                              vrp::algorithms::costs::calculate_transition_cost,
                                              vrp::algorithms::transitions::perform_transition>;

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_MODELS_HPP
