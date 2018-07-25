#ifndef VRP_HEURISTICS_OPERATORS_HPP
#define VRP_HEURISTICS_OPERATORS_HPP

#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "models/Convolution.hpp"
#include "models/Plan.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Returns invalid transition-cost model.
ANY_EXEC_UNIT inline TransitionCostModel create_invaild() {
  return thrust::make_pair(vrp::models::Transition(), -1);
}

/// Picks the next vehicle and assigns it to the task.
ANY_EXEC_UNIT inline void spawn_vehicle(const vrp::models::Problem::Shadow& problem,
                                        const vrp::models::Tasks::Shadow& tasks,
                                        int task,
                                        int vehicle) {
  tasks.times[task] = problem.customers.starts[0];
  tasks.capacities[task] = problem.resources.capacities[vehicle];
  tasks.costs[task] = problem.resources.fixedCosts[vehicle];
}

/// Creates transition and calculates cost for given customer if it is not handled.
struct create_cost_transition final {
  Step step;
  TransitionCostOp operators;
  const vrp::runtime::vector_ptr<vrp::models::Convolution> convolutions;

  ANY_EXEC_UNIT
  create_cost_transition(const Step& step,
                         const TransitionCostOp& operators,
                         const vrp::runtime::vector_ptr<vrp::models::Convolution>& convolutions) :
    step(step),
    operators(operators), convolutions(convolutions) {}

  ANY_EXEC_UNIT TransitionCostModel
  operator()(const thrust::tuple<int, vrp::models::Plan>& customer);
};

/// Compares costs of two transitions and returns the lowest.
struct compare_transition_costs {
  ANY_EXEC_UNIT TransitionCostModel& operator()(TransitionCostModel& result,
                                                const TransitionCostModel& left);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_OPERATORS_HPP
