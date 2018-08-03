#include "NearestNeighbor.hpp"
#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/transitions/Factories.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

/// Specifies information about tasks for heuristic step.
struct Step final {
  /// Base index.
  int base;
  /// Task to start from.
  int from;
  /// Task to finish.
  int to;
  /// Vehicle index.
  int vehicle;
};

/// Stores transition and its cost as single model.
using TransitionCostModel = thrust::pair<vrp::models::Transition, float>;

/// Stores transition and cost operators as one.
using TransitionCostOp = thrust::pair<vrp::algorithms::transitions::create_transition,
                                      vrp::algorithms::costs::calculate_transition_cost>;

/// Returns invalid transition-cost model.
ANY_EXEC_UNIT inline TransitionCostModel create_invaild() {
  return thrust::make_pair(vrp::models::Transition(), -1);
}

/// Picks the next vehicle and assigns it to the task.
ANY_EXEC_UNIT void spawnNewVehicle(const Problem::Shadow& problem,
                                   const Tasks::Shadow& tasks,
                                   int task,
                                   int vehicle) {
  tasks.times[task] = problem.customers.starts[0];
  tasks.capacities[task] = problem.resources.capacities[vehicle];
  tasks.costs[task] = problem.resources.fixedCosts[vehicle];
}

/// Compares costs of two transitions and returns the lowest.
struct compare_transition_costs {
  ANY_EXEC_UNIT TransitionCostModel& operator()(TransitionCostModel& result,
                                                const TransitionCostModel& left) {
    if (left.first.isValid() && (left.second < result.second || !result.first.isValid())) {
      result = left;
    }

    return result;
  }
};

/// Creates transition and calculates cost for given customer if it is not handled.
template<typename TransitionOp>
struct create_cost_transition final {
  TransitionOp transitionOp;
  const vector_ptr<Convolution> convolutions;
  Step step;

  ANY_EXEC_UNIT TransitionCostModel operator()(const thrust::tuple<int, Plan>& customer) {
    auto plan = thrust::get<1>(customer);

    if (plan.isAssigned()) return create_invaild();

    auto wrapped =
      plan.hasConvolution()
        ? variant<int, Convolution>::create<Convolution>(*(convolutions + plan.convolution()))
        : variant<int, Convolution>::create<int>(thrust::get<0>(customer));

    auto transition = transitionOp.create({step.base, step.from, step.to, wrapped, step.vehicle});

    float cost = transition.isValid() ? transitionOp.estimate(transition) : -1;

    return thrust::make_pair(transition, cost);
  }
};

/// Finds next transition.
template<typename TransitionOp>
struct find_next_transition final {
  const Context context;
  TransitionOp transitionOp;

  /// Finds next transition
  ANY_EXEC_UNIT Transition operator()(const Step& step) {
    return thrust::transform_reduce(
             exec_unit_policy{},
             thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),
                                                          context.tasks.plan + step.base)),
             thrust::make_zip_iterator(
               thrust::make_tuple(thrust::make_counting_iterator(context.problem.size),
                                  context.tasks.plan + step.base + context.problem.size)),
             create_cost_transition<TransitionOp>{transitionOp, context.convolutions, step},
             create_invaild(), compare_transition_costs())
      .first;
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace heuristics {

template<typename TransitionOp>
vrp::models::Transition nearest_neighbor<TransitionOp>::operator()(const Context& context,
                                                                   int base,
                                                                   int from,
                                                                   int to,
                                                                   int vehicle) {
  auto transitionOp = TransitionOp(context.problem, context.tasks);
  return find_next_transition<TransitionOp>{context, transitionOp}({base, from, to, vehicle});
}

template<typename TransitionOp>
void nearest_neighbor<TransitionOp>::operator()(const Context& context, int index, int shift) {
  const auto begin = index * context.problem.size;

  auto transitionOp = TransitionOp(context.problem, context.tasks);
  auto findTransition = find_next_transition<TransitionOp>{context, transitionOp};

  int vehicle = 0;
  int from = shift;
  int to = from + 1;

  do {
    auto transition = findTransition(Step{begin, from, to, vehicle});
    if (transition.isValid()) {
      auto cost = transitionOp.estimate(transition);
      from = transitionOp.perform(transition, cost);
      to = from + 1;
    } else {
      // NOTE cannot find any further customer to serve within vehicle
      if (from == 0 || vehicle == context.problem.resources.vehicles - 1) break;

      from = 0;
      spawnNewVehicle(context.problem, context.tasks, from, ++vehicle);
    }
    /// TODO end is wrong?
  } while (to < context.problem.size);
}

/// NOTE make linker happy.
template class nearest_neighbor<TransitionOperator>;

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp
