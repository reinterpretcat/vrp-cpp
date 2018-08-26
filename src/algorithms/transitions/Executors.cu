#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "iterators/Aggregates.hpp"

#include <thrust/functional.h>
#include <thrust/transform_scan.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

/// Applies transition for single customer.
struct apply_customer final {
  const Tasks::Shadow tasks;

  ANY_EXEC_UNIT int operator()(const Transition& transition, float cost) const {
    const auto& details = transition.details;
    const auto& delta = transition.delta;
    int customer = details.customer.get<int>();

    int from = details.base + details.from;
    int to = details.base + details.to;

    tasks.ids[to] = customer;
    tasks.times[to] = tasks.times[from] + delta.duration();
    tasks.capacities[to] = tasks.capacities[from] - delta.demand;
    tasks.vehicles[to] = details.vehicle;

    tasks.costs[to] = tasks.costs[from] + cost;
    tasks.plan[details.base + customer] = Plan::assign();

    return transition.details.to;
  }
};

/// Analyzes transition for single customer.
struct analyze_customer final {
  const Tasks::Shadow tasks;

  ANY_EXEC_UNIT int operator()(const Transition& transition, Transition::State& state) const {
    state.customer = transition.details.customer.get<int>();
    state.time += transition.delta.duration();
    state.capacity -= transition.delta.demand;

    return transition.details.to;
  }
};

template<typename CustomerProcessor>
struct process_convolution final {
  CustomerProcessor customers;
  create_transition factory;
  const Tasks::Shadow tasks;

  ANY_EXEC_UNIT int operator()(const Transition& transition, Transition::State& state) {
    const auto& details = transition.details;
    const auto& convolution = details.customer.get<Convolution>();

    int from = details.from;
    int to = details.to;

    auto first = tasks.ids + convolution.base + convolution.tasks.first;
    auto last = tasks.ids + convolution.base + convolution.tasks.second;

    for (auto i = first; i <= last; ++i) {
      int customer = *i;
      Plan plan = tasks.plan[details.base + customer];
      if (plan.isAssigned()) continue;

      auto newDetails = Transition::Details{
        details.base, from, to, variant<int, Convolution>::create(customer), details.vehicle};
      auto newTransition = factory(newDetails, state);
      from = customers(newTransition, state);
      to = from + 1;
    }

    return thrust::max(details.to, to - 1);
  }
};

/// Applies convolution.
struct apply_convolution final {
  calculate_transition_cost costs;
  apply_customer customers;
  const Tasks::Shadow tasks;

  ANY_EXEC_UNIT int operator()(const Transition& transition, Transition::State& state) {
    auto cost = costs(transition);
    auto next = customers(transition, cost);

    int task = transition.details.base + next;

    state.customer = tasks.ids[task];
    state.capacity = tasks.capacities[task];
    state.time = tasks.times[task];

    return next;
  }
};
/// Analyzes convolution
struct analyze_convolution final {
  analyze_customer customers;

  ANY_EXEC_UNIT int operator()(const Transition& transition, Transition::State& state) {
    return customers(transition, state);
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace transitions {

int perform_transition::operator()(const Transition& transition, float cost) const {
  if (!transition.details.customer.is<Convolution>())
    return apply_customer{tasks}(transition, cost);

  int task = transition.details.base + transition.details.from;
  auto state = Transition::State{tasks.ids[task], tasks.capacities[task], tasks.times[task]};
  auto processor = process_convolution<apply_convolution>{
    apply_convolution{calculate_transition_cost{problem, tasks}, apply_customer{tasks}, tasks},
    create_transition{problem, tasks}, tasks};

  return processor(transition, state);
}

int perform_transition::operator()(const vrp::models::Transition& transition,
                                   vrp::models::Transition::State& state) const {
  if (!transition.details.customer.is<Convolution>())
    return analyze_customer{tasks}(transition, state);

  auto processor = process_convolution<analyze_convolution>{
    analyze_convolution{analyze_customer{tasks}}, create_transition{problem, tasks}, tasks};

  return processor(transition, state);
}

}  // namespace transitions
}  // namespace algorithms
}  // namespace vrp
