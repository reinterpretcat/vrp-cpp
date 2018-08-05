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


template<typename Processor>
struct process_convolution final {
  Processor processor;

  ANY_EXEC_UNIT int operator()(const Problem::Shadow problem,
                               const Tasks::Shadow tasks,
                               const Transition& transition) {
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

      from = processor(variant<int, Convolution>::create(customer), from, to);
      to = from + 1;
    }

    return to - 1;
  }
};

struct make_transition final {
  create_transition factory;
  calculate_transition_cost costs;
  apply_customer customers;
  Transition::Details details;

  ANY_EXEC_UNIT int operator()(const variant<int, Convolution>& customer, int from, int to) {
    auto transition = factory({details.base, from, to, customer, details.vehicle});
    auto cost = costs(transition);
    return customers(transition, cost);
  }
};

struct analyze_transition final {
  create_transition factory;
  analyze_customer analyzer;
  Transition::Details details;
  Transition::State state;

  ANY_EXEC_UNIT int operator()(const variant<int, Convolution>& customer, int from, int to) {
    auto transition = factory({details.base, from, to, customer, details.vehicle}, state);
    return analyzer(transition, state);
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace transitions {

int perform_transition::operator()(const Transition& transition, float cost) const {
  return transition.details.customer.is<Convolution>()
           ? process_convolution<make_transition>{make_transition{
               create_transition{problem, tasks}, calculate_transition_cost{problem, tasks},
               apply_customer{tasks}, transition.details}}(problem, tasks, transition)
           : apply_customer{tasks}(transition, cost);
}

int perform_transition::operator()(const vrp::models::Transition& transition,
                                   vrp::models::Transition::State& state) const {
  return transition.details.customer.is<Convolution>()
           ? process_convolution<analyze_transition>{analyze_transition{
               create_transition{problem, tasks}, analyze_customer{tasks}, transition.details,
               state}}(problem, tasks, transition)
           : analyze_customer{tasks}(transition, state);
}

}  // namespace transitions
}  // namespace algorithms
}  // namespace vrp
