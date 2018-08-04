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


// ANY_EXEC_UNIT inline int moveToCustomer(const Transition& transition,
//                                        float cost,
//                                        const Tasks::Shadow& tasks) {
//  const auto& details = transition.details;
//  const auto& delta = transition.delta;
//  int customer = details.customer.get<int>();
//
//  int to = details.base + details.to;
//  int from = details.base + details.from;
//
//  tasks.ids[to] = customer;
//  tasks.times[to] = tasks.times[from] + delta.duration();
//  tasks.capacities[to] = tasks.capacities[from] - delta.demand;
//  tasks.vehicles[to] = details.vehicle;
//
//  tasks.costs[to] = tasks.costs[from] + cost;
//  tasks.plan[details.base + customer] = Plan::assign();
//
//  return transition.details.to;
//}

/// Represents customer entry where arguments are:
/// * assignment status
/// * customer id
/// * index accumulator
/// This type is introduced to calculate a proper next task index as
/// some of the customers in convolution might be already served.
// using CustomerEntry = thrust::tuple<bool, int, int>;

///// Creates CustomerEntry from customer id.
// struct create_customer_entry final {
//  const Tasks::Shadow tasks;
//  const Transition::Details details;
//
//  EXEC_UNIT CustomerEntry operator()(int customer) const {
//    Plan plan = tasks.plan[details.base + customer];
//    return thrust::make_tuple(plan.isAssigned(), customer, plan.isAssigned() ? -1 : 0);
//  }
//};
//
///// Calculates proper sequential index.
// struct calculate_seq_index final {
//  EXEC_UNIT CustomerEntry operator()(const CustomerEntry& init, const CustomerEntry& value) const
//  {
//    return thrust::get<0>(value)
//             ? thrust::make_tuple(true, thrust::get<1>(value), thrust::get<2>(init))
//             : thrust::make_tuple(false, thrust::get<1>(value), thrust::get<2>(init) + 1);
//  }
//};

///// Performs actual transition and stores the max task shift index.
// struct process_convolution_task final {
//  const Problem::Shadow problem;
//  const Tasks::Shadow tasks;
//  const Transition::Details details;
//  //vector_ptr<int> max;
//
//  EXEC_UNIT int operator()(int customer, int index) {
//    auto var = variant<int, Convolution>::create(customer);
//
//    auto transition = create_transition{problem, tasks}(
//      {details.base, details.from + index, details.to + index, var, details.vehicle});
//
//    auto cost = calculate_transition_cost{problem, tasks}(transition);
//
//    return moveToCustomer(transition, cost, tasks);
//
//    //vrp::runtime::max(vrp::runtime::raw_pointer_cast<int>(max), nextTask);
//
//    //return {};
//  }
//};

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

    return from;
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
  // TODO
  return 0;
}

}  // namespace transitions
}  // namespace algorithms
}  // namespace vrp
