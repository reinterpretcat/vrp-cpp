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
ANY_EXEC_UNIT inline int moveToCustomer(const Transition& transition,
                                        float cost,
                                        const Tasks::Shadow& tasks) {
  const auto& details = transition.details;
  const auto& delta = transition.delta;
  int customer = details.customer.get<int>();

  int to = details.base + details.to;
  int from = details.base + details.from;

  tasks.ids[to] = customer;
  tasks.times[to] = tasks.times[from] + delta.duration();
  tasks.capacities[to] = tasks.capacities[from] - delta.demand;
  tasks.vehicles[to] = details.vehicle;

  tasks.costs[to] = tasks.costs[from] + cost;
  tasks.plan[details.base + customer] = Plan::assign();

  return transition.details.to;
}

/// Represents customer entry where arguments are:
/// * assignment status
/// * customer id
/// * index accumulator
/// This type is introduced to calculate a proper next task index as
/// some of the customers in convolution might be already served.
using CustomerEntry = thrust::tuple<bool, int, int>;

/// Creates CustomerEntry from customer id.
struct create_customer_entry final {
  const Tasks::Shadow tasks;
  const Transition::Details details;

  EXEC_UNIT CustomerEntry operator()(int customer) const {
    Plan plan = tasks.plan[details.base + customer];
    return thrust::make_tuple(plan.isAssigned(), customer, plan.isAssigned() ? -1 : 0);
  }
};

/// Calculates proper sequential index.
struct calculate_seq_index final {
  EXEC_UNIT CustomerEntry operator()(const CustomerEntry& init, const CustomerEntry& value) const {
    return thrust::get<0>(value)
             ? thrust::make_tuple(true, thrust::get<1>(value), thrust::get<2>(init))
             : thrust::make_tuple(false, thrust::get<1>(value), thrust::get<2>(init) + 1);
  }
};

/// Performs actual transition and stores the max task shift index.
struct process_convolution_task final {
  const Problem::Shadow problem;
  const Tasks::Shadow tasks;
  const Transition::Details details;
  vector_ptr<int> max;

  EXEC_UNIT CustomerEntry operator()(const CustomerEntry& value) {
    if (thrust::get<0>(value)) return {};

    int customer = thrust::get<1>(value);
    int index = thrust::get<2>(value);

    auto var = variant<int, Convolution>::create(customer);

    auto transition = create_transition{problem, tasks}(
      {details.base, details.from + index, details.to + index, var, details.vehicle});

    auto cost = calculate_transition_cost{problem, tasks}(transition);

    auto nextTask = moveToCustomer(transition, cost, tasks);

    vrp::runtime::max(vrp::runtime::raw_pointer_cast<int>(max), nextTask);

    return {};
  }
};

ANY_EXEC_UNIT inline int moveToConvolution(const Transition& transition,
                                           const Problem::Shadow problem,
                                           const Tasks::Shadow tasks) {
  const auto& details = transition.details;
  const auto& convolution = details.customer.get<Convolution>();

  auto max = vrp::runtime::allocate<int>(details.from);
  auto iterator = vrp::iterators::make_aggregate_output_iterator(
    vector<CustomerEntry>::iterator{}, process_convolution_task{problem, tasks, details, max});

  thrust::transform_inclusive_scan(
    exec_unit_policy{}, tasks.ids + convolution.base + convolution.tasks.first,
    tasks.ids + convolution.base + convolution.tasks.second + 1, iterator,
    create_customer_entry{tasks, details}, calculate_seq_index{});

  return vrp::runtime::release<int>(max);
}

}  // namespace

ANY_EXEC_UNIT int perform_transition::operator()(const Transition& transition, float cost) const {
  return transition.details.customer.is<Convolution>()
           ? moveToConvolution(transition, problem, tasks)
           : moveToCustomer(transition, cost, tasks);
}
