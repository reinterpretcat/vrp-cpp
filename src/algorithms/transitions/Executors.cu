#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::utils;

namespace {

__host__ __device__ inline int base(const Tasks::Shadow& tasks, int task) {
  return (task / tasks.customers) * tasks.customers;
}

__host__ __device__ inline void moveToCustomer(const TransitionCost& transitionCost,
                                               const Tasks::Shadow& tasks) {
  const auto& transition = thrust::get<0>(transitionCost);
  const auto& details = transition.details;
  const auto& delta = transition.delta;
  int customer = details.customer.get<int>();

  tasks.ids[details.to] = customer;
  tasks.times[details.to] = tasks.times[details.from] + delta.duration();
  tasks.capacities[details.to] = tasks.capacities[details.from] - delta.demand;
  tasks.vehicles[details.to] = details.vehicle;

  tasks.costs[details.to] = tasks.costs[details.from] + thrust::get<1>(transitionCost);
  tasks.plan[base(tasks, details.to) + customer] = Plan::assign();
}

/// Process single task from convolution.
struct process_task final {
  const Problem::Shadow problem;
  const Tasks::Shadow tasks;
  const Transition::Details details;
  int base;

  template<typename Tuple>
  __device__ void operator()(const Tuple& tuple) {
    auto customer = thrust::get<1>(tuple);
    auto index = base + thrust::get<0>(tuple);

    auto variant = device_variant<int, Convolution>();
    variant.set<int>(customer);

    auto newDetails =
      Transition::Details{details.from + index, details.to + index, variant, details.vehicle};

    auto transition = create_transition{problem, tasks}(newDetails);

    auto cost = calculate_transition_cost{problem.resources}(transition);

    moveToCustomer(thrust::make_tuple(transition, cost), tasks);
  }
};

__host__ __device__ inline void moveToConvolution(const Transition& transition,
                                                  const Problem::Shadow& problem,
                                                  const Tasks::Shadow& tasks) {
  const auto& details = transition.details;
  const auto& convolution = details.customer.get<Convolution>();
  const int count = convolution.tasks.second - convolution.tasks.first + 1;

  thrust::for_each(
    thrust::device,
    thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_counting_iterator(0), tasks.ids + convolution.base + convolution.tasks.first)),
    thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator(count),
                         tasks.ids + convolution.base + convolution.tasks.second + 1)),
    process_task{problem, tasks, details, convolution.base});
}

}  // namespace

__host__ __device__ void perform_transition::operator()(
  const TransitionCost& transitionCost) const {
  if (thrust::get<0>(transitionCost).details.customer.is<Convolution>())
    moveToConvolution(thrust::get<0>(transitionCost), problem, tasks);
  else
    moveToCustomer(transitionCost, tasks);
}
