#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::utils;

namespace {
__host__ __device__ inline void moveToCustomer(const Transition& transition,
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
}

/// Process single task from convolution.
struct process_convolution_task final {
  const Problem::Shadow problem;
  const Tasks::Shadow tasks;
  const Transition::Details details;

  template<typename Tuple>
  __device__ void operator()(const Tuple& tuple) {
    int index = thrust::get<0>(tuple);
    int customer = thrust::get<1>(tuple);

    auto variant = device_variant<int, Convolution>();
    variant.set<int>(customer);

    auto transition = create_transition{problem, tasks}(
      {details.base, details.from + index, details.to + index, variant, details.vehicle});

    if (!transition.isValid()) {
      printf("Invalid transition for tasks:[%d,%d], index=%d vehicle=%d\n", details.from,
             details.to, index, details.vehicle);
    }

    auto cost = calculate_transition_cost{problem.resources}(transition);

    moveToCustomer(transition, cost, tasks);
  }
};

__host__ __device__ inline void moveToConvolution(const Transition& transition,
                                                  const Problem::Shadow& problem,
                                                  const Tasks::Shadow& tasks) {
  const auto& details = transition.details;
  const auto& convolution = details.customer.get<Convolution>();
  int count = convolution.tasks.second - convolution.tasks.first + 1;

  thrust::for_each(
    thrust::device,
    thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_counting_iterator(0), tasks.ids + convolution.base + convolution.tasks.first)),
    thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator(count),
                         tasks.ids + convolution.base + convolution.tasks.second + 1)),
    process_convolution_task{problem, tasks, details});
}

}  // namespace

__host__ __device__ void perform_transition::operator()(const Transition& transition,
                                                        float cost) const {
  if (transition.details.customer.is<Convolution>()) {
    const auto& convolution = transition.details.customer.get<Convolution>();
    printf("execute: convolution [%d, %d]\n", convolution.tasks.first, convolution.tasks.second);
    moveToConvolution(transition, problem, tasks);
  } else {
    int customer = transition.details.customer.get<int>();
    printf("execute: customer %d\n", customer);
    moveToCustomer(transition, cost, tasks);
  }
}
