#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "iterators/Aggregates.hpp"
#include "utils/memory/Allocations.hpp"

#include <thrust/functional.h>
#include <thrust/transform_scan.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::utils;

namespace {
__host__ __device__ inline int moveToCustomer(const Transition& transition,
                                              float cost,
                                              const Tasks::Shadow& tasks) {
  const auto& details = transition.details;
  const auto& delta = transition.delta;
  int customer = details.customer.get<int>();

  printf("customer:%d\n", customer);

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
using CustomerEntry = thrust::tuple<bool,int,int>;

/// Creates CustomerEntry from customer id.
struct create_customer_entry final {
  const Tasks::Shadow tasks;
  const Transition::Details details;

  __device__ CustomerEntry operator()(int customer) const {
    Plan plan = tasks.plan[details.base + customer];
    return thrust::make_tuple(plan.isAssigned(), customer, plan.isAssigned() ? -1 : 0);
  }
};

/// Calculates proper sequential index.
struct calculate_seq_index final {
  __device__ CustomerEntry operator()(const CustomerEntry& init, const CustomerEntry& value) const {
    printf("[%d,%d,%d] [%d,%d,%d]\n",
           thrust::get<0>(init), thrust::get<1>(init), thrust::get<2>(init),
           thrust::get<0>(value), thrust::get<1>(value), thrust::get<2>(value));

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
  thrust::device_ptr<int> max;

  __device__ CustomerEntry operator()(const CustomerEntry& value) {
    printf("-------------[%d,%d,%d]\n",
           thrust::get<0>(value), thrust::get<1>(value), thrust::get<2>(value));

    if(thrust::get<0>(value)) {
      printf("plan in convolution is processed for customer: %d\n", thrust::get<1>(value));
      return {};
    }

    int customer = thrust::get<1>(value);
    int index = thrust::get<2>(value);

    printf("plan in convolution is not processed for customer %d, conv index is %d\n", customer, index);

    auto variant = device_variant<int, Convolution>();
    variant.set<int>(customer);

    auto transition = create_transition{problem, tasks}(
      {details.base, details.from + index, details.to + index, variant, details.vehicle});

    auto cost = calculate_transition_cost{problem.resources}(transition);

    auto nextTask = moveToCustomer(transition, cost, tasks);

    atomicMax(max.get(), nextTask);

    return {};
  }
};

__host__ __device__ inline int moveToConvolution(const Transition& transition,
                                                 const Problem::Shadow& problem,
                                                 const Tasks::Shadow& tasks) {
  const auto& details = transition.details;
  const auto& convolution = details.customer.get<Convolution>();
  int count = convolution.tasks.second - convolution.tasks.first + 1;

  printf("-----CONVOLUTION [%d,%d]:%d customer:[%d,%d]\n", convolution.tasks.first,
         convolution.tasks.second, convolution.base, convolution.customers.first,
         convolution.customers.second);

  auto max = vrp::utils::allocate<int>(details.from);
  auto iterator = vrp::iterators::make_aggregate_output_iterator(
      thrust::device_vector<CustomerEntry>::iterator {},
      process_convolution_task{problem, tasks, details, max});

  thrust::transform_inclusive_scan(
      thrust::device,
      tasks.ids + convolution.base + convolution.tasks.first,
      tasks.ids + convolution.base + convolution.tasks.second + 1,
      iterator,
      create_customer_entry{tasks, details},
      calculate_seq_index {});

  return vrp::utils::release(max);
}

}  // namespace

__host__ __device__ int perform_transition::operator()(const Transition& transition,
                                                       float cost) const {
  return transition.details.customer.is<Convolution>()
           ? moveToConvolution(transition, problem, tasks)
           : moveToCustomer(transition, cost, tasks);
}
