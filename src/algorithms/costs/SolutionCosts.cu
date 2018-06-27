#include "algorithms/costs/SolutionCosts.hpp"
#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "iterators/Aggregates.hpp"
#include "utils/memory/Allocations.hpp"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/unique.h>

using namespace vrp::algorithms::transitions;
using namespace vrp::algorithms::costs;
using namespace vrp::iterators;
using namespace vrp::models;
using namespace vrp::utils;

namespace {

/// Contains total cost and solution shadow.
struct Model final {
  float total;
  vrp::models::Solution::Shadow solution;
};

/// Aggregates all costs.
struct aggregate_cost final {
  Model* model;
  int lastCustomer;
  int base;

  template<class Tuple>
  __device__ void operator()(const Tuple& tuple) {
    const int task = lastCustomer - thrust::get<0>(tuple);
    const int vehicle = thrust::get<1>(tuple);
    const float cost = thrust::get<2>(tuple);

    auto depot = device_variant<int, Convolution>();
    depot.set<int>(0);

    auto details = Transition::Details{base, task, -1, depot, vehicle};
    auto transition = create_transition(model->solution.problem, model->solution.tasks)(details);
    auto returnCost = calculate_transition_cost(model->solution.problem.resources)(transition);
    auto routeCost = cost + returnCost;

    // NOTE to use atomicAdd, variable has to be allocated in device memory,
    // not in registers
    atomicAdd(&model->total, routeCost);
  }
};

}  // namespace


__host__ float calculate_total_cost::operator()(Solution& solution, int index) const {
  int end = solution.tasks.customers * (index + 1);
  int rbegin = solution.tasks.size() - end;
  int rend = rbegin + solution.tasks.customers;

  auto model = vrp::utils::allocate<Model>({0, solution.getShadow()});
  auto iterator = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator(0), solution.tasks.vehicles.rbegin() + rbegin,
                       solution.tasks.costs.rbegin() + rbegin));

  thrust::unique_by_key_copy(
    thrust::device, solution.tasks.vehicles.rbegin() + rbegin,
    solution.tasks.vehicles.rbegin() + rend, iterator, thrust::make_discard_iterator(),
    make_aggregate_output_iterator(
      iterator,
      aggregate_cost{model.get(), solution.tasks.customers - 1, end - solution.tasks.customers}));

  return vrp::utils::release(model).total;
}
