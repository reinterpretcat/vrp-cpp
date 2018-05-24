#include "algorithms/costs/Models.hpp"
#include "algorithms/costs/SolutionCosts.hpp"
#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "iterators/Aggregates.hpp"
#include "utils/Memory.hpp"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/unique.h>

using namespace vrp::algorithms::transitions;
using namespace vrp::algorithms::costs;
using namespace vrp::iterators;
using namespace vrp::models;

namespace {
/// Aggregates all costs.
struct aggregate_cost final {
  Model* model;
  int lastCustomer;
  int baseTask;

  template<class Tuple>
  __device__ void operator()(const Tuple& tuple) {
    const int task = lastCustomer - thrust::get<0>(tuple);
    const int vehicle = thrust::get<1>(tuple);
    const int depot = 0;
    const float cost = thrust::get<2>(tuple);

    auto details = Transition::Details{baseTask + task, -1, depot, vehicle};
    auto transition = create_transition(model->problem, model->tasks)(details);
    auto returnCost = calculate_transition_cost(model->problem.resources)(transition);
    auto routeCost = cost + returnCost;

    // NOTE to use atomicAdd, variable has to be allocated in device memory,
    // not in registers
    atomicAdd(&model->total, routeCost);
  }
};

}  // namespace


__host__ float calculate_total_cost::operator()(const vrp::models::Problem& problem,
                                                vrp::models::Tasks& tasks,
                                                int solution) const {
  int end = tasks.customers * (solution + 1);
  int rbegin = tasks.size() - end;
  int rend = rbegin + tasks.customers;

  auto model = vrp::utils::allocate<Model>({0, problem.getShadow(), tasks.getShadow()});
  auto iterator = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0),
                                                               tasks.vehicles.rbegin() + rbegin,
                                                               tasks.costs.rbegin() + rbegin));

  thrust::unique_by_key_copy(
    thrust::device, tasks.vehicles.rbegin() + rbegin, tasks.vehicles.rbegin() + rend, iterator,
    thrust::make_discard_iterator(),
    make_aggregate_output_iterator(
      iterator, aggregate_cost{model.get(), tasks.customers - 1, end - tasks.customers}));

  return vrp::utils::release(model).total;
}
