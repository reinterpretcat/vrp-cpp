#include "algorithms/costs/SolutionCosts.hpp"
#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "iterators/Aggregates.hpp"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/unique.h>

using namespace vrp::algorithms::transitions;
using namespace vrp::algorithms::costs;
using namespace vrp::iterators;
using namespace vrp::models;
using namespace vrp::runtime;

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
  EXEC_UNIT void operator()(const Tuple& tuple) {
    const int task = lastCustomer - thrust::get<0>(tuple);
    const int vehicle = thrust::get<1>(tuple);
    const float cost = thrust::get<2>(tuple);

    auto depot = variant<int, Convolution>();
    depot.set<int>(0);

    auto details = Transition::Details{base, task, -1, depot, vehicle};
    auto transition = create_transition(model->solution.problem, model->solution.tasks)(details);
    auto returnCost =
      calculate_transition_cost(model->solution.problem, model->solution.tasks)(transition);
    auto routeCost = cost + returnCost;

    vrp::runtime::add(&model->total, routeCost);
  }
};

}  // namespace


__host__ float calculate_total_cost::operator()(Solution& solution, int index) const {
  typedef vector<int>::iterator IntIterator;
  typedef vector<float>::iterator FloatIterator;

  int start = solution.tasks.customers * index;
  int end = start + solution.tasks.customers;

  auto model = vrp::runtime::allocate<Model>({0, solution.getShadow()});
  auto iterator = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator(0),
                       thrust::reverse_iterator<IntIterator>(solution.tasks.vehicles.data() + end),
                       thrust::reverse_iterator<FloatIterator>(solution.tasks.costs.data() + end)));

  thrust::unique_by_key_copy(
    exec_unit, thrust::reverse_iterator<IntIterator>(solution.tasks.vehicles.data() + end),
    thrust::reverse_iterator<IntIterator>(solution.tasks.vehicles.data() + start), iterator,
    thrust::make_discard_iterator(),
    make_aggregate_output_iterator(
      iterator,
      aggregate_cost{model.get(), solution.tasks.customers - 1, end - solution.tasks.customers}));

  return vrp::runtime::release(model).total;
}
