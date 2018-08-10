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
  vector_ptr<Model> model;
  int lastCustomer;
  int base;

  template<class Tuple>
  EXEC_UNIT void operator()(const Tuple& tuple) {
    const int task = lastCustomer - thrust::get<0>(tuple);
    const int vehicle = thrust::get<1>(tuple);
    const float cost = thrust::get<2>(tuple);

    auto depot = variant<int, Convolution>::create(0);
    auto modelPtr = vrp::runtime::raw_pointer_cast<Model>(model);
    auto solution = modelPtr->solution;

    auto details = Transition::Details{base, task, -1, depot, vehicle};
    auto transition = create_transition{solution.problem, solution.tasks}(details);
    auto returnCost = calculate_transition_cost{solution.problem, solution.tasks}(transition);
    auto routeCost = cost + returnCost;

    vrp::runtime::add(&modelPtr->total, routeCost);
  }
};

}  // namespace


float calculate_total_cost::operator()(int index) const {
  typedef vector<int>::iterator IntIterator;
  typedef vector<float>::iterator FloatIterator;

  int start = solution.tasks.customers * index;
  int end = start + solution.tasks.customers;

  auto model = vrp::runtime::allocate<Model>({0, solution});
  auto iterator = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator(0),
                       thrust::reverse_iterator<IntIterator>(solution.tasks.vehicles + end),
                       thrust::reverse_iterator<FloatIterator>(solution.tasks.costs + end)));

  thrust::unique_by_key_copy(
    exec_unit_policy{}, thrust::reverse_iterator<IntIterator>(solution.tasks.vehicles + end),
    thrust::reverse_iterator<IntIterator>(solution.tasks.vehicles + start), iterator,
    thrust::make_discard_iterator(),
    make_aggregate_output_iterator(iterator, aggregate_cost{model, solution.tasks.customers - 1,
                                                            end - solution.tasks.customers}));

  return vrp::runtime::release<Model>(model).total;
}
