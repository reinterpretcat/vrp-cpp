#include "heuristics/NearestNeighbor.hpp"

#include "algorithms/Transitions.cu"
#include "algorithms/Costs.cu"

#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>

using namespace vrp::heuristics;
using namespace vrp::models;

namespace {

__host__ __device__
TransitionCost createInvalid() {
  return thrust::make_tuple(vrp::models::Transition(), -1);
}

/// Creates transition and calculates cost to given customer if it is not handled.
struct create_transition {
  int fromTask;
  int toTask;
  int vehicle;

  vrp::algorithms::create_transition transitionFactory;
  vrp::algorithms::calculate_cost costCalculator;

  __host__ __device__
  TransitionCost operator()(const thrust::tuple<int, bool> &customer) {
    if (thrust::get<1>(customer))
      return createInvalid();

    auto transition = transitionFactory({fromTask, toTask, thrust::get<0>(customer), vehicle });
    auto cost = costCalculator(transition);

    return thrust::make_tuple(transition, cost);
  }
};

/// Compares costs of two transitions and returns the lowest.
struct compare_transition_costs {
  __host__ __device__
  TransitionCost& operator()(TransitionCost &result, const TransitionCost &left) {
    if (thrust::get<0>(left).isValid() &&
        (thrust::get<1>(left) < thrust::get<1>(result) || !thrust::get<0>(result).isValid())) {
      result = left;
    }
    return result;
  }
};

}

TransitionCost NearestNeighbor::operator()(int fromTask, int toTask, int vehicle) {
  int base = (fromTask / tasks.customers) * tasks.customers;
  return thrust::transform_reduce(
      thrust::device,
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::make_counting_iterator(0),
          tasks.plan + base
      )),
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::make_counting_iterator(problem.size),
          tasks.plan + base + problem.size
      )),
      create_transition {fromTask,
                         toTask,
                         vehicle,
                         vrp::algorithms::create_transition {problem, tasks},
                         vrp::algorithms::calculate_cost {problem.resources}},
      createInvalid(),
      compare_transition_costs()
  );
}
