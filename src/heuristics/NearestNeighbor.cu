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

using TransitionCost = NearestNeighbor::TransitionCost;

__host__ __device__
TransitionCost createInvalid() {
  return thrust::make_pair(vrp::models::Transition(), -1);
}

/// Creates transition and calculates cost to given customer if it is not handled.
struct CreateTransition {
  int task;

  vrp::algorithms::CreateTransition transitionFactory;
  vrp::algorithms::CalculateCost costCalculator;

  __host__ __device__
  TransitionCost operator()(const thrust::tuple<int, bool> &customer) {
    if (thrust::get<1>(customer))
      return createInvalid();

    auto transition = transitionFactory(task, thrust::get<0>(customer));
    auto cost = costCalculator(transition);

    return thrust::make_pair(transition, cost);
  }
};

/// Compares costs of two transitions and returns the lowest.
struct CompareTransitionCosts {
  __host__ __device__
  TransitionCost& operator()(TransitionCost &result, const TransitionCost &left) {
    if (left.first.isValid() && (left.second < result.second || !result.first.isValid())) {
      result = left;
    }
    return result;
  }
};

}

TransitionCost NearestNeighbor::operator()(int task) {
  return thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::make_counting_iterator(0),
          tasks.plan
      )),
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::make_counting_iterator(problem.size),
          tasks.plan + problem.size
      )),
      CreateTransition {task,
                        vrp::algorithms::CreateTransition {problem, tasks},
                        vrp::algorithms::CalculateCost {problem.resources}},
      createInvalid(),
      CompareTransitionCosts()
  );
}
