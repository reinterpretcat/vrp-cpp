#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/transitions/Factories.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::utils;

namespace {

__host__ __device__ TransitionCost createInvalid() {
  return thrust::make_tuple(vrp::models::Transition(), -1);
}

/// Creates transition and calculates cost to given customer if it is not handled.
struct create_cost_transition {
  int fromTask;
  int toTask;
  int vehicle;

  thrust::device_ptr<Convolution> convolutions;

  create_transition transitionFactory;
  calculate_transition_cost costCalculator;

  __host__ __device__ TransitionCost operator()(const thrust::tuple<int, Plan>& customer) {
    auto plan = thrust::get<1>(customer);

    if (plan.isAssigned()) return createInvalid();

    auto wrapped = device_variant<int, Convolution>();
    if (plan.hasConvolution())
      wrapped.set<Convolution>(*(convolutions + plan.convolution()));
    else
      wrapped.set<int>(thrust::get<0>(customer));

    auto transition = transitionFactory({fromTask, toTask, wrapped, vehicle});
    auto cost = costCalculator(transition);

    return thrust::make_tuple(transition, cost);
  }
};

/// Compares costs of two transitions and returns the lowest.
struct compare_transition_costs {
  __host__ __device__ TransitionCost& operator()(TransitionCost& result,
                                                 const TransitionCost& left) {
    if (thrust::get<0>(left).isValid() &&
        (thrust::get<1>(left) < thrust::get<1>(result) || !thrust::get<0>(result).isValid())) {
      result = left;
    }

    return result;
  }
};

}  // namespace

TransitionCost nearest_neighbor::operator()(int fromTask, int toTask, int vehicle) {
  int base = (fromTask / tasks.customers) * tasks.customers;
  return thrust::transform_reduce(
    thrust::device,
    thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator(0), tasks.plan + base)),
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(problem.size),
                                                 tasks.plan + base + problem.size)),
    create_cost_transition{fromTask, toTask, vehicle, convolutions,
                           create_transition{problem, tasks},
                           calculate_transition_cost{problem.resources}},
    createInvalid(), compare_transition_costs());
}
