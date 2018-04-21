#ifndef VRP_ALGORITHMS_COSTS_HPP
#define VRP_ALGORITHMS_COSTS_HPP

#include "algorithms/Transitions.cu"
#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"
#include "utils/Memory.hpp"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/unique.h>

namespace vrp {
namespace algorithms {

/// Calculates cost of transition.
struct calculate_transition_cost final {
  const vrp::models::Resources::Shadow resources;

  __host__ __device__
  explicit calculate_transition_cost(const vrp::models::Resources::Shadow &resources) :
      resources(resources) {}

  __host__ __device__
  float operator()(const vrp::models::Transition &transition) const {
    int vehicle = transition.details.vehicle;

    auto distance = transition.delta.distance * resources.distanceCosts[vehicle];
    auto traveling = transition.delta.traveling * resources.timeCosts[vehicle];
    auto waiting = transition.delta.waiting * resources.waitingCosts[vehicle];
    auto serving = transition.delta.serving * resources.timeCosts[vehicle];

    return distance + traveling + waiting + serving;
  }
};

/// Calculates total cost of solution.
struct calculate_total_cost final {
  /// Aggregates all costs.
  struct AggregateCost final {
    float *total;
    vrp::models::Problem::Shadow *problem;
    vrp::models::Tasks::Shadow *tasks;
    const int maxTask;

    template<class Tuple>
    __device__
    float operator()(const Tuple &tuple) {
      const int task = maxTask - thrust::get<0>(tuple);
      const int vehicle = thrust::get<1>(tuple);
      const int depot = 0;
      const float cost = thrust::get<2>(tuple);

      auto details = vrp::models::Transition::Details{task, -1, depot, vehicle};
      auto transition = create_transition(*problem, *tasks)(details);
      auto returnCost = calculate_transition_cost(problem->resources)(transition);

      auto routeCost = cost + returnCost;

      // NOTE to use atomicAdd, variable has to be allocated in memory
      atomicAdd(total, routeCost);

      return routeCost;
    }
  };

  __host__
  float operator()(const vrp::models::Problem &problem,
                   vrp::models::Tasks &tasks,
                   int solution = 0) const {
    int end = tasks.customers * (solution + 1);
    int rbegin = tasks.size() - end;
    int rend = rbegin + tasks.customers;
    auto count = static_cast<std::size_t >(tasks.vehicles[end - 1] + 1);

    auto total = vrp::utils::allocate<float>(0);
    auto sProblem = vrp::utils::allocate(problem.getShadow());
    auto sTasks = vrp::utils::allocate(tasks.getShadow());

    thrust::unique_by_key_copy(
        thrust::device,
        tasks.vehicles.rbegin() + rbegin,
        tasks.vehicles.rbegin() + rend,
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(0),
            tasks.vehicles.rbegin() + rbegin,
            tasks.costs.rbegin() + rbegin)
        ),
        thrust::make_discard_iterator(),
        thrust::make_transform_output_iterator(
            thrust::make_discard_iterator(),
            AggregateCost{total.get(), sProblem.get(), sTasks.get(), tasks.customers - 1}
        )
    );

    thrust::device_free(sProblem);
    thrust::device_free(sTasks);

    return vrp::utils::release(total);
  }
};

}
}

#endif //VRP_ALGORITHMS_COSTS_HPP
