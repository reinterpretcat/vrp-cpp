#ifndef VRP_ALGORITHMS_COSTS_HPP
#define VRP_ALGORITHMS_COSTS_HPP

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/unique.h>

namespace vrp {
namespace algorithms {

/// Calculates costs for transition.
struct calculate_cost final {
  const vrp::models::Resources::Shadow resources;

  __host__ __device__
  explicit calculate_cost(const vrp::models::Resources::Shadow &resources) :
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

/// Calculates cost of all used vehicles separately.
struct calculate_total_cost final {

  /// Calculates total cost.
  struct RouteCost final {

    float *total;

     template<class Tuple>
    __device__
    float operator()(const Tuple &tuple) {
      const int task = thrust::get<0>(tuple);
      const int vehicle = thrust::get<1>(tuple);
      const float cost = thrust::get<2>(tuple);
      // TODO calculate return to depot cost

      atomicAdd(total, cost);

      // NOTE ignored
      return 0;
    }
  };

  float operator()(const vrp::models::Tasks &tasks, int solution = 0) const {
    int rbegin = tasks.size() - tasks.customers * (solution + 1);
    int rend = rbegin + tasks.customers;
    auto count = static_cast<std::size_t >(tasks.vehicles.back() + 1);

    // NOTE to use atomicAdd, variable has to be allocated in memory.
    thrust::device_ptr<float> total = thrust::device_malloc<float>(1);
    *total = 0;

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
            RouteCost { total.get() }
        )
    );

    float result = *total;
    thrust::device_free(total);

    return result;
  }
};

}
}

#endif //VRP_ALGORITHMS_COSTS_HPP
