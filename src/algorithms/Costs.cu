#ifndef VRP_ALGORITHMS_COSTS_HPP
#define VRP_ALGORITHMS_COSTS_HPP

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

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
struct calculate_vehicles_cost final {

  struct RouteCost final {
    template<class Tuple>
    __host__ __device__
    float operator()(const Tuple& tuple) const {
      const int task = thrust::get<0>(tuple);
      const int vehicle = thrust::get<1>(tuple);
      const float cost = thrust::get<2>(tuple);
      // TODO calculate return to depot cost
      return cost;
    }
  };

  thrust::device_vector<float> operator()(const vrp::models::Tasks &tasks, int solution = 0) const {
    int rbegin = tasks.size() - tasks.customers * (solution + 1);
    int rend = rbegin + tasks.customers;
    auto count = static_cast<std::size_t >(tasks.vehicles.back() + 1);

    thrust::device_vector<float> costs(count);

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
        thrust::make_transform_output_iterator(costs.rbegin(), RouteCost())
    );

    return std::move(costs);
  }
};

}
}

#endif //VRP_ALGORITHMS_COSTS_HPP
