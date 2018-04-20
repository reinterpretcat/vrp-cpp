#ifndef VRP_ALGORITHMS_COSTS_HPP
#define VRP_ALGORITHMS_COSTS_HPP

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

#include <thrust/iterator/transform_output_iterator.h>

#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

#include <thrust/detail/use_default.h>
#include <thrust/iterator/detail/any_assign.h>

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

struct calculate_total_cost final {

  struct Functor final {
    template<class Tuple>
    __host__ __device__
    float operator()(const Tuple& tuple) const {
      const int task = thrust::get<0>(tuple);
      const int vehicle = thrust::get<1>(tuple);
      const float cost = thrust::get<2>(tuple);
      return cost;
    }
  };

  float operator()(const vrp::models::Tasks &tasks, int solution = 0) const {
    int begin = tasks.customers * solution;
    int end = begin + tasks.customers;

    int count = tasks.vehicles[end - 1] + 1;
    thrust::device_vector<int> vehicles (count);
    thrust::device_vector<float> costs (count);

    thrust::unique_by_key_copy(
        thrust::device,
        tasks.vehicles.begin() + begin,
        tasks.vehicles.begin() + end,
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(0),
            tasks.vehicles.begin() + begin,
            tasks.costs.begin() + begin)
        ),
        vehicles.begin(),
        thrust::make_transform_output_iterator(costs.begin(), Functor())
    );

    thrust::copy(vehicles.begin(), vehicles.end(), std::ostream_iterator<int>(std::cout, ","));
    std::cout << "\n";
    thrust::copy(costs.begin(), costs.end(), std::ostream_iterator<float>(std::cout, ","));
    std::cout << "\n";

    return 0;
  }
};

}
}

#endif //VRP_ALGORITHMS_COSTS_HPP