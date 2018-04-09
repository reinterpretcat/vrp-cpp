#ifndef VRP_ALGORITHMS_COSTS_HPP
#define VRP_ALGORITHMS_COSTS_HPP

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "models/Transition.hpp"

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

}
}

#endif //VRP_ALGORITHMS_COSTS_HPP