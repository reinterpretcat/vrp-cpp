#ifndef VRP_ALGORITHMS_COSTS_HPP
#define VRP_ALGORITHMS_COSTS_HPP

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {

/// Calculates costs for transition.
struct CalculateCost {
  const vrp::models::Resources::Shadow resources;

  explicit CalculateCost(const vrp::models::Resources::Shadow &resources) :
      resources(resources) {}

  __host__ __device__
  float operator()(const vrp::models::Transition &transition) const {
    int vehicle = transition.vehicle;

    auto distance = transition.distance * resources.distanceCosts[vehicle];
    auto traveling = transition.traveling * resources.timeCosts[vehicle];
    auto waiting = transition.waiting * resources.waitingCosts[vehicle];
    auto serving = transition.serving * resources.timeCosts[vehicle];

    return distance + traveling + waiting + serving;
  }
};

}
}

#endif //VRP_ALGORITHMS_COSTS_HPP