#include "algorithms/costs/TransitionCosts.hpp"

using namespace vrp::algorithms::costs;
using namespace vrp::models;

float calculate_transition_cost::operator()(const Transition& transition) const {
  int vehicle = transition.details.vehicle;

  auto distance = transition.delta.distance * resources.distanceCosts[vehicle];
  auto traveling = transition.delta.traveling * resources.timeCosts[vehicle];
  auto waiting = transition.delta.waiting * resources.waitingCosts[vehicle];
  auto serving = transition.delta.serving * resources.timeCosts[vehicle];

  return distance + traveling + waiting + serving;
}
