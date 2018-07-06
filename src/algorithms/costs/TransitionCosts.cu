#include "algorithms/costs/TransitionCosts.hpp"

using namespace vrp::algorithms::costs;
using namespace vrp::models;

float calculate_transition_cost::operator()(const Transition& transition) const {
  assert(transition.isValid());

  int vehicle = transition.details.vehicle;

  auto distance = transition.delta.distance * problem.resources.distanceCosts[vehicle];
  auto traveling = transition.delta.traveling * problem.resources.timeCosts[vehicle];
  auto waiting = transition.delta.waiting * problem.resources.waitingCosts[vehicle];
  auto serving = transition.delta.serving * problem.resources.timeCosts[vehicle];

  return distance + traveling + waiting + serving;
}
