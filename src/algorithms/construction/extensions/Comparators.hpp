#pragma once

#include "models/extensions/solution/Comparators.hpp"

namespace vrp::algorithms::construction {

/// Compares insertion route contexts by their actor.
struct compare_insertion_route_contexts final {
  bool operator()(const InsertionRouteContext& lhs, const InsertionRouteContext& rhs) const {
    if (lhs.route->actor->vehicle->id == rhs.route->actor->vehicle->id)
      return models::solution::compare_actor_details{}(lhs.route->actor->detail, rhs.route->actor->detail);

    return lhs.route->actor->vehicle->id < rhs.route->actor->vehicle->id;
  }
};
}