#pragma once

#include "algorithms/construction/InsertionRouteState.hpp"
#include "models/solution/Route.hpp"

#include <memory>

namespace vrp::algorithms::construction {

/// Specifies insertion context for route.
struct InsertionRouteContext final {
  /// Route
  std::shared_ptr<models::solution::Route> route;

  /// Insertion state.
  std::shared_ptr<InsertionRouteState> state;
};

}  // namespace vrp::algorithms::construction