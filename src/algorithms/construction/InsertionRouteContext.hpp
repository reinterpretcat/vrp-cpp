#pragma once

#include "algorithms/construction/InsertionRouteState.hpp"
#include "models/common/Timestamp.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Actor.hpp"
#include "models/solution/Route.hpp"

#include <memory>

namespace vrp::algorithms::construction {

/// Specifies insertion context for route.
struct InsertionRouteContext final {
  /// Specifies type which keeps reference to route and state together.
  using RouteState = std::pair<std::shared_ptr<models::solution::Route>, std::shared_ptr<InsertionRouteState>>;

  /// A route where job is supposed to be inserted.
  RouteState route;

  /// A proposed actor to be used. Might be different from one used with route.
  models::solution::Route::Actor actor;

  /// New departure time from start.
  models::common::Timestamp departure;
};

}  // namespace vrp::algorithms::construction