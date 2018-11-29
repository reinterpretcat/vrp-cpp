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
  /// A route where job is supposed to be inserted.
  InsertionContext::RouteState route;

  /// A proposed actor to be used. Might be different from one used with route.
  models::solution::Route::Actor actor;

  /// New departure time from start.
  models::common::Timestamp departure;
};

}  // namespace vrp::algorithms::construction