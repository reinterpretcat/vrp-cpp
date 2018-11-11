#pragma once

#include "models/common/Timestamp.hpp"
#include "models/problem/Actor.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Route.hpp"

#include <memory>

namespace vrp::algorithms::construction {

/// Specifies insertion context for route.
struct InsertionRouteContext final {
  /// A proposed actor to be used.
  std::shared_ptr<models::problem::Actor> actor;

  /// A route where job is supposed to be inserted.
  std::shared_ptr<models::solution::Route> route;

  /// Moment in time (departure from last activity).
  models::common::Timestamp time;
};

}  // namespace vrp::algorithms::construction