#pragma once

#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionRouteState.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Route.hpp"

#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Contains information needed to performed insertions in solution.
struct InsertionContext final {
  /// Specfies type which stores together route and its insertion state.
  using RouteState = std::pair<std::shared_ptr<models::solution::Route>, std::shared_ptr<InsertionRouteState>>;

  /// Solution progress.
  InsertionProgress progress;

  /// List of jobs which still require assignment.
  std::vector<models::problem::Job> jobs;

  /// List of routes and their states.
  std::vector<RouteState> routes;
};
}