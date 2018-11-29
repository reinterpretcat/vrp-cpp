#pragma once

#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionRouteState.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Route.hpp"

#include <set>
#include <utility>

namespace vrp::algorithms::construction {

/// Contains information needed to performed insertions in solution.
struct InsertionContext final {
  /// Specifies type which stores together route and its insertion state.
  using RouteState = std::pair<std::shared_ptr<models::solution::Route>, std::shared_ptr<InsertionRouteState>>;
  /// Specifies type which stores together job and its unassignment reason.
  using JobState = std::pair<models::problem::Job, int>;

  /// Solution progress.
  InsertionProgress progress;

  /// Set of jobs which require assignment.
  std::set<models::problem::Job, models::problem::compare_jobs> jobs;

  /// List of unassigned jobs.
  std::vector<JobState> unassigned;

  /// List of routes and their states.
  std::vector<RouteState> routes;
};
}