#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionRouteState.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Registry.hpp"
#include "models/solution/Route.hpp"

#include <map>
#include <set>
#include <utility>

namespace vrp::algorithms::construction {

/// Contains information needed to performed insertions in solution.
struct InsertionContext final {
  /// Solution progress.
  InsertionProgress progress;

  /// Keeps track of used resources.
  std::shared_ptr<models::solution::Registry> registry;

  /// Used constraint.
  std::shared_ptr<InsertionConstraint> constraint;

  /// Set of jobs which require assignment.
  std::set<models::problem::Job, models::problem::compare_jobs> jobs;

  /// Map of unassigned jobs within reason code.
  std::map<models::problem::Job, int, models::problem::compare_jobs> unassigned;

  /// Map of routes within their state.
  std::map<std::shared_ptr<models::solution::Route>, std::shared_ptr<InsertionRouteState>> routes;
};
}