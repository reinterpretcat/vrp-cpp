#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionRouteState.hpp"
#include "algorithms/construction/extensions/Comparators.hpp"
#include "models/Problem.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Registry.hpp"
#include "models/solution/Route.hpp"
#include "utils/Random.hpp"

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Contains information needed to performed insertions in solution.
struct InsertionContext final {
  /// Solution progress.
  InsertionProgress progress;

  /// Keeps track of used resources.
  std::shared_ptr<models::solution::Registry> registry;

  /// Original problem.
  std::shared_ptr<const models::Problem> problem;

  /// List of jobs which require assignment.
  std::vector<models::problem::Job> jobs;

  /// Map of unassigned jobs within reason code.
  std::map<models::problem::Job, int, models::problem::compare_jobs> unassigned;

  /// Set of routes within their state.
  std::set<InsertionRouteContext, compare_insertion_route_contexts> routes;

  /// Random generator.
  std::shared_ptr<utils::Random> random;
};
}