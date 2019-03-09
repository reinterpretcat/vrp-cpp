#pragma once

#include "algorithms/construction/InsertionRouteContext.hpp"
#include "algorithms/construction/extensions/Comparators.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Route.hpp"

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace vrp::algorithms::construction {

/// Contains information regarding insertion solution.
struct InsertionSolutionContext final {
  /// List of jobs which require permanent assignment.
  std::vector<models::problem::Job> required;

  /// List of jobs which at the moment does not require assignment and might be ignored.
  std::set<models::problem::Job, models::problem::compare_jobs> optional;

  /// Map of jobs within reason code.
  std::map<models::problem::Job, int, models::problem::compare_jobs> unassigned;

  /// Set of routes within their state.
  std::set<InsertionRouteContext, compare_insertion_route_contexts> routes;
};
}
