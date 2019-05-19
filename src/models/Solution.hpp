#pragma once

#include "models/common/Cost.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Registry.hpp"
#include "models/solution/Route.hpp"
#include "models/solution/Statistic.hpp"

#include <any>
#include <map>
#include <memory>
#include <vector>

namespace vrp::models {

/// Represents a VRP solution.
struct Solution final {
  /// Actor's registry.
  std::shared_ptr<const solution::Registry> registry;

  /// List of assigned routes.
  std::vector<std::shared_ptr<const solution::Route>> routes;

  /// Map of unassigned jobs within reason code.
  std::map<models::problem::Job, int, models::problem::compare_jobs> unassigned;

  /// Specifies index for storing extra data of arbitrary type.
  std::shared_ptr<const std::map<std::string, std::any>> extras;
};

/// Specifies solution with its cost.
using EstimatedSolution = std::pair<std::shared_ptr<const models::Solution>, models::common::ObjectiveCost>;

}  // namespace vrp::models::solution
