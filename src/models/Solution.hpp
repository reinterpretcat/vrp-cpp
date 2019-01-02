#pragma once

#include "models/common/Cost.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Registry.hpp"
#include "models/solution/Route.hpp"
#include "models/solution/Statistic.hpp"

#include <memory>
#include <vector>

namespace vrp::models {

/// Represents VRP solution.
struct Solution final {
  /// Solution cost.
  common::Cost cost;

  /// Actor's registry.
  std::shared_ptr<solution::Registry> registry;

  /// List of assigned routes.
  std::vector<std::shared_ptr<solution::Route>> routes;

  /// Collection of unassigned jobs.
  std::set<problem::Job, problem::compare_jobs> unassigned;
};

}  // namespace vrp::models::solution
