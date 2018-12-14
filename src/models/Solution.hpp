#pragma once

#include "models/common/Cost.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Route.hpp"
#include "models/solution/Statistic.hpp"

#include <memory>
#include <vector>

namespace vrp::models {

/// Represents VRP solution.
struct Solution final {
  /// Solution cost.
  common::Cost cost;

  /// List of assigned routes.
  std::vector<std::shared_ptr<solution::Route>> routes;

  /// List of unassigned jobs.
  std::vector<std::shared_ptr<problem::Job>> unassignedJobs;
};

}  // namespace vrp::models::solution
