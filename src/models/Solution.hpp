#pragma once

#include "models/problem/Job.hpp"
#include "models/solution/Route.hpp"
#include "models/solution/Statistic.hpp"

#include <memory>
#include <vector>

namespace vrp::models {

// TODO define it better

/// Represents VRP solution.
struct Solution final {
  /// Solution statistic.
  Statistic statisic;

  /// List of assigned routes.
  std::vector<std::shared_ptr<Route>> routes;

  /// List of unassigned jobs.
  std::vector<std::shared_ptr<vrp::models::problem::Job>> unassignedJobs;
};

}  // namespace vrp::models::solution
