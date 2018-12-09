#pragma once

#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/problem/Fleet.hpp"
#include "models/problem/Job.hpp"

#include <memory>
#include <set>

namespace vrp::models {

/// Defines VRP problem.
struct Problem final {
  /// Specifies used fleet.
  std::shared_ptr<problem::Fleet> fleet;

  /// Specifies used jobs.
  std::set<models::problem::Job, models::problem::compare_jobs> jobs;

  /// Specifies activity costs.
  std::shared_ptr<costs::ActivityCosts> activity;

  /// Specifies transport costs.
  std::shared_ptr<costs::TransportCosts> transport;
};
}
