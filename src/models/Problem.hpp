#pragma once

#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "models/problem/Jobs.hpp"

#include <memory>

namespace vrp::models {

/// Defines VRP problem.
struct Problem final {
  /// Specifies used fleet.
  std::shared_ptr<problem::Fleet> fleet;

  /// Specifies used jobs.
  std::shared_ptr<problem::Jobs> jobs;

  /// Specifies activity costs.
  std::shared_ptr<costs::ActivityCosts> activity;

  /// Specifies transport costs.
  std::shared_ptr<costs::TransportCosts> transport;
};
}
