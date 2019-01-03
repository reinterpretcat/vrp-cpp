#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "models/problem/Jobs.hpp"

#include <memory>

namespace vrp::models {

/// Defines VRP problem.
struct Problem final {
  /// Specifies used fleet.
  std::shared_ptr<const problem::Fleet> fleet;

  /// Specifies used jobs.
  std::shared_ptr<const problem::Jobs> jobs;

  /// Specifies constraints.
  std::shared_ptr<const algorithms::construction::InsertionConstraint> constraint;

  /// Specifies activity costs.
  std::shared_ptr<const costs::ActivityCosts> activity;

  /// Specifies transport costs.
  std::shared_ptr<const costs::TransportCosts> transport;
};
}
