#pragma once

#include "algorithms/construction/InsertionConstraint.hpp"
#include "algorithms/objectives/ObjectiveFunction.hpp"
#include "models/Lock.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "models/problem/Jobs.hpp"

#include <any>
#include <map>
#include <memory>
#include <vector>

namespace vrp::models {

/// Defines VRP problem.
struct Problem final {
  /// Specifies used fleet.
  std::shared_ptr<const problem::Fleet> fleet;

  /// Specifies all job.
  std::shared_ptr<const problem::Jobs> jobs;

  /// Specifies jobs which preassigned to specific vehicles and/or drivers.
  std::shared_ptr<const std::vector<Lock>> locks;

  /// Specifies constraints.
  std::shared_ptr<const algorithms::construction::InsertionConstraint> constraint;

  /// Specifies objective function.
  std::shared_ptr<const algorithms::objectives::ObjectiveFunction> objective;

  /// Specifies activity costs.
  std::shared_ptr<const costs::ActivityCosts> activity;

  /// Specifies transport costs.
  std::shared_ptr<const costs::TransportCosts> transport;

  /// Specifies index for storing extra parameters of arbitrary type.
  std::shared_ptr<const std::map<std::string, std::any>> extras;
};
}
