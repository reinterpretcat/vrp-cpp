#pragma once

#include "models/common/Schedule.hpp"
#include "models/problem/Costs.hpp"

namespace vrp::models::problem {

/// Represents a driver, person who drives Vehicle
/// Introduced to allow future extensions:
/// * reuse vehicle multiple times with different drivers
/// * solve best driver-vehicle match problem.
struct Driver {
  /// Specifies driver costs.
  Costs costs;

  /// Specifies departure/arrival schedule limits.
  vrp::models::common::Schedule schedule;
};

}  // namespace vrp::models::problem::fleet
