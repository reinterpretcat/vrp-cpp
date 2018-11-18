#pragma once

#include "models/common/Dimension.hpp"
#include "models/common/Schedule.hpp"
#include "models/problem/Costs.hpp"

namespace vrp::models::problem {

/// Represents a driver, person who drives Vehicle
/// Introduced to allow future extensions:
/// * reuse vehicle multiple times with different drivers
/// * solve best driver-vehicle match problem.
struct Driver {
  /// Specifies driver id.
  std::string id;

  /// Specifies driver costs.
  Costs costs;

  /// Specifies dimensions supported by driver.
  vrp::models::common::Dimensions dimensions;
};

}  // namespace vrp::models::problem
