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
  /// Represents driver detail.
  struct Detail final {
    // TODO
  };

  /// Specifies driver costs.
  Costs costs;

  /// Specifies dimensions supported by driver.
  vrp::models::common::Dimensions dimens;
};

}  // namespace vrp::models::problem
