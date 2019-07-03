#pragma once

#include "models/common/Dimension.hpp"
#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/common/TimeWindow.hpp"
#include "models/problem/Costs.hpp"

#include <vector>

namespace vrp::models::problem {

/// Represents a driver, person who drives Vehicle.
/// Introduced to allow the following scenarious:
/// * reuse vehicle multiple times with different drivers
/// * solve best driver-vehicle match problem.
struct Driver {
  /// Represents driver detail.
  struct Detail final {
    /// Time windows when driver can work.
    common::TimeWindow time;
  };

  /// Specifies driver costs.
  Costs costs;

  /// Specifies dimensions supported by driver.
  vrp::models::common::Dimensions dimens;

  /// Specifies driver details.
  std::vector<Detail> details;
};

}  // namespace vrp::models::problem
