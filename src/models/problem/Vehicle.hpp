#pragma once

#include "Costs.hpp"
#include "Driver.hpp"
#include "models/common/Dimension.hpp"
#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"

#include <optional>
#include <string>
#include <vector>

namespace vrp::models::problem {

/// Represents a vehicle.
struct Vehicle final {
  /// Vehicle id.
  std::string id;

  /// Vehicle profile.
  std::string profile;

  /// Specifies vehicle costs.
  Costs costs;

  /// Specifies departure/arrival schedule limits.
  vrp::models::common::Schedule schedule;

  /// Specifies dimensions supported by vehicle.
  vrp::models::common::Dimensions dimensions;

  /// Start vehicle location.
  vrp::models::common::Location start;

  /// End vehicle location.
  std::optional<vrp::models::common::Location> end;
};

}  // namespace vrp::models::problem
