#pragma once

#include "models/common/Duration.hpp"
#include "models/common/Location.hpp"
#include "models/common/TimeWindow.hpp"

#include <optional>
#include <vector>

namespace vrp::models::problem {

/// Represents vehicle detail.
struct VehicleDetail {
  /// Location where vehicle starts.
  common::Location start;

  /// Location where vehicle ends.
  std::optional<common::Location> end;

  /// Time windows when vehicle can be used.
  common::TimeWindow time;
};
}