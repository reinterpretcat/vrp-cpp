#pragma once

#include "Costs.hpp"
#include "Driver.hpp"
#include "models/common/Dimension.hpp"
#include "models/common/Location.hpp"
#include "models/common/TimeWindow.hpp"
#include "models/problem/VehicleDetail.hpp"

#include <optional>
#include <string>
#include <vector>

namespace vrp::models::problem {

/// Represents a vehicle.
struct Vehicle final {
  /// Vehicle id.
  std::string id;

  /// Specifies vehicle transport cost profile.
  std::string profile;

  /// Specifies vehicle costs.
  Costs costs;

  /// Specifies dimensions supported by vehicle.
  vrp::models::common::Dimensions dimens;

  /// Specifies vehicle details.
  std::vector<VehicleDetail> details;
};

}  // namespace vrp::models::problem
