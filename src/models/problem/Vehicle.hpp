#pragma once

#include "Costs.hpp"
#include "Driver.hpp"
#include "models/common/Dimension.hpp"
#include "models/common/Location.hpp"
#include "models/common/Profile.hpp"
#include "models/common/TimeWindow.hpp"

#include <optional>
#include <string>
#include <vector>

namespace vrp::models::problem {

/// Represents a vehicle.
struct Vehicle final {
  /// Represents vehicle detail.
  struct Detail {
    /// Location where vehicle starts.
    common::Location start;

    /// Location where vehicle ends.
    std::optional<common::Location> end;

    /// Time windows when vehicle can be used.
    common::TimeWindow time;
  };

  /// Specifies vehicle transport cost profile id.
  common::Profile profile;

  /// Specifies vehicle costs.
  Costs costs;

  /// Specifies dimensions supported by vehicle.
  vrp::models::common::Dimensions dimens;

  /// Specifies vehicle details.
  std::vector<Detail> details;
};

}  // namespace vrp::models::problem
