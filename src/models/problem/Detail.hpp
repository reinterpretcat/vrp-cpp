#pragma once

#include "models/common/Duration.hpp"
#include "models/common/Location.hpp"
#include "models/common/TimeWindow.hpp"

#include <optional>
#include <vector>

namespace vrp::models::problem {

/// Represents a work which has to be performed.
struct Detail final {
  /// Location where work has to be performed.
  std::optional<common::Location> location;

  /// Time has to be spend performing work.
  common::Duration duration;

  /// Time windows when work can be performed.
  std::vector<common::TimeWindow> times;
};
}
