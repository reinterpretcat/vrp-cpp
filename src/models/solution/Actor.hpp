#pragma once

#include "models/common/TimeWindow.hpp"
#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>

namespace vrp::models::solution {

/// Represents an actor.
struct Actor final {
  /// Represents actor detail.
  struct Detail final {
    /// Location where actor starts.
    common::Location start;

    /// Location where actor ends.
    std::optional<common::Location> end;

    /// Time windows when actor can work.
    common::TimeWindow time;
  };

  /// A vehicle associated within actor.
  std::shared_ptr<const problem::Vehicle> vehicle;

  /// A driver associated within actor.
  std::shared_ptr<const problem::Driver> driver;

  /// Specifies actor detail.
  Detail detail;
};

}  // namespace vrp::models::problem
