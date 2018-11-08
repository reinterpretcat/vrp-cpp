#pragma once

#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>

namespace vrp::models::problem {

/// Represents an actor.
struct Actor final {
  /// A vehicle associated within actor.
  std::shared_ptr<const Vehicle> vehicle;

  /// A driver associated within actor.
  std::shared_ptr<const Driver> driver;
};

}  // namespace vrp::models::problem