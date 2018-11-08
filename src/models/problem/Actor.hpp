#pragma once

#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"

#include <memory>

namespace vrp::models::problem {

/// Represents an actor.
struct Actor final {
  std::shared_ptr<const Vehicle> vehicle;
  std::shared_ptr<const Driver> driver;
};

}  // namespace vrp::models::problem