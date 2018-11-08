#pragma once

#include "models/problem/Actor.hpp"
#include "models/solution/Tour.hpp"

#include <memory>

namespace vrp::models::solution {

/// Represents a vehicle tour.
struct Route final {
  /// An actor associated within route.
  problem::Actor actor;

  /// Specifies job tour assigned to this route.
  solution::Tour tour;
};

}  // namespace vrp::models::solution
