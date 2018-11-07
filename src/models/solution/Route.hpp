#pragma once

#include "models/problem/Driver.hpp"
#include "models/problem/Vehicle.hpp"
#include "models/solution/Tour.hpp"

#include <memory>

namespace vrp::models::solution {
/// Represents a vehicle tour.
struct Route final {
  /// Vehicle associated within route.
  std::shared_ptr<problem::Vehicle> vehicle;

  /// Driver associated within route.
  std::shared_ptr<problem::Driver> driver;

  /// Specifies job tour assigned to this route.
  Tour tour;
};

}  // namespace vrp::models::solution
