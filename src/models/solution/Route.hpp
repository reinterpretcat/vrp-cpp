#pragma once

#include "models/solution/Actor.hpp"
#include "models/solution/Tour.hpp"

#include <memory>

namespace vrp::models::solution {

/// Represents a vehicle tour.
struct Route final {
  using Actor = std::shared_ptr<const solution::Actor>;

  /// An actor associated within route.
  Route::Actor actor;

  /// Route start activity.
  Tour::Activity start;

  /// Route end activity.
  Tour::Activity end;

  /// Specifies job tour assigned to this route.
  Tour tour;
};

}  // namespace vrp::models::solution
