#pragma once

#include "models/solution/Actor.hpp"
#include "models/solution/Tour.hpp"

#include <memory>

namespace vrp::models::solution {

/// Represents a tour performing jobs.
struct Route final {
  using Actor = std::shared_ptr<const solution::Actor>;

  /// An actor associated within route.
  Route::Actor actor;

  /// Specifies job tour assigned to this route.
  Tour tour;
};

}  // namespace vrp::models::solution
