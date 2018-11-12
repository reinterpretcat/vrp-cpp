#pragma once

#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"

namespace vrp::models::solution {

/// Specifies stop information.
struct Stop final {
  /// Specifies stop's schedule: actual arrival and departure time.
  common::Schedule schedule;

  /// Location where stop is performed.
  common::Location location;
};
}
