#pragma once

#include "Timestamp.hpp"

namespace vrp::models::common {

/// Represents a schedule.
struct Schedule final {
  /// Arrival time.
  Timestamp arrival;

  /// Departure time.
  Timestamp departure;
};

}  // namespace vrp::models::common
