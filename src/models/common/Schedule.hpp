#pragma once

#include "Timestamp.hpp"

namespace vrp::models::common {

/// Represents a schedule.
struct Schedule final {
  /// Departure time.
  Timestamp departure;

  /// Arrival time.
  Timestamp arrival;
};

}  // namespace vrp::models::common
