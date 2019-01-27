#pragma once

#include "models/common/Schedule.hpp"
#include "models/common/Timestamp.hpp"
#include "models/solution/Tour.hpp"

namespace vrp::algorithms::construction {

/// Specifies insertion context for activity.
struct InsertionActivityContext final {
  /// Insertion index.
  size_t index;

  /// A new departure from previous activity.
  models::common::Timestamp departure;

  /// Previous activity.
  models::solution::Tour::Activity prev;

  /// Target activity.
  models::solution::Tour::Activity target;

  /// Next activity.
  models::solution::Tour::Activity next;
};
}
