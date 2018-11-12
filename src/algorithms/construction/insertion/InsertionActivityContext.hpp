#pragma once

#include "models/common/Schedule.hpp"
#include "models/solution/Tour.hpp"

namespace vrp::algorithms::construction {

/// Specifies insertion context for activity.
struct InsertionActivityContext final {
  /// Insertion index.
  int index;

  /// Proposed activity schedule
  models::common::Schedule schedule;

  /// Previous activity.
  models::solution::Tour::Activity previous;

  /// Target activity.
  models::solution::Tour::Activity target;

  /// Next activity.
  models::solution::Tour::Activity next;
};
}
