#pragma once

#include "models/common/Schedule.hpp"
#include "models/solution/Tour.hpp"

#include <optional>

namespace vrp::algorithms::construction {

/// Specifies insertion context for activity.
struct InsertionActivityContext final {
  /// Insertion index.
  size_t index;

  /// Previous activity.
  models::solution::Tour::Activity prev;

  /// Target activity.
  models::solution::Tour::Activity target;

  /// Next activity. Absent if tour is open and target activity inserted last.
  std::optional<models::solution::Tour::Activity> next;
};
}
