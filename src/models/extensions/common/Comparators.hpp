#pragma once

#include "models/common/Schedule.hpp"
#include "models/common/TimeWindow.hpp"

#include <utility>

namespace vrp::models::common {

/// Compares time windows.
struct compare_time_windows final {
  bool operator()(const TimeWindow& lhs, const TimeWindow& rhs) const {
    return lhs.start == rhs.start ? lhs.end < rhs.end : lhs.start < rhs.start;
  }
};

/// Compares schedules.
struct compare_schedules final {
  bool operator()(const Schedule& lhs, const Schedule& rhs) const {
    return lhs.arrival == rhs.arrival ? lhs.departure < rhs.departure : lhs.arrival < rhs.arrival;
  }
};

}  // namespace vrp::models::solution
