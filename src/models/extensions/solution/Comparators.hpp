#pragma once

#include "models/solution/Activity.hpp"

namespace vrp::models::solution {

/// Compares activities.
struct compare_activities final {
  bool operator()(const Activity& lhs, const Activity& rhs) const {
    if (lhs.location == rhs.location) {
      if (lhs.schedule.arrival == rhs.schedule.arrival) { return lhs.schedule.departure < rhs.schedule.departure; }
      return lhs.schedule.arrival < rhs.schedule.arrival;
    }
    // TODO compare jobs as well?
    return lhs.location < rhs.location;
  }
};

}  // namespace vrp::models::solution
