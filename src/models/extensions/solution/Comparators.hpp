#pragma once

#include "models/solution/Activity.hpp"

#include <utility>

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

/// Compares pairs of activity and key.
struct compare_activities_with_key final {
  bool operator()(const std::pair<Activity, std::string>& lhs, const std::pair<Activity, std::string>& rhs) const {
    return lhs.second == rhs.second ? compare_activities{}(lhs.first, rhs.first) : lhs.second < rhs.second;
  }
};

}  // namespace vrp::models::solution
