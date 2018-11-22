#pragma once

#include "models/extensions/common/Comparators.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Actor.hpp"

#include <utility>

namespace vrp::models::solution {

/// Compares activities.
struct compare_activities final {
  bool operator()(const Activity& lhs, const Activity& rhs) const {
    // TODO compare jobs as well?
    return lhs.detail.location == rhs.detail.location ? common::compare_schedules{}(lhs.schedule, rhs.schedule)
                                                      : lhs.detail.location < rhs.detail.location;
  }
};

/// Compares pairs of activity and key.
struct compare_activities_with_key final {
  bool operator()(const std::pair<Activity, std::string>& lhs, const std::pair<Activity, std::string>& rhs) const {
    return lhs.second == rhs.second ? compare_activities{}(lhs.first, rhs.first) : lhs.second < rhs.second;
  }
};

/// Compares pairs of activity and key.
struct compare_actor_details final {
  bool operator()(const Actor::Detail& lhs, const Actor::Detail& rhs) const {
    if (lhs.start == rhs.start)
      return lhs.end == rhs.end ? common::compare_time_windows{}(lhs.time, rhs.time) : lhs.end < rhs.end;

    return lhs.start < rhs.start;
  }
};

}  // namespace vrp::models::solution
