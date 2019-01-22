#pragma once

#include "models/extensions/common/Comparators.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Actor.hpp"
#include "models/solution/Tour.hpp"

#include <memory>
#include <utility>

namespace vrp::models::solution {

/// Compares pairs of activity and key.
struct compare_activities_with_key final {
  bool operator()(const std::pair<Tour::Activity, std::string>& lhs,
                  const std::pair<Tour::Activity, std::string>& rhs) const {
    return lhs.second == rhs.second ? lhs.first.get() < rhs.first.get() : lhs.second < rhs.second;
  }
};

/// Compares pairs of actor details and key.
struct compare_actor_details final {
  bool operator()(const Actor::Detail& lhs, const Actor::Detail& rhs) const {
    if (lhs.start == rhs.start)
      return lhs.end == rhs.end ? common::compare_time_windows{}(lhs.time, rhs.time) : lhs.end < rhs.end;

    return lhs.start < rhs.start;
  }
};

}  // namespace vrp::models::solution
