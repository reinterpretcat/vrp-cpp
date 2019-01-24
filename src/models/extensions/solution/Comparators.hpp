#pragma once

#include "models/extensions/common/Comparators.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Actor.hpp"
#include "models/solution/Tour.hpp"
#include "utils/extensions/Hash.hpp"

#include <memory>
#include <utility>

namespace vrp::models::solution {

/// Creates a hash from activity and key.
struct hash_activities_with_key final {
  std::size_t operator()(const std::pair<Tour::Activity, std::string>& item) const {
    auto hash1 = std::hash<Tour::Activity>{}(item.first);
    auto hash2 = std::hash<std::string>{}(item.second);
    return hash1 | utils::hash_combine<size_t>{hash2};
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
