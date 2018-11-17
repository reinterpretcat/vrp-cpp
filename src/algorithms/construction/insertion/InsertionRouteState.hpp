#pragma once

#include "models/extensions/solution/Comparators.hpp"
#include "models/solution/Activity.hpp"

#include <any>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>

namespace vrp::algorithms::construction {

/// Provides the way to associate arbirtary data within route and activity.
struct InsertionRouteState final {
  // region Predefined states

  inline static const std::string FutureWaiting = "future_waiting";

  // endregion

  // region State getters

  template<typename T>
  std::optional<T> get(const std::string& key) const {
    auto value = routeStates_.find(key);
    return value == routeStates_.end() ? std::optional<T>{} : std::make_optional(std::any_cast<T>(value->second));
  }

  template<typename T>
  std::optional<T> get(const std::string& key, const models::solution::Activity& activity) const {
    auto value = activityStates_.find(std::pair{activity, key});
    return value == activityStates_.end() ? std::optional<T>{} : std::make_optional(std::any_cast<T>(value->second));
  }

  // endregion

private:
  using Activity = models::solution::Activity;
  using ActivityComparator = models::solution::compare_activities_with_key;

  std::unordered_map<std::string, std::any> routeStates_;
  std::map<std::pair<Activity, std::string>, std::any, ActivityComparator> activityStates_;
};
}