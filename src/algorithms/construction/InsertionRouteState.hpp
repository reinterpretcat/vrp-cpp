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
  // region State getters

  template<typename T>
  std::optional<T> get(const std::string& key) const {
    auto value = routeStates_.find(key);
    return value == routeStates_.end() ? std::optional<T>{} : std::make_optional(std::any_cast<T>(value->second));
  }

  template<typename T>
  std::optional<T> get(const std::string& key, const models::solution::Tour::Activity& activity) const {
    auto value = activityStates_.find(std::pair{activity, key});
    return value == activityStates_.end() ? std::optional<T>{} : std::make_optional(std::any_cast<T>(value->second));
  }

  // endregion

  // region State setters

  template<typename T>
  void put(const std::string& key, const T& value) {
    routeStates_[key] = value;
  }

  template<typename T>
  void put(const std::string& key, const models::solution::Tour::Activity& activity, const T& value) {
    activityStates_[std::pair{activity, key}] = value;
  }

  // endregion

private:
  using ActivityHasher = models::solution::hash_activities_with_key;
  using ActivityWithKey = std::pair<models::solution::Tour::Activity, std::string>;

  std::unordered_map<std::string, std::any> routeStates_;
  std::unordered_map<ActivityWithKey, std::any, ActivityHasher> activityStates_;
};
}
