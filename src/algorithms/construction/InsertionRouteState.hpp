#pragma once

#include "models/extensions/solution/Comparators.hpp"
#include "models/solution/Activity.hpp"

#include <any>
#include <map>
#include <optional>
#include <range/v3/all.hpp>
#include <set>
#include <string>
#include <unordered_map>

namespace vrp::algorithms::construction {

/// Provides the way to associate arbitrary data within route and activity.
struct InsertionRouteState final {
  InsertionRouteState() = default;

  explicit InsertionRouteState(std::pair<size_t, size_t> sizes) : keys_(), routeStates_(), activityStates_() {
    routeStates_.reserve(sizes.first);
    activityStates_.reserve(sizes.second);
  }

  // region State getters

  /// Gets value associated with key.
  template<typename T>
  std::optional<T> get(const std::string& key) const {
    auto value = routeStates_.find(key);
    return value == routeStates_.end() ? std::optional<T>{} : std::make_optional(std::any_cast<T>(value->second));
  }

  std::optional<std::any> get(const std::string& key) const {
    auto value = routeStates_.find(key);
    return value == routeStates_.end() ? std::optional<std::any>{} : std::make_optional(value->second);
  }

  /// Gets typed value associated with key and specific activity.
  template<typename T>
  std::optional<T> get(const std::string& key, const models::solution::Tour::Activity& activity) const {
    auto value = activityStates_.find(std::pair{activity, key});
    return value == activityStates_.end() ? std::optional<T>{} : std::make_optional(std::any_cast<T>(value->second));
  }

  /// Gets untyped value associated with key and specific activity
  std::optional<std::any> get(const std::string& key, const models::solution::Tour::Activity& activity) const {
    auto value = activityStates_.find(std::pair{activity, key});
    return value == activityStates_.end() ? std::optional<std::any>{} : std::optional<std::any>{value->second};
  }

  // endregion

  // region State setters

  /// Puts value associated with key.
  template<typename T>
  void put(const std::string& key, const T& value) {
    routeStates_[key] = value;
    keys_.insert(key);
  }

  /// Puts value associated with key and specific activity.
  template<typename T>
  void put(const std::string& key, const models::solution::Tour::Activity& activity, const T& value) {
    activityStates_[std::pair{activity, key}] = value;
    keys_.insert(key);
  }

  /// Removes all states for given activity.
  void remove(const models::solution::Tour::Activity& activity) {
    ranges::for_each(keys_, [&](const auto& key) {
      activityStates_.erase({std::pair{activity, key}});
    });
  }

  // endregion

  // region Discovery

  /// Returns all registered keys.
  ranges::any_view<std::string> keys() const { return keys_; }

  /// Returns size of internal storages.
  std::pair<size_t, size_t> sizes() const { return {routeStates_.size(), activityStates_.size()}; }

  // endregion

private:
  using ActivityHasher = models::solution::hash_activities_with_key;
  using ActivityWithKey = std::pair<models::solution::Tour::Activity, std::string>;

  std::set<std::string> keys_;
  std::unordered_map<std::string, std::any> routeStates_;
  std::unordered_map<ActivityWithKey, std::any, ActivityHasher> activityStates_;
};
}
