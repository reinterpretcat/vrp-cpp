#pragma once

#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Route.hpp"

#include <memory>

namespace vrp::models::solution {

/// A helper class to build activity.
class build_activity {
public:
  build_activity& type(const Activity::Type& value) {
    activity_.type = value;
    return *this;
  }

  build_activity& time(const common::TimeWindow& value) {
    activity_.time = value;
    return *this;
  }

  build_activity& duration(const common::Duration& value) {
    activity_.duration = value;
    return *this;
  }

  build_activity& location(const common::Location& value) {
    activity_.location = value;
    return *this;
  }

  build_activity& schedule(const common::Schedule& value) {
    activity_.schedule = value;
    return *this;
  }

  build_activity& job(const problem::Job& value) {
    activity_.job = std::make_optional<problem::Job>(value);
    activity_.type = Activity::Type::Job;
    return *this;
  }

  Activity&& owned() { return std::move(activity_); }

  std::shared_ptr<Activity> shared() { return std::make_shared<Activity>(std::move(activity_)); }

private:
  Activity activity_;
};

/// A helper class to build route.
class build_route {
public:
  build_route& actor(problem::Actor&& value) {
    route_.actor = value;
    return *this;
  }

  build_route& start(solution::Tour::Activity value) {
    assert(value->type == Activity::Type::Start);
    route_.start = std::move(value);
    return *this;
  }

  build_route& end(solution::Tour::Activity value) {
    assert(value->type == Activity::Type::End);
    route_.end = std::move(value);
    return *this;
  }

  build_route& tour(solution::Tour&& value) {
    route_.tour = value;
    return *this;
  }

  Route&& owned() { return std::move(route_); }

  std::shared_ptr<Route> shared() { return std::make_shared<Route>(std::move(route_)); }

private:
  Route route_;
};

}  // namespace vrp::models::solution
