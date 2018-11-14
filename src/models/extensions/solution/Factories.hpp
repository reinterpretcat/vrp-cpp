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
  build_activity& withType(const Activity::Type& type) {
    activity_.type = type;
    return *this;
  }

  build_activity& withTime(common::TimeWindow&& time) {
    activity_.time = time;
    return *this;
  }

  build_activity& withLocation(const common::Location& location) {
    activity_.location = location;
    return *this;
  }

  build_activity& withSchedule(const common::Schedule& schedule) {
    activity_.schedule = schedule;
    return *this;
  }

  build_activity& withJob(const problem::Job& job) {
    activity_.job = std::make_optional<problem::Job>(job);
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
  build_route& withActor(problem::Actor&& actor) {
    route_.actor = actor;
    return *this;
  }

  build_route& withStart(solution::Tour::Activity start) {
    route_.start = std::move(start);
    return *this;
  }

  build_route& withEnd(solution::Tour::Activity end) {
    route_.end = std::move(end);
    return *this;
  }

  build_route& withTour(solution::Tour&& tour) {
    route_.tour = tour;
    return *this;
  }

  Route&& owned() { return std::move(route_); }

  std::shared_ptr<Route> shared() { return std::make_shared<Route>(std::move(route_)); }

private:
  Route route_;
};

}  // namespace vrp::models::solution
