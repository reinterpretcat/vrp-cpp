#pragma once

#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Route.hpp"

#include <memory>

namespace vrp::models::solution {

/// A helper class to build stop.
class build_stop {
public:
  build_stop& withLocation(const common::Location& location) {
    stop_.location = location;
    return *this;
  }

  build_stop& withSchedule(const common::Schedule& schedule) {
    stop_.schedule = schedule;
    return *this;
  }

  Stop&& owned() { return std::move(stop_); }

private:
  Stop stop_;
};

/// A helper class to build activity.
class build_activity {
public:
  build_activity& withInterval(common::TimeWindow&& interval) {
    activity_.interval = interval;
    return *this;
  }

  build_activity& withStop(solution::Stop&& stop) {
    activity_.stop = stop;
    return *this;
  }

  build_activity& withJob(const problem::Job& job) {
    activity_.job = std::make_optional<problem::Job>(job);
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

  build_route& withStart(solution::Stop&& start) {
    route_.start = start;
    return *this;
  }

  build_route& withEnd(solution::Stop&& end) {
    route_.end = end;
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
