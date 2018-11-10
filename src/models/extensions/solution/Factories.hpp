#pragma once

#include "models/solution/Activity.hpp"
#include "models/solution/Route.hpp"

#include <memory>

namespace vrp::models::solution {

/// A helper class to build activity.
class build_activity {
public:
  build_activity& withSchedule(common::Schedule&& schedule) {
    activity_.schedule = schedule;
    return *this;
  }

  build_activity& withLocation(common::Location&& location) {
    activity_.location = location;
    return *this;
  }

  build_activity& withJob(std::shared_ptr<const models::problem::Job> job) {
    activity_.job = job;
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
