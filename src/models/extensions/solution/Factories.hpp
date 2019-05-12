#pragma once

#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Actor.hpp"
#include "models/solution/Route.hpp"

#include <gsl/gsl>
#include <memory>
#include <numeric>

namespace vrp::models::solution {

/// A helper class to build activity.
class build_activity {
public:
  build_activity& detail(const Activity::Detail& value) {
    activity_.detail = value;
    return *this;
  }

  build_activity& schedule(const common::Schedule& value) {
    activity_.schedule = value;
    return *this;
  }

  build_activity& service(const std::shared_ptr<const problem::Service>& value) {
    activity_.service = value;
    return *this;
  }

  Activity&& owned() { return std::move(activity_); }

  std::shared_ptr<Activity> shared() { return std::make_shared<Activity>(std::move(activity_)); }

protected:
  Activity activity_;
};

/// A helper class to build actor.
class build_actor {
public:
  build_actor& driver(const std::shared_ptr<const problem::Driver>& value) {
    actor_.driver = value;
    return *this;
  }

  build_actor& vehicle(const std::shared_ptr<const problem::Vehicle>& value) {
    actor_.vehicle = value;
    return *this;
  }

  build_actor& detail(const Actor::Detail& detail) {
    actor_.detail = detail;
    return *this;
  }

  Actor&& owned() { return std::move(actor_); }

  std::shared_ptr<Actor> shared() { return std::make_shared<Actor>(std::move(actor_)); }

private:
  Actor actor_;
};

/// A helper class to build route.
class build_route {
public:
  build_route& actor(Route::Actor value) {
    route_->actor = std::move(value);
    return *this;
  }

  build_route& start(Tour::Activity value) {
    route_->tour.start(value);
    return *this;
  }

  build_route& end(Tour::Activity value) {
    route_->tour.end(value);
    return *this;
  }

  std::shared_ptr<Route> shared() {
    Expects(!route_->tour.empty());
    return route_;
  }

private:
  std::shared_ptr<Route> route_ = std::make_shared<Route>();
};

}  // namespace vrp::models::solution
