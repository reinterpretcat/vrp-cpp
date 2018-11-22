#pragma once

#include "models/common/Location.hpp"
#include "models/common/Schedule.hpp"
#include "models/solution/Activity.hpp"
#include "models/solution/Actor.hpp"
#include "models/solution/Route.hpp"

#include <memory>
#include <numeric>

namespace vrp::models::solution {

/// A helper class to build activity.
class build_activity {
public:
  build_activity& type(const Activity::Type& value) {
    activity_.type = value;
    return *this;
  }

  build_activity& detail(const Activity::Detail& value) {
    activity_.detail = value;
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

  build_actor& start(const common::Location& value) {
    actor_.start = value;
    return *this;
  }

  build_actor& end(const common::Location& value) {
    actor_.end = value;
    return *this;
  }

  build_actor& time(const common::TimeWindow& value) {
    actor_.time = value;
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
    route_.actor = std::move(value);
    return *this;
  }

  build_route& start(Tour::Activity value) {
    assert(value->type == Activity::Type::Start);
    route_.start = std::move(value);
    return *this;
  }

  build_route& end(Tour::Activity value) {
    assert(value->type == Activity::Type::End);
    route_.end = std::move(value);
    return *this;
  }

  build_route& tour(Tour&& value) {
    route_.tour = value;
    return *this;
  }

  Route&& owned() { return build(); }

  std::shared_ptr<Route> shared() { return std::make_shared<Route>(build()); }

private:
  Route&& build() {
    assert(route_.start != nullptr && route_.end != nullptr);
    return std::move(route_);
  }

  Route route_;
};

}  // namespace vrp::models::solution
