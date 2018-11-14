#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"
#pragma once

#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/solution/Factories.hpp"

namespace vrp::test {

constexpr vrp::models::common::Duration DefaultDuration = 1;
constexpr vrp::models::common::Timestamp DefaultTime = 0;
constexpr vrp::models::common::Location DefaultActorLocation = 0;
constexpr vrp::models::common::Location DefaultJobLocation = 1;
constexpr vrp::models::common::TimeWindow DefaultTimeWindow = {0, 1000};
constexpr vrp::models::problem::Costs DefaultCosts = {100, 1, 1, 1, 1};
const vrp::models::common::Schedule DefaultSchedule = {5, 10};
const vrp::models::common::Dimension DefaultDimension = {"capacity", 1};
const vrp::models::problem::Detail DefaultDetail = {{DefaultJobLocation}, DefaultDuration, {DefaultTimeWindow}};

class test_build_service : public vrp::models::problem::build_service {
public:
  explicit test_build_service() : vrp::models::problem::build_service() {
    withId("service").withDetails({DefaultDetail}).withDimensions({DefaultDimension});
  }
};

inline vrp::models::problem::Job DefaultService = vrp::models::problem::as_job(test_build_service{}.shared());

class test_build_activity : public vrp::models::solution::build_activity {
public:
  explicit test_build_activity() : vrp::models::solution::build_activity() {
    withSchedule(DefaultSchedule).withLocation(DefaultJobLocation).withJob(DefaultService);
  }
};

inline std::shared_ptr<vrp::models::solution::Activity> DefaultActivity = test_build_activity{}.shared();

class test_build_vehicle : public vrp::models::problem::build_vehicle {
public:
  explicit test_build_vehicle() : vrp::models::problem::build_vehicle() {
    withId("vehicle1")
      .withProfile("car")
      .withStart(static_cast<vrp::models::common::Location>(DefaultActorLocation))
      .withEnd(static_cast<vrp::models::common::Location>(DefaultActorLocation))
      .withDimensions({DefaultDimension})
      .withCosts(static_cast<vrp::models::problem::Costs>(DefaultCosts));
  }
};

class test_build_driver : public vrp::models::problem::build_driver {
public:
  explicit test_build_driver() : vrp::models::problem::build_driver() {
    withCosts(static_cast<vrp::models::problem::Costs>(DefaultCosts));
  }
};

inline std::shared_ptr<const vrp::models::problem::Driver> DefaultDriver = test_build_driver{}.shared();

class test_build_actor : public vrp::models::problem::build_actor {
public:
  explicit test_build_actor() : vrp::models::problem::build_actor() {
    withDriver(test_build_driver{}.shared()).withVehicle(test_build_vehicle{}.shared());
  }
};

inline std::shared_ptr<vrp::models::problem::Actor> DefaultActor = test_build_actor{}.shared();

class test_build_route : public vrp::models::solution::build_route {
public:
  explicit test_build_route() : vrp::models::solution::build_route() { withActor(test_build_actor{}.owned()); }
};

}  // namespace vrp::test

#pragma clang diagnostic pop
