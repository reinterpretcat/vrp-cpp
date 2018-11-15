#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"
#pragma once

#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/solution/Factories.hpp"

namespace vrp::test {

constexpr vrp::models::common::Duration DefaultDuration = 1;
constexpr vrp::models::common::Timestamp DefaultTime = 0;
constexpr vrp::models::common::Location DefaultActorLocation = 0;
constexpr vrp::models::common::Location DefaultJobLocation = 5;
constexpr vrp::models::common::TimeWindow DefaultTimeWindow = {0, 1000};
constexpr vrp::models::problem::Costs DefaultCosts = {100, 1, 1, 1, 1};
const vrp::models::common::Dimension DefaultDimension = {"capacity", 1};
const vrp::models::problem::Detail DefaultDetail = {{DefaultJobLocation}, DefaultDuration, {DefaultTimeWindow}};

class test_build_service : public vrp::models::problem::build_service {
public:
  explicit test_build_service() : vrp::models::problem::build_service() {
    id("service").details({DefaultDetail}).dimensions({DefaultDimension});
  }
};

inline vrp::models::problem::Job DefaultService = vrp::models::problem::as_job(test_build_service{}.shared());

class test_build_activity : public models::solution::build_activity {
public:
  explicit test_build_activity() : models::solution::build_activity() {
    duration(DefaultDuration)
      .type(models::solution::Activity::Type::Job)
      .location(DefaultJobLocation)
      .job(DefaultService);
  }
};

inline std::shared_ptr<vrp::models::solution::Activity> DefaultActivity = test_build_activity{}.shared();

class test_build_vehicle : public vrp::models::problem::build_vehicle {
public:
  explicit test_build_vehicle() : vrp::models::problem::build_vehicle() {
    id("vehicle1").profile("car").start(DefaultActorLocation).dimensions({DefaultDimension}).costs(DefaultCosts);
  }
};

class test_build_driver : public vrp::models::problem::build_driver {
public:
  explicit test_build_driver() : vrp::models::problem::build_driver() { costs({0, 0, 0, 0}); }
};

inline std::shared_ptr<const vrp::models::problem::Driver> DefaultDriver = test_build_driver{}.shared();

class test_build_actor : public vrp::models::problem::build_actor {
public:
  explicit test_build_actor() : vrp::models::problem::build_actor() {
    driver(test_build_driver{}.shared()).vehicle(test_build_vehicle{}.shared());
  }
};

inline std::shared_ptr<vrp::models::problem::Actor> DefaultActor = test_build_actor{}.shared();

class test_build_route : public vrp::models::solution::build_route {
public:
  explicit test_build_route() : vrp::models::solution::build_route() {
    using namespace vrp::models::solution;
    actor(test_build_actor{}.owned())
      .start(test_build_activity{}
               .duration(0)
               .schedule({0, 1000})
               .location(DefaultActorLocation)
               .type(Activity::Type::Start)
               .shared())
      .end(test_build_activity{}
             .duration(0)
             .schedule({0, 1000})
             .location(DefaultActorLocation)
             .type(Activity::Type::End)
             .shared());
  }
};

}  // namespace vrp::test

#pragma clang diagnostic pop
