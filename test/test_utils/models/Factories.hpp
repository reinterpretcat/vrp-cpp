#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"
#pragma once

#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/problem/Fleet.hpp"

namespace vrp::test {

constexpr vrp::models::common::Duration DefaultDuration = 0;
constexpr vrp::models::common::Timestamp DefaultTime = 0;
constexpr vrp::models::common::Location DefaultActorLocation = 0;
constexpr vrp::models::common::Location DefaultJobLocation = 5;
constexpr vrp::models::common::TimeWindow DefaultTimeWindow = {0, 1000};
constexpr vrp::models::problem::Costs DefaultCosts = {100, 1, 1, 1, 1};
const vrp::models::common::Dimension DefaultDimension = {"capacity", 1};

const vrp::models::problem::JobDetail DefaultJobDetail = {{DefaultJobLocation}, DefaultDuration, {DefaultTimeWindow}};
const vrp::models::problem::VehicleDetail DefaultVehicleDetail = {DefaultActorLocation, {}, DefaultTimeWindow};

class test_build_service : public vrp::models::problem::build_service {
public:
  explicit test_build_service() : vrp::models::problem::build_service() {
    id("service").details({DefaultJobDetail}).dimensions({DefaultDimension});
  }
};

class test_build_detail : public vrp::models::problem::build_detail {
public:
  explicit test_build_detail() : vrp::models::problem::build_detail() {
    location(DefaultJobLocation).duration(DefaultDuration).times({DefaultTimeWindow});
  }
};

inline vrp::models::problem::Job DefaultService = vrp::models::problem::as_job(test_build_service{}.shared());

class test_build_activity : public models::solution::build_activity {
public:
  explicit test_build_activity() : models::solution::build_activity() {
    type(models::solution::Activity::Type::Job)
      .detail({DefaultJobLocation, DefaultDuration, DefaultTimeWindow})
      .schedule({0, 0})
      .job(DefaultService);
  }

  test_build_activity& location(const models::common::Location& value) {
    activity_.detail.location = value;
    return *this;
  }

  test_build_activity& duration(const models::common::Duration& value) {
    activity_.detail.duration = value;
    return *this;
  }

  test_build_activity& time(const models::common::TimeWindow& value) {
    activity_.detail.time = value;
    return *this;
  }
};

inline std::shared_ptr<vrp::models::solution::Activity> DefaultActivity = test_build_activity{}.shared();

class test_build_vehicle : public vrp::models::problem::build_vehicle {
public:
  explicit test_build_vehicle() : vrp::models::problem::build_vehicle() {
    id("vehicle1").profile("car").details({DefaultVehicleDetail}).dimensions({DefaultDimension}).costs(DefaultCosts);
  }
};

inline std::shared_ptr<const vrp::models::problem::Vehicle> DefaultVehicle = test_build_vehicle{}.shared();

class test_build_driver : public vrp::models::problem::build_driver {
public:
  explicit test_build_driver() : vrp::models::problem::build_driver() { id("driver").costs({0, 0, 0, 0}); }
};

inline std::shared_ptr<const vrp::models::problem::Driver> DefaultDriver = test_build_driver{}.shared();

class test_build_actor : public vrp::models::solution::build_actor {
public:
  explicit test_build_actor() : vrp::models::solution::build_actor() {
    driver(test_build_driver{}.shared())
      .vehicle(test_build_vehicle{}.shared())
      .detail({DefaultActorLocation, {}, DefaultTimeWindow});
  }
};

inline std::shared_ptr<const vrp::models::solution::Actor> DefaultActor = test_build_actor{}.shared();

class test_build_route : public vrp::models::solution::build_route {
public:
  explicit test_build_route() : vrp::models::solution::build_route() {
    using namespace vrp::models::solution;
    actor(test_build_actor{}.shared())
      .start(test_build_activity{}
               .detail({DefaultActorLocation, 0, DefaultTimeWindow})
               .schedule({0, 0})
               .type(Activity::Type::Start)
               .shared())
      .end(test_build_activity{}
             .detail({DefaultActorLocation, 0, DefaultTimeWindow})
             .schedule({0, 1000})
             .type(Activity::Type::End)
             .shared());
  }
};

inline std::shared_ptr<const vrp::models::problem::Fleet> DefaultFleet =
  std::make_shared<vrp::models::problem::Fleet>(vrp::models::problem::Fleet{}.add(*DefaultDriver).add(*DefaultVehicle));


}  // namespace vrp::test

#pragma clang diagnostic pop
