#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"
#pragma once

#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/solution/Factories.hpp"

namespace vrp::test {

constexpr vrp::models::common::Location DefaultLocation = 0;
constexpr vrp::models::common::TimeWindow DefaultTimeWindow = {0, 1000};
const vrp::models::common::Schedule DefaultSchedule = {5, 10};
const vrp::models::common::Dimension DefaultDimension = {"capacity", 1};

class test_build_service : public vrp::models::problem::build_service {
public:
  explicit test_build_service() : vrp::models::problem::build_service() {
    withId("service").withLocation({DefaultLocation}).withTimes({DefaultTimeWindow}).withDimensions({DefaultDimension});
  }
};

std::shared_ptr<const vrp::models::problem::Service> DefaultService = test_build_service{}.shared();

class test_build_activity : public vrp::models::solution::build_activity {
public:
  explicit test_build_activity() : vrp::models::solution::build_activity() {
    withLocation(static_cast<vrp::models::common::Location>(DefaultLocation))
      .withSchedule(static_cast<vrp::models::common::Schedule>(DefaultSchedule))
      .withJob(DefaultService);
  }
};

std::shared_ptr<vrp::models::solution::Activity> DefaultActivity = test_build_activity{}.shared();

}  // namespace vrp::test

#pragma clang diagnostic pop
