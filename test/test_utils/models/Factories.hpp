#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"
#pragma once

#include "algorithms/construction/InsertionEvaluator.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
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
constexpr models::common::Profile DefaultProfile = 0;

const vrp::models::problem::Service::Detail DefaultJobDetail = {{DefaultJobLocation},
                                                                DefaultDuration,
                                                                {DefaultTimeWindow}};
const vrp::models::problem::Vehicle::Detail DefaultVehicleDetail = {DefaultActorLocation,
                                                                    {DefaultActorLocation},
                                                                    DefaultTimeWindow};

class test_build_service : public vrp::models::problem::build_service {
public:
  explicit test_build_service() : vrp::models::problem::build_service() {
    dimens({{"id", "service"}}).details({DefaultJobDetail});
  }

  test_build_service& id(const std::string& value) {
    service_.dimens["id"] = value;
    return *this;
  }

  template<typename Size>
  test_build_service& demand(const typename algorithms::construction::VehicleActivitySize<Size>::Demand& value) {
    service_.dimens[algorithms::construction::VehicleActivitySize<Size>::DimKeyDemand] = value;
    return *this;
  }

  test_build_service& location(const models::common::Location& value) {
    service_.details.front().location = value;
    return *this;
  }

  test_build_service& duration(const models::common::Duration& value) {
    service_.details.front().duration = value;
    return *this;
  }

  test_build_service& time(const models::common::TimeWindow& value) {
    service_.details.front().times = {value};
    return *this;
  }
};

inline vrp::models::problem::Job DefaultService = vrp::models::problem::as_job(test_build_service{}.shared());

class test_build_sequence : public vrp::models::problem::build_sequence {
  using PermutationFunc = algorithms::construction::InsertionEvaluator::PermutationFunc;

public:
  explicit test_build_sequence() : vrp::models::problem::build_sequence() { dimens({{"id", "service"}}); }

  test_build_sequence& id(const std::string& value) {
    sequence_->dimens["id"] = value;
    return *this;
  }

  test_build_sequence& permutations(std::vector<std::vector<int>> perms) {
    sequence_->dimens["prm"] = std::make_shared<PermutationFunc>([perms](const auto&) {
      return ranges::view::transform(perms, [](const auto& perm) -> const std::vector<int>& { return perm; });
    });

    return *this;
  }

  test_build_sequence& permutations(PermutationFunc permutationFunc) {
    sequence_->dimens["prm"] = std::make_shared<PermutationFunc>(permutationFunc);
    return *this;
  }

  test_build_sequence& diIndex(int index) {
    sequence_->dimens["di"] = index;
    return *this;
  }
};

inline vrp::models::problem::Job DefaultSequence =
  vrp::models::problem::as_job(test_build_sequence{}
                                 .id("sequence")
                                 .service(test_build_service{}.id("s1").location(3).shared())
                                 .service(test_build_service{}.id("s2").location(7).shared())
                                 .shared());

class test_build_activity : public models::solution::build_activity {
public:
  explicit test_build_activity() : models::solution::build_activity() {
    detail({DefaultJobLocation, DefaultDuration, DefaultTimeWindow})
      .schedule({0, 0})
      .service(ranges::get<0>(DefaultService));
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
    dimens({{"id", "vehicle1"}}).profile(DefaultProfile).details({DefaultVehicleDetail}).costs(DefaultCosts);
  }

  test_build_vehicle& id(const std::string& value) {
    vehicle_.dimens["id"] = value;
    return *this;
  }
};

inline std::shared_ptr<const vrp::models::problem::Vehicle> DefaultVehicle = test_build_vehicle{}.shared();

class test_build_driver : public vrp::models::problem::build_driver {
public:
  explicit test_build_driver() : vrp::models::problem::build_driver() {
    dimens({{"id", "driver"}}).costs({0, 0, 0, 0});
  }

  test_build_driver& id(const std::string& value) {
    driver_.dimens["id"] = value;
    return *this;
  }
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
      .start(build_activity{}.detail({DefaultActorLocation, 0, DefaultTimeWindow}).schedule({0, 0}).shared())
      .end(build_activity{}.detail({DefaultActorLocation, 0, DefaultTimeWindow}).schedule({0, 1000}).shared());
  }
};

}  // namespace vrp::test

#pragma clang diagnostic pop
