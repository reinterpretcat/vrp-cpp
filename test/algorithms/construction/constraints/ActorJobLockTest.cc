#include "algorithms/construction/constraints/ActorJobLock.hpp"

#include "models/problem/Fleet.hpp"
#include "models/solution/Registry.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace ranges;

namespace {
const Registry::SharedActor
getActorFromRegistry(const std::string& vehicleId, const Registry& registry) {
  return ranges::front(registry.available() |
                       view::remove_if([&](const auto& a) { return get_vehicle_id{}(*a->vehicle) != vehicleId; }) |
                       to_vector);
}
}

namespace vrp::test {

SCENARIO("actor job lock can manage actor-job locks", "[algorithms][construction][constraints]") {
  auto [locked, used, expected] = GENERATE(table<std::string, std::string, HardRouteConstraint::Result>({
    {"v1", "v1", {}},
    {"v1", "v2", {3}},
  }));

  GIVEN("fleet with 2 vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details(asDetails(0, {}, {0, 100})).owned())
      .add(test_build_vehicle{}.id("v2").details(asDetails(0, {}, {0, 100})).owned());
    auto registry = Registry(*fleet);

    WHEN("has job lock for one actor") {
      auto actorJobLock =
        ActorJobLock{{Lock{[locked = locked](const auto& a) { return get_vehicle_id{}(*a.vehicle) == locked; },
                           {Lock::Detail{Lock::Order::Any, {DefaultService}}}}}};

      THEN("returns expected constraint check") {
        auto result = actorJobLock.hard(
          InsertionRouteContext{std::make_shared<Route>(Route{getActorFromRegistry(used, registry), {}}),
                                std::make_shared<InsertionRouteState>()},
          DefaultService);

        REQUIRE(result == expected);
      }
    }
  }
}

// TODO
}