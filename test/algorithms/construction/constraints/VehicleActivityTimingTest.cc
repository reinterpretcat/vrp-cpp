#include "algorithms/construction/constraints/VehicleActivityTiming.hpp"

#include "algorithms/construction/extensions/States.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "test_utils/algorithms/construction/Contexts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Comparators.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>
#include <utility>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;
using namespace vrp::models::problem;
using namespace vrp::test;
using namespace Catch::Generators;

namespace {

std::string
operationTimeKey(const std::string& id, const Fleet& fleet) {
  return actorSharedKey(VehicleActivityTiming::StateKey, *getActor(id, fleet));
}
}

namespace vrp::test {

SCENARIO("vehicle activity timing checks states", "[algorithms][construction][constraints]") {
  auto createRoute = [](const auto& fleet) {
    auto route = test_build_route{}.actor(getActor("v1", fleet)).shared();
    route->tour
      .add(test_build_activity{}.location(10).shared())  //
      .add(test_build_activity{}.location(20).shared())  //
      .add(test_build_activity{}.location(30).shared());

    return route;
  };

  GIVEN("fleet with 4 vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_vehicle{}.id("v1").details(asDetails(0, {}, {0, 100})).owned())
      .add(test_build_vehicle{}.id("v2").details(asDetails(0, {}, {0, 60})).owned())
      .add(test_build_vehicle{}.id("v3").details(asDetails(40, {}, {0, 100})).owned())
      .add(test_build_vehicle{}.id("v4").details(asDetails(40, {}, {0, 100})).owned());

    WHEN("accept route for first vehicle with three activities") {
      auto state = InsertionRouteState{};
      auto route = createRoute(*fleet);
      VehicleActivityTiming(fleet,
                            std::make_shared<TestTransportCosts>(),  //
                            std::make_shared<ActivityCosts>())
        .accept(*route, state);

      auto [vehicle, activity, time] = GENERATE(table<std::string, size_t, Timestamp>({
        {"v1", 2, 70},
        {"v2", 2, 30},
        {"v3", 2, 90},
        {"v4", 2, 90},
        {"v1", 1, 60},
        {"v2", 1, 20},
        {"v3", 1, 80},
        {"v4", 1, 80},
        {"v1", 0, 50},
        {"v2", 0, 10},
        {"v3", 0, 70},
        {"v4", 0, 70}  //
      }));

      THEN("should update latest operation time") {
        auto result = state.get<Timestamp>(operationTimeKey(vehicle, *fleet), *route->tour.get(activity)).value_or(0);

        REQUIRE(result == time);
      }
    }
  }

  GIVEN("fleet with 6 vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_vehicle{}.id("v1").details(asDetails(0, {}, {0, 100})).owned())
      .add(test_build_vehicle{}.id("v2").details(asDetails(0, {}, {0, 60})).owned())
      .add(test_build_vehicle{}.id("v3").details(asDetails(0, {}, {0, 50})).owned())
      .add(test_build_vehicle{}.id("v4").details(asDetails(0, {}, {0, 10})).owned())
      .add(test_build_vehicle{}.id("v5").details(asDetails(0, {}, {60, 100})).owned())
      .add(test_build_vehicle{}.id("v6").details(asDetails(0, {40}, {0, 40})).owned());

    WHEN("accept and checks route for first vehicle with three activities") {
      auto state = std::make_shared<InsertionRouteState>();
      auto route = createRoute(*fleet);
      auto timing = VehicleActivityTiming(fleet,
                                          std::make_shared<TestTransportCosts>(),  //
                                          std::make_shared<ActivityCosts>());
      timing.accept(*route, *state);

      auto [vehicle, location, departure, prev, next, expected] =
        GENERATE(table<std::string, Location, Timestamp, int, int, HardActivityConstraint::Result>(
          {{"v1", 50, 30, 2, EndActivityIndex, success()},  //
           {"v1", 1000, 30, 2, EndActivityIndex, stop(1)},
           {"v1", 50, 20, 1, 2, success()},
           {"v1", 51, 20, 1, 2, stop(1)},
           {"v2", 40, 30, 2, EndActivityIndex, stop(1)},
           {"v3", 40, 30, 2, EndActivityIndex, fail(1)},
           {"v4", 40, 30, 2, EndActivityIndex, fail(1)},
           {"v5", 40, 90, 2, EndActivityIndex, fail(1)},
           {"v6", 40, 30, 1, 2, fail(1)},
           {"v6", 40, 10, 0, 1, stop(1)},
           {"v6", 40, 30, 2, EndActivityIndex, success()}}));

      THEN("returns fulfilled for insertion at the end") {
        auto routeCtx = test_build_insertion_route_context{}  //
                          .actor(getActor(vehicle, *fleet))
                          .route({route, state})
                          .owned();
        auto actCtx = test_build_insertion_activity_context{}
                        .departure(departure)
                        .prev(getActivity(routeCtx, prev))
                        .target(test_build_activity{}.location(location).shared())
                        .next(getActivity(routeCtx, next))
                        .owned();

        auto result = timing.check(routeCtx, actCtx);

        REQUIRE(result == expected);
      }
    }
  }
}

SCENARIO("vehicle activity timing updates activity schedule", "[algorithms][construction][constraints]") {
  auto fleet = std::make_shared<Fleet>();
  (*fleet)  //
    .add(test_build_vehicle{}.id("v1").details(asDetails(0, 0, {0, 1000})).owned());

  GIVEN("route with two activities with waiting time") {
    auto state = InsertionRouteState{};
    auto route = test_build_route{}.actor(getActor("v1", *fleet)).shared();
    route->tour
      .add(test_build_activity{}.location(10).time({20, 30}).duration(5).shared())  //
      .add(test_build_activity{}.location(20).time({50, 10}).duration(10).shared());

    WHEN("accept route") {
      VehicleActivityTiming(fleet,
                            std::make_shared<TestTransportCosts>(),  //
                            std::make_shared<ActivityCosts>())
        .accept(*route, state);

      THEN("activity schedule are updated") {
        REQUIRE(compare_schedules{}(route->tour.get(0)->schedule, {10, 25}));
        REQUIRE(compare_schedules{}(route->tour.get(1)->schedule, {35, 60}));
      }
    }
  }
}
}
