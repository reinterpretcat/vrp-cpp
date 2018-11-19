#include "algorithms/construction/insertion/constraints/VehicleActivityTiming.hpp"

#include "algorithms/construction/extensions/States.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "test_utils/algorithms/construction/Contexts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>
#include <utility>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;
using namespace vrp::models::problem;

namespace {

std::string
generateKey(const std::string& key, const Vehicle& vehicle) {
  return vehicleKey(key, vehicle);
}

std::string
operationTimeKey(const std::string& id, const Fleet& fleet) {
  return generateKey(VehicleActivityTiming::StateKey, *fleet.vehicle(id));
}
}

namespace vrp::test {

SCENARIO("vehicle activity timing accepts route modifying its state", "[algorithms][construction][insertion]") {
  GIVEN("fleet with 4 vehicles, three of them are unique") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)                                                                //
      .add(test_build_vehicle{}.id("v1").start(0).time({0, 100}).owned())   //
      .add(test_build_vehicle{}.id("v2").start(0).time({0, 60}).owned())    //
      .add(test_build_vehicle{}.id("v3").start(40).time({0, 100}).owned())  //
      .add(test_build_vehicle{}.id("v4").start(40).time({0, 100}).owned());

    WHEN("accept route for first vehicle with three activities") {
      auto state = InsertionRouteState{};
      auto route = test_build_route{}.actor({fleet->vehicle("v1"), DefaultDriver}).owned();
      route.tour
        .add(test_build_activity{}.location(10).shared())  //
        .add(test_build_activity{}.location(20).shared())  //
        .add(test_build_activity{}.location(30).shared());
      VehicleActivityTiming(fleet,
                            std::make_shared<TestTransportCosts>(),  //
                            std::make_shared<ActivityCosts>())
        .accept(route, state);

      THEN("should update latest operation time of third activity") {
        auto time = state.get<Timestamp>(operationTimeKey("v1", *fleet), *route.tour.get(2)).value_or(0);

        REQUIRE(time == 70);
      }
    }
  }
}
}
