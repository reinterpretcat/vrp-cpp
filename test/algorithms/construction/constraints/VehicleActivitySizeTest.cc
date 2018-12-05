#include "algorithms/construction/constraints/VehicleActivitySize.hpp"

#include "algorithms/construction/extensions/States.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "test_utils/algorithms/constraints/Helpers.hpp"
#include "test_utils/algorithms/construction/Contexts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::solution;
using namespace vrp::models::problem;
using namespace vrp::test;

namespace {

const auto CurrentKey = VehicleActivitySize<int>::StateKeyCurrent;

Tour::Activity
activity(const std::string& id, Timestamp departure, int size) {
  return test_build_activity{}
    .schedule({0, departure})
    .job(as_job(test_build_service{}.id(id).dimens({{"size", size}}).shared()))
    .shared();
}
}

namespace vrp::test {

SCENARIO("vehicle activity size", "[algorithms][construction][constraints]") {
  GIVEN("fleet with 1 vehicle and service jobs") {
    auto fleet = std::make_shared<Fleet>();
    fleet->add(test_build_vehicle{}.id("v1").dimens({{"size", 10}}).details(asDetails(0, {}, {0, 100})).owned());

    WHEN("accept route with three service activities") {
      auto state = InsertionRouteState{};
      auto route = test_build_route{}.actor(getActor("v1", *fleet)).shared();

      auto [s1, s2, s3, start, expS1, expS2, expS3, end] = GENERATE(table<int, int, int, int, int, int, int, int>({
        {-1, 2, -3, 4, 3, 5, 2, 2},  //
        {1, -2, 3, 2, 3, 1, 4, 4},
        {0, 1, 0, 0, 0, 1, 1, 1},
      }));

      route->tour.add(activity("s1", 1, s1)).add(activity("s2", 2, s2)).add(activity("s3", 3, s3));
      VehicleActivitySize<int>{}.accept(*route, state);

      THEN("has correct load at start") { REQUIRE(state.get<int>(CurrentKey, *route->start).value_or(-1) == start); }

      THEN("has correct load at end") { REQUIRE(state.get<int>(CurrentKey, *route->end).value_or(-1) == end); }

      THEN("has correct current load at each activity") {
        REQUIRE(state.get<int>(CurrentKey, *route->tour.get(0)).value_or(-1) == expS1);
        REQUIRE(state.get<int>(CurrentKey, *route->tour.get(1)).value_or(-1) == expS2);
        REQUIRE(state.get<int>(CurrentKey, *route->tour.get(2)).value_or(-1) == expS3);
      }
    }
  }
}
}
