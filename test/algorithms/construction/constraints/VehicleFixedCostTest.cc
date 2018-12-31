#include "algorithms/construction/constraints/VehicleFixedCost.hpp"

#include "test_utils/algorithms/construction/Contexts.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::problem;

namespace vrp::test {

SCENARIO("vehicle fixed cost", "[algorithms][construction][constraints]") {
  auto fixedCost = VehicleFixedCost{};
  auto fleet = std::make_shared<Fleet>();
  (*fleet)
    .add(test_build_vehicle{}.id("v1").dimens({{"size", 10}}).details(asDetails(0, {}, {0, 100})).owned())
    .add(test_build_vehicle{}.id("v2").dimens({{"size", 10}}).details(asDetails(0, {}, {0, 100})).owned());

  GIVEN("different actors in route context") {
    auto routeCtx =
      test_build_insertion_route_context{}  //
        .actor(getActor("v2", *fleet))
        .route({test_build_route{}.actor(getActor("v1", *fleet)).shared(), std::make_shared<InsertionRouteState>()})
        .owned();

    THEN("returns fixed cost as extra") { REQUIRE(fixedCost.check(routeCtx, DefaultService) == 100); }
  }

  GIVEN("same actors in route context") {
    auto routeCtx =
      test_build_insertion_route_context{}  //
        .actor(getActor("v1", *fleet))
        .route({test_build_route{}.actor(getActor("v1", *fleet)).shared(), std::make_shared<InsertionRouteState>()})
        .owned();

    THEN("returns zero as extra cost") { REQUIRE(fixedCost.check(routeCtx, DefaultService) == 0); }
  }
}
}
