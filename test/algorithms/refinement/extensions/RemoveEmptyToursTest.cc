#include "algorithms/refinement/extensions/RemoveEmptyTours.hpp"

#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::algorithms::refinement;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::utils;

namespace vrp::test {

SCENARIO("remove empty tours works", "[algorithms][refinement][extensions]") {
  auto fleet = Fleet{};
  fleet.add(test_build_driver{}.owned())
    .add(test_build_vehicle{}.id("v1").owned())
    .add(test_build_vehicle{}.id("v2").owned());
  auto registry = std::make_shared<Registry>(fleet);

  GIVEN("empty and non empty routes") {
    auto actors = registry->available() | ranges::to_vector;
    auto actor1 = actors.front();
    auto actor2 = actors.back();
    registry->use(actor1);
    registry->use(actor2);


    auto route1 = test_build_route{}.actor(actor1).shared();
    auto route2 = test_build_route{}.actor(actor2).shared();
    route2->tour.insert(test_build_activity{}.location(10).shared(), 1);

    auto ctx = InsertionContext{{}, registry, {}, {}, {}, {{route1, {}}, {route2, {}}}, {}};

    WHEN("remove empty tours") {
      remove_empty_tours{}(ctx);

      THEN("only non empty tour is left") {
        REQUIRE(ctx.routes.size() == 1);
        REQUIRE(ctx.routes.begin()->route->actor == actor2);
      }

      THEN("empty route's actor is released in registry") {
        auto available = ctx.registry->available() | ranges::to_vector;
        REQUIRE(available.size() == 1);
        REQUIRE(available.front() == actor1);
      }
    }
  }
}
}