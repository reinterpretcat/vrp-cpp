#include "algorithms/refinement/extensions/RemoveEmptyTours.hpp"

#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace vrp::models::solution;

namespace vrp::test {

SCENARIO("empty tours removal cleans empty routes from solution", "[algorithms][refinement][extensions]") {
  auto fleet = Fleet{};
  fleet.add(test_build_driver{}.owned())
    .add(test_build_vehicle{}.id("v1").owned())
    .add(test_build_vehicle{}.id("v2").owned());
  auto registry = std::make_shared<Registry>(fleet);

  GIVEN("solution with empty and non empty tours tour") {
    auto actor1 = getActor("v1", fleet);
    auto actor2 = getActor("v2", fleet);
    registry->use(*actor1);
    registry->use(*actor2);

    auto route1 = test_build_route{}.actor(actor1).shared();
    auto route2 = test_build_route{}.actor(actor2).shared();
    route2->tour.add(test_build_activity{}.location(10).shared());

    auto solution = Solution{{}, registry, {route1, route2}, {}};
    WHEN("clean is used") {
      remove_empty_tours{}.operator()(solution);

      THEN("only non empty tour is left") {
        REQUIRE(solution.routes.size() == 1);
        REQUIRE(solution.routes.front() == route2);
      }

      THEN("empty route's actor is released in registry") {
        auto actors = solution.registry->actors() | ranges::to_vector;
        REQUIRE(actors.size() == 1);
        REQUIRE(actors.front()->vehicle->id == "v1");
      }
    }
  }
}
}