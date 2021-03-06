#include "algorithms/refinement/extensions/RestoreInsertionContext.hpp"

#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/algorithms/refinement/MatrixRoutes.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::algorithms::refinement;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::utils;

namespace vrp::test {

SCENARIO("restore insertion context skips empty routes", "[algorithms][refinement][extensions]") {
  auto fleet = Fleet{};
  fleet.add(test_build_driver{}.owned())
    .add(test_build_vehicle{}.id("v1").owned())
    .add(test_build_vehicle{}.id("v2").owned());
  auto registry = std::make_shared<Registry>(fleet);

  GIVEN("solution with empty and non empty tours") {
    auto actors = registry->available() | ranges::to_vector;
    auto actor1 = actors.front();
    auto actor2 = actors.back();
    registry->use(actor1);
    registry->use(actor2);

    auto route1 = test_build_route{}.actor(actor1).shared();
    auto route2 = test_build_route{}.actor(actor2).shared();
    route2->tour.insert(test_build_activity{}.location(10).shared(), 1);

    auto refinementCtx = RefinementContext{generate_matrix_routes{}(1, 1).first,
                                           std::make_shared<Random>(),
                                           std::make_shared<std::set<Job, compare_jobs>>(),
                                           {},
                                           0};
    auto solution = Solution{registry, {route1, route2}, {}};
    WHEN("context is restored") {
      auto insertionCtx = restore_insertion_context{}.operator()(refinementCtx, solution);

      THEN("only non empty tour is left") {
        REQUIRE(insertionCtx.solution->routes.size() == 1);
        REQUIRE(insertionCtx.solution->routes.begin()->route->actor == actor2);
      }

      THEN("empty route's actor is released in registry") {
        auto available = insertionCtx.solution->registry->available() | ranges::to_vector;
        REQUIRE(available.size() == 1);
        REQUIRE(available.front() == actor1);
      }
    }
  }
}
}