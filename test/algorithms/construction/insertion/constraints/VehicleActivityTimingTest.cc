#include "algorithms/construction/insertion/constraints/VehicleActivityTimingTest.hpp"

//#include "models/costs/ActivityCosts.hpp"
//#include "test_utils/algorithms/construction/Contexts.hpp"
//#include "test_utils/algorithms/construction/Insertions.hpp"
//#include "test_utils/fakes/TestTransportCosts.hpp"
//#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>
#include <utility>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;

using namespace vrp::models::solution;

namespace vrp::test {

SCENARIO("todohtop"
         "",
         "[algorithms][construction][insertion]") {
  GIVEN("tour with two activities") {
    //    auto progress = test_build_insertion_progress{}.owned();
    //    auto prev = test_build_activity{}.location(10).duration(0).schedule({0, 10}).shared();
    //    auto target = test_build_activity{}.location(30).duration(10).shared();
    //    auto next = test_build_activity{}.location(20).duration(0).time({40, 70}).shared();
    //
    //    // old: d(10 + 20) + t(10 + 20 + 20) = 80
    //    // new: d(10 + 10 + 30) + t(20 + 10 + 30) = 110
    //    WHEN("inserting in between new activity with the same actor") {
    //      auto [routeCtx, actCtx] = sameActor(prev, target, next);
    //      routeCtx->route->tour.add(prev).add(next);
    //
    //      THEN("cost for activity is correct") {
    //        auto cost = evaluator.testActivityCosts(routeCtx, actCtx, progress);
    //
    //        REQUIRE(cost == 30);
    //      }
    //    }
  }
}
}
